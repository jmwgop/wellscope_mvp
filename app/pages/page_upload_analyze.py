# app/pages/page_upload_analyze.py

from __future__ import annotations
from typing import Dict, Any, Optional
import time
import traceback

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import pandas as pd

# Import all our components
from app.components.upload_panel import render_upload_panel
from app.components.filter_panel import render_filter_panel
from app.components.cluster_controls import render_all_controls
from app.components.plots import render_interactive_plots
from app.components.tables import render_well_data_table, render_cluster_summary_table, render_similarity_ranking_table, render_export_options

# Import services
from app.services.pipeline_driver import run_complete_pipeline, run_configurable_pipeline
from app.services.caching import cached_run_pipeline, clear_pipeline_cache, get_cache_info
from app.state.session import get_session_value, set_session_value

# Import utilities
from app.utils.formatting import format_duration, format_dataframe_summary, format_pipeline_stats
from app.utils.validation import validate_uploaded_data

def _get_session():
    """Get session state, handling cases where Streamlit is not available."""
    try:
        import streamlit as st
        return st.session_state
    except (ImportError, AttributeError):
        return {}

def render_page():
    """
    Main page rendering function that orchestrates the entire analysis workflow.
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_page_render()

    try:
        st.set_page_config(
            page_title="WellScope MVP - Well Analysis",
            page_icon="🛢️",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Header
        st.title("🛢️ WellScope MVP - Well Similarity Analysis")
        st.markdown("Upload well data and discover similar production patterns using machine learning.")
        
        # Create main layout
        main_col, sidebar_col = st.columns([3, 1])
        
        with sidebar_col:
            render_sidebar()
        
        with main_col:
            render_main_workflow()
            
        return {'page_rendered': True, 'streamlit_available': True}
        
    except Exception:
        # If Streamlit fails (e.g., in testing), return mock
        return _mock_page_render()

def render_sidebar():
    """Render the sidebar with analysis controls and status."""
    if not STREAMLIT_AVAILABLE:
        return
        
    st.sidebar.header("⚙️ Analysis Settings")
    
    # Get session state
    session = _get_session()
    
    # Show current data status
    headers_df = get_session_value(session, 'headers_df')
    monthly_df = get_session_value(session, 'monthly_df')
    joined_df = get_session_value(session, 'joined_df')
    
    if headers_df is not None and monthly_df is not None:
        st.sidebar.success("✅ Data uploaded successfully")
        
        with st.sidebar.expander("📊 Data Summary", expanded=False):
            st.write("**Headers:**", format_dataframe_summary(headers_df)['rows'], "wells")
            st.write("**Monthly:**", format_dataframe_summary(monthly_df)['rows'], "records")
            if joined_df is not None:
                st.write("**Joined:**", format_dataframe_summary(joined_df)['rows'], "wells")
    else:
        st.sidebar.info("📁 Upload data to begin analysis")
    
    # Show analysis controls if data is available
    if joined_df is not None and len(joined_df) > 0:
        st.sidebar.divider()
        
        # Get filtered data for intelligent controls
        filters_cfg = get_session_value(session, 'filters_cfg', {})
        filtered_df = None
        if filters_cfg:
            try:
                from wellscope_mvp.pipeline.filter_inputs import apply_filters, FilterConfig
                filter_config = FilterConfig(**filters_cfg)
                filter_result = apply_filters(joined_df, filter_config)
                filtered_df = filter_result['filtered']
            except Exception:
                # If filtering fails, use original data
                filtered_df = joined_df
        else:
            filtered_df = joined_df
        
        # Analysis controls (now data-aware with filter integration)
        configs, configs_valid, config_errors = render_all_controls(filtered_df, filters_cfg)
        set_session_value(session, 'analysis_configs', configs)
        set_session_value(session, 'configs_valid', configs_valid)
        
        # Show run analysis button
        if configs_valid:
            if st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True):
                set_session_value(session, 'run_analysis', True)
        else:
            st.sidebar.error(f"❌ Fix {len(config_errors)} configuration issues first")

def render_main_workflow():
    """Render the main workflow steps."""
    if not STREAMLIT_AVAILABLE:
        return
    
    # Get session state
    session = _get_session()
    
    # Step 1: Upload Panel
    st.header("📁 Step 1: Upload Data")
    upload_results = render_upload_panel()
    
    if upload_results.get('files_uploaded'):
        headers_df = upload_results.get('headers_df')
        monthly_df = upload_results.get('monthly_df')
        
        # Validate uploaded data
        validation_errors = validate_uploaded_data(headers_df, monthly_df)
        if validation_errors:
            st.error("❌ Data validation failed:")
            for error in validation_errors:
                st.error(f"• {error}")
            return
        
        # Store in session
        set_session_value(session, 'headers_df', headers_df)
        set_session_value(session, 'monthly_df', monthly_df)
        
        # Auto-join data for filtering
        try:
            from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
            joined_df, join_stats = join_headers_and_monthly(headers_df, monthly_df)
            set_session_value(session, 'joined_df', joined_df)
            set_session_value(session, 'join_stats', join_stats)
            
            st.success(f"✅ Successfully joined {len(joined_df):,} wells")
            
            # Show join statistics
            with st.expander("📊 Data Integration Summary", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Headers", f"{join_stats.get('headers_rows', 0):,}")
                with col2:
                    st.metric("Monthly Records", f"{join_stats.get('monthly_rows', 0):,}")
                with col3:
                    st.metric("Joined Wells", f"{join_stats.get('joined_rows', 0):,}")
                with col4:
                    match_rate = join_stats.get('matched_api14', 0) / max(join_stats.get('headers_rows', 1), 1) * 100
                    st.metric("Match Rate", f"{match_rate:.1f}%")
                    
        except Exception as e:
            st.error(f"❌ Failed to join data: {str(e)}")
            return
    
    # Get current session data
    joined_df = get_session_value(session, 'joined_df')
    if joined_df is None or len(joined_df) == 0:
        st.info("👆 Upload data files to continue")
        return
    
    st.divider()
    
    # Step 2: Filtering
    st.header("🔍 Step 2: Filter Data")
    filters_cfg = render_filter_panel(joined_df)
    set_session_value(session, 'filters_cfg', filters_cfg)
    
    st.divider()
    
    # Step 3: Analysis Configuration (handled in sidebar)
    st.header("⚙️ Step 3: Configure Analysis")
    configs_valid = get_session_value(session, 'configs_valid', False)
    analysis_configs = get_session_value(session, 'analysis_configs', {})
    
    if not configs_valid:
        st.info("👈 Configure analysis parameters in the sidebar")
        return
    else:
        st.success("✅ Analysis configuration valid")
        
        # Show configuration summary
        with st.expander("📋 Configuration Summary", expanded=False):
            if 'vector' in analysis_configs:
                vector_cfg = analysis_configs['vector']
                st.write(f"**Vector:** {vector_cfg.months} months of {vector_cfg.stream} production")
            if 'cluster' in analysis_configs:
                cluster_cfg = analysis_configs['cluster']
                algo = "HDBSCAN" if cluster_cfg.use_hdbscan else "DBSCAN"
                st.write(f"**Clustering:** {algo} with min cluster size {cluster_cfg.min_cluster_size}")
    
    st.divider()
    
    # Step 4: Run Analysis
    st.header("🚀 Step 4: Run Analysis")
    
    # Check if analysis should run
    run_analysis = get_session_value(session, 'run_analysis', False)
    pipeline_results = get_session_value(session, 'pipeline_results')
    
    if run_analysis or st.button("Run Full Analysis", type="primary"):
        set_session_value(session, 'run_analysis', False)  # Reset flag
        
        with st.spinner("Running ML pipeline analysis..."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Prepare configurations
                filters_cfg = get_session_value(session, 'filters_cfg', {})
                analysis_configs = get_session_value(session, 'analysis_configs', {})
                
                status_text.text("🔄 Starting pipeline...")
                progress_bar.progress(10)
                
                # Run pipeline (caching is handled internally)
                start_time = time.time()
                
                status_text.text("🔄 Running configurable pipeline...")
                progress_bar.progress(30)
                
                pipeline_results = run_configurable_pipeline(
                    headers_df=get_session_value(session, 'headers_df'),
                    monthly_df=get_session_value(session, 'monthly_df'),
                    filters_cfg=filters_cfg,
                    vector_cfg=analysis_configs.get('vector'),
                    cluster_cfg=analysis_configs.get('cluster'),
                    projection_cfg=analysis_configs.get('projection')
                )
                
                progress_bar.progress(100)
                
                duration = time.time() - start_time
                st.success(f"🎉 Analysis completed in {format_duration(duration)}!")
                
                # Store results
                set_session_value(session, 'pipeline_results', pipeline_results)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.expander("🐛 Error Details").code(traceback.format_exc())
                return
    
    # Step 5: Show Results
    if pipeline_results:
        st.divider()
        st.header("📊 Step 5: Analysis Results")
        render_analysis_results(pipeline_results)

def render_analysis_results(pipeline_results: Dict[str, Any]):
    """Render the analysis results with plots and tables."""
    if not STREAMLIT_AVAILABLE:
        return
    
    # Get key DataFrames
    labels_df = pipeline_results.get('labels_df')
    scores_df = pipeline_results.get('scores_df')
    coords_df = pipeline_results.get('coords_df')
    joined_df = pipeline_results.get('joined_df')
    
    # Summary metrics
    st.subheader("📈 Analysis Summary")
    
    if labels_df is not None:
        unique_clusters = labels_df['label'].unique()
        n_clusters = len([c for c in unique_clusters if c != -1])  # Exclude noise
        n_noise = len(labels_df[labels_df['label'] == -1])
        n_wells = len(labels_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Wells", f"{n_wells:,}")
        with col2:
            st.metric("Clusters Found", f"{n_clusters}")
        with col3:
            st.metric("Wells in Clusters", f"{n_wells - n_noise:,}")
        with col4:
            if scores_df is not None and 'similarity' in scores_df.columns:
                avg_sim = scores_df['similarity'].mean()
                st.metric("Avg Similarity", f"{avg_sim:.2f}")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Cluster Map", "📊 Data Tables", "📈 Charts", "💾 Export"])
    
    with tab1:
        st.subheader("Interactive Cluster Visualization")
        render_interactive_plots(pipeline_results)
    
    with tab2:
        st.subheader("Analysis Data Tables")
        
        # Well data table
        if joined_df is not None:
            st.write("**Well Data with Cluster Assignments**")
            render_well_data_table(joined_df, labels_df, scores_df, page_size=20)
        
        st.divider()
        
        # Cluster summary table
        if labels_df is not None:
            st.write("**Cluster Summary Statistics**")
            render_cluster_summary_table(labels_df, scores_df)
        
        st.divider()
        
        # Similarity ranking table
        if scores_df is not None:
            st.write("**Top Similar Wells**")
            render_similarity_ranking_table(scores_df, joined_df, top_n=50)
    
    with tab3:
        st.subheader("Additional Analysis Charts")
        st.info("🚧 Additional charts coming soon - production curves, similarity distributions, etc.")
    
    with tab4:
        st.subheader("Export Analysis Results")
        render_export_options(pipeline_results)

def _generate_cache_key(joined_df: pd.DataFrame, filters_cfg: Dict, analysis_configs: Dict) -> str:
    """Generate a cache key based on data and configuration."""
    import hashlib
    import json
    
    # Create a deterministic string from inputs
    key_parts = [
        f"data_shape_{joined_df.shape}",
        f"data_hash_{hash(tuple(joined_df.columns))}",
        f"filters_{json.dumps(filters_cfg, sort_keys=True)}",
        f"configs_{json.dumps(str(analysis_configs), sort_keys=True)}"
    ]
    
    key_string = "_".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()

def _mock_page_render():
    """Mock page render for testing environments."""
    return {
        'page_rendered': True,
        'streamlit_available': False
    }

if __name__ == "__main__":
    render_page()