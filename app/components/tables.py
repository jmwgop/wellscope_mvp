# app/components/tables.py

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from app.utils.formatting import (
    format_number, format_percentage, format_cluster_label, 
    format_similarity_score, get_similarity_color, format_api_display
)
from app.services.io import save_dataframe_csv

def render_well_data_table(joined_df: pd.DataFrame, 
                          labels_df: Optional[pd.DataFrame] = None,
                          scores_df: Optional[pd.DataFrame] = None,
                          page_size: int = 20) -> None:
    """
    Render paginated table of well data with clustering results.
    
    Args:
        joined_df: Main dataset with well information
        labels_df: Optional cluster labels
        scores_df: Optional similarity scores
        page_size: Number of rows per page
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    if joined_df is None or len(joined_df) == 0:
        st.warning("No well data available")
        return
    
    st.subheader("🗂️ Well Data Table")
    
    # Merge with clustering results if available
    display_df = joined_df.copy()
    
    if labels_df is not None:
        api_col = [col for col in joined_df.columns if 'API' in col.upper()][0]
        label_api_col = [col for col in labels_df.columns if 'API' in col.upper()][0]
        
        display_df = display_df.merge(
            labels_df[[label_api_col, 'label', 'cluster_size']], 
            left_on=api_col, 
            right_on=label_api_col, 
            how='left',
            suffixes=('', '_cluster')
        )
        display_df['cluster_name'] = display_df['label'].apply(
            lambda x: format_cluster_label(x) if pd.notna(x) else 'Unknown'
        )
    
    if scores_df is not None and labels_df is not None:
        scores_api_col = [col for col in scores_df.columns if 'API' in col.upper()][0]
        display_df = display_df.merge(
            scores_df[[scores_api_col, 'similarity']], 
            left_on=api_col, 
            right_on=scores_api_col, 
            how='left',
            suffixes=('', '_sim')
        )
    
    # Column selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Filter controls
        filter_col1, filter_col2 = st.columns(2)
        
        with filter_col1:
            # Formation filter
            if 'Target Formation' in display_df.columns:
                formations = sorted(display_df['Target Formation'].dropna().unique())
                selected_formations = st.multiselect(
                    "Filter by Formation",
                    options=formations,
                    default=None,
                    key="table_formation_filter"
                )
                if selected_formations:
                    display_df = display_df[display_df['Target Formation'].isin(selected_formations)]
        
        with filter_col2:
            # Cluster filter
            if 'cluster_name' in display_df.columns:
                clusters = sorted(display_df['cluster_name'].dropna().unique())
                selected_clusters = st.multiselect(
                    "Filter by Cluster",
                    options=clusters,
                    default=None,
                    key="table_cluster_filter"
                )
                if selected_clusters:
                    display_df = display_df[display_df['cluster_name'].isin(selected_clusters)]
    
    with col2:
        # Download button
        if len(display_df) > 0:
            csv_data = save_dataframe_csv(display_df)
            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name="well_data.csv",
                mime="text/csv",
                key="download_well_data"
            )
    
    # Select key columns for display
    display_columns = []
    
    # Always include API
    api_col = [col for col in display_df.columns if 'API' in col.upper()][0]
    display_columns.append(api_col)
    
    # Add important columns if they exist
    important_cols = [
        'Target Formation', 'Operator (Reported)', 'Completion Date',
        'DI Lateral Length', 'Horizontal Length'
    ]
    
    for col in important_cols:
        if col in display_df.columns:
            display_columns.append(col)
    
    # Add clustering results if available
    if 'cluster_name' in display_df.columns:
        display_columns.extend(['cluster_name', 'cluster_size'])
    
    if 'similarity' in display_df.columns:
        display_columns.append('similarity')
    
    # Limit to first N columns to avoid overcrowding
    display_columns = display_columns[:8]
    
    # Format display DataFrame
    formatted_df = display_df[display_columns].copy()
    
    # Format specific columns
    if api_col in formatted_df.columns:
        formatted_df[api_col] = formatted_df[api_col].apply(
            lambda x: format_api_display(str(x)) if pd.notna(x) else 'N/A'
        )
    
    if 'Completion Date' in formatted_df.columns:
        formatted_df['Completion Date'] = pd.to_datetime(
            formatted_df['Completion Date'], errors='coerce'
        ).dt.strftime('%Y-%m-%d')
    
    if 'DI Lateral Length' in formatted_df.columns:
        formatted_df['DI Lateral Length'] = formatted_df['DI Lateral Length'].apply(
            lambda x: format_number(x, 0) + ' ft' if pd.notna(x) else 'N/A'
        )
    
    if 'similarity' in formatted_df.columns:
        formatted_df['similarity'] = formatted_df['similarity'].apply(format_similarity_score)
    
    # Pagination
    total_rows = len(formatted_df)
    total_pages = (total_rows - 1) // page_size + 1 if total_rows > 0 else 1
    
    if total_pages > 1:
        page = st.selectbox(
            f"Page (showing {page_size} of {total_rows} wells)",
            options=list(range(1, total_pages + 1)),
            key="well_table_page"
        )
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_rows)
        page_df = formatted_df.iloc[start_idx:end_idx]
    else:
        page_df = formatted_df
    
    # Display table
    st.dataframe(
        page_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary stats
    if total_rows > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Wells", f"{total_rows:,}")
        
        with col2:
            if 'cluster_name' in display_df.columns:
                n_clusters = display_df['cluster_name'].nunique()
                st.metric("Clusters", n_clusters)
        
        with col3:
            if 'Target Formation' in display_df.columns:
                n_formations = display_df['Target Formation'].nunique()
                st.metric("Formations", n_formations)
        
        with col4:
            if 'similarity' in display_df.columns:
                avg_sim = display_df['similarity'].mean()
                st.metric("Avg Similarity", format_similarity_score(avg_sim))

def render_cluster_summary_table(labels_df: pd.DataFrame, 
                                scores_df: Optional[pd.DataFrame] = None) -> None:
    """
    Render cluster summary statistics table.
    
    Args:
        labels_df: DataFrame with cluster labels
        scores_df: Optional similarity scores for additional stats
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    if labels_df is None or len(labels_df) == 0:
        st.warning("No cluster data available")
        return
    
    st.subheader("🎯 Cluster Summary")
    
    # Calculate cluster statistics
    cluster_stats = labels_df.groupby('label').agg({
        'cluster_size': 'first'  # cluster_size is the same for all wells in cluster
    })
    cluster_stats['well_count'] = labels_df.groupby('label').size()
    
    # Add similarity stats if available
    if scores_df is not None:
        api_col = [col for col in scores_df.columns if 'API' in col.upper()][0]
        label_api_col = [col for col in labels_df.columns if 'API' in col.upper()][0]
        
        # Merge to get cluster labels with similarity scores
        scores_with_labels = scores_df.merge(
            labels_df[[label_api_col, 'label']], 
            left_on=api_col, 
            right_on=label_api_col, 
            how='left',
            suffixes=('', '_from_labels')
        )
        
        # Use the label column from the merge
        label_col = 'label' if 'label' in scores_with_labels.columns else 'label_from_labels'
        
        if label_col in scores_with_labels.columns:
            sim_stats = scores_with_labels.groupby(label_col)['similarity'].agg([
                'mean', 'std', 'min', 'max'
            ]).round(3)
        else:
            # Fallback if no label column
            sim_stats = pd.DataFrame()
        
        cluster_stats = cluster_stats.join(sim_stats, how='left')
    
    # Format the display
    display_stats = cluster_stats.reset_index()
    display_stats['cluster_name'] = display_stats['label'].apply(format_cluster_label)
    display_stats['well_count_formatted'] = display_stats['well_count'].apply(
        lambda x: f"{x:,}"
    )
    
    # Reorder and select columns
    display_columns = ['cluster_name', 'well_count_formatted']
    column_names = ['Cluster', 'Wells']
    
    if 'mean' in display_stats.columns:
        display_stats['mean_similarity'] = display_stats['mean'].apply(format_similarity_score)
        display_stats['std_similarity'] = display_stats['std'].apply(
            lambda x: f"{x:.3f}" if pd.notna(x) else 'N/A'
        )
        display_columns.extend(['mean_similarity', 'std_similarity'])
        column_names.extend(['Avg Similarity', 'Std Dev'])
    
    # Rename columns for display
    display_df = display_stats[display_columns].copy()
    display_df.columns = column_names
    
    # Display table
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )
    
    # Summary metrics
    total_wells = cluster_stats['well_count'].sum()
    n_clusters = len(cluster_stats[cluster_stats.index != -1])  # Exclude noise
    noise_wells = cluster_stats.get(-1, {}).get('well_count', 0)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Valid Clusters", n_clusters)
    
    with col2:
        clustered_wells = total_wells - noise_wells
        st.metric("Clustered Wells", f"{clustered_wells:,}")
    
    with col3:
        if noise_wells > 0:
            st.metric("Noise Wells", f"{noise_wells:,}")

def render_similarity_ranking_table(scores_df: pd.DataFrame, 
                                  joined_df: Optional[pd.DataFrame] = None,
                                  top_n: int = 20) -> None:
    """
    Render table showing wells ranked by similarity scores.
    
    Args:
        scores_df: DataFrame with similarity scores
        joined_df: Optional well data for additional context
        top_n: Number of top wells to show
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    if scores_df is None or len(scores_df) == 0:
        st.warning("No similarity score data available")
        return
    
    st.subheader("🏆 Top Similar Wells")
    
    # Sort by similarity score
    top_wells = scores_df.nlargest(top_n, 'similarity').copy()
    
    # Add well information if available
    if joined_df is not None:
        api_col = [col for col in scores_df.columns if 'API' in col.upper()][0]
        well_api_col = [col for col in joined_df.columns if 'API' in col.upper()][0]
        
        # Select key columns from well data
        well_cols = [well_api_col]
        for col in ['Target Formation', 'Operator (Reported)', 'Completion Date']:
            if col in joined_df.columns:
                well_cols.append(col)
        
        top_wells = top_wells.merge(
            joined_df[well_cols],
            left_on=api_col,
            right_on=well_api_col,
            how='left'
        )
    
    # Format for display
    display_df = top_wells.copy()
    
    # Format API
    api_col = [col for col in display_df.columns if 'API' in col.upper()][0]
    display_df[api_col] = display_df[api_col].apply(
        lambda x: format_api_display(str(x)) if pd.notna(x) else 'N/A'
    )
    
    # Format similarity score
    display_df['similarity_formatted'] = display_df['similarity'].apply(format_similarity_score)
    
    # Format cluster label
    if 'label' in display_df.columns:
        display_df['cluster_formatted'] = display_df['label'].apply(format_cluster_label)
    
    # Select display columns
    display_columns = [api_col, 'similarity_formatted']
    column_names = ['Well API', 'Similarity Score']
    
    if 'cluster_formatted' in display_df.columns:
        display_columns.append('cluster_formatted')
        column_names.append('Cluster')
    
    for col in ['Target Formation', 'Operator (Reported)']:
        if col in display_df.columns:
            display_columns.append(col)
            column_names.append(col.replace(' (Reported)', ''))
    
    # Create final display DataFrame
    final_df = display_df[display_columns].copy()
    final_df.columns = column_names
    
    # Add ranking
    final_df.insert(0, 'Rank', range(1, len(final_df) + 1))
    
    # Display table
    st.dataframe(
        final_df,
        use_container_width=True,
        hide_index=True
    )

def render_export_options(pipeline_results: Dict[str, Any]) -> None:
    """
    Render data export options.
    
    Args:
        pipeline_results: Complete pipeline results
    """
    if not STREAMLIT_AVAILABLE:
        return
    
    st.subheader("💾 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Export cluster results
        if 'labels_df' in pipeline_results and pipeline_results['labels_df'] is not None:
            labels_csv = save_dataframe_csv(pipeline_results['labels_df'])
            st.download_button(
                label="📊 Cluster Labels",
                data=labels_csv,
                file_name="cluster_labels.csv",
                mime="text/csv",
                key="download_labels"
            )
    
    with col2:
        # Export similarity scores
        if 'scores_df' in pipeline_results and pipeline_results['scores_df'] is not None:
            scores_csv = save_dataframe_csv(pipeline_results['scores_df'])
            st.download_button(
                label="📏 Similarity Scores",
                data=scores_csv,
                file_name="similarity_scores.csv",
                mime="text/csv",
                key="download_scores"
            )
    
    with col3:
        # Export 2D coordinates
        if 'coords_df' in pipeline_results and pipeline_results['coords_df'] is not None:
            coords_csv = save_dataframe_csv(pipeline_results['coords_df'])
            st.download_button(
                label="🗺️ 2D Coordinates",
                data=coords_csv,
                file_name="umap_coordinates.csv",
                mime="text/csv",
                key="download_coords"
            )

def create_analysis_summary_table(pipeline_results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a comprehensive analysis summary table.
    
    Args:
        pipeline_results: Complete pipeline results
        
    Returns:
        Summary DataFrame
    """
    summary_data = []
    
    # Join statistics
    if 'join_stats' in pipeline_results:
        stats = pipeline_results['join_stats']
        summary_data.extend([
            ['Data Integration', 'Headers Loaded', format_number(stats.get('headers_rows', 0), 0)],
            ['Data Integration', 'Monthly Records', format_number(stats.get('monthly_rows', 0), 0)],
            ['Data Integration', 'Joined Records', format_number(stats.get('joined_rows', 0), 0)],
            ['Data Integration', 'Matched Wells', format_number(stats.get('matched_api14', 0), 0)]
        ])
    
    # Filter statistics
    if 'filter_stats' in pipeline_results and 'stats' in pipeline_results['filter_stats']:
        stats = pipeline_results['filter_stats']['stats']
        summary_data.extend([
            ['Filtering', 'Input Wells', format_number(stats.get('input_rows', 0), 0)],
            ['Filtering', 'Filtered Wells', format_number(stats.get('output_rows', 0), 0)],
            ['Filtering', 'Retention Rate', format_percentage(stats.get('kept_fraction', 0))]
        ])
    
    # Clustering statistics
    if 'cluster_meta' in pipeline_results:
        stats = pipeline_results['cluster_meta']
        summary_data.extend([
            ['Clustering', 'Algorithm', 'HDBSCAN' if stats.get('algorithm') == 'hdbscan' else 'DBSCAN'],
            ['Clustering', 'Clusters Found', str(stats.get('n_clusters', 0))],
            ['Clustering', 'Noise Wells', str(stats.get('n_noise', 0))]
        ])
    
    # Similarity statistics
    if 'similarity_meta' in pipeline_results:
        stats = pipeline_results['similarity_meta']
        summary_data.extend([
            ['Similarity', 'Wells Scored', str(stats.get('n_scored', 0))],
            ['Similarity', 'Mean Score', format_similarity_score(stats.get('mean_similarity', 0))]
        ])
    
    if summary_data:
        return pd.DataFrame(summary_data, columns=['Category', 'Metric', 'Value'])
    else:
        return pd.DataFrame(columns=['Category', 'Metric', 'Value'])