# app/components/smart_analysis_config.py

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from app.utils.data_analyzer import analyze_filtered_data, estimate_processing_time
from app.utils.smart_recommendations import generate_smart_recommendations, format_confidence_display
from app.config.ui_defaults import DEFAULTS


def render_smart_analysis_config(filtered_df: pd.DataFrame, 
                                filters_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Render simplified, data-driven analysis configuration.
    
    This replaces the complex clustering parameter interface with smart recommendations
    and simple user controls that non-ML users can understand.
    
    Args:
        filtered_df: Filtered dataset from Step 2
        filters_cfg: Filter configuration for context
        
    Returns:
        (smart_configs, is_valid) - Contains user preferences converted to technical configs
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_smart_config()
    
    # Step 1: Immediate Data Analysis
    st.subheader("📊 Data Analysis Results")
    
    with st.spinner("Analyzing your filtered data..."):
        data_analysis = analyze_filtered_data(filtered_df, filters_cfg)
    
    # Display data analysis in user-friendly format
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Wells to Analyze", f"{data_analysis['n_wells']:,}")
    with col2:
        production_type = data_analysis.get('production_type', 'Mixed')
        st.metric("Production Type", production_type)
    with col3:
        similarity_pct = data_analysis.get('similarity_mean', 0.0) * 100
        st.metric("Well Similarity", f"{similarity_pct:.0f}%")
    with col4:
        confidence_level = format_confidence_display(data_analysis.get('confidence', 0.5))
        st.metric("Analysis Confidence", confidence_level)
    
    # Step 2: Smart Recommendations
    recommendations = generate_smart_recommendations(data_analysis)
    
    # Show key insights
    if recommendations.get('insights'):
        st.success("🔍 **Key Insights:** " + " • ".join(recommendations['insights']))
    
    if recommendations.get('production_optimization'):
        opt_msg = recommendations['production_optimization']
        st.info(f"🛢️ **Production Data Optimization:** {opt_msg}")
    
    if recommendations.get('warnings'):
        st.warning("⚠️ **Consider:** " + " • ".join(recommendations['warnings']))
    
    st.divider()
    
    # Step 3: Smart Recommendations Display
    st.subheader("🎯 Recommended Analysis")
    
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.metric("Recommended Clusters", recommendations['recommended_clusters'])
        st.metric("Expected Group Size", f"{recommendations['expected_group_size_min']}-{recommendations['expected_group_size_max']} wells")
    
    with rec_col2:
        st.metric("Optimal History Length", f"{recommendations['optimal_months']} months")
        st.metric("Processing Time", recommendations['estimated_time'])
    
    # Technical details in expander (optional)
    with st.expander("🔧 Technical Details", expanded=False):
        st.write(f"**Algorithm:** {recommendations['algorithm_choice']}")
        st.write(f"**Optimization:** {recommendations['optimization_reason']}")
        st.write(f"**Similarity Method:** {recommendations['similarity_method']}")
    
    st.divider()
    
    # Step 4: Simple User Controls
    st.subheader("⚙️ Adjust Your Preferences")
    st.write("Fine-tune the analysis based on your specific needs:")
    
    # User preference controls
    pref_col1, pref_col2 = st.columns(2)
    
    with pref_col1:
        # Cluster count preference
        min_clusters = max(2, recommendations['recommended_clusters'] - 5)
        max_clusters = min(20, recommendations['recommended_clusters'] + 5)
        default_clusters = recommendations['recommended_clusters']
        
        user_clusters = st.slider(
            "Number of Well Groups",
            min_value=min_clusters,
            max_value=max_clusters,
            value=default_clusters,
            step=1,
            help=f"Recommended: {default_clusters} groups based on your data similarity",
            key="smart_cluster_count"
        )
        
        # Group size preference
        group_size_pref = st.select_slider(
            "Preferred Group Size",
            options=["smaller", "balanced", "larger"],
            value="balanced",
            format_func=lambda x: {
                'smaller': 'Smaller Groups (more specific)',
                'balanced': 'Balanced Groups (recommended)',
                'larger': 'Larger Groups (broader patterns)'
            }[x],
            help="Smaller groups find more specific patterns, larger groups find broader similarities",
            key="smart_group_size"
        )
    
    with pref_col2:
        # Production history length
        min_months = max(6, recommendations['optimal_months'] - 6)
        max_months = min(36, recommendations['optimal_months'] + 12)
        default_months = recommendations['optimal_months']
        
        user_months = st.slider(
            "Production History Length",
            min_value=min_months,
            max_value=max_months,
            value=default_months,
            step=6,
            help=f"Recommended: {default_months} months based on your data maturity",
            key="smart_months"
        )
        
        # Analysis focus
        analysis_focus = st.radio(
            "Analysis Focus",
            options=['oil', 'gas', 'combined'],
            format_func=lambda x: {
                'oil': '🛢️ Oil Production',
                'gas': '⛽ Gas Production', 
                'combined': '📊 Combined (BOE)'
            }[x],
            index=0 if recommendations.get('production_stream') == 'oil' else 
                  1 if recommendations.get('production_stream') == 'gas' else 2,
            help="Focus analysis on specific production stream",
            key="smart_analysis_focus"
        )
    
    st.divider()
    
    # Step 5: Real-time Preview
    st.subheader("📋 Final Configuration Preview")
    
    # Calculate updated recommendations based on user choices
    updated_analysis = _update_analysis_with_user_preferences(
        data_analysis, recommendations, 
        user_clusters, group_size_pref, user_months, analysis_focus
    )
    
    # Show preview
    preview_col1, preview_col2, preview_col3 = st.columns(3)
    
    with preview_col1:
        st.info(f"**Will Create:** ~{user_clusters} distinct well groups")
    with preview_col2:
        group_size_range = _calculate_expected_group_size(data_analysis['n_wells'], user_clusters, group_size_pref)
        st.info(f"**Group Sizes:** {group_size_range['min']}-{group_size_range['max']} wells each")
    with preview_col3:
        processing_time = estimate_processing_time(data_analysis['n_wells'], user_months)
        st.info(f"**Processing Time:** ~{processing_time}")
    
    # Technical configuration summary
    with st.expander("📋 Technical Configuration (Auto-Generated)", expanded=False):
        tech_config = updated_analysis['technical_config']
        st.json(tech_config)
    
    # Build final configuration
    smart_config = {
        'user_preferences': {
            'cluster_count': user_clusters,
            'group_size_preference': group_size_pref,
            'months': user_months,
            'analysis_focus': analysis_focus
        },
        'data_analysis': data_analysis,
        'recommendations': updated_analysis,
        'technical_configs': updated_analysis['technical_config']
    }
    
    # Configuration is always valid in smart mode (we handle edge cases automatically)
    is_valid = True
    
    # Success message
    st.success("✅ Smart analysis configuration complete! Ready to discover well patterns.")
    
    return smart_config, is_valid


def _update_analysis_with_user_preferences(data_analysis: Dict[str, Any],
                                           base_recommendations: Dict[str, Any],
                                           user_clusters: int,
                                           group_size_pref: str,
                                           user_months: int,
                                           analysis_focus: str) -> Dict[str, Any]:
    """Update analysis and technical config based on user preferences."""
    
    from app.utils.smart_recommendations import convert_to_technical_config
    
    # Create updated preferences
    user_prefs = {
        'target_clusters': user_clusters,
        'group_size_preference': group_size_pref,
        'months': user_months,
        'production_stream': analysis_focus
    }
    
    # Convert to technical configuration
    technical_config = convert_to_technical_config(data_analysis, base_recommendations, user_prefs)
    
    # Update recommendations
    updated_recommendations = base_recommendations.copy()
    updated_recommendations.update({
        'final_clusters': user_clusters,
        'final_months': user_months,
        'final_stream': analysis_focus,
        'technical_config': technical_config
    })
    
    return updated_recommendations


def _calculate_expected_group_size(n_wells: int, n_clusters: int, size_pref: str) -> Dict[str, int]:
    """Calculate expected group size range based on preferences."""
    avg_size = n_wells // n_clusters
    
    if size_pref == 'smaller':
        # More variation, smaller average
        min_size = max(2, int(avg_size * 0.3))
        max_size = int(avg_size * 1.2)
    elif size_pref == 'larger':
        # Less variation, larger average  
        min_size = max(5, int(avg_size * 0.7))
        max_size = int(avg_size * 1.5)
    else:  # balanced
        min_size = max(3, int(avg_size * 0.5))
        max_size = int(avg_size * 1.3)
    
    return {'min': min_size, 'max': max_size}


def _mock_smart_config() -> Tuple[Dict[str, Any], bool]:
    """Mock configuration for testing environments."""
    return {
        'user_preferences': {
            'cluster_count': 5,
            'group_size_preference': 'balanced',
            'months': 12,
            'analysis_focus': 'oil'
        },
        'technical_configs': {
            'vector': {'months': 12, 'stream': 'oil'},
            'cluster': {'use_hdbscan': True, 'min_cluster_size': 10},
            'projection': {'n_neighbors': 15}
        }
    }, True