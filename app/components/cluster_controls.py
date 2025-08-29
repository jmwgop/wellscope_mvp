# app/components/cluster_controls.py

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from wellscope_mvp.pipeline.vector_builder import VectorConfig
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
from wellscope_mvp.pipeline.umap_projector import ProjectionConfig
from app.utils.validation import validate_vector_config, validate_cluster_config, validate_projection_config, format_validation_errors
from app.config.ui_defaults import DEFAULTS
from app.utils.clustering_intelligence import (
    calculate_data_characteristics,
    calculate_intelligent_cluster_params,
    get_user_friendly_suggestions,
    convert_to_cluster_config,
    get_vector_length_suggestions
)

def render_vector_controls(filtered_df: Optional[pd.DataFrame] = None,
                          filters_cfg: Dict[str, Any] = None) -> Tuple[VectorConfig, bool, List[str]]:
    """
    Render data-aware vector configuration controls.
    
    Args:
        filtered_df: Filtered dataset for intelligent vector length calculation
        filters_cfg: Filter configuration for vector length intelligence
        
    Returns:
        (vector_config, is_valid, error_messages)
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_vector_config()
    
    st.subheader("📈 Production Curve Vectors")
    
    # Get vector length suggestions
    if filtered_df is not None and filters_cfg is not None:
        vector_suggestions = get_vector_length_suggestions(filtered_df, filters_cfg)
        optimal_length = vector_suggestions['optimal_length']
        max_safe_length = vector_suggestions['max_safe_length']
        
        # Show suggestions
        if vector_suggestions['recommendations']:
            st.info("💡 **Recommendations:** " + " • ".join(vector_suggestions['recommendations']))
        if vector_suggestions['warnings']:
            st.warning("⚠️ **Warnings:** " + " • ".join(vector_suggestions['warnings']))
    else:
        optimal_length = DEFAULTS.vector_months
        max_safe_length = 36
        vector_suggestions = {'recommendations': [], 'warnings': []}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Vector length (months) - now data-aware
        months = st.slider(
            "Vector Length (months)",
            min_value=6,
            max_value=min(max_safe_length + 12, 48),  # Allow some flexibility beyond max_safe
            value=min(optimal_length, DEFAULTS.vector_months),
            step=6,
            key="vector_months",
            help=f"Recommended: {optimal_length} months based on your data and filters. Captures production curve decline pattern."
        )
        
        # Production stream (simplified for oil focus)
        stream = st.selectbox(
            "Production Stream",
            options=['oil', 'gas', 'boe'],  # Removed water - rarely used for clustering
            format_func=lambda x: {
                'oil': 'Oil (bbls) - RECOMMENDED',
                'gas': 'Gas (Mcf)',
                'boe': 'BOE (barrels oil equivalent)'
            }[x],
            index=0,  # Default to oil - best for production curve clustering
            key="vector_stream", 
            help="Oil production is typically best for decline curve clustering"
        )
    
    with col2:
        # Normalization method (simplified)
        normalize = st.selectbox(
            "Curve Normalization",
            options=['q_over_qmax', 'pct_decline'],
            format_func=lambda x: {
                'q_over_qmax': 'Ratio to Peak (q/qmax) - RECOMMENDED',
                'pct_decline': 'Percent Decline'
            }[x],
            index=0,  # Default to q_over_qmax - best for shape comparison
            key="vector_normalize",
            help="Ratio to Peak normalizes curves by peak production - ideal for shape clustering"
        )
        
        # BOE gas factor (only show if BOE selected)
        if stream == 'boe':
            boe_gas_factor = st.number_input(
                "BOE Gas Factor",
                min_value=1.0,
                max_value=10.0,
                value=DEFAULTS.boe_gas_factor,
                step=0.1,
                key="boe_gas_factor",
                help="Gas to oil conversion factor for BOE calculation"
            )
        else:
            boe_gas_factor = DEFAULTS.boe_gas_factor
    
    # Create configuration
    config_dict = {
        'months': months,
        'stream': stream,
        'normalize': normalize,
        'boe_gas_factor': boe_gas_factor
    }
    
    # Validate configuration
    errors = validate_vector_config(config_dict)
    
    # Create VectorConfig object
    vector_config = VectorConfig(
        months=months,
        stream=stream,
        normalize=normalize,
        boe_gas_factor=boe_gas_factor
    )
    
    # Show validation results
    if errors:
        st.error(format_validation_errors(errors, "Vector Configuration"))
    else:
        st.success("✅ Vector configuration valid")
    
    return vector_config, len(errors) == 0, errors

def render_clustering_controls(filtered_df: Optional[pd.DataFrame] = None) -> Tuple[ClusterConfig, bool, List[str]]:
    """
    Render data-aware clustering configuration controls.

    Args:
        filtered_df: Filtered dataset from Step 2 for intelligent parameter calculation

    Returns:
        (cluster_config, is_valid, error_messages)
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_cluster_config()
    
    st.subheader("🎯 Clustering Configuration")
    
    # Calculate data characteristics for intelligent suggestions
    data_chars = calculate_data_characteristics(filtered_df)
    suggestions = get_user_friendly_suggestions(data_chars)
    
    # Show data summary and suggestions
    if filtered_df is not None and len(filtered_df) > 0:
        n_wells = data_chars['n_wells']
        confidence = suggestions['confidence']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Wells to Cluster", f"{n_wells:,}")
        with col2:
            st.metric("Suggested Groups", suggestions['recommended_clusters'])
        with col3:
            confidence_color = (
                "normal" if confidence >= DEFAULTS.good_confidence_threshold 
                else "inverse" if confidence >= DEFAULTS.min_confidence_warning
                else "off"
            )
            st.metric("Success Confidence", f"{confidence:.1f}", delta_color=confidence_color)
        
        # Show suggestions and warnings
        if suggestions['suggestions']:
            st.info("💡 **Suggestions:** " + " • ".join(suggestions['suggestions']))
        if suggestions['warnings']:
            st.warning("⚠️ **Warnings:** " + " • ".join(suggestions['warnings']))
    
    st.divider()
    
    # Simplified production-focused controls
    st.markdown("**🎯 Clustering Strategy**")
    st.info("💡 Optimized for production decline curve clustering using HDBSCAN + Cosine similarity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Expected number of clusters (primary control)
        max_clusters = max(3, min(15, data_chars['n_wells'] // 4)) if data_chars['n_wells'] > 0 else 8
        expected_clusters = st.slider(
            "Expected Well Groups",
            min_value=2,
            max_value=max_clusters,
            value=min(suggestions.get('recommended_clusters', 5), max_clusters-1) if filtered_df is not None else 5,
            step=1,
            key="expected_clusters",
            help="How many distinct production behavior groups do you expect? System will auto-optimize parameters."
        )
        
        # Group size preference (simplified)
        group_size_preference = st.select_slider(
            "Group Size",
            options=["small", "medium", "large"],
            format_func=lambda x: {
                'small': 'Small Groups (5-15 wells)',
                'medium': 'Medium Groups (10-30 wells)', 
                'large': 'Large Groups (20+ wells)'
            }[x],
            value=DEFAULTS.default_group_size_preference,
            key="group_size_preference",
            help="Smaller groups = more specific patterns, Larger groups = broader patterns"
        )
    
    with col2:
        # Clustering sensitivity (key control)
        sensitivity = st.select_slider(
            "Clustering Strictness",
            options=["loose", "balanced", "strict"],
            format_func=lambda x: {
                'loose': 'Loose - Find more groups',
                'balanced': 'Balanced - Recommended', 
                'strict': 'Strict - Fewer tight groups'
            }[x],
            value=DEFAULTS.default_clustering_sensitivity,
            key="clustering_sensitivity",
            help="Loose: Finds more diverse groups. Strict: Only very similar wells grouped together."
        )
        
        # Production curve focus (show but lock to optimal)
        st.markdown("**Similarity Method**")
        st.success("🎯 **Shape-based (Cosine)** - Optimal for production curves")
        metric = 'cosine'  # Hard-coded to optimal choice
        
        # Hidden advanced option toggle
        show_advanced = st.checkbox("Show advanced options", value=False, key="show_clustering_advanced")
    
    # Calculate intelligent parameters
    intelligent_params = calculate_intelligent_cluster_params(
        data_chars,
        expected_clusters=expected_clusters,
        group_size_preference=group_size_preference,
        sensitivity=sensitivity
    )
    
    # Advanced options - only show if requested or for debugging
    if show_advanced:
        with st.expander("⚙️ Advanced Options (Auto-calculated)", expanded=True):
            st.warning("⚠️ **Expert Mode** - These parameters are auto-optimized. Only adjust if you understand clustering algorithms.")
            
            # Allow manual metric override in advanced mode
            metric = st.selectbox(
                "Distance Metric (Advanced)",
                options=['cosine', 'euclidean', 'manhattan'],
                format_func=lambda x: {
                    'cosine': 'Shape-based (Cosine) - RECOMMENDED',
                    'euclidean': 'Standard (Euclidean)',
                    'manhattan': 'Robust (Manhattan)'
                }[x],
                index=0,
                key="advanced_clustering_metric",
                help="Distance metric for measuring well similarity"
            )
    else:
        # Use optimal default when not in advanced mode
        metric = 'cosine'
    
    # Always show calculated parameters (collapsed by default)
    with st.expander("📊 Calculated Parameters", expanded=False):
        st.info("These parameters are automatically calculated based on your data and preferences above.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Min Cluster Size", intelligent_params['min_cluster_size'])
            st.metric("Algorithm", intelligent_params['algorithm_recommendation'])
        with col2:
            st.metric("Min Samples", intelligent_params.get('min_samples', 'Auto'))
            st.metric("Confidence Score", f"{intelligent_params.get('confidence_score', 0.5):.2f}")
        
        # Allow manual override
        manual_override = st.checkbox(
            "Override automatic parameters",
            key="manual_clustering_override",
            help="Check this to manually set clustering parameters (for advanced users)"
        )
        
        if manual_override:
            st.warning("⚠️ Manual override enabled - you are responsible for parameter selection")
            
            col1, col2 = st.columns(2)
            with col1:
                manual_min_cluster_size = st.slider(
                    "Manual Min Cluster Size",
                    min_value=2,
                    max_value=max(2, data_chars['n_wells'] // 2) if data_chars['n_wells'] > 0 else 50,
                    value=intelligent_params['min_cluster_size'],
                    key="manual_min_cluster_size"
                )
                
                manual_use_hdbscan = st.radio(
                    "Manual Algorithm",
                    options=[True, False],
                    format_func=lambda x: "HDBSCAN" if x else "DBSCAN",
                    index=0 if intelligent_params['use_hdbscan'] else 1,
                    key="manual_use_hdbscan"
                )
            
            with col2:
                manual_min_samples = st.number_input(
                    "Manual Min Samples",
                    min_value=1,
                    max_value=max(1, data_chars['n_wells'] // 4) if data_chars['n_wells'] > 0 else 25,
                    value=intelligent_params.get('min_samples', intelligent_params['min_cluster_size']),
                    key="manual_min_samples"
                )
                
                manual_eps = st.slider(
                    "Manual DBSCAN Epsilon",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key="manual_dbscan_eps"
                )
            
            # Use manual parameters
            intelligent_params.update({
                'min_cluster_size': manual_min_cluster_size,
                'min_samples': manual_min_samples,
                'use_hdbscan': manual_use_hdbscan,
            })
    
    # Create cluster config from intelligent parameters
    cluster_config = convert_to_cluster_config(intelligent_params, metric)
    
    # Validate configuration
    errors = validate_cluster_config(cluster_config)
    
    if errors:
        st.error(format_validation_errors(errors, "Clustering Configuration"))
    else:
        st.success("✅ Clustering configuration valid")
    
    return cluster_config, len(errors) == 0, errors

def render_projection_controls() -> Tuple[ProjectionConfig, bool, List[str]]:
    """
    Render simplified UMAP projection configuration controls.
    
    Returns:
        (projection_config, is_valid, error_messages)
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_projection_config()
    
    # Simplified projection - most users don't need to adjust these
    st.markdown("**🗺️ 2D Visualization**")
    st.info("💡 Auto-optimized UMAP projection for well cluster visualization")
    
    show_projection_advanced = st.checkbox("Show projection options", value=False, key="show_projection_advanced")
    
    if show_projection_advanced:
        with st.expander("🗺️ 2D Projection Settings", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                # Number of neighbors
                n_neighbors = st.slider(
                    "N Neighbors",
                    min_value=2,
                    max_value=100,
                    value=DEFAULTS.umap_n_neighbors,
                    step=1,
                    key="umap_n_neighbors",
                    help="Number of neighbors for UMAP projection"
                )
                
                # Minimum distance
                min_dist = st.slider(
                    "Minimum Distance",
                    min_value=0.0,
                    max_value=1.0,
                    value=DEFAULTS.umap_min_dist,
                    step=0.05,
                    key="umap_min_dist",
                    help="Minimum distance between points in projection"
                )
            
            with col2:
                # Number of components (usually 2 for visualization)
                n_components = st.selectbox(
                    "Dimensions",
                    options=[2, 3],
                    index=0,
                    key="umap_n_components",
                    help="Number of dimensions for projection (2D or 3D)"
                )
                
                # Random state for reproducibility
                random_state = st.number_input(
                    "Random State",
                    min_value=0,
                    max_value=9999,
                    value=DEFAULTS.random_state,
                    step=1,
                    key="umap_random_state",
                    help="Random seed for reproducible results"
                )
    else:
        # Use defaults when advanced options are hidden
        n_neighbors = DEFAULTS.umap_n_neighbors
        min_dist = DEFAULTS.umap_min_dist
        n_components = 2  # Default 2D visualization
        random_state = DEFAULTS.random_state
    
    # Create configuration
    config_dict = {
        'n_components': n_components,
        'n_neighbors': n_neighbors,
        'min_dist': min_dist,
        'random_state': random_state
    }
    
    # Validate configuration
    errors = validate_projection_config(config_dict)
    
    # Create ProjectionConfig object
    projection_config = ProjectionConfig(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=DEFAULTS.umap_metric,
        random_state=random_state
    )
    
    return projection_config, len(errors) == 0, errors

def render_all_controls(filtered_df: Optional[pd.DataFrame] = None,
                       filters_cfg: Dict[str, Any] = None) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Render all analysis controls together.
    
    Args:
        filtered_df: Filtered dataset for intelligent parameter calculation
        filters_cfg: Filter configuration for vector length intelligence
    
    Returns:
        (all_configs, all_valid, all_errors)
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_all_controls()
    
    st.header("⚙️ Analysis Configuration")
    
    # Vector controls - now data-aware based on filters and data
    vector_config, vector_valid, vector_errors = render_vector_controls(filtered_df, filters_cfg)
    
    st.divider()
    
    # Clustering controls - now data-aware
    cluster_config, cluster_valid, cluster_errors = render_clustering_controls(filtered_df)
    
    # Projection controls
    projection_config, projection_valid, projection_errors = render_projection_controls()
    
    # Combine results
    all_configs = {
        'vector': vector_config,
        'cluster': cluster_config,
        'projection': projection_config
    }
    
    all_valid = vector_valid and cluster_valid and projection_valid
    all_errors = vector_errors + cluster_errors + projection_errors
    
    # Show overall status
    if all_valid:
        st.success("🎉 All configurations are valid! Ready to run analysis.")
    else:
        st.warning(f"⚠️ Found {len(all_errors)} configuration issue(s). Please review settings.")
    
    return all_configs, all_valid, all_errors

def get_recommended_settings(data_size: int) -> Dict[str, Any]:
    """Get recommended settings based on data size."""
    recommendations = {}
    
    if data_size < 100:
        recommendations.update({
            'min_cluster_size': max(2, data_size // 10),
            'vector_months': 12,
            'umap_n_neighbors': min(15, data_size // 3)
        })
    elif data_size < 1000:
        recommendations.update({
            'min_cluster_size': 10,
            'vector_months': 24,
            'umap_n_neighbors': 15
        })
    else:
        recommendations.update({
            'min_cluster_size': 20,
            'vector_months': 24,
            'umap_n_neighbors': 30
        })
    
    return recommendations

def _mock_vector_config() -> Tuple[VectorConfig, bool, List[str]]:
    """Mock vector config for testing."""
    return VectorConfig(), True, []

def _mock_cluster_config() -> Tuple[ClusterConfig, bool, List[str]]:
    """Mock cluster config for testing."""
    return ClusterConfig(), True, []

def _mock_projection_config() -> Tuple[ProjectionConfig, bool, List[str]]:
    """Mock projection config for testing."""
    return ProjectionConfig(), True, []

def _mock_all_controls() -> Tuple[Dict[str, Any], bool, List[str]]:
    """Mock all controls for testing."""
    return {
        'vector': VectorConfig(),
        'cluster': ClusterConfig(), 
        'projection': ProjectionConfig()
    }, True, []