# app/utils/smart_recommendations.py

from __future__ import annotations
from typing import Dict, Any
import math

from wellscope_mvp.pipeline.vector_builder import VectorConfig
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
from wellscope_mvp.pipeline.umap_projector import ProjectionConfig
from app.config.ui_defaults import DEFAULTS


def generate_smart_recommendations(data_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate user-friendly recommendations based on data analysis.
    
    Converts technical data analysis into actionable, understandable recommendations
    for non-ML users.
    
    Args:
        data_analysis: Output from analyze_filtered_data()
        
    Returns:
        Dictionary with user-friendly recommendations and insights
    """
    
    n_wells = data_analysis['n_wells']
    similarity = data_analysis.get('similarity_mean', 0.6)
    production_type = data_analysis.get('production_type', 'Mixed')
    formation_diversity = data_analysis.get('formation_diversity', 5)
    maturity_fraction = data_analysis.get('mature_well_fraction', 0.2)
    
    # Generate insights
    insights = []
    warnings = []
    
    # Dataset size insights
    if n_wells < 30:
        insights.append(f"Small dataset ({n_wells} wells) - will focus on finding clear patterns")
        warnings.append("Consider combining with similar datasets for more robust clustering")
    elif n_wells > 1000:
        insights.append(f"Large dataset ({n_wells:,} wells) - can detect subtle patterns")
    
    # Production similarity insights  
    if similarity > 0.85:
        insights.append(f"Very high well similarity ({similarity:.0%}) - typical for same formation/completion")
        production_optimization_msg = "Using specialized parameters optimized for high-similarity oil & gas data"
    elif similarity > 0.75:
        insights.append(f"High well similarity ({similarity:.0%}) - wells likely from similar plays")
        production_optimization_msg = "Using production-optimized clustering for similar decline curves"
    elif similarity > 0.65:
        insights.append(f"Moderate well similarity ({similarity:.0%}) - mix of production patterns expected")
        production_optimization_msg = None
    else:
        insights.append(f"Diverse well patterns ({similarity:.0%} similarity) - will find distinct groups")
        warnings.append("Low similarity may indicate mixed formations or completion types")
        production_optimization_msg = None
    
    # Formation insights
    if formation_diversity < 2:
        insights.append("Single formation dataset - wells likely have similar geology")
    elif formation_diversity > 6:
        insights.append("Multiple formations detected - expect diverse production behavior")
        warnings.append("Consider analyzing formations separately for more specific insights")
    
    # Maturity insights
    if maturity_fraction > 0.6:
        insights.append("Mostly mature wells (24+ months) - can analyze full decline curves")
    elif maturity_fraction < 0.2:
        insights.append("Mostly new wells (<24 months) - will focus on early production patterns")
        warnings.append("Limited production history - consider updating analysis as wells mature")
    
    # Calculate recommended clusters
    recommended_clusters = _calculate_recommended_clusters(n_wells, similarity, formation_diversity)
    
    # Calculate optimal months
    optimal_months = data_analysis.get('optimal_vector_length', 12)
    
    # Calculate expected group sizes
    avg_group_size = n_wells // recommended_clusters
    group_size_min = max(3, int(avg_group_size * 0.6))
    group_size_max = int(avg_group_size * 1.4)
    
    # Determine algorithm choice
    if similarity > 0.85 and n_wells > 50:
        algorithm_choice = "DBSCAN (optimized for high-similarity production data)"
        optimization_reason = f"High similarity ({similarity:.0%}) detected - using aggressive clustering proven for oil & gas data"
    elif n_wells < 50:
        algorithm_choice = "HDBSCAN (better for small datasets)"
        optimization_reason = f"Small dataset ({n_wells} wells) - using HDBSCAN for robust clustering"
    else:
        algorithm_choice = "HDBSCAN (general-purpose clustering)"
        optimization_reason = f"Moderate similarity ({similarity:.0%}) - using flexible HDBSCAN algorithm"
    
    # Processing time estimate
    from app.utils.data_analyzer import estimate_processing_time
    estimated_time = estimate_processing_time(n_wells, optimal_months)
    
    return {
        'insights': insights,
        'warnings': warnings,
        'production_optimization': production_optimization_msg,
        'recommended_clusters': recommended_clusters,
        'expected_group_size_min': group_size_min,
        'expected_group_size_max': group_size_max,
        'optimal_months': optimal_months,
        'production_stream': _recommend_production_stream(production_type),
        'algorithm_choice': algorithm_choice,
        'optimization_reason': optimization_reason,
        'similarity_method': "Shape-based (cosine similarity)",
        'estimated_time': estimated_time,
        'confidence_level': _calculate_recommendation_confidence(data_analysis)
    }


def _calculate_recommended_clusters(n_wells: int, similarity: float, formation_diversity: float) -> int:
    """Calculate recommended number of clusters based on data characteristics."""
    
    # Base clusters on dataset size
    if n_wells < 30:
        base_clusters = max(2, n_wells // 8)  # Very conservative for small datasets
    elif n_wells < 100:
        base_clusters = max(3, n_wells // 15)  # Conservative 
    elif n_wells < 500:
        base_clusters = max(4, n_wells // 25)  # Standard
    elif n_wells < 1000:
        base_clusters = max(5, n_wells // 40)  # Large dataset
    else:
        base_clusters = max(8, min(20, n_wells // 60))  # Very large dataset
    
    # Adjust for similarity (high similarity = fewer meaningful clusters)
    if similarity > 0.85:
        similarity_multiplier = 0.7  # Fewer clusters for high similarity
    elif similarity > 0.75:
        similarity_multiplier = 0.8  # Slightly fewer clusters
    elif similarity < 0.6:
        similarity_multiplier = 1.3  # More clusters for diverse data
    else:
        similarity_multiplier = 1.0  # Standard
    
    # Adjust for formation diversity
    if formation_diversity < 2:
        formation_multiplier = 0.8  # Single formation = fewer distinct patterns
    elif formation_diversity > 6:
        formation_multiplier = 1.2  # Multiple formations = more patterns
    else:
        formation_multiplier = 1.0  # Standard
    
    # Calculate final recommendation
    final_clusters = int(base_clusters * similarity_multiplier * formation_multiplier)
    
    # Apply reasonable bounds
    final_clusters = max(2, min(20, final_clusters))
    
    # Ensure clusters make sense for dataset size
    if final_clusters > n_wells // 5:
        final_clusters = max(2, n_wells // 5)  # At least 5 wells per cluster
    
    return final_clusters


def _recommend_production_stream(production_type: str) -> str:
    """Recommend production stream focus based on detected type."""
    
    if production_type.lower() in ['oil', 'crude', 'liquid']:
        return 'oil'
    elif production_type.lower() in ['gas', 'natural gas', 'mcf']:
        return 'gas'
    else:
        return 'oil'  # Default to oil for mixed/unknown


def _calculate_recommendation_confidence(data_analysis: Dict[str, Any]) -> str:
    """Calculate and format recommendation confidence level."""
    
    confidence = data_analysis.get('confidence', 0.5)
    
    if confidence >= 0.8:
        return 'High'
    elif confidence >= 0.6:
        return 'Medium'
    else:
        return 'Low'


def format_confidence_display(confidence: float) -> str:
    """Format confidence score for user display."""
    
    if confidence >= 0.8:
        return "High"
    elif confidence >= 0.6:
        return "Medium" 
    else:
        return "Low"


def convert_to_technical_config(data_analysis: Dict[str, Any], 
                               recommendations: Dict[str, Any],
                               user_preferences: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert user preferences and data analysis into technical configurations.
    
    This bridges the gap between simple user controls and complex technical parameters.
    """
    
    # Extract user preferences
    target_clusters = user_preferences.get('target_clusters', recommendations['recommended_clusters'])
    group_size_pref = user_preferences.get('group_size_preference', 'balanced')
    months = user_preferences.get('months', recommendations['optimal_months'])
    production_stream = user_preferences.get('production_stream', 'oil')
    
    # Create vector configuration
    vector_config = VectorConfig(
        months=months,
        stream=production_stream,
        normalize='q_over_qmax',  # Best for shape clustering
        boe_gas_factor=DEFAULTS.boe_gas_factor
    )
    
    # Create clustering configuration with intelligent parameters
    cluster_config = _create_intelligent_cluster_config(
        data_analysis, target_clusters, group_size_pref
    )
    
    # Create projection configuration (simplified)
    n_neighbors = min(30, max(5, data_analysis['n_wells'] // 10))
    projection_config = ProjectionConfig(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=DEFAULTS.umap_min_dist,
        metric=DEFAULTS.umap_metric,
        random_state=DEFAULTS.random_state
    )
    
    return {
        'vector': vector_config,
        'cluster': cluster_config,
        'projection': projection_config
    }


def _create_intelligent_cluster_config(data_analysis: Dict[str, Any],
                                      target_clusters: int, 
                                      group_size_pref: str) -> ClusterConfig:
    """Create intelligent clustering configuration."""
    
    n_wells = data_analysis['n_wells']
    similarity = data_analysis.get('similarity_mean', 0.6)
    
    # Determine algorithm based on similarity and size
    if similarity > 0.85 and n_wells > 50:
        # High similarity - use production-optimized DBSCAN
        use_hdbscan = False
        
        # Production-optimized parameters based on similarity
        if similarity > 0.90:
            eps = 0.05  # Very aggressive for very high similarity
        elif similarity > 0.87:
            eps = 0.08  # Aggressive 
        else:
            eps = 0.12  # Moderately aggressive
            
        min_samples = 2  # Minimal requirement for production optimization
        min_cluster_size = min_samples  # Not used for DBSCAN
        metric = "euclidean"  # Works best with small eps values
        
    else:
        # Use HDBSCAN for other cases
        use_hdbscan = True
        eps = 0.5  # Default (not used for HDBSCAN)
        metric = "cosine"  # Best for production curve shapes
        
        # Calculate min_cluster_size based on target clusters and preferences
        avg_cluster_size = n_wells // target_clusters
        
        # Adjust based on group size preference
        if group_size_pref == 'smaller':
            min_cluster_size = max(2, int(avg_cluster_size * 0.6))
        elif group_size_pref == 'larger':
            min_cluster_size = max(5, int(avg_cluster_size * 1.2))
        else:  # balanced
            min_cluster_size = max(3, int(avg_cluster_size * 0.8))
        
        # min_samples typically 50-80% of min_cluster_size for HDBSCAN
        min_samples = max(2, int(min_cluster_size * 0.6))
        
        # Apply reasonable bounds
        min_cluster_size = max(2, min(min_cluster_size, n_wells // 3))
        min_samples = max(2, min(min_samples, min_cluster_size))
    
    return ClusterConfig(
        use_hdbscan=use_hdbscan,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        eps=eps
    )