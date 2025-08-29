# wellscope_mvp/pipeline/production_clustering.py

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


def detect_production_data(vectors_df: pd.DataFrame, vector_prefix: str = "v") -> Dict[str, float]:
    """
    Detect if the dataset represents oil & gas production curves.
    
    Args:
        vectors_df: DataFrame with production vectors
        vector_prefix: Prefix for vector columns
        
    Returns:
        Dictionary with detection results and similarity statistics
    """
    vector_cols = [c for c in vectors_df.columns if c.startswith(vector_prefix)]
    if len(vector_cols) < 6:
        return {
            'is_production_data': False,
            'confidence': 0.0,
            'cosine_similarity_mean': 0.0,
            'reason': 'insufficient_vector_length'
        }
    
    # Extract vector data
    X = vectors_df[vector_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if X.shape[0] < 10:
        return {
            'is_production_data': False,
            'confidence': 0.0,
            'cosine_similarity_mean': 0.0,
            'reason': 'insufficient_data_points'
        }
    
    # Calculate cosine similarity statistics
    try:
        cos_sim_matrix = cosine_similarity(X)
        # Get upper triangle (exclude diagonal)
        upper_tri = np.triu_indices_from(cos_sim_matrix, k=1)
        similarities = cos_sim_matrix[upper_tri]
        
        similarity_mean = float(similarities.mean())
        similarity_std = float(similarities.std())
        
        # Check for production curve characteristics
        production_indicators = []
        
        # 1. High inter-well similarity (production curves are inherently similar)
        if similarity_mean > 0.75:
            production_indicators.append('high_similarity')
        
        # 2. Declining pattern check (most vectors should decline)
        decline_count = 0
        for i in range(X.shape[0]):
            curve = X[i]
            if len(curve) >= 6:
                # Check if curve generally declines from early to late values
                early_avg = curve[:3].mean()
                late_avg = curve[-3:].mean()
                if early_avg > late_avg and early_avg > 0:
                    decline_count += 1
        
        decline_fraction = decline_count / X.shape[0]
        if decline_fraction > 0.6:  # 60% of wells show decline
            production_indicators.append('decline_pattern')
        
        # 3. Variance pattern (early months typically have more variance)
        if len(vector_cols) >= 12:
            early_var = np.var(X[:, :6], axis=0).mean()
            late_var = np.var(X[:, -6:], axis=0).mean()
            if early_var > late_var * 1.2:  # Early variance > late variance
                production_indicators.append('variance_pattern')
        
        # Determine if this is production data
        confidence = 0.0
        is_production = False
        reason = "low_similarity"
        
        if similarity_mean > 0.80:
            # Very high similarity - almost certainly production data
            is_production = True
            confidence = min(0.95, similarity_mean + 0.1)
            reason = f"high_similarity_{similarity_mean:.3f}"
            
            if len(production_indicators) >= 2:
                confidence = 0.98
                reason = f"high_similarity_with_production_patterns"
        elif similarity_mean > 0.70 and len(production_indicators) >= 2:
            # Moderate similarity with production patterns
            is_production = True
            confidence = 0.8
            reason = "moderate_similarity_with_patterns"
        
        return {
            'is_production_data': is_production,
            'confidence': confidence,
            'cosine_similarity_mean': similarity_mean,
            'cosine_similarity_std': similarity_std,
            'decline_fraction': decline_fraction,
            'production_indicators': production_indicators,
            'reason': reason
        }
        
    except Exception as e:
        return {
            'is_production_data': False,
            'confidence': 0.0,
            'cosine_similarity_mean': 0.0,
            'reason': f'calculation_error_{str(e)}'
        }


def get_production_optimized_config(
    vectors_df: pd.DataFrame,
    similarity_stats: Dict[str, float],
    base_config: Optional['ClusterConfig'] = None
) -> 'ClusterConfig':
    """
    Get optimized clustering configuration for oil & gas production data.
    
    Based on experimental findings:
    - High similarity data (>85%): DBSCAN with eps=0.05, min_samples=2
    - Medium-high similarity (75-85%): DBSCAN with eps=0.1, min_samples=2  
    - Moderate similarity (65-75%): DBSCAN with eps=0.2, min_samples=3
    
    Args:
        vectors_df: Production vectors DataFrame
        similarity_stats: Output from detect_production_data()
        base_config: Base configuration to modify
        
    Returns:
        Optimized ClusterConfig for production data
    """
    from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
    n_wells = len(vectors_df)
    similarity_mean = similarity_stats.get('cosine_similarity_mean', 0.0)
    
    # Production-optimized parameters based on experimental results
    if similarity_mean >= 0.85:
        # Very high similarity (like Eagleford) - use most aggressive parameters
        config = ClusterConfig(
            use_hdbscan=False,    # DBSCAN proven superior for high similarity
            eps=0.05,             # Experimental optimum from clustering_summary.md
            min_samples=2,        # Minimal requirement allows detection of subtle differences
            metric="euclidean",   # Works best with small eps values
            vector_prefix="v"
        )
    elif similarity_mean >= 0.80:
        # High similarity - moderately aggressive
        config = ClusterConfig(
            use_hdbscan=False,
            eps=0.08,
            min_samples=2,
            metric="euclidean",
            vector_prefix="v"
        )
    elif similarity_mean >= 0.75:
        # Medium-high similarity - less aggressive
        config = ClusterConfig(
            use_hdbscan=False,
            eps=0.12,
            min_samples=3,
            metric="euclidean", 
            vector_prefix="v"
        )
    elif similarity_mean >= 0.65:
        # Moderate similarity - can try HDBSCAN with aggressive params
        min_cluster_size = max(2, min(5, n_wells // 20))  # Very small clusters
        config = ClusterConfig(
            use_hdbscan=True,
            min_cluster_size=min_cluster_size,
            min_samples=2,
            metric="cosine",      # Better for shape-based clustering
            eps=0.2,              # DBSCAN fallback
            vector_prefix="v"
        )
    else:
        # Low similarity for production data - use standard parameters
        # This case is rare for actual production data
        min_cluster_size = max(3, min(10, n_wells // 15))
        config = ClusterConfig(
            use_hdbscan=True,
            min_cluster_size=min_cluster_size,
            min_samples=max(2, min_cluster_size // 2),
            metric="cosine",
            eps=0.3,
            vector_prefix="v"
        )
    
    return config


def calculate_production_similarity_stats(vectors_df: pd.DataFrame, vector_prefix: str = "v") -> Dict[str, float]:
    """
    Calculate detailed similarity statistics for production data.
    
    Args:
        vectors_df: DataFrame with production vectors
        vector_prefix: Prefix for vector columns
        
    Returns:
        Dictionary with comprehensive similarity statistics
    """
    vector_cols = [c for c in vectors_df.columns if c.startswith(vector_prefix)]
    if not vector_cols:
        return {'error': 'no_vector_columns'}
    
    X = vectors_df[vector_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    if X.shape[0] < 2:
        return {'error': 'insufficient_wells'}
    
    try:
        # Cosine similarity analysis
        cos_sim_matrix = cosine_similarity(X)
        upper_tri = np.triu_indices_from(cos_sim_matrix, k=1)
        similarities = cos_sim_matrix[upper_tri]
        
        # Euclidean distance analysis
        from sklearn.metrics.pairwise import euclidean_distances  # type: ignore
        euc_dist_matrix = euclidean_distances(X)
        distances = euc_dist_matrix[upper_tri]
        
        # Calculate percentiles for similarity distribution
        similarity_percentiles = {
            f'p{p}': float(np.percentile(similarities, p))
            for p in [10, 25, 50, 75, 85, 90, 95, 99]
        }
        
        # High similarity analysis
        high_sim_95 = (similarities > 0.95).mean()
        high_sim_90 = (similarities > 0.90).mean()
        high_sim_85 = (similarities > 0.85).mean()
        
        return {
            'n_wells': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'cosine_similarity_mean': float(similarities.mean()),
            'cosine_similarity_std': float(similarities.std()),
            'cosine_similarity_min': float(similarities.min()),
            'cosine_similarity_max': float(similarities.max()),
            'euclidean_distance_mean': float(distances.mean()),
            'euclidean_distance_std': float(distances.std()),
            'high_similarity_95_pct': float(high_sim_95),
            'high_similarity_90_pct': float(high_sim_90),
            'high_similarity_85_pct': float(high_sim_85),
            **similarity_percentiles
        }
        
    except Exception as e:
        return {'error': f'calculation_failed_{str(e)}'}


def get_production_clustering_recommendation(vectors_df: pd.DataFrame) -> Dict[str, any]:
    """
    Get comprehensive clustering recommendation for production data.
    
    Args:
        vectors_df: DataFrame with production vectors
        
    Returns:
        Dictionary with recommendation details and confidence scores
    """
    # Detect production data characteristics
    detection_result = detect_production_data(vectors_df)
    similarity_stats = calculate_production_similarity_stats(vectors_df)
    
    # Get optimized configuration
    if detection_result['is_production_data']:
        optimized_config = get_production_optimized_config(vectors_df, detection_result)
        
        # Estimate expected clusters based on similarity and dataset size
        n_wells = len(vectors_df)
        similarity_mean = detection_result['cosine_similarity_mean']
        
        if similarity_mean >= 0.90:
            expected_clusters = max(2, min(8, n_wells // 25))   # High similarity: fewer, larger clusters
        elif similarity_mean >= 0.80:
            expected_clusters = max(3, min(12, n_wells // 20))  # Medium-high: moderate clusters
        else:
            expected_clusters = max(4, min(15, n_wells // 15))  # Lower similarity: more clusters
        
        return {
            'is_production_data': True,
            'confidence': detection_result['confidence'],
            'similarity_mean': similarity_mean,
            'recommended_config': optimized_config,
            'expected_clusters': expected_clusters,
            'algorithm_choice': 'DBSCAN' if not optimized_config.use_hdbscan else 'HDBSCAN',
            'optimization_reason': detection_result['reason'],
            'user_message': _generate_user_message(detection_result, similarity_stats, expected_clusters),
            'similarity_stats': similarity_stats
        }
    else:
        return {
            'is_production_data': False,
            'confidence': detection_result['confidence'], 
            'reason': detection_result['reason'],
            'user_message': "Standard clustering parameters recommended - data does not appear to be oil & gas production curves."
        }


def _generate_user_message(
    detection_result: Dict[str, float],
    similarity_stats: Dict[str, float], 
    expected_clusters: int
) -> str:
    """Generate user-friendly message about production data optimization."""
    similarity_mean = detection_result['cosine_similarity_mean']
    confidence = detection_result['confidence']
    
    if similarity_mean >= 0.90:
        similarity_desc = "very high"
        optimization_desc = "aggressive"
    elif similarity_mean >= 0.85:
        similarity_desc = "high" 
        optimization_desc = "optimized"
    elif similarity_mean >= 0.80:
        similarity_desc = "moderately high"
        optimization_desc = "adjusted"
    else:
        similarity_desc = "moderate"
        optimization_desc = "standard"
    
    message = (
        f"🛢️ **Production Data Detected** ({similarity_mean:.1%} well similarity) - "
        f"Using {optimization_desc} clustering parameters optimized for oil & gas decline curves. "
        f"Expecting ~{expected_clusters} distinct production behavior groups."
    )
    
    if similarity_mean >= 0.85:
        message += " High similarity is normal for wells in the same formation with similar completions."
    
    return message