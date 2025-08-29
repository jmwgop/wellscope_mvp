# app/utils/mature_well_clustering.py

from __future__ import annotations
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from wellscope_mvp.pipeline.vector_builder import VectorConfig, build_shape_vectors
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig, run_clustering
from wellscope_mvp.pipeline.filter_inputs import compute_months_produced


def analyze_well_maturity(filtered_df: pd.DataFrame, 
                         maturity_threshold: int = 24,
                         api_col: str = "API14",
                         monthly_date_col: str = "Monthly Production Date") -> Dict[str, Any]:
    """
    Analyze well maturity distribution to inform clustering strategy.
    
    Args:
        filtered_df: Filtered dataset from pipeline
        maturity_threshold: Minimum months for "mature" wells
        api_col: API column name
        monthly_date_col: Monthly production date column
        
    Returns:
        Dictionary with maturity analysis results
    """
    if filtered_df is None or len(filtered_df) == 0:
        return {
            'total_wells': 0,
            'mature_wells': 0,
            'young_wells': 0,
            'mature_fraction': 0.0,
            'recommended_strategy': 'insufficient_data'
        }
    
    # Calculate months produced for each well
    months_produced = compute_months_produced(filtered_df, api_col, monthly_date_col)
    
    # Combine with API information
    well_maturity = pd.DataFrame({
        api_col: filtered_df[api_col],
        'months_produced': months_produced
    }).drop_duplicates(subset=[api_col])
    
    total_wells = len(well_maturity)
    mature_wells = len(well_maturity[well_maturity['months_produced'] >= maturity_threshold])
    young_wells = total_wells - mature_wells
    mature_fraction = mature_wells / total_wells if total_wells > 0 else 0
    
    # Determine recommended strategy
    if mature_fraction >= 0.4 and mature_wells >= 20:
        strategy = 'mature_first'  # Enough mature wells to establish meaningful clusters
    elif mature_fraction >= 0.2 and mature_wells >= 10:
        strategy = 'mixed_length'  # Some mature wells, use variable vector lengths
    else:
        strategy = 'uniform_short'  # Too few mature wells, use shorter vectors for all
    
    return {
        'total_wells': total_wells,
        'mature_wells': mature_wells,
        'young_wells': young_wells,
        'mature_fraction': mature_fraction,
        'maturity_threshold': maturity_threshold,
        'recommended_strategy': strategy,
        'mature_well_apis': well_maturity[well_maturity['months_produced'] >= maturity_threshold][api_col].tolist(),
        'young_well_apis': well_maturity[well_maturity['months_produced'] < maturity_threshold][api_col].tolist()
    }


def run_mature_first_clustering(
    filtered_df: pd.DataFrame,
    vector_config: VectorConfig,
    cluster_config: ClusterConfig,
    maturity_threshold: int = 24,
    api_col: str = "API14"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run mature-well-first clustering strategy.
    
    Strategy:
    1. Identify wells with >= maturity_threshold months of data
    2. Cluster mature wells using full vector length
    3. Build shorter vectors for young wells  
    4. Assign young wells to existing clusters based on similarity
    
    Args:
        filtered_df: Filtered dataset from pipeline
        vector_config: Vector generation configuration
        cluster_config: Clustering configuration
        maturity_threshold: Minimum months for mature wells
        api_col: API column name
        
    Returns:
        (labels_df, metadata) - Same format as standard clustering
    """
    # Analyze well maturity
    maturity_analysis = analyze_well_maturity(filtered_df, maturity_threshold, api_col)
    
    if maturity_analysis['recommended_strategy'] != 'mature_first':
        # Fall back to standard clustering if not enough mature wells
        return _run_fallback_clustering(filtered_df, vector_config, cluster_config, maturity_analysis)
    
    mature_apis = set(maturity_analysis['mature_well_apis'])
    young_apis = set(maturity_analysis['young_well_apis'])
    
    # Step 1: Extract mature wells and build full-length vectors
    mature_wells_df = filtered_df[filtered_df[api_col].isin(mature_apis)]
    mature_vectors_df, mature_vector_meta = build_shape_vectors(mature_wells_df, vector_config)
    
    # Step 2: Cluster mature wells
    mature_labels_df, mature_cluster_meta = run_clustering(mature_vectors_df, cluster_config)
    
    # Check if mature clustering was successful
    n_mature_clusters = mature_cluster_meta['n_clusters']
    if n_mature_clusters == 0:
        # All mature wells are noise - fall back to standard approach
        return _run_fallback_clustering(filtered_df, vector_config, cluster_config, maturity_analysis)
    
    # Step 3: Build shorter vectors for young wells (use available data)
    if young_apis:
        young_wells_df = filtered_df[filtered_df[api_col].isin(young_apis)]
        
        # Determine optimal vector length for young wells based on their data
        young_months_available = _calculate_available_months(young_wells_df, api_col)
        optimal_young_length = max(6, min(12, int(np.percentile(young_months_available, 75))))
        
        # Create shorter vector config for young wells
        young_vector_config = VectorConfig(
            months=optimal_young_length,
            normalize=vector_config.normalize,
            stream=vector_config.stream,
            boe_gas_factor=vector_config.boe_gas_factor
        )
        
        young_vectors_df, young_vector_meta = build_shape_vectors(young_wells_df, young_vector_config)
        
        # Step 4: Assign young wells to mature clusters
        young_labels_df = _assign_to_existing_clusters(
            young_vectors_df, mature_vectors_df, mature_labels_df, 
            optimal_young_length, vector_config.months
        )
        
        # Combine results
        combined_labels_df = pd.concat([mature_labels_df, young_labels_df], ignore_index=True)
        
    else:
        # No young wells - just use mature results
        combined_labels_df = mature_labels_df
        young_vector_meta = {}
    
    # Recalculate cluster sizes for combined data
    combined_labels_df = _recalculate_cluster_sizes(combined_labels_df)
    
    # Create combined metadata
    combined_meta = {
        **mature_cluster_meta,
        'algorithm_used': f"{mature_cluster_meta['algorithm_used']}_mature_first",
        'maturity_analysis': maturity_analysis,
        'mature_vector_meta': mature_vector_meta,
        'young_vector_meta': young_vector_meta,
        'clustering_strategy': 'mature_first',
        'young_vector_length': optimal_young_length if young_apis else 0
    }
    
    return combined_labels_df, combined_meta


def _run_fallback_clustering(filtered_df: pd.DataFrame,
                           vector_config: VectorConfig,
                           cluster_config: ClusterConfig,
                           maturity_analysis: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run fallback clustering when mature-first strategy isn't viable."""
    # Use shorter vectors for all wells when insufficient mature wells
    fallback_length = max(6, min(12, int(maturity_analysis.get('maturity_threshold', 24) * 0.6)))
    
    fallback_vector_config = VectorConfig(
        months=fallback_length,
        normalize=vector_config.normalize,
        stream=vector_config.stream,
        boe_gas_factor=vector_config.boe_gas_factor
    )
    
    # Run standard pipeline with adjusted vector length
    vectors_df, vector_meta = build_shape_vectors(filtered_df, fallback_vector_config)
    labels_df, cluster_meta = run_clustering(vectors_df, cluster_config)
    
    # Add strategy metadata
    cluster_meta.update({
        'algorithm_used': f"{cluster_meta['algorithm_used']}_fallback_uniform",
        'maturity_analysis': maturity_analysis,
        'clustering_strategy': 'fallback_uniform',
        'fallback_vector_length': fallback_length
    })
    
    return labels_df, cluster_meta


def _calculate_available_months(df: pd.DataFrame, api_col: str) -> np.ndarray:
    """Calculate available months of data for each well."""
    months_produced = compute_months_produced(df, api_col)
    well_months = df.groupby(api_col)['months_produced'].first() if 'months_produced' in df.columns else months_produced
    return well_months.values


def _assign_to_existing_clusters(young_vectors_df: pd.DataFrame,
                               mature_vectors_df: pd.DataFrame,
                               mature_labels_df: pd.DataFrame,
                               young_length: int,
                               mature_length: int) -> pd.DataFrame:
    """
    Assign young wells to existing mature clusters based on truncated vector similarity.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Get API column name
    api_col = [c for c in young_vectors_df.columns if not c.startswith('v')][0]
    
    # Extract vector columns
    young_vector_cols = [c for c in young_vectors_df.columns if c.startswith('v')]
    mature_vector_cols = [c for c in mature_vectors_df.columns if c.startswith('v')]
    
    # Truncate mature vectors to match young vector length
    mature_truncated_cols = mature_vector_cols[:young_length]
    young_vector_cols_adj = young_vector_cols[:young_length]  # In case young vectors are even shorter
    
    # Get mature cluster centroids (excluding noise wells)
    mature_non_noise = mature_labels_df[mature_labels_df['label'] >= 0]
    cluster_centroids = {}
    
    for cluster_id in mature_non_noise['label'].unique():
        cluster_wells = mature_non_noise[mature_non_noise['label'] == cluster_id][api_col].tolist()
        cluster_vectors = mature_vectors_df[mature_vectors_df[api_col].isin(cluster_wells)]
        
        if len(cluster_vectors) > 0:
            # Calculate centroid using truncated vectors
            centroid = cluster_vectors[mature_truncated_cols].mean().values
            cluster_centroids[cluster_id] = centroid
    
    # Assign young wells to clusters
    young_assignments = []
    
    for _, young_well in young_vectors_df.iterrows():
        young_vector = young_well[young_vector_cols_adj].values
        
        if len(cluster_centroids) == 0:
            # No clusters available - assign as noise
            assigned_cluster = -1
        else:
            # Find most similar cluster
            similarities = {}
            for cluster_id, centroid in cluster_centroids.items():
                # Ensure vector lengths match for similarity calculation
                min_len = min(len(young_vector), len(centroid))
                similarity = cosine_similarity([young_vector[:min_len]], [centroid[:min_len]])[0][0]
                similarities[cluster_id] = similarity
            
            # Assign to most similar cluster if similarity is reasonable (>0.3)
            best_cluster = max(similarities.keys(), key=lambda k: similarities[k])
            if similarities[best_cluster] > 0.3:  # Threshold for reasonable similarity
                assigned_cluster = best_cluster
            else:
                assigned_cluster = -1  # Assign as noise if too dissimilar
        
        young_assignments.append({
            api_col: young_well[api_col],
            'label': int(assigned_cluster),
            'is_noise': assigned_cluster == -1,
            'cluster_size': 0  # Will be recalculated later
        })
    
    return pd.DataFrame(young_assignments)


def _recalculate_cluster_sizes(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Recalculate cluster sizes after combining mature and young wells."""
    # Count non-noise wells per cluster
    cluster_counts = labels_df[labels_df['label'] >= 0]['label'].value_counts().to_dict()
    
    # Update cluster sizes
    labels_df['cluster_size'] = labels_df['label'].map(lambda l: cluster_counts.get(l, 0))
    
    return labels_df


def get_mature_clustering_recommendations(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get recommendations for mature-well-first clustering approach.
    
    Returns:
        Dictionary with recommendations and analysis
    """
    maturity_analysis = analyze_well_maturity(filtered_df)
    
    recommendations = []
    warnings = []
    
    strategy = maturity_analysis['recommended_strategy']
    mature_count = maturity_analysis['mature_wells']
    total_count = maturity_analysis['total_wells']
    mature_fraction = maturity_analysis['mature_fraction']
    
    if strategy == 'mature_first':
        recommendations.append(f"Excellent! {mature_count} mature wells (24+ months) will establish primary clusters")
        recommendations.append(f"Remaining {maturity_analysis['young_wells']} newer wells will be matched to established patterns")
        
    elif strategy == 'mixed_length':
        recommendations.append(f"Good! {mature_count} wells have longer history for partial cluster establishment")
        warnings.append(f"Limited mature wells ({mature_fraction:.1%}) - consider shorter vectors for all wells")
        
    else:  # uniform_short
        warnings.append(f"Few mature wells ({mature_count} of {total_count}) - using uniform short vectors")
        recommendations.append("Consider expanding date range to include more mature wells")
    
    return {
        'strategy': strategy,
        'recommendations': recommendations,
        'warnings': warnings,
        'maturity_analysis': maturity_analysis
    }