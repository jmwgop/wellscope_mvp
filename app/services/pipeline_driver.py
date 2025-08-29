# app/services/pipeline_driver.py

from __future__ import annotations
from typing import Dict, Any, Tuple
import pandas as pd

from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
from wellscope_mvp.pipeline.filter_inputs import apply_filters, FilterConfig
from wellscope_mvp.pipeline.vector_builder import build_shape_vectors, VectorConfig
from wellscope_mvp.pipeline.cluster_runner import run_clustering, ClusterConfig
from wellscope_mvp.pipeline.umap_projector import project_vectors, ProjectionConfig
from wellscope_mvp.pipeline.similarity_score import score_similarity, SimilarityConfig
from app.utils.mature_well_clustering import run_mature_first_clustering, get_mature_clustering_recommendations

def run_complete_pipeline(
    headers_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    filters_cfg: Dict[str, Any] = None,
    use_mature_first: bool = True,
    use_production_optimization: bool = True
) -> Dict[str, Any]:
    """
    Run the complete ML pipeline from raw data to similarity scores with production optimization.
    
    Args:
        headers_df: Well headers DataFrame
        monthly_df: Monthly production DataFrame  
        filters_cfg: Optional filter configuration (uses defaults if None)
        use_mature_first: Whether to use mature-first clustering
        use_production_optimization: Whether to enable production data optimization
        
    Returns:
        Dictionary containing all pipeline artifacts:
        - joined_df: Joined headers + monthly data
        - vectors_df: Production curve vectors
        - labels_df: Cluster labels
        - coords_df: 2D projection coordinates
        - scores_df: Similarity scores
        - join_stats: Join operation statistics
        - filter_stats: Filter operation statistics
        - vector_meta: Vector generation metadata
        - cluster_meta: Clustering metadata
        - projection_meta: Projection metadata
        - similarity_meta: Similarity scoring metadata
        
    Raises:
        ValueError: If input data is invalid or pipeline fails
    """
    if not isinstance(headers_df, pd.DataFrame) or len(headers_df) == 0:
        raise ValueError("Headers DataFrame is empty or invalid")
    
    if not isinstance(monthly_df, pd.DataFrame) or len(monthly_df) == 0:
        raise ValueError("Monthly DataFrame is empty or invalid")
    
    # Use default config if none provided
    if filters_cfg is None:
        filters_cfg = {}
    
    try:
        # Step 1: Join inputs
        joined_df, join_stats = join_headers_and_monthly(headers_df, monthly_df)
        if len(joined_df) == 0:
            raise ValueError("No wells remain after joining data")
        
        # Step 2: Apply filters
        filter_config = FilterConfig(**filters_cfg) if filters_cfg else FilterConfig()
        filter_result = apply_filters(joined_df, filter_config)
        filtered_df = filter_result['filtered']
        filter_stats = filter_result
        if len(filtered_df) == 0:
            raise ValueError("No wells remain after filtering")
        
        # Step 3: Generate vectors
        vector_config = VectorConfig()
        vectors_df, vector_meta = build_shape_vectors(filtered_df, vector_config)
        if len(vectors_df) == 0:
            raise ValueError("Vector generation failed")
        
        # Step 4: Run clustering (mature-well-first or standard)
        cluster_config = ClusterConfig()
        if use_mature_first:
            labels_df, cluster_meta = run_mature_first_clustering(
                filtered_df, vector_config, cluster_config
            )
        else:
            labels_df, cluster_meta = run_clustering(vectors_df, cluster_config, use_production_optimization)
        
        # Step 5: Generate 2D projection
        # For mature-first clustering, we need to use the appropriate vectors
        projection_config = ProjectionConfig()
        if use_mature_first and 'clustering_strategy' in cluster_meta:
            # Use the same vectors that were used for clustering
            if vectors_df is not None and len(vectors_df) > 0:
                coords_df, projection_meta = project_vectors(vectors_df, projection_config)
            else:
                # Fallback: rebuild vectors if needed
                coords_df, projection_meta = project_vectors(vectors_df, projection_config)
        else:
            coords_df, projection_meta = project_vectors(vectors_df, projection_config)
        
        # Step 6: Compute similarity scores  
        # Merge vectors and labels for similarity computation
        api_col = [c for c in vectors_df.columns if not c.startswith("v")][0]
        vectors_with_labels = vectors_df.merge(
            labels_df[[api_col, "label", "cluster_size"]], 
            on=api_col, 
            how="left"
        )
        similarity_config = SimilarityConfig()
        scores_df, similarity_meta = score_similarity(vectors_with_labels, similarity_config)
        
        return {
            # DataFrames
            'joined_df': filtered_df,  # Use filtered for downstream
            'vectors_df': vectors_df,
            'labels_df': labels_df,
            'coords_df': coords_df,
            'scores_df': scores_df,
            
            # Metadata
            'join_stats': join_stats,
            'filter_stats': filter_stats,
            'vector_meta': vector_meta,
            'cluster_meta': cluster_meta,
            'projection_meta': projection_meta,
            'similarity_meta': similarity_meta
        }
        
    except Exception as e:
        raise ValueError(f"Pipeline execution failed: {str(e)}") from e

def run_configurable_pipeline(
    headers_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
    filters_cfg: Dict[str, Any] = None,
    vector_cfg: VectorConfig = None,
    cluster_cfg: ClusterConfig = None,
    projection_cfg: ProjectionConfig = None,
    similarity_cfg: SimilarityConfig = None,
    use_mature_first: bool = True,
    use_production_optimization: bool = True
) -> Dict[str, Any]:
    """
    Run the ML pipeline with custom configurations and production optimization.
    
    Args:
        headers_df: Well headers DataFrame
        monthly_df: Monthly production DataFrame  
        filters_cfg: Filter configuration dictionary
        vector_cfg: Vector generation configuration
        cluster_cfg: Clustering configuration
        projection_cfg: Projection configuration
        similarity_cfg: Similarity scoring configuration
        use_mature_first: Whether to use mature-first clustering
        use_production_optimization: Whether to enable production data optimization
        
    Returns:
        Dictionary containing all pipeline artifacts (same as run_complete_pipeline)
        
    Raises:
        ValueError: If input data is invalid or pipeline fails
    """
    if not isinstance(headers_df, pd.DataFrame) or len(headers_df) == 0:
        raise ValueError("Headers DataFrame is empty or invalid")
    
    if not isinstance(monthly_df, pd.DataFrame) or len(monthly_df) == 0:
        raise ValueError("Monthly DataFrame is empty or invalid")
    
    # Use default configs if none provided
    if filters_cfg is None:
        filters_cfg = {}
    if vector_cfg is None:
        vector_cfg = VectorConfig()
    if cluster_cfg is None:
        cluster_cfg = ClusterConfig()
    if projection_cfg is None:
        projection_cfg = ProjectionConfig()
    if similarity_cfg is None:
        similarity_cfg = SimilarityConfig()
    
    try:
        # Step 1: Join inputs
        joined_df, join_stats = join_headers_and_monthly(headers_df, monthly_df)
        if len(joined_df) == 0:
            raise ValueError("No wells remain after joining data")
        
        # Step 2: Apply filters
        filter_config = FilterConfig(**filters_cfg) if filters_cfg else FilterConfig()
        filter_result = apply_filters(joined_df, filter_config)
        filtered_df = filter_result['filtered']
        filter_stats = filter_result
        if len(filtered_df) == 0:
            raise ValueError("No wells remain after filtering")
        
        # Step 3: Generate vectors with custom config
        vectors_df, vector_meta = build_shape_vectors(filtered_df, vector_cfg)
        if len(vectors_df) == 0:
            raise ValueError("Vector generation failed")
        
        # Step 4: Run clustering with custom config (mature-first or standard) + production optimization
        if use_mature_first:
            labels_df, cluster_meta = run_mature_first_clustering(
                filtered_df, vector_cfg, cluster_cfg
            )
        else:
            labels_df, cluster_meta = run_clustering(vectors_df, cluster_cfg, use_production_optimization)
        
        # Step 5: Generate 2D projection with custom config
        coords_df, projection_meta = project_vectors(vectors_df, projection_cfg)
        
        # Step 6: Compute similarity scores with custom config
        api_col = [c for c in vectors_df.columns if not c.startswith("v")][0]
        vectors_with_labels = vectors_df.merge(
            labels_df[[api_col, "label", "cluster_size"]], 
            on=api_col, 
            how="left"
        )
        scores_df, similarity_meta = score_similarity(vectors_with_labels, similarity_cfg)
        
        return {
            # DataFrames
            'joined_df': filtered_df,  # Use filtered for downstream
            'vectors_df': vectors_df,
            'labels_df': labels_df,
            'coords_df': coords_df,
            'scores_df': scores_df,
            
            # Metadata
            'join_stats': join_stats,
            'filter_stats': filter_stats,
            'vector_meta': vector_meta,
            'cluster_meta': cluster_meta,
            'projection_meta': projection_meta,
            'similarity_meta': similarity_meta
        }
        
    except Exception as e:
        raise ValueError(f"Configurable pipeline execution failed: {str(e)}") from e