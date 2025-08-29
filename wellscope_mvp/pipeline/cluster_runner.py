# wellscope_mvp/pipeline/cluster_runner.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ClusterConfig:
    # Prefer HDBSCAN if available; fallback to DBSCAN otherwise
    use_hdbscan: bool = True
    min_cluster_size: int = 20
    min_samples: Optional[int] = None
    metric: str = "euclidean"

    # Fallback DBSCAN params
    eps: float = 0.5

    # Vector selection
    api_col: Optional[str] = None            # if None, auto-detect first non-v* column
    vector_prefix: str = "v"                 # columns starting with this are used as features


def _detect_api_col(df: pd.DataFrame, explicit: Optional[str], vector_prefix: str) -> str:
    if explicit and explicit in df.columns:
        return explicit
    # Heuristic: api column is the first non-vector column
    for c in df.columns:
        if not c.startswith(vector_prefix):
            return c
    raise ValueError("Could not detect API column; provide ClusterConfig.api_col.")


def _extract_vectors(df: pd.DataFrame, vector_prefix: str) -> Tuple[np.ndarray, List[str]]:
    vec_cols = [c for c in df.columns if c.startswith(vector_prefix)]
    if not vec_cols:
        raise ValueError(f"No vector columns found with prefix '{vector_prefix}'.")
    X = df[vec_cols].to_numpy(dtype=float)
    # Replace NaNs/Infs with zeros (conservative)
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, vec_cols


def run_clustering(
    vectors_df: pd.DataFrame,
    cfg: ClusterConfig,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Cluster vectorized wells using HDBSCAN (if installed) or DBSCAN fallback.
    Includes intelligent fallback logic for better results.

    Returns:
      labels_df: columns = [api_col, label, is_noise, cluster_size]
      meta: {'n_samples','n_clusters','n_noise','algorithm_used','fallback_attempts'}
    """
    api_col = _detect_api_col(vectors_df, cfg.api_col, cfg.vector_prefix)
    X, vec_cols = _extract_vectors(vectors_df, cfg.vector_prefix)

    labels: np.ndarray
    algorithm_used = "dbscan_fallback"
    fallback_attempts = 0
    n_samples = len(X)

    # Try HDBSCAN if requested
    if cfg.use_hdbscan:
        try:
            import hdbscan  # type: ignore

            min_samples = cfg.min_samples if cfg.min_samples is not None else cfg.min_cluster_size
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(cfg.min_cluster_size),
                min_samples=int(min_samples),
                metric=cfg.metric,
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(X)
            algorithm_used = "hdbscan"
            
            # Check if HDBSCAN found any clusters
            n_noise = (labels == -1).sum()
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Intelligent fallback: if all noise, try with relaxed parameters
            if n_clusters == 0 and n_samples >= 10:
                fallback_attempts += 1
                relaxed_min_cluster_size = max(2, cfg.min_cluster_size // 2)
                relaxed_min_samples = max(2, min_samples // 2)
                
                clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=int(relaxed_min_cluster_size),
                    min_samples=int(relaxed_min_samples),
                    metric=cfg.metric,
                    cluster_selection_method="leaf",  # More permissive selection
                )
                labels = clusterer.fit_predict(X)
                algorithm_used = "hdbscan_relaxed"
                
                # If still all noise, fall back to DBSCAN
                n_clusters_relaxed = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters_relaxed == 0:
                    fallback_attempts += 1
                    algorithm_used = "dbscan_auto_fallback"
                    labels = _try_dbscan_with_adaptive_eps(X, cfg, n_samples)
                    
        except Exception:
            # Fallback to DBSCAN
            fallback_attempts += 1
            from sklearn.cluster import DBSCAN  # type: ignore

            min_samples = cfg.min_samples if cfg.min_samples is not None else max(5, cfg.min_cluster_size // 2)
            clusterer = DBSCAN(eps=float(cfg.eps), min_samples=int(min_samples), metric=cfg.metric)
            labels = clusterer.fit_predict(X)
            algorithm_used = "dbscan_fallback"
    else:
        # Directly use DBSCAN with intelligent parameter selection
        labels = _try_dbscan_with_adaptive_eps(X, cfg, n_samples)
        algorithm_used = "dbscan_direct"

    # Compute cluster sizes (excluding noise)
    labels_series = pd.Series(labels)
    counts = labels_series[labels_series >= 0].value_counts().to_dict()

    # Build output DataFrame
    out = pd.DataFrame({
        api_col: vectors_df[api_col].values,
        "label": labels.astype(int),
    })
    out["is_noise"] = (out["label"] == -1)
    out["cluster_size"] = out["label"].map(lambda l: counts.get(l, 0))

    n_samples = int(len(out))
    n_noise = int((out["label"] == -1).sum())
    n_clusters = int(len([k for k in counts.keys() if k >= 0]))

    meta = {
        "n_samples": n_samples,
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "algorithm_used": algorithm_used,
        "fallback_attempts": fallback_attempts,
        "noise_fraction": n_noise / n_samples if n_samples > 0 else 1.0,
    }
    return out, meta


def _try_dbscan_with_adaptive_eps(X: np.ndarray, cfg: ClusterConfig, n_samples: int) -> np.ndarray:
    """
    Try DBSCAN with adaptive epsilon selection for better results.
    """
    from sklearn.cluster import DBSCAN  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
    
    min_samples = cfg.min_samples if cfg.min_samples is not None else max(3, min(cfg.min_cluster_size // 2, n_samples // 10))
    
    # If user provided eps, try it first
    if hasattr(cfg, 'eps') and cfg.eps != 0.5:  # 0.5 is default, so try user value first
        clusterer = DBSCAN(eps=float(cfg.eps), min_samples=int(min_samples), metric=cfg.metric)
        labels = clusterer.fit_predict(X)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 0:  # Found clusters, return
            return labels
    
    # Adaptive epsilon selection using k-distance method
    try:
        # Calculate k-nearest neighbor distances
        k = min(min_samples + 1, n_samples - 1)
        nbrs = NearestNeighbors(n_neighbors=k, metric=cfg.metric).fit(X)
        distances, _ = nbrs.kneighbors(X)
        
        # Sort k-distances and find elbow point (simple heuristic)
        k_distances = np.sort(distances[:, k-1])  # k-th nearest neighbor distances
        
        # Try a few epsilon values around the median and 75th percentile
        candidate_eps = [
            np.percentile(k_distances, 50),   # Median
            np.percentile(k_distances, 75),   # 75th percentile 
            np.percentile(k_distances, 85),   # 85th percentile
        ]
        
        best_labels = None
        best_n_clusters = 0
        
        for eps in candidate_eps:
            if eps <= 0:
                continue
            clusterer = DBSCAN(eps=eps, min_samples=int(min_samples), metric=cfg.metric)
            labels = clusterer.fit_predict(X)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Prefer solutions with reasonable number of clusters (not too many, not zero)
            if n_clusters > best_n_clusters and n_clusters <= n_samples // 3:
                best_labels = labels
                best_n_clusters = n_clusters
        
        if best_labels is not None:
            return best_labels
            
    except Exception:
        pass  # Fall through to default
    
    # Final fallback: use default parameters
    clusterer = DBSCAN(eps=float(cfg.eps), min_samples=int(min_samples), metric=cfg.metric)
    return clusterer.fit_predict(X)