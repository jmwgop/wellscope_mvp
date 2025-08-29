# wellscope_mvp/pipeline/similarity_score.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.linalg import norm


@dataclass(frozen=True)
class SimilarityConfig:
    api_col: Optional[str] = None            # if None, auto-detect first non-vector column
    label_col: str = "label"                 # clustering label column
    vector_prefix: str = "v"                 # feature columns start with this prefix
    within_cluster_only: bool = True         # ignore noise and compute per-cluster
    drop_noise: bool = True                  # drop label == -1 from results


def _detect_api_col(df: pd.DataFrame, explicit: Optional[str], vector_prefix: str) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for c in df.columns:
        if not c.startswith(vector_prefix):
            return c
    raise ValueError("Could not detect API column for similarity scoring.")


def _extract_vectors(df: pd.DataFrame, vector_prefix: str) -> Tuple[np.ndarray, List[str]]:
    vec_cols = [c for c in df.columns if c.startswith(vector_prefix)]
    if not vec_cols:
        raise ValueError(f"No vector columns found with prefix '{vector_prefix}'.")
    X = df[vec_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, vec_cols


def _cosine_sim(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row in A to vector b."""
    b_norm = norm(b)
    if b_norm == 0:
        return np.zeros(A.shape[0], dtype=float)
    A_norms = norm(A, axis=1)
    # Avoid divide-by-zero
    denom = np.where(A_norms == 0, 1.0, A_norms) * b_norm
    sims = (A @ b) / denom
    # If any A row is zero, force similarity to 0
    sims = np.where(A_norms == 0, 0.0, sims)
    # Clamp numerical noise
    return np.clip(sims, -1.0, 1.0)


def score_similarity(
    vectors_with_labels: pd.DataFrame,
    cfg: SimilarityConfig,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute cosine similarity of each well to its cluster centroid.

    Returns:
      scores_df: [api_col, label, similarity, cluster_size, cluster_avg_similarity]
      meta: {'n_scored','n_clusters','mean_similarity'}
    """
    df = vectors_with_labels.copy()

    if cfg.label_col not in df.columns:
        raise ValueError(f"Label column '{cfg.label_col}' not found.")

    api_col = _detect_api_col(df, cfg.api_col, cfg.vector_prefix)
    X, vec_cols = _extract_vectors(df, cfg.vector_prefix)

    labels = df[cfg.label_col].astype(int).to_numpy()
    is_noise = (labels == -1)

    # Optionally drop noise
    if cfg.drop_noise:
        keep_mask = ~is_noise
        df = df.loc[keep_mask].reset_index(drop=True)
        X = X[keep_mask]
        labels = labels[keep_mask]

    # If no rows left, return empty
    if len(df) == 0:
        out = pd.DataFrame(columns=[api_col, cfg.label_col, "similarity", "cluster_size", "cluster_avg_similarity"])
        return out, {"n_scored": 0, "n_clusters": 0, "mean_similarity": 0.0}

    # Compute centroids per cluster (vector mean)
    centroids: Dict[int, np.ndarray] = {}
    sizes: Dict[int, int] = {}
    for lab in np.unique(labels):
        if lab == -1 and cfg.within_cluster_only:
            continue
        mask = (labels == lab)
        if not mask.any():
            continue
        centroids[lab] = X[mask].mean(axis=0)
        sizes[lab] = int(mask.sum())

    # Score similarity to own centroid
    sims = np.zeros(len(df), dtype=float)
    for lab in np.unique(labels):
        mask = (labels == lab)
        if not mask.any():
            continue
        c = centroids.get(lab, None)
        if c is None:
            sims[mask] = 0.0
        else:
            sims[mask] = _cosine_sim(X[mask], c)

    out = pd.DataFrame({
        api_col: df[api_col].values,
        cfg.label_col: labels,
        "similarity": sims,
    })
    out["cluster_size"] = out[cfg.label_col].map(lambda l: sizes.get(int(l), 0))

    # Cluster average similarity
    cluster_avg = out.groupby(cfg.label_col)["similarity"].mean().to_dict()
    out["cluster_avg_similarity"] = out[cfg.label_col].map(lambda l: float(cluster_avg.get(int(l), 0.0)))

    meta = {
        "n_scored": int(len(out)),
        "n_clusters": int(len([k for k in sizes.keys()])),
        "mean_similarity": float(out["similarity"].mean()) if len(out) else 0.0,
    }
    return out, meta