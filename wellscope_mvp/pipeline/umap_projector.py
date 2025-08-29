# wellscope_mvp/pipeline/umap_projector.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectionConfig:
    n_components: int = 2
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    api_col: Optional[str] = None
    vector_prefix: str = "v"


def _detect_api_col(df: pd.DataFrame, explicit: Optional[str], vector_prefix: str) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for c in df.columns:
        if not c.startswith(vector_prefix):
            return c
    raise ValueError("Could not detect API column for projection.")


def _extract_vectors(df: pd.DataFrame, vector_prefix: str) -> Tuple[np.ndarray, List[str]]:
    vec_cols = [c for c in df.columns if c.startswith(vector_prefix)]
    if not vec_cols:
        raise ValueError(f"No vector columns found with prefix '{vector_prefix}'.")
    X = df[vec_cols].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, vec_cols


def project_vectors(
    vectors_df: pd.DataFrame,
    cfg: ProjectionConfig,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Project high-dimensional vectors to 2D (or nD). Uses UMAP if available; falls back to PCA.
    Returns:
      coords_df: [api_col, x, y] (or x0..x{n-1})
      meta: {'algorithm_used': 'umap'|'pca_fallback'}
    """
    api_col = _detect_api_col(vectors_df, cfg.api_col, cfg.vector_prefix)
    X, vec_cols = _extract_vectors(vectors_df, cfg.vector_prefix)

    algorithm_used = "pca_fallback"
    components: np.ndarray

    if cfg.n_components < 1:
        raise ValueError("n_components must be >= 1")

    # Try UMAP
    try:
        import umap  # type: ignore

        reducer = umap.UMAP(
            n_neighbors=int(cfg.n_neighbors),
            n_components=int(cfg.n_components),
            min_dist=float(cfg.min_dist),
            metric=cfg.metric,
            random_state=int(cfg.random_state),
            verbose=False,
        )
        components = reducer.fit_transform(X)
        algorithm_used = "umap"
    except Exception:
        # PCA fallback
        from sklearn.decomposition import PCA  # type: ignore

        reducer = PCA(n_components=int(cfg.n_components), random_state=int(cfg.random_state))
        components = reducer.fit_transform(X)
        algorithm_used = "pca_fallback"

    # Build output frame
    coord_cols = [f"x{j}" if cfg.n_components > 2 else ("x" if j == 0 else "y") for j in range(cfg.n_components)]
    coords_df = pd.DataFrame(components, columns=coord_cols)
    coords_df.insert(0, api_col, vectors_df[api_col].values)

    meta = {"algorithm_used": algorithm_used}
    return coords_df, meta