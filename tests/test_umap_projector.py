# tests/test_umap_projector.py

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
from wellscope_mvp.pipeline.vector_builder import VectorConfig, build_shape_vectors
from wellscope_mvp.pipeline.umap_projector import ProjectionConfig, project_vectors


FIXTURES = Path(__file__).parent / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"


@pytest.fixture(scope="module")
def vectors_df():
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)
    joined_df, stats = join_headers_and_monthly(headers_df, monthly_df, how="inner")
    assert stats["joined_rows"] > 0

    vcfg = VectorConfig(months=12, normalize="q_over_qmax", stream="oil")
    vec_df, meta = build_shape_vectors(joined_df, vcfg)
    return vec_df


def test_project_vectors_default(vectors_df):
    pcfg = ProjectionConfig(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    coords_df, meta = project_vectors(vectors_df, pcfg)

    # shape checks
    assert coords_df.shape[0] == len(vectors_df)
    assert {"x", "y"}.issubset(set(coords_df.columns))
    # numeric & finite
    arr = coords_df[["x", "y"]].to_numpy()
    assert np.isfinite(arr).all()
    # algo info present
    assert meta["algorithm_used"] in ("umap", "pca_fallback")


def test_project_vectors_n_components_3(vectors_df):
    pcfg = ProjectionConfig(n_components=3, random_state=42)
    coords_df, meta = project_vectors(vectors_df, pcfg)
    assert {"x0", "x1", "x2"}.issubset(set(coords_df.columns))
    arr = coords_df[["x0", "x1", "x2"]].to_numpy()
    assert np.isfinite(arr).all()