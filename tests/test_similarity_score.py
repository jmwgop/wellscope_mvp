# tests/test_similarity_score.py

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
from wellscope_mvp.pipeline.vector_builder import VectorConfig, build_shape_vectors
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig, run_clustering
from wellscope_mvp.pipeline.similarity_score import SimilarityConfig, score_similarity


FIXTURES = Path(__file__).parent / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"


@pytest.fixture(scope="module")
def clustered_vectors():
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)
    joined_df, stats = join_headers_and_monthly(headers_df, monthly_df, how="inner")
    assert stats["joined_rows"] > 0

    vcfg = VectorConfig(months=12, normalize="q_over_qmax", stream="oil")
    vec_df, _ = build_shape_vectors(joined_df, vcfg)

    # Use DBSCAN fallback to avoid dependency on hdbscan
    ccfg = ClusterConfig(use_hdbscan=False, min_cluster_size=5, eps=0.6)
    labels_df, meta = run_clustering(vec_df, ccfg)

    # Merge labels back to vectors
    api_col = [c for c in vec_df.columns if not c.startswith("v")][0]
    merged = vec_df.merge(labels_df[[api_col, "label", "cluster_size"]], on=api_col, how="left")
    return merged


def test_similarity_scores_basic(clustered_vectors):
    scfg = SimilarityConfig(within_cluster_only=True, drop_noise=True)
    scores_df, meta = score_similarity(clustered_vectors, scfg)

    # Shape and columns
    assert len(scores_df) == meta["n_scored"]
    assert {"similarity", "cluster_size", "cluster_avg_similarity", "label"}.issubset(scores_df.columns)

    # Similarity is finite and within [-1, 1]
    sims = scores_df["similarity"].to_numpy()
    assert np.isfinite(sims).all()
    assert (sims >= -1.0).all() and (sims <= 1.0).all()

    # Cluster averages computed where cluster_size > 0
    if (scores_df["cluster_size"] > 0).any():
        grouped = scores_df.groupby("label")["similarity"].mean()
        # compare against reported column
        check = scores_df.groupby("label")["cluster_avg_similarity"].first()
        # allow tiny numeric noise
        diff = (grouped - check).abs().max()
        assert diff < 1e-9


def test_similarity_including_noise(clustered_vectors):
    scfg = SimilarityConfig(within_cluster_only=True, drop_noise=False)
    scores_df, meta = score_similarity(clustered_vectors, scfg)

    # Noise rows should exist if any -1 labels present
    has_noise = (clustered_vectors["label"] == -1).any()
    if has_noise:
        assert (scores_df["label"] == -1).any()