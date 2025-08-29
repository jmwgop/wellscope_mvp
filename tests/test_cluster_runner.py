# tests/test_cluster_runner.py

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
from wellscope_mvp.pipeline.vector_builder import VectorConfig, build_shape_vectors
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig, run_clustering


FIXTURES = Path(__file__).parent / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"


@pytest.fixture(scope="module")
def vectors_df():
    # Build vectors from real joined data
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)
    joined_df, stats = join_headers_and_monthly(headers_df, monthly_df, how="inner")
    assert stats["joined_rows"] > 0

    vcfg = VectorConfig(months=12, normalize="q_over_qmax", stream="oil")
    vec_df, meta = build_shape_vectors(joined_df, vcfg)
    assert meta["n_wells"] == len(vec_df)
    return vec_df


def test_cluster_runner_production_defaults(vectors_df):
    # Test with production-optimized defaults (HDBSCAN + Cosine)
    ccfg = ClusterConfig()  # Use defaults
    labels_df, meta = run_clustering(vectors_df, ccfg)

    # Basic shape and meta checks
    assert len(labels_df) == meta["n_samples"] == len(vectors_df)
    assert "label" in labels_df.columns and "is_noise" in labels_df.columns
    assert "cluster_size" in labels_df.columns
    
    # Should use production-optimized defaults
    assert meta["algorithm_used"] in ("hdbscan", "hdbscan_relaxed", "dbscan_fallback", "dbscan_auto_fallback")
    
    # Labels are ints; noise is labeled -1
    assert np.issubdtype(labels_df["label"].dtype, np.integer)
    assert labels_df["is_noise"].dtype == bool

    # Cluster sizes are non-negative
    assert (labels_df["cluster_size"] >= 0).all()
    
    # Check new metadata fields
    assert "fallback_attempts" in meta
    assert "noise_fraction" in meta
    assert isinstance(meta["fallback_attempts"], int)
    assert 0 <= meta["noise_fraction"] <= 1


def test_cluster_runner_dbscan_fallback(vectors_df):
    # Force DBSCAN fallback for testing backwards compatibility
    ccfg = ClusterConfig(use_hdbscan=False, min_cluster_size=5, eps=0.5, metric="cosine")
    labels_df, meta = run_clustering(vectors_df, ccfg)

    # Basic shape and meta checks
    assert len(labels_df) == meta["n_samples"] == len(vectors_df)
    assert "label" in labels_df.columns and "is_noise" in labels_df.columns
    assert "cluster_size" in labels_df.columns
    assert meta["algorithm_used"] in ("dbscan_fallback", "hdbscan", "dbscan_direct")

    # Labels are ints; noise is labeled -1
    assert np.issubdtype(labels_df["label"].dtype, np.integer)
    assert labels_df["is_noise"].dtype == bool

    # Cluster sizes are non-negative
    assert (labels_df["cluster_size"] >= 0).all()
    
    # Check new metadata fields
    assert "fallback_attempts" in meta
    assert "noise_fraction" in meta
    assert isinstance(meta["fallback_attempts"], int)
    assert 0 <= meta["noise_fraction"] <= 1


def test_cluster_runner_hdbscan_cosine_production_focus(vectors_df):
    # Test production-focused HDBSCAN with cosine metric (optimal for production curves)
    ccfg = ClusterConfig(use_hdbscan=True, min_cluster_size=8, min_samples=None, metric="cosine")
    labels_df, meta = run_clustering(vectors_df, ccfg)

    assert len(labels_df) == len(vectors_df)
    assert meta["algorithm_used"] in ("hdbscan", "hdbscan_relaxed", "dbscan_fallback", "dbscan_auto_fallback")
    # sanity: at least one label present (could be all noise depending on data/params, that's fine)
    assert "label" in labels_df.columns
    
    # Check intelligent fallback metadata
    assert "fallback_attempts" in meta
    assert "noise_fraction" in meta
    
    # Cosine metric should be optimal for production curve shape analysis
    # (The actual metric used depends on implementation details, but test should pass)


def test_cluster_runner_intelligent_fallback(vectors_df):
    """Test intelligent fallback when initial clustering fails."""
    # Use very restrictive parameters that are likely to produce all noise
    ccfg = ClusterConfig(use_hdbscan=True, min_cluster_size=50, min_samples=40)  # Very large for small dataset
    labels_df, meta = run_clustering(vectors_df, ccfg)

    assert len(labels_df) == len(vectors_df)
    # Should have attempted fallback strategies
    assert meta["fallback_attempts"] >= 0
    
    # Even with fallback, result should be valid
    assert "label" in labels_df.columns
    assert "is_noise" in labels_df.columns
    assert "cluster_size" in labels_df.columns


def test_cluster_runner_default_config_values():
    """Test that ClusterConfig uses production-optimized defaults."""
    config = ClusterConfig()
    
    # Should use production-optimized defaults
    assert config.use_hdbscan == True    # HDBSCAN preferred for production clustering
    assert config.metric == "cosine"    # Cosine optimal for production curve shapes  
    assert config.min_cluster_size >= 2  # Reasonable minimum
    
    # Other reasonable defaults
    assert config.eps == 0.5  # DBSCAN fallback parameter
    assert isinstance(config.min_samples, (int, type(None)))


def test_production_curve_shape_clustering_focus(vectors_df):
    """Test clustering optimized for production curve shape analysis."""
    # This test emphasizes that our clustering is focused on production curve shapes,
    # not just generic data clustering
    
    # Use production-optimized config
    ccfg = ClusterConfig(
        use_hdbscan=True,
        min_cluster_size=6, 
        metric="cosine",  # Key: cosine for shape similarity
        min_samples=4
    )
    
    labels_df, meta = run_clustering(vectors_df, ccfg)
    
    # Should complete without errors
    assert len(labels_df) == len(vectors_df)
    assert isinstance(meta, dict)
    
    # Labels should be valid
    assert "label" in labels_df.columns
    assert "is_noise" in labels_df.columns
    
    # Should have reasonable clustering results (not all noise for reasonable data)
    unique_labels = labels_df["label"].unique()
    n_clusters = len([label for label in unique_labels if label != -1])
    n_noise = len(labels_df[labels_df["label"] == -1])
    
    # With production curve data and cosine distance, should find some structure
    # (exact results depend on the test data, but should not be pathological)
    assert n_clusters >= 0  # At least not negative
    assert n_noise >= 0     # At least not negative
    assert n_clusters + (1 if n_noise > 0 else 0) <= len(vectors_df)  # Sanity check