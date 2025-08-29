# tests/test_vector_builder.py

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
from wellscope_mvp.pipeline.vector_builder import VectorConfig, build_shape_vectors


FIXTURES = Path(__file__).parent / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"


@pytest.fixture(scope="module")
def joined():
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)
    joined_df, stats = join_headers_and_monthly(headers_df, monthly_df, how="inner")
    assert stats["joined_rows"] > 0
    return joined_df


def test_build_vectors_q_over_qmax_oil(joined):
    cfg = VectorConfig(months=12, normalize="q_over_qmax", stream="oil")
    vec_df, meta = build_shape_vectors(joined, cfg)

    # Shape assertions
    assert meta["n_features"] == 12
    assert vec_df.shape[1] == 1 + 12  # API col + 12 features
    assert meta["n_wells"] == len(vec_df)

    # Values in [0,1] for q_over_qmax
    feature_cols = [c for c in vec_df.columns if c.startswith("v")]
    arr = vec_df[feature_cols].to_numpy()
    assert np.isfinite(arr).all()
    assert (arr >= 0.0).all() and (arr <= 1.0).all()


def test_build_vectors_pct_decline_boe(joined):
    cfg = VectorConfig(months=18, normalize="pct_decline", stream="boe", boe_gas_factor=6.0)
    vec_df, meta = build_shape_vectors(joined, cfg)

    assert meta["n_features"] == 18
    feature_cols = [c for c in vec_df.columns if c.startswith("v")]
    arr = vec_df[feature_cols].to_numpy()

    # First element should be 0 (by construction), length matches n
    assert np.allclose(arr[:, 0], 0.0, atol=1e-12)
    assert arr.shape[1] == 18
    # Finite numbers
    assert np.isfinite(arr).all()


def test_vector_builder_raises_without_date(joined):
    # Remove monthly date columns to force error
    bad = joined.drop(columns=[c for c in joined.columns if "Monthly Production Date" in c], errors="ignore")
    cfg = VectorConfig(months=6)
    with pytest.raises(ValueError):
        build_shape_vectors(bad, cfg)