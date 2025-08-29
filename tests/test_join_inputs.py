# tests/test_join_inputs.py

from pathlib import Path

import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly


FIXTURES = Path(__file__).parent / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"


def test_join_happy_path_inner():
    """Inner join on API-14 should produce rows and sensible coverage stats."""
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)

    joined, stats = join_headers_and_monthly(headers_df, monthly_df, how="inner")

    # Basic shape checks
    assert stats["headers_rows"] > 0
    assert stats["monthly_rows"] > 0
    assert stats["joined_rows"] > 0
    assert stats["matched_api14"] <= stats["headers_unique_api14"]
    assert stats["unmatched_headers_api14"] == stats["headers_unique_api14"] - stats["matched_api14"]

    # Ensure the normalized join keys exist post-merge
    # (Names depend on the detected candidate columns; assert existence by suffix pattern)
    join_keys = [c for c in joined.columns if c.endswith("__norm14")]
    assert len(join_keys) >= 2, "Expected normalized join keys to be present in joined data"

    # Sanity: API keys should be 14-digit where present
    norm14_cols = [c for c in join_keys if isinstance(joined[c].iloc[0], (str, type(None)))]
    for c in norm14_cols:
        sample = joined[c].dropna().astype(str)
        if not sample.empty:
            assert sample.map(lambda s: len(s) == 14 and s.isdigit()).all()


def test_join_left_increases_or_equals_rows():
    """Left join should not reduce monthly rows; it should maintain or expand due to headers duplicates."""
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)

    joined_inner, stats_inner = join_headers_and_monthly(headers_df, monthly_df, how="inner")
    joined_left, stats_left = join_headers_and_monthly(headers_df, monthly_df, how="left")

    assert stats_left["joined_rows"] >= stats_inner["joined_rows"]
    # Left join should have at least as many rows as monthly input
    assert stats_left["joined_rows"] >= stats_left["monthly_rows"]


def test_join_raises_when_missing_keys():
    """If neither dataset exposes an API key, join should raise a clear error."""
    # Remove key columns artificially
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)

    # Drop API from headers copy
    bad_headers = headers_df.drop(columns=[c for c in headers_df.columns if c == "API14"], errors="ignore")
    # Drop API candidates from monthly copy
    bad_monthly = monthly_df.drop(
        columns=[c for c in monthly_df.columns if c in ("API14_norm", "API_UWI_norm", "API/UWI")],
        errors="ignore",
    )

    with pytest.raises(ValueError):
        join_headers_and_monthly(bad_headers, monthly_df)

    with pytest.raises(ValueError):
        join_headers_and_monthly(headers_df, bad_monthly)