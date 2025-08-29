# tests/test_filter_inputs.py

from pathlib import Path
import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.pipeline.join_inputs import join_headers_and_monthly
from wellscope_mvp.pipeline.filter_inputs import FilterConfig, apply_filters, compute_months_produced


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


def test_compute_months_produced_basic(joined):
    # Expect non-negative integers and at least some wells with months >= 1
    # Determine API column used in join (prefer normalized keys)
    api_cols = [c for c in joined.columns if c.endswith("__norm14")] or ["API14"]
    api_col = api_cols[0] if api_cols[0] in joined.columns else "API14"
    months = compute_months_produced(joined, api_col=api_col)
    assert (months >= 0).all()
    assert months.dtype.kind in ("i",)  # integer
    assert months.max() >= 1  # at least some wells have production history


def test_filter_by_formation_when_present(joined):
    if "Target Formation" not in joined.columns:
        pytest.skip("No formation column in joined data")
    # pick a formation value that exists
    sample_formations = (
        joined["Target Formation"].dropna().astype(str).value_counts().head(1).index.tolist()
    )
    cfg = FilterConfig(formations=sample_formations, min_months_produced=0)
    result = apply_filters(joined, cfg)
    filtered = result["filtered"]

    assert len(filtered) <= len(joined)
    if sample_formations:
        assert filtered["Target Formation"].isin(sample_formations).all()


def test_filter_by_completion_year_range(joined):
    if "Completion Date" not in joined.columns:
        pytest.skip("No completion date in joined data")
    # derive year range from data
    years = pd.to_datetime(joined["Completion Date"], errors="coerce").dt.year.dropna()
    if years.empty:
        pytest.skip("No valid completion years in data")
    lo, hi = int(years.min()), int(years.max())
    mid = max(lo, hi - 1)  # a narrow range near the top to induce some filtering
    cfg = FilterConfig(completion_year_range=(mid, hi))
    result = apply_filters(joined, cfg)
    filtered = result["filtered"]

    assert len(filtered) <= len(joined)
    fyears = pd.to_datetime(filtered["Completion Date"], errors="coerce").dt.year.dropna()
    if not fyears.empty:
        assert (fyears >= mid).all() and (fyears <= hi).all()


def test_filter_by_lateral_range_if_available(joined):
    # find a lateral column present
    lat_col = None
    for c in ("DI Lateral Length", "Horizontal Length"):
        if c in joined.columns:
            lat_col = c
            break
    if lat_col is None:
        pytest.skip("No lateral length column in joined data")

    vals = pd.to_numeric(joined[lat_col], errors="coerce").dropna()
    if vals.empty:
        pytest.skip("No numeric lateral values")
    lo, hi = float(vals.quantile(0.25)), float(vals.quantile(0.75))
    cfg = FilterConfig(lateral_ft_range=(lo, hi))
    result = apply_filters(joined, cfg)
    filtered = result["filtered"]
    fvals = pd.to_numeric(filtered[lat_col], errors="coerce").dropna()

    assert len(filtered) <= len(joined)
    # Allow NaNs to pass through; check bounds where present
    if not fvals.empty:
        assert (fvals >= lo).all() and (fvals <= hi).all()


def test_filter_by_min_months_produced(joined):
    # choose a threshold that likely filters something but not everything
    cfg = FilterConfig(min_months_produced=6)
    result = apply_filters(joined, cfg)
    filtered = result["filtered"]

    assert len(filtered) <= len(joined)
    # Verify months_produced >= threshold for filtered rows
    api_cols = [c for c in joined.columns if c.endswith("__norm14")] or ["API14"]
    api_col = api_cols[0] if api_cols[0] in joined.columns else "API14"
    months = compute_months_produced(filtered, api_col=api_col)
    if len(filtered) > 0:
        assert (months >= 6).all()