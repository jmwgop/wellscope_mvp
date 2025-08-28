# tests/test_monthly_loader.py

from pathlib import Path
import re

import pandas as pd
import pytest

from wellscope_mvp.data.monthly_loader import load_monthly_csv
from wellscope_mvp.schema.monthly_schema import (
    REQUIRED_FIELDS,
    DATE_FIELDS,
    FLOAT_FIELDS,
    INT_FIELDS,
    KEY_FIELDS,
    validate_columns,
)

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "Producing Entity Monthly Production.CSV"


def test_fixture_has_required_columns():
    """The fixture CSV itself should include all required monthly columns."""
    sample = pd.read_csv(FIXTURE_PATH, nrows=1)
    ok, missing = validate_columns(sample.columns.tolist())
    assert ok, f"Fixture missing required monthly columns: {missing}"


def test_load_monthly_csv_happy_path():
    """Loader returns a DataFrame with required columns and sensible coercions."""
    df, meta = load_monthly_csv(FIXTURE_PATH)

    # Meta sanity
    assert isinstance(meta, dict)
    assert meta.get("n_rows", 0) > 0

    # Required columns preserved (plus pass-through extras)
    for col in REQUIRED_FIELDS:
        assert col in df.columns, f"Missing required column after load: {col}"

    # Normalized API/UWI column exists and is digits-only
    assert "API_UWI_norm" in df.columns, "Expected normalized API column 'API_UWI_norm'"
    # All should be digits (14-digit API format)
    mask_digits = df["API_UWI_norm"].dropna().map(lambda s: str(s).isdigit())
    assert mask_digits.all(), "API_UWI_norm should be digits-only"

    # Date coercion
    for c in DATE_FIELDS:
        if c in df.columns:
            assert pd.api.types.is_datetime64_any_dtype(df[c]), f"{c} should be datetime64"

    # Float coercion (allow NaN but dtype must be numeric)
    for c in FLOAT_FIELDS:
        if c in df.columns:
            assert (
                pd.api.types.is_float_dtype(df[c]) or pd.api.types.is_integer_dtype(df[c])
            ), f"{c} should be numeric after coercion"

    # Int coercion (nullable Int64)
    for c in INT_FIELDS:
        if c in df.columns:
            assert str(df[c].dtype) == "Int64", f"{c} should be pandas nullable Int64"


def test_loader_keeps_extra_columns():
    """Loader should not drop extra columns beyond the required set."""
    raw = pd.read_csv(FIXTURE_PATH, nrows=1)
    df, _ = load_monthly_csv(FIXTURE_PATH)
    assert set(REQUIRED_FIELDS).issubset(df.columns)
    assert len(df.columns) >= len(raw.columns), "Loader must preserve extra columns"


def test_missing_required_column_raises(tmp_path: Path):
    """If a required monthly column is missing, loader should raise ValueError."""
    raw = pd.read_csv(FIXTURE_PATH)
    # Drop a non-key required column to simulate schema failure
    to_drop = next(col for col in REQUIRED_FIELDS if col not in KEY_FIELDS)
    broken = raw.drop(columns=[to_drop])

    broken_path = tmp_path / "broken_monthly.csv"
    broken.to_csv(broken_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        load_monthly_csv(broken_path)

    msg = str(excinfo.value)
    assert "Missing required monthly columns" in msg
    assert to_drop in msg