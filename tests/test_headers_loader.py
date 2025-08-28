from pathlib import Path
import pandas as pd
import pytest

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.schema.headers_schema import REQUIRED_FIELDS, validate_columns

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "Well Headers.CSV"


def test_fixture_has_required_columns():
    """Fixture itself should contain all required headers."""
    sample = pd.read_csv(FIXTURE_PATH, nrows=1)
    ok, missing = validate_columns(sample.columns.tolist())
    assert ok, f"Fixture missing required columns: {missing}"


def test_load_headers_csv_happy_path():
    """Loader returns a DataFrame with required columns and no missing API14s."""
    df, meta = load_headers_csv(FIXTURE_PATH)

    # Meta sanity
    assert meta["n_rows"] > 0

    # Required columns
    for col in REQUIRED_FIELDS:
        assert col in df.columns

    # API14 normalized and 14 digits
    assert df["API14"].notna().all()
    assert df["API14"].map(lambda s: len(str(s)) == 14 and str(s).isdigit()).all()


def test_loader_keeps_extra_columns():
    """Extras in the CSV should be preserved."""
    raw = pd.read_csv(FIXTURE_PATH, nrows=1)
    df, _ = load_headers_csv(FIXTURE_PATH)
    assert set(REQUIRED_FIELDS).issubset(df.columns)
    assert len(df.columns) >= len(raw.columns)


def test_missing_required_column_raises(tmp_path: Path):
    """If a required column is missing, loader should raise ValueError."""
    raw = pd.read_csv(FIXTURE_PATH)
    # drop one required field
    broken = raw.drop(columns=[REQUIRED_FIELDS[0]])
    broken_path = tmp_path / "broken.csv"
    broken.to_csv(broken_path, index=False)

    with pytest.raises(ValueError):
        load_headers_csv(broken_path)