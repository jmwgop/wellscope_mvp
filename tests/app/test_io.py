# tests/app/test_io.py

from pathlib import Path
from io import BytesIO
import pandas as pd
import pytest

from app.services.io import (
    validate_file_size, validate_csv_format, load_headers_from_file,
    load_monthly_from_file, save_dataframe_csv, get_dataframe_summary
)

FIXTURES = Path(__file__).parent.parent.parent / "tests" / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"

def test_validate_file_size():
    """Test file size validation."""
    # Small file should pass
    small_data = b"col1,col2\n1,2\n3,4\n"
    small_file = BytesIO(small_data)
    validate_file_size(small_file, max_size_mb=1)  # Should not raise
    
    # Large file should fail
    with pytest.raises(ValueError, match="exceeds limit"):
        validate_file_size(small_file, max_size_mb=0.000001)  # Tiny limit

def test_validate_csv_format():
    """Test CSV format validation."""
    # Valid CSV
    valid_csv = BytesIO(b"col1,col2\n1,2\n3,4\n")
    validate_csv_format(valid_csv)  # Should not raise
    
    # Invalid CSV (too few columns)
    invalid_csv = BytesIO(b"single_col\n1\n2\n")
    with pytest.raises(ValueError, match="at least 2 columns"):
        validate_csv_format(invalid_csv)
    
    # Completely invalid
    bad_csv = BytesIO(b"not,csv,data\x00\xff")
    with pytest.raises(ValueError, match="Invalid CSV format"):
        validate_csv_format(bad_csv)

def test_load_headers_from_file():
    """Test headers loading from file path."""
    df, meta = load_headers_from_file(HEADERS_CSV)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "API14" in df.columns
    assert isinstance(meta, dict)
    assert "n_rows" in meta

def test_load_monthly_from_file():
    """Test monthly loading from file path.""" 
    df, meta = load_monthly_from_file(MONTHLY_CSV)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "API/UWI" in df.columns
    assert isinstance(meta, dict)
    assert "n_rows" in meta

def test_load_from_file_object():
    """Test loading from BytesIO object."""
    # Read actual fixture into memory
    with open(HEADERS_CSV, 'rb') as f:
        data = f.read()
    
    file_obj = BytesIO(data)
    df, meta = load_headers_from_file(file_obj)
    
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "API14" in df.columns

def test_save_dataframe_csv():
    """Test DataFrame to CSV conversion."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    csv_bytes = save_dataframe_csv(df)
    
    assert isinstance(csv_bytes, bytes)
    csv_str = csv_bytes.decode('utf-8')
    assert "col1,col2" in csv_str
    assert "1,a" in csv_str

def test_get_dataframe_summary():
    """Test DataFrame summary generation."""
    df = pd.DataFrame({
        "int_col": [1, 2, 3],
        "str_col": ["a", "b", "c"],
        "float_col": [1.1, 2.2, 3.3]
    })
    
    summary = get_dataframe_summary(df)
    
    assert summary["n_rows"] == 3
    assert summary["n_cols"] == 3
    assert "int_col" in summary["columns"]
    assert "memory_usage_mb" in summary
    assert "dtypes" in summary
    assert len(summary["dtypes"]) == 3