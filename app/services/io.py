# app/services/io.py

from __future__ import annotations
from pathlib import Path
from typing import Union, BinaryIO, Tuple, Dict, Any
import pandas as pd

from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv
from app.config.ui_defaults import DEFAULTS

def validate_file_size(file_obj: BinaryIO, max_size_mb: int = DEFAULTS.max_file_size_mb) -> None:
    """Validate uploaded file size."""
    file_obj.seek(0, 2)  # Seek to end
    size_bytes = file_obj.tell()
    file_obj.seek(0)  # Reset to beginning
    
    size_mb = size_bytes / (1024 * 1024)
    if size_mb > max_size_mb:
        raise ValueError(f"File size ({size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)")

def validate_csv_format(file_obj: BinaryIO) -> None:
    """Basic CSV format validation."""
    file_obj.seek(0)
    try:
        # Try to read first few lines as CSV
        sample = pd.read_csv(file_obj, nrows=5)
        if len(sample.columns) < 2:
            raise ValueError("CSV must have at least 2 columns")
        file_obj.seek(0)
    except Exception as e:
        file_obj.seek(0)
        raise ValueError(f"Invalid CSV format: {str(e)}")

def load_headers_from_file(file_source: Union[Path, str, BinaryIO]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load headers CSV from file path or uploaded file object.
    
    Args:
        file_source: File path, string path, or uploaded file object
        
    Returns:
        Tuple of (dataframe, metadata)
        
    Raises:
        ValueError: If file validation fails or required columns missing
    """
    if hasattr(file_source, 'read'):  # File-like object
        validate_file_size(file_source)
        validate_csv_format(file_source)
        # For file objects, we need to save temporarily or use StringIO
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            file_source.seek(0)
            tmp.write(file_source.read())
            tmp_path = tmp.name
        
        try:
            return load_headers_csv(tmp_path)
        finally:
            Path(tmp_path).unlink()  # Clean up
    else:
        # File path
        return load_headers_csv(file_source)

def load_monthly_from_file(file_source: Union[Path, str, BinaryIO]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load monthly production CSV from file path or uploaded file object.
    
    Args:
        file_source: File path, string path, or uploaded file object
        
    Returns:
        Tuple of (dataframe, metadata)
        
    Raises:
        ValueError: If file validation fails or required columns missing
    """
    if hasattr(file_source, 'read'):  # File-like object
        validate_file_size(file_source)
        validate_csv_format(file_source)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.csv', delete=False) as tmp:
            file_source.seek(0)
            tmp.write(file_source.read())
            tmp_path = tmp.name
        
        try:
            return load_monthly_csv(tmp_path)
        finally:
            Path(tmp_path).unlink()  # Clean up
    else:
        # File path
        return load_monthly_csv(file_source)

def save_dataframe_csv(df: pd.DataFrame) -> bytes:
    """Convert DataFrame to CSV bytes for download."""
    return df.to_csv(index=False).encode('utf-8')

def get_dataframe_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Get summary statistics for a DataFrame."""
    return {
        'n_rows': len(df),
        'n_cols': len(df.columns),
        'columns': list(df.columns),
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024 * 1024),
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
    }