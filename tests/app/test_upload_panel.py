# tests/app/test_upload_panel.py

import pandas as pd
import pytest

from app.components.upload_panel import (
    render_upload_panel, validate_uploaded_files, get_upload_progress,
    _mock_upload_panel
)

def test_render_upload_panel():
    """Test upload panel renders without error."""
    # This should work in both Streamlit and non-Streamlit environments
    result = render_upload_panel()
    
    # Should return dictionary with expected keys
    expected_keys = {
        'headers_df', 'monthly_df', 'headers_meta', 'monthly_meta', 
        'upload_status', 'files_uploaded', 'errors'
    }
    assert set(result.keys()) == expected_keys
    
    # Initial state should be pending
    assert result['upload_status'] in ['pending', 'error']
    assert isinstance(result['errors'], list)

def test_validate_uploaded_files():
    """Test file validation logic."""
    # Test with None files
    is_valid, errors = validate_uploaded_files(None, None)
    assert not is_valid
    assert len(errors) == 2
    assert any('Headers file is required' in error for error in errors)
    assert any('Monthly production file is required' in error for error in errors)

def test_get_upload_progress():
    """Test upload progress tracking."""
    # No files uploaded
    progress = get_upload_progress(None, None)
    assert progress['progress_pct'] == 0.0
    assert progress['status_text'] == 'No files uploaded'
    
    # Headers only
    headers_df = pd.DataFrame({'API14': ['123']})
    progress = get_upload_progress(headers_df, None)
    assert progress['progress_pct'] == 0.5
    assert progress['headers_uploaded'] is True
    assert progress['monthly_uploaded'] is False
    
    # Both files uploaded
    monthly_df = pd.DataFrame({'API/UWI': ['123']})
    progress = get_upload_progress(headers_df, monthly_df)
    assert progress['progress_pct'] == 1.0
    assert progress['headers_uploaded'] is True
    assert progress['monthly_uploaded'] is True
    assert 'successfully' in progress['status_text']

def test_mock_upload_panel():
    """Test mock upload panel for testing."""
    result = _mock_upload_panel()
    
    expected_keys = {
        'headers_df', 'monthly_df', 'headers_meta', 'monthly_meta', 
        'upload_status', 'files_uploaded', 'errors'
    }
    assert set(result.keys()) == expected_keys
    assert result['upload_status'] == 'pending'
    assert len(result['errors']) > 0  # Should have Streamlit not available error