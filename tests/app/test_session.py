# tests/app/test_session.py

import pandas as pd
import pytest

from app.state.session import (
    SessionKeys, get_session_value, set_session_value, clear_session_keys,
    has_required_data, get_dataframe, set_dataframe, clear_pipeline_outputs,
    has_uploaded_data, has_analysis_results
)

def test_session_keys():
    """Test that session keys are properly defined."""
    assert hasattr(SessionKeys, 'HEADERS_DF')
    assert hasattr(SessionKeys, 'MONTHLY_DF')
    assert isinstance(SessionKeys.HEADERS_DF, str)

def test_get_set_session_value():
    """Test basic session value operations."""
    session = {}
    
    # Set and get
    set_session_value(session, "test_key", "test_value")
    assert get_session_value(session, "test_key") == "test_value"
    
    # Default value for missing key
    assert get_session_value(session, "missing_key", "default") == "default"
    assert get_session_value(session, "missing_key") is None

def test_dataframe_operations():
    """Test DataFrame-specific session operations."""
    session = {}
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    
    # Set DataFrame
    set_dataframe(session, SessionKeys.HEADERS_DF, df)
    retrieved_df = get_dataframe(session, SessionKeys.HEADERS_DF)
    
    assert isinstance(retrieved_df, pd.DataFrame)
    assert len(retrieved_df) == 3
    assert list(retrieved_df.columns) == ["col1", "col2"]
    
    # Non-DataFrame returns None
    session["not_df"] = {"key": "value"}
    assert get_dataframe(session, "not_df") is None
    
    # Type validation
    with pytest.raises(TypeError):
        set_dataframe(session, "test", "not_a_dataframe")

def test_clear_operations():
    """Test session clearing operations."""
    session = {
        "key1": "value1",
        "key2": "value2", 
        "key3": "value3"
    }
    
    # Clear specific keys
    clear_session_keys(session, ["key1", "key3"])
    assert "key1" not in session
    assert "key2" in session
    assert "key3" not in session
    
    # Clear pipeline outputs
    session.update({
        SessionKeys.HEADERS_DF: pd.DataFrame(),
        SessionKeys.JOINED_DF: pd.DataFrame(),
        SessionKeys.VECTORS_DF: pd.DataFrame(),
        SessionKeys.JOIN_STATS: {"test": 1}
    })
    
    clear_pipeline_outputs(session)
    assert SessionKeys.HEADERS_DF in session  # preserved
    assert SessionKeys.JOINED_DF not in session  # cleared
    assert SessionKeys.VECTORS_DF not in session  # cleared
    assert SessionKeys.JOIN_STATS not in session  # cleared

def test_has_required_data():
    """Test validation functions."""
    session = {
        SessionKeys.HEADERS_DF: pd.DataFrame({"col": [1, 2]}),
        SessionKeys.MONTHLY_DF: pd.DataFrame({"col": [3, 4]}),
        SessionKeys.VECTORS_DF: None
    }
    
    # Has uploaded data
    assert has_uploaded_data(session) is True
    
    # Missing required data for analysis
    assert has_analysis_results(session) is False
    
    # Generic required data check
    assert has_required_data(session, [SessionKeys.HEADERS_DF, SessionKeys.MONTHLY_DF]) is True
    assert has_required_data(session, [SessionKeys.VECTORS_DF]) is False
    assert has_required_data(session, ["missing_key"]) is False