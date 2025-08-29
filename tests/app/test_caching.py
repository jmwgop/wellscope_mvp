# tests/app/test_caching.py

from pathlib import Path
import time
import pandas as pd
import pytest

from app.services.caching import (
    cached_run_pipeline, clear_pipeline_cache, get_cache_info,
    _hash_dataframe, _generate_cache_key
)
from wellscope_mvp.data.headers_loader import load_headers_csv
from wellscope_mvp.data.monthly_loader import load_monthly_csv

FIXTURES = Path(__file__).parent.parent.parent / "tests" / "fixtures"
HEADERS_CSV = FIXTURES / "Well Headers.CSV"
MONTHLY_CSV = FIXTURES / "Producing Entity Monthly Production.CSV"

@pytest.fixture
def sample_data():
    """Load sample data for testing."""
    headers_df, _ = load_headers_csv(HEADERS_CSV)
    monthly_df, _ = load_monthly_csv(MONTHLY_CSV)
    return headers_df, monthly_df

def test_hash_dataframe_consistency():
    """Test that DataFrame hashing is consistent."""
    df1 = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    df2 = pd.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
    df3 = pd.DataFrame({'col1': [1, 2, 4], 'col2': ['a', 'b', 'c']})
    
    # Same content should have same hash
    assert _hash_dataframe(df1) == _hash_dataframe(df2)
    
    # Different content should have different hash
    assert _hash_dataframe(df1) != _hash_dataframe(df3)
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    assert _hash_dataframe(empty_df) == "empty_df"

def test_generate_cache_key():
    """Test cache key generation."""
    df1 = pd.DataFrame({'col1': [1, 2]})
    df2 = pd.DataFrame({'col2': ['a', 'b']})
    
    # Same inputs should generate same key
    key1 = _generate_cache_key(df1, df2, {'param': 'value'})
    key2 = _generate_cache_key(df1, df2, {'param': 'value'})
    assert key1 == key2
    
    # Different filters should generate different keys
    key3 = _generate_cache_key(df1, df2, {'param': 'different'})
    assert key1 != key3
    
    # No filters vs empty dict should be the same
    key4 = _generate_cache_key(df1, df2, None)
    key5 = _generate_cache_key(df1, df2, {})
    assert key4 == key5

def test_cached_pipeline_fallback_mode(sample_data):
    """Test cached pipeline when Streamlit is not available (fallback mode)."""
    headers_df, monthly_df = sample_data
    
    # Clear cache before test
    clear_pipeline_cache()
    
    # First call (cache miss) - force fallback mode for testing
    start_time = time.time()
    result1 = cached_run_pipeline(headers_df, monthly_df, force_fallback=True)
    first_call_time = time.time() - start_time
    
    # Verify result structure
    expected_keys = {
        'joined_df', 'vectors_df', 'labels_df', 'coords_df', 'scores_df',
        'join_stats', 'filter_stats', 'vector_meta', 'cluster_meta',
        'projection_meta', 'similarity_meta'
    }
    assert set(result1.keys()) == expected_keys
    
    # Second call (should be cached and faster)
    start_time = time.time()
    result2 = cached_run_pipeline(headers_df, monthly_df, force_fallback=True)
    second_call_time = time.time() - start_time
    
    # Results should be equivalent
    assert set(result1.keys()) == set(result2.keys())
    
    # Check key DataFrame shapes match
    for key in ['joined_df', 'vectors_df', 'labels_df', 'coords_df']:
        if len(result1[key]) > 0 and len(result2[key]) > 0:  # Skip empty DataFrames
            assert result1[key].shape == result2[key].shape, f"{key} shapes don't match"
    
    # Cache info should show cached item (when using fallback)
    cache_info = get_cache_info()
    assert cache_info['fallback_cache_size'] == 1

def test_cached_pipeline_different_inputs(sample_data):
    """Test that different inputs produce different cached results."""
    headers_df, monthly_df = sample_data
    
    clear_pipeline_cache()
    
    # First call with no filters - force fallback mode
    result1 = cached_run_pipeline(headers_df, monthly_df, None, force_fallback=True)
    
    # Second call with filters (should be different cache entry)
    filters_cfg = {'completion_year_range': [2015, 2020]}
    result2 = cached_run_pipeline(headers_df, monthly_df, filters_cfg, force_fallback=True)
    
    # Should have 2 cache entries now
    cache_info = get_cache_info()
    assert cache_info['fallback_cache_size'] == 2
    
    # Results should potentially be different (filtered data)
    # At minimum, the filter_stats should be different
    assert result1['filter_stats']['stats'] != result2['filter_stats']['stats']

def test_clear_cache(sample_data):
    """Test cache clearing functionality."""
    headers_df, monthly_df = sample_data
    
    # Add something to cache (use fallback mode)
    cached_run_pipeline(headers_df, monthly_df, force_fallback=True)
    assert get_cache_info()['fallback_cache_size'] > 0
    
    # Clear cache
    clear_pipeline_cache()
    assert get_cache_info()['fallback_cache_size'] == 0

def test_cache_mutation_protection(sample_data):
    """Test that cached results are protected from mutation."""
    headers_df, monthly_df = sample_data
    
    clear_pipeline_cache()
    
    # Get cached result (use fallback mode)
    result1 = cached_run_pipeline(headers_df, monthly_df, force_fallback=True)
    original_shape = result1['joined_df'].shape
    
    # Try to mutate the result
    if len(result1['joined_df']) > 0:
        result1['joined_df'].iloc[0, 0] = 'MUTATED'
    
    # Get same result again from cache
    result2 = cached_run_pipeline(headers_df, monthly_df, force_fallback=True)
    
    # Should not be mutated
    assert result2['joined_df'].shape == original_shape
    if len(result2['joined_df']) > 0:
        assert result2['joined_df'].iloc[0, 0] != 'MUTATED'

def test_empty_dataframe_handling():
    """Test handling of empty DataFrames."""
    empty_df = pd.DataFrame()
    regular_df = pd.DataFrame({'col': [1, 2, 3]})
    
    # This should not crash
    with pytest.raises(ValueError):  # Pipeline should reject empty inputs
        cached_run_pipeline(empty_df, regular_df)

def test_cache_info_structure():
    """Test that cache info returns expected structure."""
    cache_info = get_cache_info()
    
    expected_keys = {'fallback_cache_size', 'fallback_cache_keys', 'streamlit_available'}
    assert set(cache_info.keys()) == expected_keys
    
    assert isinstance(cache_info['fallback_cache_size'], int)
    assert isinstance(cache_info['fallback_cache_keys'], list)
    assert isinstance(cache_info['streamlit_available'], bool)

def test_large_filter_config_hashing():
    """Test that complex filter configurations are handled correctly."""
    df = pd.DataFrame({'col': [1, 2, 3]})
    
    # Complex filter config
    complex_filters = {
        'formations': ['EAGLEFORD', 'AUSTIN CHALK'],
        'completion_year_range': [2015, 2020],
        'lateral_ft_range': [5000, 15000],
        'min_months_produced': 12,
        'operators': ['OPERATOR_A', 'OPERATOR_B', 'OPERATOR_C']
    }
    
    # Should not crash and should generate consistent keys
    key1 = _generate_cache_key(df, df, complex_filters)
    key2 = _generate_cache_key(df, df, complex_filters)
    assert key1 == key2