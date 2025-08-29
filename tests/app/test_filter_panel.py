# tests/app/test_filter_panel.py

import pandas as pd
import pytest

from app.components.filter_panel import (
    render_filter_panel, render_advanced_filters, get_filter_options_from_data,
    apply_filters_preview, _mock_filter_panel
)

def test_render_filter_panel():
    """Test filter panel renders without error."""
    # Test without data
    result = render_filter_panel(None)
    
    # Now returns config dict directly
    assert isinstance(result, dict)
    # Should have common filter keys
    common_keys = {'completion_year_range', 'lateral_ft_range', 'min_months_produced'}
    assert common_keys.issubset(set(result.keys()))

def test_render_advanced_filters():
    """Test advanced filters render."""
    result = render_advanced_filters(None)
    assert isinstance(result, dict)

def test_get_filter_options_from_data():
    """Test extracting filter options from data."""
    df = pd.DataFrame({
        'Target Formation': ['EAGLEFORD', 'AUSTIN CHALK', 'EAGLEFORD'],
        'Operator (Reported)': ['OP1', 'OP2', 'OP1'],
        'Well Status': ['ACTIVE', 'INACTIVE', 'ACTIVE'],
        'Completion Date': ['2020-01-01', '2021-06-15', '2019-12-01'],
        'DI Lateral Length': [8000, 12000, 9500]
    })
    
    options = get_filter_options_from_data(df)
    
    assert 'formations' in options
    assert set(options['formations']) == {'AUSTIN CHALK', 'EAGLEFORD'}
    
    assert 'operators' in options
    assert set(options['operators']) == {'OP1', 'OP2'}
    
    assert 'well_statuses' in options
    assert set(options['well_statuses']) == {'ACTIVE', 'INACTIVE'}
    
    assert 'year_range' in options
    assert options['year_range'] == (2019, 2021)
    
    assert 'lateral_range' in options
    assert options['lateral_range'] == (8000.0, 12000.0)

def test_apply_filters_preview():
    """Test filter preview functionality."""
    df = pd.DataFrame({
        'API14': ['123', '456', '789'],
        'Target Formation': ['EAGLEFORD', 'AUSTIN CHALK', 'EAGLEFORD'],
        'Operator (Reported)': ['OP1', 'OP2', 'OP1'],
        'Completion Date': ['2020-01-01', '2021-06-15', '2019-12-01'],
        'DI Lateral Length': [8000, 12000, 9500]
    })
    
    # Test formation filter
    config = {'formations': ['EAGLEFORD']}
    filtered_df, stats = apply_filters_preview(df, config)
    
    assert len(filtered_df) == 2  # Only EAGLEFORD wells
    assert stats['input_rows'] == 3
    assert stats['output_rows'] == 2
    assert 'after_formation_filter' in stats
    
    # Test year range filter
    config = {'completion_year_range': (2020, 2021)}
    filtered_df, stats = apply_filters_preview(df, config)
    
    assert len(filtered_df) == 2  # Wells from 2020-2021
    assert 'after_year_filter' in stats
    
    # Test lateral range filter
    config = {'lateral_ft_range': (8000, 10000)}
    filtered_df, stats = apply_filters_preview(df, config)
    
    assert len(filtered_df) == 2  # Wells with lateral 8000-10000
    assert 'after_lateral_filter' in stats
    
    # Test empty DataFrame
    empty_df = pd.DataFrame()
    filtered_df, stats = apply_filters_preview(empty_df, {})
    assert len(filtered_df) == 0
    assert stats['input_rows'] == 0

def test_mock_filter_panel():
    """Test mock filter panel for testing."""
    result = _mock_filter_panel()
    
    # Mock now returns config dict directly
    assert isinstance(result, dict)
    expected_keys = {'completion_year_range', 'lateral_ft_range', 'min_months_produced'}
    assert set(result.keys()) == expected_keys
    assert 'completion_year_range' in result