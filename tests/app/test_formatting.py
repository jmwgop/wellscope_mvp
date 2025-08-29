# tests/app/test_formatting.py

import pandas as pd
import pytest

from app.utils.formatting import (
    format_number, format_percentage, format_date, format_file_size, format_duration,
    get_cluster_colors, format_cluster_label, format_similarity_score,
    get_similarity_color, format_dataframe_summary, format_filter_summary,
    format_pipeline_stats, truncate_text, format_api_display
)

def test_format_number():
    """Test number formatting with units."""
    assert format_number(123) == "123.0"
    assert format_number(1234) == "1.2K"
    assert format_number(1234567) == "1.2M"
    assert format_number(1234567890) == "1.2B"
    assert format_number(None) == "N/A"
    assert format_number("invalid") == "invalid"

def test_format_percentage():
    """Test percentage formatting."""
    assert format_percentage(0.75) == "75.0%"
    assert format_percentage(0.123) == "12.3%"
    assert format_percentage(None) == "N/A"
    assert format_percentage("invalid") == "invalid"

def test_format_date():
    """Test date formatting."""
    assert format_date("2023-01-15") == "2023-01-15"
    assert format_date(pd.Timestamp("2023-01-15")) == "2023-01-15"
    assert format_date(None) == "N/A"

def test_format_file_size():
    """Test file size formatting."""
    assert format_file_size(512) == "512 B"
    assert format_file_size(2048) == "2.0 KB"
    assert format_file_size(2048 * 1024) == "2.0 MB"
    assert format_file_size(2048 * 1024 * 1024) == "2.0 GB"

def test_format_duration():
    """Test duration formatting."""
    assert format_duration(0.5) == "500ms"
    assert format_duration(2.5) == "2.5s"
    assert format_duration(75) == "1.2m"
    assert format_duration(3665) == "1.0h"

def test_get_cluster_colors():
    """Test cluster color generation."""
    colors = get_cluster_colors(5)
    assert len(colors) == 6  # 5 clusters + 1 noise
    assert colors[0] == "#808080"  # Noise color
    
    colors_no_noise = get_cluster_colors(3, include_noise=False)
    assert len(colors_no_noise) == 3

def test_format_cluster_label():
    """Test cluster label formatting."""
    assert format_cluster_label(-1) == "Noise"
    assert format_cluster_label(0) == "Cluster 0"
    assert format_cluster_label(5) == "Cluster 5"

def test_format_similarity_score():
    """Test similarity score formatting."""
    assert format_similarity_score(0.75) == "0.750"
    assert format_similarity_score(None) == "N/A"
    assert format_similarity_score(float('nan')) == "N/A"

def test_get_similarity_color():
    """Test similarity color coding."""
    assert get_similarity_color(0.8) == "#2ECC40"  # High similarity (green)
    assert get_similarity_color(0.5) == "#FF851B"  # Medium similarity (orange)
    assert get_similarity_color(0.2) == "#FF4136"  # Low similarity (red)
    assert get_similarity_color(None) == "#CCCCCC"  # N/A (gray)

def test_format_dataframe_summary():
    """Test DataFrame summary formatting."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', None]
    })
    
    summary = format_dataframe_summary(df)
    assert summary['rows'] == "3"  # format_number with 0 precision doesn't show .0
    assert summary['columns'] == "2"
    assert 'memory' in summary
    assert 'completeness' in summary
    
    # Test empty DataFrame
    empty_summary = format_dataframe_summary(pd.DataFrame())
    assert empty_summary['rows'] == "0"
    assert empty_summary['columns'] == "0"

def test_format_filter_summary():
    """Test filter summary formatting."""
    filters = {
        'formations': ['EAGLEFORD'],
        'completion_year_range': (2018, 2022),
        'lateral_ft_range': (5000, 15000),
        'min_months_produced': 12
    }
    
    summary = format_filter_summary(filters, well_count=100)
    assert len(summary) > 0
    assert any('Formation: EAGLEFORD' in item for item in summary)
    assert any('Wells: 100' in item for item in summary)  # Remove .0
    
    # Test empty filters
    empty_summary = format_filter_summary({})
    assert empty_summary == ["No filters applied"]

def test_format_pipeline_stats():
    """Test pipeline statistics formatting."""
    stats = {
        'input_rows': 1000,
        'output_rows': 800,
        'kept_fraction': 0.8,
        'processing_time': 5.5,
        'memory_usage': 1024 * 1024
    }
    
    formatted = format_pipeline_stats(stats)
    assert formatted['input_rows'] == "1K"  # format_number with 0 precision
    assert formatted['output_rows'] == "800"
    assert formatted['kept_fraction'] == "80.0%"
    assert '5.5s' in formatted['processing_time']

def test_truncate_text():
    """Test text truncation."""
    long_text = "This is a very long text that should be truncated"
    assert truncate_text(long_text, 20) == "This is a very lo..."
    assert truncate_text("Short", 20) == "Short"

def test_format_api_display():
    """Test API number formatting."""
    assert format_api_display("42041324790000") == "42-041-32479-00-00"
    assert format_api_display("123") == "123"  # Too short, return as-is
    assert format_api_display("") == ""