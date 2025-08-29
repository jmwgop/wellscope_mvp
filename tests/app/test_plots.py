# tests/app/test_plots.py

import pandas as pd
import pytest

from app.components.plots import (
    render_cluster_scatter, render_production_curves, render_cluster_summary_chart,
    render_similarity_histogram, render_interactive_plots, render_3d_scatter,
    get_plot_download_data
)

# Create sample data for testing
@pytest.fixture
def sample_coords_df():
    return pd.DataFrame({
        'API14_norm': ['123', '456', '789'],
        'x': [1.0, 2.0, 3.0],
        'y': [1.5, 2.5, 3.5]
    })

@pytest.fixture
def sample_labels_df():
    return pd.DataFrame({
        'API14_norm': ['123', '456', '789'],
        'label': [0, 1, 0],
        'cluster_size': [2, 1, 2]
    })

@pytest.fixture
def sample_vectors_df():
    return pd.DataFrame({
        'API14_norm': ['123', '456', '789'],
        'v1': [0.8, 0.6, 0.7],
        'v2': [0.6, 0.4, 0.5],
        'v3': [0.4, 0.2, 0.3],
        'v4': [0.2, 0.1, 0.15]
    })

@pytest.fixture
def sample_scores_df():
    return pd.DataFrame({
        'API14_norm': ['123', '456', '789'],
        'label': [0, 1, 0],
        'similarity': [0.85, 0.65, 0.75]
    })

def test_render_cluster_scatter(sample_coords_df, sample_labels_df):
    """Test cluster scatter plot rendering."""
    fig = render_cluster_scatter(sample_coords_df, sample_labels_df)
    
    # Should return None if plotting not available, or figure if available
    assert fig is None or hasattr(fig, 'data')
    
    # Test with empty data
    empty_df = pd.DataFrame()
    fig_empty = render_cluster_scatter(empty_df, sample_labels_df)
    assert fig_empty is None

def test_render_production_curves(sample_vectors_df, sample_labels_df):
    """Test production curves rendering."""
    fig = render_production_curves(sample_vectors_df, sample_labels_df)
    
    # Should return None if plotting not available, or figure if available
    assert fig is None or hasattr(fig, 'data')
    
    # Test with selected cluster
    fig_cluster = render_production_curves(sample_vectors_df, sample_labels_df, selected_cluster=0)
    assert fig_cluster is None or hasattr(fig_cluster, 'data')
    
    # Test with empty data
    empty_df = pd.DataFrame()
    fig_empty = render_production_curves(empty_df, sample_labels_df)
    assert fig_empty is None

def test_render_cluster_summary_chart(sample_labels_df):
    """Test cluster summary chart rendering."""
    fig = render_cluster_summary_chart(sample_labels_df)
    
    # Should return None if plotting not available, or figure if available
    assert fig is None or hasattr(fig, 'data')
    
    # Test with empty data
    fig_empty = render_cluster_summary_chart(pd.DataFrame())
    assert fig_empty is None

def test_render_similarity_histogram(sample_scores_df):
    """Test similarity histogram rendering."""
    fig = render_similarity_histogram(sample_scores_df)
    
    # Should return None if plotting not available, or figure if available
    assert fig is None or hasattr(fig, 'data')
    
    # Test with empty data
    fig_empty = render_similarity_histogram(pd.DataFrame())
    assert fig_empty is None

def test_render_interactive_plots(sample_coords_df, sample_labels_df, 
                                sample_vectors_df, sample_scores_df):
    """Test interactive plots rendering."""
    pipeline_results = {
        'coords_df': sample_coords_df,
        'labels_df': sample_labels_df,
        'vectors_df': sample_vectors_df,
        'scores_df': sample_scores_df
    }
    
    # Should not raise error
    render_interactive_plots(pipeline_results)
    
    # Test with missing data
    render_interactive_plots({})

def test_render_3d_scatter(sample_coords_df, sample_labels_df):
    """Test 3D scatter plot rendering."""
    # Add z coordinate
    coords_3d = sample_coords_df.copy()
    coords_3d['z'] = [1.0, 2.0, 3.0]
    
    fig = render_3d_scatter(coords_3d, sample_labels_df)
    assert fig is None or hasattr(fig, 'data')
    
    # Test without z coordinate (should return None)
    fig_2d = render_3d_scatter(sample_coords_df, sample_labels_df)
    assert fig_2d is None

def test_get_plot_download_data():
    """Test plot download data generation."""
    # Test with None figure (when plotting not available)
    data = get_plot_download_data(None, 'png')
    assert data == b""
    
    data = get_plot_download_data(None, 'html')
    assert data == b""
    
    data = get_plot_download_data(None, 'svg')
    assert data == b""
    
    # Test with invalid format
    data = get_plot_download_data(None, 'invalid')
    assert data == b""