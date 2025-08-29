# tests/app/test_tables.py

import pandas as pd
import pytest

from app.components.tables import (
    render_well_data_table, render_cluster_summary_table, render_similarity_ranking_table,
    render_export_options, create_analysis_summary_table
)

# Create sample data for testing
@pytest.fixture
def sample_joined_df():
    return pd.DataFrame({
        'API14': ['123', '456', '789'],
        'Target Formation': ['EAGLEFORD', 'AUSTIN CHALK', 'EAGLEFORD'],
        'Operator (Reported)': ['OP1', 'OP2', 'OP1'],
        'Completion Date': ['2020-01-01', '2021-06-15', '2019-12-01'],
        'DI Lateral Length': [8000, 12000, 9500]
    })

@pytest.fixture
def sample_labels_df():
    return pd.DataFrame({
        'API14': ['123', '456', '789'],
        'label': [0, 1, 0],
        'cluster_size': [2, 1, 2]
    })

@pytest.fixture
def sample_scores_df():
    return pd.DataFrame({
        'API14': ['123', '456', '789'],
        'label': [0, 1, 0],
        'similarity': [0.85, 0.65, 0.75],
        'cluster_size': [2, 1, 2],
        'cluster_avg_similarity': [0.80, 0.65, 0.80]
    })

@pytest.fixture
def sample_pipeline_results():
    return {
        'join_stats': {
            'headers_rows': 100,
            'monthly_rows': 5000,
            'joined_rows': 4500,
            'matched_api14': 95
        },
        'filter_stats': {
            'stats': {
                'input_rows': 4500,
                'output_rows': 3000,
                'kept_fraction': 0.667
            }
        },
        'cluster_meta': {
            'algorithm': 'hdbscan',
            'n_clusters': 5,
            'n_noise': 100
        },
        'similarity_meta': {
            'n_scored': 2900,
            'mean_similarity': 0.735
        }
    }

def test_render_well_data_table(sample_joined_df, sample_labels_df, sample_scores_df):
    """Test well data table rendering."""
    # Should not raise error in either Streamlit or non-Streamlit environment
    render_well_data_table(sample_joined_df, sample_labels_df, sample_scores_df)
    
    # Test with empty data
    render_well_data_table(pd.DataFrame())
    
    # Test with None data
    render_well_data_table(None)

def test_render_cluster_summary_table(sample_labels_df, sample_scores_df):
    """Test cluster summary table rendering."""
    # Should not raise error
    render_cluster_summary_table(sample_labels_df, sample_scores_df)
    
    # Test without similarity scores
    render_cluster_summary_table(sample_labels_df)
    
    # Test with empty data
    render_cluster_summary_table(pd.DataFrame())
    
    # Test with None data
    render_cluster_summary_table(None)

def test_render_similarity_ranking_table(sample_scores_df, sample_joined_df):
    """Test similarity ranking table rendering."""
    # Should not raise error
    render_similarity_ranking_table(sample_scores_df, sample_joined_df)
    
    # Test without well data
    render_similarity_ranking_table(sample_scores_df)
    
    # Test with empty data
    render_similarity_ranking_table(pd.DataFrame())
    
    # Test with None data
    render_similarity_ranking_table(None)

def test_render_export_options(sample_pipeline_results):
    """Test export options rendering."""
    # Add DataFrames to pipeline results
    pipeline_results = sample_pipeline_results.copy()
    pipeline_results.update({
        'labels_df': pd.DataFrame({'col': [1, 2, 3]}),
        'scores_df': pd.DataFrame({'col': [4, 5, 6]}),
        'coords_df': pd.DataFrame({'col': [7, 8, 9]})
    })
    
    # Should not raise error
    render_export_options(pipeline_results)
    
    # Test with empty results
    render_export_options({})

def test_create_analysis_summary_table(sample_pipeline_results):
    """Test analysis summary table creation."""
    summary_df = create_analysis_summary_table(sample_pipeline_results)
    
    assert isinstance(summary_df, pd.DataFrame)
    assert 'Category' in summary_df.columns
    assert 'Metric' in summary_df.columns
    assert 'Value' in summary_df.columns
    
    # Should have data from all sections
    categories = summary_df['Category'].unique()
    expected_categories = ['Data Integration', 'Filtering', 'Clustering', 'Similarity']
    
    for category in expected_categories:
        assert category in categories
    
    # Test with empty results
    empty_summary = create_analysis_summary_table({})
    assert isinstance(empty_summary, pd.DataFrame)
    assert len(empty_summary) == 0

def test_table_functionality_with_real_data():
    """Test table components work with realistic data structure."""
    # Create more realistic test data
    joined_df = pd.DataFrame({
        'API14': [f"4204132479000{i}" for i in range(5)],
        'Target Formation': ['EAGLEFORD'] * 3 + ['AUSTIN CHALK'] * 2,
        'Operator (Reported)': ['OPERATOR_A'] * 2 + ['OPERATOR_B'] * 3,
        'Completion Date': ['2020-01-01', '2020-02-01', '2020-03-01', '2019-12-01', '2021-01-01'],
        'DI Lateral Length': [8000, 9000, 10000, 7500, 8500],
        'Well Status': ['ACTIVE'] * 4 + ['INACTIVE']
    })
    
    labels_df = pd.DataFrame({
        'API14': [f"4204132479000{i}" for i in range(5)],
        'label': [0, 0, 1, -1, 1],  # Include noise (-1)
        'cluster_size': [2, 2, 2, 0, 2]
    })
    
    scores_df = pd.DataFrame({
        'API14': [f"4204132479000{i}" for i in range(4)],  # One less (noise well excluded)
        'label': [0, 0, 1, 1],
        'similarity': [0.85, 0.82, 0.78, 0.80],
        'cluster_size': [2, 2, 2, 2],
        'cluster_avg_similarity': [0.835, 0.835, 0.79, 0.79]
    })
    
    # Test all table components
    render_well_data_table(joined_df, labels_df, scores_df, page_size=3)
    render_cluster_summary_table(labels_df, scores_df)
    render_similarity_ranking_table(scores_df, joined_df, top_n=3)
    
    # Test analysis summary with complete pipeline results
    pipeline_results = {
        'join_stats': {
            'headers_rows': 150,
            'monthly_rows': 8000,
            'joined_rows': 7500,
            'matched_api14': 148
        },
        'filter_stats': {
            'stats': {
                'input_rows': 7500,
                'output_rows': 5000,
                'kept_fraction': 0.667
            }
        },
        'cluster_meta': {
            'algorithm': 'hdbscan',
            'n_clusters': 4,
            'n_noise': 250
        },
        'similarity_meta': {
            'n_scored': 4750,
            'mean_similarity': 0.712
        },
        'labels_df': labels_df,
        'scores_df': scores_df,
        'coords_df': pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    }
    
    summary_df = create_analysis_summary_table(pipeline_results)
    assert len(summary_df) > 0
    
    render_export_options(pipeline_results)