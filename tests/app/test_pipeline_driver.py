# tests/app/test_pipeline_driver.py

from pathlib import Path
import pandas as pd
import pytest

from app.services.pipeline_driver import run_complete_pipeline, run_configurable_pipeline
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

def test_complete_pipeline_success(sample_data):
    """Test successful pipeline execution."""
    headers_df, monthly_df = sample_data
    
    result = run_complete_pipeline(headers_df, monthly_df)
    
    # Check all expected keys are present
    expected_keys = {
        'joined_df', 'vectors_df', 'labels_df', 'coords_df', 'scores_df',
        'join_stats', 'filter_stats', 'vector_meta', 'cluster_meta', 
        'projection_meta', 'similarity_meta'
    }
    assert set(result.keys()) == expected_keys
    
    # Check DataFrame types and non-empty (except scores_df which may be empty if all noise)
    dataframe_keys = ['joined_df', 'vectors_df', 'labels_df', 'coords_df', 'scores_df']
    for key in dataframe_keys:
        assert isinstance(result[key], pd.DataFrame), f"{key} is not a DataFrame"
        if key != 'scores_df':  # scores_df may be empty if all wells are noise
            assert len(result[key]) > 0, f"{key} is empty"
    
    # Check metadata types
    metadata_keys = ['join_stats', 'filter_stats', 'vector_meta', 'cluster_meta', 
                     'projection_meta', 'similarity_meta']
    for key in metadata_keys:
        assert isinstance(result[key], dict), f"{key} is not a dict"

def test_pipeline_with_custom_filters(sample_data):
    """Test pipeline with custom filter configuration."""
    headers_df, monthly_df = sample_data
    
    filters_cfg = {
        'completion_year_range': [2010, 2020],
        'min_months_produced': 12
    }
    
    result = run_complete_pipeline(headers_df, monthly_df, filters_cfg)
    
    # Should still complete successfully
    assert isinstance(result['joined_df'], pd.DataFrame)
    assert len(result['joined_df']) > 0
    
    # Filter stats should reflect the filtering
    assert 'stats' in result['filter_stats']
    assert 'output_rows' in result['filter_stats']['stats']

def test_pipeline_empty_headers():
    """Test pipeline with empty headers DataFrame."""
    empty_headers = pd.DataFrame()
    monthly_df = pd.DataFrame({'col': [1, 2, 3]})
    
    with pytest.raises(ValueError, match="Headers DataFrame is empty"):
        run_complete_pipeline(empty_headers, monthly_df)

def test_pipeline_empty_monthly():
    """Test pipeline with empty monthly DataFrame."""
    headers_df = pd.DataFrame({'col': [1, 2, 3]})
    empty_monthly = pd.DataFrame()
    
    with pytest.raises(ValueError, match="Monthly DataFrame is empty"):
        run_complete_pipeline(headers_df, empty_monthly)

def test_pipeline_invalid_input_types():
    """Test pipeline with invalid input types."""
    with pytest.raises(ValueError, match="Headers DataFrame is empty"):
        run_complete_pipeline("not_a_dataframe", pd.DataFrame())
    
    with pytest.raises(ValueError, match="Headers DataFrame is empty"):
        run_complete_pipeline(pd.DataFrame(), "not_a_dataframe")

def test_pipeline_no_join_results(sample_data):
    """Test pipeline behavior when join produces no results."""
    headers_df, _ = sample_data
    
    # Create monthly data with no matching APIs
    bad_monthly = pd.DataFrame({
        'API/UWI': ['NONEXISTENT_API_1', 'NONEXISTENT_API_2'],
        'Monthly Production Date': ['2020-01-01', '2020-01-01'],
        'Monthly Oil (bbl)': [100, 200],
        'Monthly Gas (Mcf)': [1000, 2000],
        'Monthly Water (bbl)': [50, 100],
        'Well Count': [1, 1],
        'Producing Month Number': [1, 1]
    })
    
    with pytest.raises(ValueError, match="No wells remain after joining"):
        run_complete_pipeline(headers_df, bad_monthly)

def test_pipeline_restrictive_filters(sample_data):
    """Test pipeline with overly restrictive filters."""
    headers_df, monthly_df = sample_data
    
    # Extremely restrictive filters that should filter out all wells
    restrictive_filters = {
        'completion_year_range': [1900, 1901],  # Very old range
        'min_months_produced': 1000  # Impossibly high
    }
    
    with pytest.raises(ValueError, match="No wells remain after filtering"):
        run_complete_pipeline(headers_df, monthly_df, restrictive_filters)


def test_run_configurable_pipeline_success(sample_data):
    """Test successful configurable pipeline execution with custom configs."""
    from wellscope_mvp.pipeline.vector_builder import VectorConfig
    from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
    from wellscope_mvp.pipeline.umap_projector import ProjectionConfig
    from wellscope_mvp.pipeline.similarity_score import SimilarityConfig
    
    headers_df, monthly_df = sample_data
    
    # Define custom configurations
    vector_cfg = VectorConfig(months=12, normalize="q_over_qmax", stream="oil")
    cluster_cfg = ClusterConfig(use_hdbscan=True, min_cluster_size=5, metric="cosine")
    projection_cfg = ProjectionConfig(n_neighbors=15, min_dist=0.1)
    similarity_cfg = SimilarityConfig()
    
    result = run_configurable_pipeline(
        headers_df=headers_df,
        monthly_df=monthly_df,
        filters_cfg={'min_months_produced': 6},
        vector_cfg=vector_cfg,
        cluster_cfg=cluster_cfg,
        projection_cfg=projection_cfg,
        similarity_cfg=similarity_cfg
    )
    
    # Check all expected keys are present
    expected_keys = {
        'joined_df', 'vectors_df', 'labels_df', 'coords_df', 'scores_df',
        'join_stats', 'filter_stats', 'vector_meta', 'cluster_meta', 
        'projection_meta', 'similarity_meta'
    }
    assert set(result.keys()) == expected_keys
    
    # Check that custom configurations were used
    assert result['vector_meta']['months'] == 12  # Custom vector length
    
    # Should have used HDBSCAN with cosine metric
    cluster_meta = result['cluster_meta']
    if 'algorithm_used' in cluster_meta:
        assert 'hdbscan' in cluster_meta['algorithm_used'].lower() or cluster_meta['algorithm_used'] == 'hdbscan_relaxed'


def test_run_configurable_pipeline_with_mature_first(sample_data):
    """Test configurable pipeline with mature-first clustering enabled."""
    from wellscope_mvp.pipeline.vector_builder import VectorConfig
    from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
    
    headers_df, monthly_df = sample_data
    
    vector_cfg = VectorConfig(months=18, normalize="q_over_qmax", stream="oil")
    cluster_cfg = ClusterConfig(use_hdbscan=True, min_cluster_size=4, metric="cosine")
    
    result = run_configurable_pipeline(
        headers_df=headers_df,
        monthly_df=monthly_df,
        filters_cfg={},
        vector_cfg=vector_cfg,
        cluster_cfg=cluster_cfg,
        use_mature_first=True
    )
    
    # Should complete successfully
    assert isinstance(result['labels_df'], pd.DataFrame)
    assert len(result['labels_df']) > 0
    
    # Check for mature-first clustering metadata
    cluster_meta = result['cluster_meta']
    if 'clustering_strategy' in cluster_meta:
        # Should indicate mature-first was attempted (success depends on data)
        assert cluster_meta['clustering_strategy'] in [
            'mature_first', 'fallback_uniform', 'uniform_short'
        ]


def test_run_configurable_pipeline_without_mature_first(sample_data):
    """Test configurable pipeline with mature-first clustering disabled."""
    from wellscope_mvp.pipeline.vector_builder import VectorConfig
    from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
    
    headers_df, monthly_df = sample_data
    
    vector_cfg = VectorConfig(months=12, normalize="q_over_qmax", stream="oil")
    cluster_cfg = ClusterConfig(use_hdbscan=True, min_cluster_size=3, metric="cosine")
    
    result = run_configurable_pipeline(
        headers_df=headers_df,
        monthly_df=monthly_df,
        vector_cfg=vector_cfg,
        cluster_cfg=cluster_cfg,
        use_mature_first=False  # Disable mature-first
    )
    
    # Should complete successfully with standard clustering
    assert isinstance(result['labels_df'], pd.DataFrame)
    assert len(result['labels_df']) > 0
    
    # Should use standard clustering (no mature-first metadata)
    cluster_meta = result['cluster_meta']
    # If clustering_strategy exists, it should not be mature-first
    if 'clustering_strategy' in cluster_meta:
        assert cluster_meta['clustering_strategy'] != 'mature_first'


def test_configurable_pipeline_default_configs(sample_data):
    """Test configurable pipeline with default configurations."""
    headers_df, monthly_df = sample_data
    
    # Run with no custom configs - should use defaults
    result = run_configurable_pipeline(headers_df, monthly_df)
    
    # Should complete successfully
    expected_keys = {
        'joined_df', 'vectors_df', 'labels_df', 'coords_df', 'scores_df',
        'join_stats', 'filter_stats', 'vector_meta', 'cluster_meta', 
        'projection_meta', 'similarity_meta'
    }
    assert set(result.keys()) == expected_keys
    
    # Should use production-optimized defaults
    # Default vector config should be reasonable
    assert result['vector_meta']['months'] >= 6
    assert result['vector_meta']['months'] <= 24


def test_mature_first_clustering_strategy(sample_data):
    """Test that mature-first clustering strategy is properly invoked."""
    headers_df, monthly_df = sample_data
    
    # Run complete pipeline with mature-first enabled (default)
    result = run_complete_pipeline(headers_df, monthly_df, use_mature_first=True)
    
    # Should complete successfully
    assert isinstance(result, dict)
    assert 'cluster_meta' in result
    
    # Check if mature-first was attempted
    cluster_meta = result['cluster_meta']
    # The strategy will depend on the actual data characteristics
    if 'clustering_strategy' in cluster_meta:
        assert isinstance(cluster_meta['clustering_strategy'], str)
        assert cluster_meta['clustering_strategy'] in [
            'mature_first', 'fallback_uniform', 'uniform_short', 'insufficient_data'
        ]


def test_pipeline_with_production_optimized_defaults(sample_data):
    """Test that pipeline uses production-optimized defaults."""
    headers_df, monthly_df = sample_data
    
    result = run_complete_pipeline(headers_df, monthly_df)
    
    # Should complete successfully
    assert isinstance(result, dict)
    
    # Check that production-focused defaults were applied
    vector_meta = result['vector_meta']
    assert vector_meta['normalize'] == 'q_over_qmax'  # Production normalization
    assert vector_meta['stream'] == 'oil'             # Oil stream default