# tests/app/test_cluster_controls.py

import pytest

from app.components.cluster_controls import (
    render_vector_controls, render_clustering_controls, render_projection_controls,
    render_all_controls,
    _mock_vector_config, _mock_cluster_config, _mock_projection_config, _mock_all_controls
)
from wellscope_mvp.pipeline.vector_builder import VectorConfig
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
from wellscope_mvp.pipeline.umap_projector import ProjectionConfig

def test_render_vector_controls():
    """Test vector controls render without error."""
    config, is_valid, errors = render_vector_controls()
    
    assert isinstance(config, VectorConfig)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)

def test_render_clustering_controls():
    """Test clustering controls render without error."""
    config, is_valid, errors = render_clustering_controls()
    
    assert isinstance(config, ClusterConfig)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    
    # Should use production-optimized metric default
    assert config.metric == 'cosine'  # Optimal for production curve shapes
    
    # Without data, may use simple defaults, but should still be valid
    assert config.min_cluster_size >= 2  # Reasonable minimum


def test_render_clustering_controls_with_data():
    """Test clustering controls render with filtered data for intelligent parameter selection."""
    # Create sample filtered data with production data
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create more realistic well data with production history
    wells = []
    for i in range(20):
        api = f'42{i:08d}'
        completion_date = datetime(2020, 1, 1) + timedelta(days=30*i)
        
        # Add production data for each well (12-24 months)
        for month in range(12 + i):
            wells.append({
                'API14': api,
                'Target Formation': 'EAGLEFORD' if i < 15 else 'AUSTIN CHALK',
                'Completion Date': completion_date.strftime('%Y-%m-%d'),
                'DI Lateral Length': 8000 + i * 200,
                'Monthly Production Date': (completion_date + timedelta(days=30*month)).strftime('%Y-%m-%d'),
                'Monthly Oil': max(50, 1000 * (0.95 ** month)),  # Decline curve
                'Monthly Gas': max(100, 2000 * (0.93 ** month)),
                'Monthly Water': max(10, 200 * (1.1 ** month))
            })
    
    filtered_df = pd.DataFrame(wells)
    
    result = render_clustering_controls(filtered_df)
    
    # Should return (config, is_valid, errors) tuple
    assert len(result) == 3
    config, is_valid, errors = result
    
    # Should return intelligent config based on data characteristics
    assert isinstance(config, ClusterConfig)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)
    
    # Intelligent clustering should adjust parameters based on data size
    assert config.min_cluster_size >= 2  # Reasonable for 20 wells
    assert config.min_cluster_size <= 8   # Not too large for small dataset
    assert config.metric == 'cosine'      # Production-optimized default
    assert config.use_hdbscan == True     # Preferred algorithm

def test_render_projection_controls():
    """Test projection controls render without error."""
    config, is_valid, errors = render_projection_controls()
    
    assert isinstance(config, ProjectionConfig)
    assert isinstance(is_valid, bool)
    assert isinstance(errors, list)

def test_render_all_controls():
    """Test all controls render together."""
    configs, all_valid, all_errors = render_all_controls()
    
    expected_keys = {'vector', 'cluster', 'projection'}
    assert set(configs.keys()) == expected_keys
    
    assert isinstance(configs['vector'], VectorConfig)
    assert isinstance(configs['cluster'], ClusterConfig)
    assert isinstance(configs['projection'], ProjectionConfig)
    
    assert isinstance(all_valid, bool)
    assert isinstance(all_errors, list)


def test_render_all_controls_with_data():
    """Test all controls render with filtered data and filters configuration."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create sample data with production history
    wells = []
    for i in range(10):
        api = f'42{i:08d}'
        completion_date = datetime(2020, 1, 1) + timedelta(days=60*i)
        
        # Add production data
        for month in range(18):  # 18 months of data
            wells.append({
                'API14': api,
                'Target Formation': 'EAGLEFORD',
                'Completion Date': completion_date.strftime('%Y-%m-%d'),
                'DI Lateral Length': 8000 + i * 100,
                'Monthly Production Date': (completion_date + timedelta(days=30*month)).strftime('%Y-%m-%d'),
                'Monthly Oil': max(50, 800 * (0.96 ** month)),
                'Monthly Gas': max(100, 1600 * (0.94 ** month)),
                'Monthly Water': max(10, 100 * (1.05 ** month))
            })
    
    filtered_df = pd.DataFrame(wells)
    filters_cfg = {'min_months_produced': 12}  # Example filter config
    
    configs, all_valid, all_errors = render_all_controls(filtered_df, filters_cfg)
    
    expected_keys = {'vector', 'cluster', 'projection'}
    assert set(configs.keys()) == expected_keys
    
    assert isinstance(configs['vector'], VectorConfig)
    assert isinstance(configs['cluster'], ClusterConfig)
    assert isinstance(configs['projection'], ProjectionConfig)
    
    assert isinstance(all_valid, bool)
    assert isinstance(all_errors, list)
    
    # Check that intelligent parameters are applied
    # Vector length should be influenced by filter settings
    assert configs['vector'].months >= 6  # Should have reasonable vector length
    assert configs['vector'].months <= 18  # Should not exceed available data
    
    # Clustering should use production-optimized settings
    assert configs['cluster'].metric == 'cosine'
    assert configs['cluster'].use_hdbscan == True

def test_mock_vector_config():
    """Test mock vector config."""
    config, is_valid, errors = _mock_vector_config()
    
    assert isinstance(config, VectorConfig)
    assert is_valid is True
    assert len(errors) == 0

def test_mock_cluster_config():
    """Test mock cluster config."""
    config, is_valid, errors = _mock_cluster_config()
    
    assert isinstance(config, ClusterConfig)
    assert is_valid is True
    assert len(errors) == 0

def test_mock_projection_config():
    """Test mock projection config."""
    config, is_valid, errors = _mock_projection_config()
    
    assert isinstance(config, ProjectionConfig)
    assert is_valid is True
    assert len(errors) == 0

def test_mock_all_controls():
    """Test mock all controls."""
    configs, all_valid, all_errors = _mock_all_controls()
    
    expected_keys = {'vector', 'cluster', 'projection'}
    assert set(configs.keys()) == expected_keys
    assert all_valid is True
    assert len(all_errors) == 0

def test_intelligent_clustering_parameters():
    """Test that intelligent clustering parameters are calculated correctly."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create dataset with known characteristics
    wells = []
    for i in range(50):  # Medium-sized dataset
        api = f'42{i:08d}'
        completion_date = datetime(2019, 1, 1) + timedelta(days=30*i)
        
        # Generate 24+ months of production data for mature wells
        for month in range(26):
            wells.append({
                'API14': api,
                'Target Formation': 'EAGLEFORD',
                'Completion Date': completion_date.strftime('%Y-%m-%d'),
                'DI Lateral Length': 8000,
                'Monthly Production Date': (completion_date + timedelta(days=30*month)).strftime('%Y-%m-%d'),
                'Monthly Oil': max(50, 1000 * (0.95 ** month)),
                'Monthly Gas': max(100, 2000 * (0.93 ** month)),
            })
    
    filtered_df = pd.DataFrame(wells)
    
    # Test clustering controls with this data
    config, is_valid, errors = render_clustering_controls(filtered_df)
    
    # Should calculate intelligent parameters for 50 wells
    assert config.min_cluster_size >= 3  # Reasonable minimum for 50 wells
    assert config.min_cluster_size <= 15  # Not too large
    assert config.use_hdbscan == True     # HDBSCAN for medium datasets
    assert config.metric == 'cosine'     # Production-optimized

def test_vector_length_intelligence():
    """Test that vector length is intelligently selected based on data."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create wells with varying amounts of production data
    wells = []
    for i in range(10):
        api = f'42{i:08d}'
        completion_date = datetime(2020, 1, 1)
        
        # First 5 wells have 36 months, last 5 have only 12 months
        months = 36 if i < 5 else 12
        
        for month in range(months):
            wells.append({
                'API14': api,
                'Completion Date': completion_date.strftime('%Y-%m-%d'),
                'Monthly Production Date': (completion_date + timedelta(days=30*month)).strftime('%Y-%m-%d'),
                'Monthly Oil': max(50, 800 * (0.96 ** month))
            })
    
    filtered_df = pd.DataFrame(wells)
    filters_cfg = {'min_months_produced': 12}
    
    configs, all_valid, all_errors = render_all_controls(filtered_df, filters_cfg)
    
    # Vector length should be intelligently chosen based on data availability
    vector_months = configs['vector'].months
    assert vector_months >= 6   # Minimum reasonable length
    assert vector_months <= 24  # Should not exceed practical limits for mixed data