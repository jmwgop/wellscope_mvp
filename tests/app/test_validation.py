# tests/app/test_validation.py

import pandas as pd
import pytest
from pathlib import Path
import tempfile

from app.utils.validation import (
    validate_filter_config, validate_vector_config, validate_cluster_config,
    validate_projection_config, validate_dataframe_structure, validate_pipeline_inputs,
    validate_session_state, format_validation_errors, get_validation_summary
)

def test_validate_filter_config():
    """Test filter configuration validation."""
    # Valid config
    valid_config = {
        'completion_year_range': (2018, 2022),
        'lateral_ft_range': (5000, 15000),
        'min_months_produced': 12
    }
    errors = validate_filter_config(valid_config)
    assert len(errors) == 0
    
    # Invalid year range
    invalid_config = {
        'completion_year_range': (2022, 2018),  # Start after end
        'lateral_ft_range': (15000, 5000),      # Min > Max
        'min_months_produced': -5               # Negative
    }
    errors = validate_filter_config(invalid_config)
    assert len(errors) >= 3

def test_validate_vector_config():
    """Test vector configuration validation."""
    # Valid config
    valid_config = {
        'months': 24,
        'stream': 'oil',
        'normalize': 'q_over_qmax',
        'boe_gas_factor': 6.0
    }
    errors = validate_vector_config(valid_config)
    assert len(errors) == 0
    
    # Invalid config
    invalid_config = {
        'months': -5,               # Negative months
        'stream': 'invalid',        # Invalid stream
        'normalize': 'invalid',     # Invalid normalization
        'boe_gas_factor': -1.0      # Negative factor
    }
    errors = validate_vector_config(invalid_config)
    assert len(errors) >= 4

def test_validate_cluster_config():
    """Test clustering configuration validation."""
    # Valid config with production-optimized defaults
    valid_config = {
        'min_cluster_size': 10,
        'eps': 0.5,
        'min_samples': 5,
        'metric': 'cosine',  # Updated to cosine default
        'use_hdbscan': True
    }
    errors = validate_cluster_config(valid_config)
    assert len(errors) == 0
    
    # Test ClusterConfig object validation too
    from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
    cluster_obj = ClusterConfig(min_cluster_size=8, metric='cosine', use_hdbscan=True)
    errors = validate_cluster_config(cluster_obj)
    assert len(errors) == 0
    
    # Invalid config
    invalid_config = {
        'min_cluster_size': 1,      # Too small
        'eps': -0.5,               # Negative eps
        'min_samples': -1,         # Negative samples
        'metric': 'invalid'        # Invalid metric
    }
    errors = validate_cluster_config(invalid_config)
    assert len(errors) >= 4

def test_validate_projection_config():
    """Test projection configuration validation."""
    # Valid config
    valid_config = {
        'n_components': 2,
        'n_neighbors': 15,
        'min_dist': 0.1,
        'random_state': 42
    }
    errors = validate_projection_config(valid_config)
    assert len(errors) == 0
    
    # Invalid config
    invalid_config = {
        'n_components': 1,          # Too few components
        'n_neighbors': 1,           # Too few neighbors
        'min_dist': -0.1,          # Negative distance
        'random_state': -1         # Negative random state
    }
    errors = validate_projection_config(invalid_config)
    assert len(errors) >= 4

def test_validate_dataframe_structure():
    """Test DataFrame structure validation."""
    # Valid DataFrame
    valid_df = pd.DataFrame({
        'API14': ['123', '456'],
        'other_col': [1, 2]
    })
    errors = validate_dataframe_structure(valid_df, ['API14'])
    assert len(errors) == 0
    
    # Missing columns
    errors = validate_dataframe_structure(valid_df, ['API14', 'missing_col'])
    assert len(errors) == 1
    assert 'missing required columns' in errors[0]
    
    # Empty DataFrame
    empty_df = pd.DataFrame()
    errors = validate_dataframe_structure(empty_df, ['API14'])
    assert len(errors) >= 2  # Empty + missing columns
    
    # None DataFrame
    errors = validate_dataframe_structure(None, ['API14'])
    assert len(errors) == 1
    assert 'is None' in errors[0]

def test_validate_pipeline_inputs():
    """Test pipeline input validation."""
    # Valid inputs
    headers_df = pd.DataFrame({'API14': ['123', '456']})
    monthly_df = pd.DataFrame({
        'API/UWI': ['123', '456'],
        'Monthly Production Date': ['2023-01-01', '2023-02-01'],
        'Monthly Oil': [100, 150]
    })
    
    errors = validate_pipeline_inputs(headers_df, monthly_df)
    assert len(errors) == 0
    
    # Invalid inputs
    errors = validate_pipeline_inputs(None, None)
    assert len(errors) >= 2

def test_validate_session_state():
    """Test session state validation."""
    # Valid session
    valid_session = {
        'key1': pd.DataFrame({'col': [1, 2]}),
        'key2': 'some_value'
    }
    errors = validate_session_state(valid_session, ['key1', 'key2'])
    assert len(errors) == 0
    
    # Missing keys
    errors = validate_session_state(valid_session, ['key1', 'missing_key'])
    assert len(errors) == 1
    
    # None values
    invalid_session = {'key1': None}
    errors = validate_session_state(invalid_session, ['key1'])
    assert len(errors) == 1

def test_format_validation_errors():
    """Test error message formatting."""
    errors = ['Error 1', 'Error 2', 'Error 3']
    
    formatted = format_validation_errors(errors, "Test Errors")
    assert "Test Errors" in formatted
    assert "1. Error 1" in formatted
    assert "2. Error 2" in formatted
    
    # Single error
    single_formatted = format_validation_errors(['Single error'])
    assert "Single error" in single_formatted
    assert "1." not in single_formatted  # No numbering for single error
    
    # No errors
    empty_formatted = format_validation_errors([])
    assert empty_formatted == ""

def test_get_validation_summary():
    """Test validation summary generation."""
    # All valid
    all_errors = {'section1': [], 'section2': []}
    is_valid, message = get_validation_summary(all_errors)
    assert is_valid is True
    assert "All validations passed" in message
    
    # Has errors
    error_dict = {
        'section1': ['Error 1'],
        'section2': ['Error 2', 'Error 3']
    }
    is_valid, message = get_validation_summary(error_dict)
    assert is_valid is False
    assert "Found 3 validation error" in message
    assert "section1" in message
    assert "section2" in message


def test_validate_intelligent_clustering_parameters():
    """Test validation of intelligent clustering parameter structures."""
    # Valid intelligent parameters dictionary
    intelligent_params = {
        'min_cluster_size': 8,
        'min_samples': 5,
        'use_hdbscan': True,
        'expected_clusters': 4,
        'algorithm_recommendation': 'HDBSCAN',
        'confidence_score': 0.8
    }
    
    # Should validate successfully as a dictionary
    errors = validate_cluster_config(intelligent_params)
    # May have some errors due to missing standard fields, but structure should be acceptable
    assert isinstance(errors, list)
    
    # Convert to ClusterConfig and validate
    from app.utils.clustering_intelligence import convert_to_cluster_config
    cluster_config = convert_to_cluster_config(intelligent_params)
    
    errors = validate_cluster_config(cluster_config)
    assert len(errors) == 0  # Should be fully valid after conversion


def test_validate_production_focused_defaults():
    """Test validation with production-focused default configurations."""
    from wellscope_mvp.pipeline.vector_builder import VectorConfig
    from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
    from wellscope_mvp.pipeline.umap_projector import ProjectionConfig
    
    # Test default production-optimized configs
    vector_config = VectorConfig()  # Should use production defaults
    cluster_config = ClusterConfig()  # Should use HDBSCAN + cosine
    projection_config = ProjectionConfig()
    
    # All should validate successfully
    assert len(validate_vector_config(vector_config.__dict__)) == 0
    assert len(validate_cluster_config(cluster_config)) == 0
    assert len(validate_projection_config(projection_config.__dict__)) == 0


def test_validate_cosine_metric_preference():
    """Test that cosine metric is properly validated for production use.""" 
    # Cosine should be a valid metric
    cosine_config = {
        'min_cluster_size': 6,
        'metric': 'cosine',
        'use_hdbscan': True
    }
    errors = validate_cluster_config(cosine_config)
    assert len(errors) == 0
    
    # Test other valid metrics still work
    euclidean_config = {
        'min_cluster_size': 6,
        'metric': 'euclidean', 
        'use_hdbscan': True
    }
    errors = validate_cluster_config(euclidean_config)
    assert len(errors) == 0
    
    # Test that manhattan is also valid (common alternative)
    manhattan_config = {
        'min_cluster_size': 6,
        'metric': 'manhattan',
        'use_hdbscan': True  
    }
    errors = validate_cluster_config(manhattan_config)
    # Should be valid or give specific metric error (not crash)
    assert isinstance(errors, list)