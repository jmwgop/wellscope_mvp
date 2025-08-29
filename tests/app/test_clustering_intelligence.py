# tests/app/test_clustering_intelligence.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.utils.clustering_intelligence import (
    calculate_data_characteristics,
    calculate_intelligent_cluster_params,
    get_user_friendly_suggestions,
    convert_to_cluster_config
)
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig


@pytest.fixture
def sample_filtered_data():
    """Create sample filtered data with various characteristics."""
    # Create 100 wells with some diversity
    n_wells = 100
    
    # Generate dates spread over several years
    base_date = datetime(2018, 1, 1)
    completion_dates = [
        (base_date + timedelta(days=np.random.randint(0, 1095))).strftime('%Y-%m-%d')
        for _ in range(n_wells)
    ]
    
    # Generate formations (mostly Eagle Ford)
    formations = ['EAGLEFORD'] * 70 + ['AUSTIN CHALK'] * 20 + ['BUDA'] * 10
    np.random.shuffle(formations)
    
    # Generate lateral lengths with some variance
    lateral_lengths = np.random.normal(8000, 2000, n_wells)
    lateral_lengths = np.clip(lateral_lengths, 3000, 15000)
    
    df = pd.DataFrame({
        'API14': [f'42{i:08d}' for i in range(n_wells)],
        'Target Formation': formations,
        'Completion Date': completion_dates,
        'DI Lateral Length': lateral_lengths,
        'Operator (Reported)': ['OPERATOR_A'] * 60 + ['OPERATOR_B'] * 40
    })
    
    return df


def test_calculate_data_characteristics_normal_data(sample_filtered_data):
    """Test data characteristics calculation with normal dataset."""
    characteristics = calculate_data_characteristics(sample_filtered_data)
    
    assert characteristics['n_wells'] == 100
    assert characteristics['formation_count'] == 3  # EAGLEFORD, AUSTIN CHALK, BUDA
    assert 0.5 <= characteristics['diversity_score'] <= 2.0
    assert characteristics['year_spread'] >= 2  # Should span multiple years
    assert characteristics['lateral_cv'] > 0  # Should have some variance


def test_calculate_data_characteristics_small_dataset():
    """Test with very small dataset."""
    small_df = pd.DataFrame({
        'API14': ['001', '002', '003'],
        'Target Formation': ['EAGLEFORD', 'EAGLEFORD', 'EAGLEFORD'],
        'Completion Date': ['2020-01-01', '2020-06-01', '2020-12-01']
    })
    
    characteristics = calculate_data_characteristics(small_df)
    
    assert characteristics['n_wells'] == 3
    assert characteristics['formation_count'] == 1
    assert characteristics['diversity_score'] < 1.0  # Low diversity


def test_calculate_data_characteristics_empty_data():
    """Test with empty dataset."""
    empty_df = pd.DataFrame()
    
    characteristics = calculate_data_characteristics(empty_df)
    
    assert characteristics['n_wells'] == 0
    assert characteristics['diversity_score'] == 1.0
    assert characteristics['formation_count'] == 0


def test_calculate_intelligent_cluster_params_small_dataset():
    """Test intelligent parameters for small dataset."""
    small_chars = {'n_wells': 20, 'diversity_score': 0.8, 'formation_count': 1, 'year_spread': 2, 'lateral_cv': 0.3}
    
    params = calculate_intelligent_cluster_params(small_chars)
    
    assert 2 <= params['min_cluster_size'] <= 8  # Should be reasonable for 20 wells
    assert params['expected_clusters'] <= 6  # Not too many clusters for small data
    assert params['algorithm_recommendation'] in ['HDBSCAN', 'DBSCAN']
    assert 0 <= params['confidence_score'] <= 1


def test_calculate_intelligent_cluster_params_medium_dataset():
    """Test intelligent parameters for medium dataset."""
    medium_chars = {'n_wells': 200, 'diversity_score': 1.2, 'formation_count': 2, 'year_spread': 5, 'lateral_cv': 0.4}
    
    params = calculate_intelligent_cluster_params(medium_chars)
    
    assert 4 <= params['min_cluster_size'] <= 30  # Reasonable for 200 wells
    assert params['expected_clusters'] >= 3  # Should find multiple clusters
    assert params['algorithm_recommendation'] == 'HDBSCAN'  # Preferred for medium datasets
    assert params['confidence_score'] > 0.5  # Should be confident


def test_calculate_intelligent_cluster_params_large_dataset():
    """Test intelligent parameters for large dataset."""
    large_chars = {'n_wells': 2000, 'diversity_score': 1.4, 'formation_count': 3, 'year_spread': 8, 'lateral_cv': 0.6}
    
    params = calculate_intelligent_cluster_params(large_chars)
    
    assert 15 <= params['min_cluster_size'] <= 100  # Reasonable for 2000 wells
    assert params['expected_clusters'] >= 8  # Should find many clusters
    assert params['algorithm_recommendation'] == 'HDBSCAN'
    assert params['confidence_score'] > 0.7  # Should be very confident


def test_calculate_intelligent_cluster_params_with_preferences():
    """Test intelligent parameters with different user preferences."""
    chars = {'n_wells': 100, 'diversity_score': 1.0, 'formation_count': 2, 'year_spread': 3, 'lateral_cv': 0.4}
    
    # Test different group size preferences
    small_params = calculate_intelligent_cluster_params(chars, group_size_preference="small")
    large_params = calculate_intelligent_cluster_params(chars, group_size_preference="large")
    
    assert small_params['min_cluster_size'] < large_params['min_cluster_size']
    
    # Test different sensitivity preferences
    loose_params = calculate_intelligent_cluster_params(chars, sensitivity="loose")
    strict_params = calculate_intelligent_cluster_params(chars, sensitivity="strict")
    
    assert loose_params['min_cluster_size'] < strict_params['min_cluster_size']


def test_get_user_friendly_suggestions(sample_filtered_data):
    """Test user-friendly suggestions generation."""
    characteristics = calculate_data_characteristics(sample_filtered_data)
    suggestions = get_user_friendly_suggestions(characteristics)
    
    assert 'suggestions' in suggestions
    assert 'warnings' in suggestions
    assert 'recommended_clusters' in suggestions
    assert 'confidence' in suggestions
    assert 'algorithm' in suggestions
    
    assert isinstance(suggestions['suggestions'], list)
    assert isinstance(suggestions['warnings'], list)
    assert isinstance(suggestions['recommended_clusters'], int)
    assert 0 <= suggestions['confidence'] <= 1
    assert suggestions['algorithm'] in ['HDBSCAN', 'DBSCAN']


def test_get_user_friendly_suggestions_small_dataset():
    """Test suggestions for small dataset."""
    small_chars = {'n_wells': 15, 'diversity_score': 0.7, 'formation_count': 1, 'year_spread': 1, 'lateral_cv': 0.2}
    suggestions = get_user_friendly_suggestions(small_chars)
    
    assert len(suggestions['warnings']) > 0  # Should warn about small dataset
    assert 'Small dataset' in suggestions['warnings'][0]


def test_get_user_friendly_suggestions_high_diversity():
    """Test suggestions for high diversity dataset."""
    diverse_chars = {'n_wells': 500, 'diversity_score': 1.8, 'formation_count': 5, 'year_spread': 10, 'lateral_cv': 0.8}
    suggestions = get_user_friendly_suggestions(diverse_chars)
    
    assert len(suggestions['suggestions']) > 0
    assert any('diversity' in suggestion.lower() for suggestion in suggestions['suggestions'])


def test_convert_to_cluster_config():
    """Test conversion to ClusterConfig object."""
    intelligent_params = {
        'min_cluster_size': 10,
        'min_samples': 6,
        'use_hdbscan': True,
        'expected_clusters': 5,
        'algorithm_recommendation': 'HDBSCAN',
        'confidence_score': 0.8
    }
    
    config = convert_to_cluster_config(intelligent_params, metric='cosine')
    
    assert isinstance(config, ClusterConfig)
    assert config.min_cluster_size == 10
    assert config.min_samples == 6
    assert config.use_hdbscan == True
    assert config.metric == 'cosine'
    assert config.eps == 0.5  # Default fallback


def test_convert_to_cluster_config_without_min_samples():
    """Test conversion when min_samples is not provided."""
    intelligent_params = {
        'min_cluster_size': 8,
        'use_hdbscan': False,
        'expected_clusters': 4,
        'algorithm_recommendation': 'DBSCAN',
        'confidence_score': 0.6
    }
    
    config = convert_to_cluster_config(intelligent_params, metric='cosine')  # Updated to cosine default
    
    assert isinstance(config, ClusterConfig)
    assert config.min_cluster_size == 8
    assert config.min_samples is None
    assert config.use_hdbscan == False
    assert config.metric == 'cosine'  # Updated expectation


def test_edge_cases_zero_wells():
    """Test edge case with zero wells."""
    characteristics = calculate_data_characteristics(pd.DataFrame())
    params = calculate_intelligent_cluster_params(characteristics)
    
    assert params['min_cluster_size'] == 2  # Minimum possible
    assert params['algorithm_recommendation'] == 'insufficient_data'


def test_edge_cases_single_well():
    """Test edge case with single well."""
    single_well_df = pd.DataFrame({
        'API14': ['001'],
        'Target Formation': ['EAGLEFORD'],
        'Completion Date': ['2020-01-01']
    })
    
    characteristics = calculate_data_characteristics(single_well_df)
    params = calculate_intelligent_cluster_params(characteristics)
    
    assert params['min_cluster_size'] == 2  # Still minimum possible
    assert params['confidence_score'] <= 0.7  # Should be reasonable but not high


def test_data_characteristics_missing_columns():
    """Test data characteristics calculation with missing columns."""
    minimal_df = pd.DataFrame({
        'API14': ['001', '002', '003', '004', '005']
    })
    
    characteristics = calculate_data_characteristics(minimal_df)
    
    assert characteristics['n_wells'] == 5
    assert characteristics['formation_count'] == 1  # Default when no formation column
    # Diversity score includes base + formation + year + lateral defaults
    assert 0.5 <= characteristics['diversity_score'] <= 1.5  # Reasonable range


def test_intelligent_params_extreme_diversity():
    """Test intelligent parameters with extreme diversity values."""
    # Very low diversity
    low_diversity_chars = {'n_wells': 100, 'diversity_score': 0.3, 'formation_count': 1, 'year_spread': 1, 'lateral_cv': 0.1}
    low_params = calculate_intelligent_cluster_params(low_diversity_chars)
    
    # Very high diversity
    high_diversity_chars = {'n_wells': 100, 'diversity_score': 2.2, 'formation_count': 8, 'year_spread': 15, 'lateral_cv': 1.2}
    high_params = calculate_intelligent_cluster_params(high_diversity_chars)
    
    # High diversity should require larger clusters
    assert high_params['min_cluster_size'] > low_params['min_cluster_size']


def test_calculate_optimal_vector_length():
    """Test optimal vector length calculation based on data maturity."""
    from app.utils.clustering_intelligence import calculate_optimal_vector_length
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create dataset with varying production lengths
    wells = []
    base_date = datetime(2020, 1, 1)
    
    # 10 wells with 36 months, 10 wells with 12 months, 5 wells with 6 months
    well_configs = [(10, 36), (10, 12), (5, 6)]
    
    well_id = 0
    for count, months in well_configs:
        for i in range(count):
            api = f'42{well_id:08d}'
            for month in range(months):
                prod_date = base_date + timedelta(days=30*month)
                wells.append({
                    'API14': api,
                    'Monthly Production Date': prod_date.strftime('%Y-%m-%d'),
                    'Monthly Oil': max(50, 1000 * (0.95 ** month)),
                    'Producing Month Number': month + 1
                })
            well_id += 1
    
    filtered_df = pd.DataFrame(wells)
    
    # Test with min_months_filter = 12
    result = calculate_optimal_vector_length(filtered_df, min_months_filter=12)
    
    assert 'recommended_length' in result
    assert 'max_safe_length' in result
    assert 'data_completeness' in result
    assert 'warnings' in result
    
    # Should recommend length that most wells can support
    recommended = result['recommended_length']
    assert 6 <= recommended <= 24  # Reasonable range
    
    # Max safe should be >= recommended
    assert result['max_safe_length'] >= recommended
    
    # Should have completeness data for common lengths
    completeness = result['data_completeness']
    assert 6 in completeness
    assert 12 in completeness
    assert 24 in completeness


def test_get_vector_length_suggestions():
    """Test user-friendly vector length suggestions."""
    from app.utils.clustering_intelligence import get_vector_length_suggestions
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create simple dataset with consistent 18-month data
    wells = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(15):
        api = f'42{i:08d}'
        for month in range(18):  # 18 months each
            wells.append({
                'API14': api,
                'Monthly Production Date': (base_date + timedelta(days=30*month)).strftime('%Y-%m-%d'),
                'Monthly Oil': max(50, 800 * (0.96 ** month)),
                'Producing Month Number': month + 1
            })
    
    filtered_df = pd.DataFrame(wells)
    filters_cfg = {'min_months_produced': 12}
    
    result = get_vector_length_suggestions(filtered_df, filters_cfg)
    
    assert 'recommendations' in result
    assert 'warnings' in result
    assert 'optimal_length' in result
    assert 'max_safe_length' in result
    assert 'vector_analysis' in result
    
    # Should provide reasonable suggestions
    assert isinstance(result['recommendations'], list)
    assert isinstance(result['warnings'], list)
    assert 6 <= result['optimal_length'] <= 18
    assert result['max_safe_length'] >= result['optimal_length']


def test_vector_length_with_insufficient_data():
    """Test vector length calculation with insufficient data."""
    from app.utils.clustering_intelligence import calculate_optimal_vector_length
    import pandas as pd
    
    # Create dataset where most wells have very little data
    wells = []
    for i in range(10):
        api = f'42{i:08d}'
        # Only 3-4 months of data per well
        for month in range(3 + i % 2):
            wells.append({
                'API14': api,
                'Monthly Production Date': f'2020-{month+1:02d}-01',
                'Monthly Oil': 100,
                'Producing Month Number': month + 1
            })
    
    filtered_df = pd.DataFrame(wells)
    
    result = calculate_optimal_vector_length(filtered_df, min_months_filter=6)
    
    # Should fall back to minimum safe length
    assert result['recommended_length'] == 6  # Minimum
    assert len(result['warnings']) > 0  # Should warn about data quality
    
    # Should still provide completeness data
    assert 'data_completeness' in result
    assert result['wells_analyzed'] == 10


def test_vector_length_filter_compatibility():
    """Test vector length recommendations respect filter constraints.""" 
    from app.utils.clustering_intelligence import get_vector_length_suggestions
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Create dataset with 30 months of data per well
    wells = []
    base_date = datetime(2020, 1, 1)
    
    for i in range(8):
        api = f'42{i:08d}'
        for month in range(30):
            wells.append({
                'API14': api,
                'Monthly Production Date': (base_date + timedelta(days=30*month)).strftime('%Y-%m-%d'),
                'Monthly Oil': max(50, 1200 * (0.94 ** month)),
                'Producing Month Number': month + 1
            })
    
    filtered_df = pd.DataFrame(wells)
    
    # Test with restrictive filter
    filters_cfg = {'min_months_produced': 18}
    result = get_vector_length_suggestions(filtered_df, filters_cfg)
    
    # Recommended length should respect the 18-month filter constraint
    assert result['optimal_length'] <= 18
    
    # Should provide filter-aware suggestions
    suggestions_text = ' '.join(result['recommendations'])
    assert '18' in suggestions_text or 'month' in suggestions_text.lower()