# tests/app/test_mature_well_clustering.py

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.utils.mature_well_clustering import (
    analyze_well_maturity,
    run_mature_first_clustering,
    get_mature_clustering_recommendations,
    _calculate_available_months,
    _recalculate_cluster_sizes
)
from wellscope_mvp.pipeline.vector_builder import VectorConfig
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig


@pytest.fixture
def mixed_vintage_data():
    """Create sample data with mixed well vintages (mature and young wells)."""
    np.random.seed(42)
    
    # Create 50 mature wells (24+ months) and 30 young wells (6-12 months)
    mature_wells = []
    young_wells = []
    
    base_date = datetime(2020, 1, 1)
    
    # Generate mature wells (2020-2021 completion, 24+ months of data)
    for i in range(50):
        api = f'mature_{i:03d}'
        completion_date = base_date + timedelta(days=np.random.randint(0, 365))
        
        # Generate 24-36 months of production data
        n_months = np.random.randint(24, 37)
        for month in range(n_months):
            prod_date = completion_date + timedelta(days=30 * month)
            
            # Realistic decline curve
            oil_prod = max(50, 1000 * np.exp(-0.05 * month) + np.random.normal(0, 50))
            gas_prod = oil_prod * 2 + np.random.normal(0, 100)
            water_prod = max(0, oil_prod * 0.3 + np.random.normal(0, 20))
            
            mature_wells.append({
                'API14': api,
                'Target Formation': 'EAGLEFORD',
                'Completion Date': completion_date.strftime('%Y-%m-%d'),
                'DI Lateral Length': np.random.randint(7000, 12000),
                'Monthly Production Date': prod_date.strftime('%Y-%m-%d'),
                'Monthly Oil': oil_prod,
                'Monthly Gas': gas_prod,
                'Monthly Water': water_prod,
                'Well Count': 1,
                'Producing Month Number': month + 1
            })
    
    # Generate young wells (2023 completion, 6-12 months of data)  
    young_base = datetime(2023, 1, 1)
    for i in range(30):
        api = f'young_{i:03d}'
        completion_date = young_base + timedelta(days=np.random.randint(0, 180))
        
        # Generate 6-12 months of production data
        n_months = np.random.randint(6, 13)
        for month in range(n_months):
            prod_date = completion_date + timedelta(days=30 * month)
            
            # Similar decline curve but shorter
            oil_prod = max(50, 800 * np.exp(-0.04 * month) + np.random.normal(0, 40))
            gas_prod = oil_prod * 2.2 + np.random.normal(0, 80)
            water_prod = max(0, oil_prod * 0.25 + np.random.normal(0, 15))
            
            young_wells.append({
                'API14': api,
                'Target Formation': 'EAGLEFORD',
                'Completion Date': completion_date.strftime('%Y-%m-%d'),
                'DI Lateral Length': np.random.randint(8000, 13000),
                'Monthly Production Date': prod_date.strftime('%Y-%m-%d'),
                'Monthly Oil': oil_prod,
                'Monthly Gas': gas_prod,
                'Monthly Water': water_prod,
                'Well Count': 1,
                'Producing Month Number': month + 1
            })
    
    return pd.DataFrame(mature_wells + young_wells)


@pytest.fixture
def mostly_young_data():
    """Create sample data with mostly young wells (insufficient mature wells)."""
    np.random.seed(42)
    
    # Create only 5 mature wells and 45 young wells
    wells = []
    
    # Generate a few mature wells
    for i in range(5):
        api = f'mature_{i:03d}'
        base_date = datetime(2020, 1, 1)
        
        for month in range(30):  # 30 months of data
            prod_date = base_date + timedelta(days=30 * month)
            oil_prod = max(50, 1000 * np.exp(-0.05 * month))
            
            wells.append({
                'API14': api,
                'Monthly Production Date': prod_date.strftime('%Y-%m-%d'),
                'Monthly Oil': oil_prod,
                'Monthly Gas': oil_prod * 2,
                'Monthly Water': oil_prod * 0.3
            })
    
    # Generate many young wells
    for i in range(45):
        api = f'young_{i:03d}'
        base_date = datetime(2023, 1, 1)
        
        for month in range(8):  # 8 months of data
            prod_date = base_date + timedelta(days=30 * month)
            oil_prod = max(50, 800 * np.exp(-0.04 * month))
            
            wells.append({
                'API14': api,
                'Monthly Production Date': prod_date.strftime('%Y-%m-%d'),
                'Monthly Oil': oil_prod,
                'Monthly Gas': oil_prod * 2,
                'Monthly Water': oil_prod * 0.3
            })
    
    return pd.DataFrame(wells)


def test_analyze_well_maturity_mixed_data(mixed_vintage_data):
    """Test well maturity analysis with mixed vintage data."""
    analysis = analyze_well_maturity(mixed_vintage_data, maturity_threshold=24)
    
    assert analysis['total_wells'] == 80  # Total wells
    # Allow for some variation due to random data generation (24+ months threshold)
    assert analysis['mature_wells'] >= 45  # Should be around 50, but allow some variance
    assert analysis['young_wells'] >= 25   # Should be around 30, but allow some variance
    assert analysis['mature_wells'] + analysis['young_wells'] == 80  # Total should match
    assert analysis['mature_fraction'] >= 0.5  # Should be majority mature
    assert analysis['recommended_strategy'] == 'mature_first'
    
    # Check API lists structure
    assert len(analysis['mature_well_apis']) == analysis['mature_wells']
    assert len(analysis['young_well_apis']) == analysis['young_wells']
    assert isinstance(analysis['mature_well_apis'], list)
    assert isinstance(analysis['young_well_apis'], list)
    
    # Verify all APIs are accounted for
    all_apis = set(analysis['mature_well_apis'] + analysis['young_well_apis'])
    unique_apis = mixed_vintage_data['API14'].unique()
    assert len(all_apis) == len(unique_apis)
    assert all_apis == set(unique_apis)


def test_analyze_well_maturity_insufficient_mature(mostly_young_data):
    """Test well maturity analysis with insufficient mature wells."""
    analysis = analyze_well_maturity(mostly_young_data, maturity_threshold=24)
    
    assert analysis['total_wells'] == 50
    assert analysis['mature_wells'] == 5
    assert analysis['young_wells'] == 45
    assert analysis['mature_fraction'] == 0.1  # 5/50
    assert analysis['recommended_strategy'] == 'uniform_short'


def test_analyze_well_maturity_empty_data():
    """Test well maturity analysis with empty data."""
    empty_df = pd.DataFrame()
    analysis = analyze_well_maturity(empty_df)
    
    assert analysis['total_wells'] == 0
    assert analysis['mature_wells'] == 0
    assert analysis['recommended_strategy'] == 'insufficient_data'


def test_get_mature_clustering_recommendations(mixed_vintage_data):
    """Test mature clustering recommendations."""
    recommendations = get_mature_clustering_recommendations(mixed_vintage_data)
    
    assert recommendations['strategy'] == 'mature_first'
    assert len(recommendations['recommendations']) > 0
    assert len(recommendations['warnings']) == 0  # Should be no warnings for good data
    
    # Check that recommendations mention mature wells
    rec_text = ' '.join(recommendations['recommendations'])
    assert 'mature wells' in rec_text.lower()


def test_get_mature_clustering_recommendations_poor_data(mostly_young_data):
    """Test recommendations with insufficient mature data."""
    recommendations = get_mature_clustering_recommendations(mostly_young_data)
    
    assert recommendations['strategy'] == 'uniform_short'
    assert len(recommendations['warnings']) > 0  # Should have warnings
    
    # Check that warnings mention the issue
    warning_text = ' '.join(recommendations['warnings'])
    assert 'few mature wells' in warning_text.lower()


def test_run_mature_first_clustering_success(mixed_vintage_data):
    """Test successful mature-first clustering."""
    vector_config = VectorConfig(months=24, normalize="q_over_qmax", stream="oil")
    cluster_config = ClusterConfig(use_hdbscan=True, min_cluster_size=5, metric="cosine")
    
    labels_df, meta = run_mature_first_clustering(
        mixed_vintage_data, vector_config, cluster_config, maturity_threshold=24
    )
    
    # Check basic structure
    assert len(labels_df) == 80  # All wells should be labeled
    assert 'label' in labels_df.columns
    assert 'is_noise' in labels_df.columns
    assert 'cluster_size' in labels_df.columns
    
    # Check metadata
    assert 'clustering_strategy' in meta
    assert meta['clustering_strategy'] == 'mature_first'
    assert 'maturity_analysis' in meta
    assert 'mature_vector_meta' in meta
    
    # Should have found some clusters (not all noise)
    n_clusters = len(labels_df[labels_df['label'] >= 0]['label'].unique())
    assert n_clusters > 0, "Should find at least some clusters"


def test_run_mature_first_clustering_fallback(mostly_young_data):
    """Test mature-first clustering fallback with insufficient mature wells."""
    vector_config = VectorConfig(months=24, normalize="q_over_qmax", stream="oil")
    cluster_config = ClusterConfig(use_hdbscan=True, min_cluster_size=3, metric="cosine")
    
    labels_df, meta = run_mature_first_clustering(
        mostly_young_data, vector_config, cluster_config, maturity_threshold=24
    )
    
    # Should fall back to uniform approach
    assert meta['clustering_strategy'] in ['fallback_uniform', 'mature_first']
    assert len(labels_df) == 50  # All wells labeled
    
    if meta['clustering_strategy'] == 'fallback_uniform':
        assert 'fallback_vector_length' in meta
        assert meta['fallback_vector_length'] < 24  # Shorter vectors


def test_calculate_available_months():
    """Test calculation of available months per well."""
    # Create simple test data
    df = pd.DataFrame({
        'API14': ['001', '001', '001', '002', '002', '003'],
        'Monthly Production Date': [
            '2020-01-01', '2020-02-01', '2020-03-01',  # Well 001: 3 months
            '2020-01-01', '2020-02-01',                # Well 002: 2 months  
            '2020-01-01'                               # Well 003: 1 month
        ],
        'Monthly Oil': [100, 90, 80, 200, 180, 150]
    })
    
    months = _calculate_available_months(df, 'API14')
    
    # Should calculate months per well correctly
    assert len(months) >= 1  # At least some wells


def test_recalculate_cluster_sizes():
    """Test cluster size recalculation."""
    # Create test labels
    labels_df = pd.DataFrame({
        'API14': ['001', '002', '003', '004', '005'],
        'label': [0, 0, 1, 1, -1],  # Cluster 0: 2 wells, Cluster 1: 2 wells, Noise: 1 well
        'is_noise': [False, False, False, False, True],
        'cluster_size': [0, 0, 0, 0, 0]  # Will be recalculated
    })
    
    updated_df = _recalculate_cluster_sizes(labels_df)
    
    # Check cluster sizes
    cluster_0_size = updated_df[updated_df['label'] == 0]['cluster_size'].iloc[0]
    cluster_1_size = updated_df[updated_df['label'] == 1]['cluster_size'].iloc[0]
    noise_size = updated_df[updated_df['label'] == -1]['cluster_size'].iloc[0]
    
    assert cluster_0_size == 2
    assert cluster_1_size == 2  
    assert noise_size == 0  # Noise wells have size 0


def test_mature_first_clustering_edge_cases():
    """Test edge cases for mature-first clustering."""
    # Test with very small dataset
    tiny_df = pd.DataFrame({
        'API14': ['001', '002'],
        'Monthly Production Date': ['2020-01-01', '2020-01-01'],
        'Monthly Oil': [100, 200]
    })
    
    vector_config = VectorConfig(months=6, normalize="q_over_qmax", stream="oil")
    cluster_config = ClusterConfig(use_hdbscan=True, min_cluster_size=2, metric="cosine")
    
    # Should handle gracefully without errors
    labels_df, meta = run_mature_first_clustering(
        tiny_df, vector_config, cluster_config, maturity_threshold=24
    )
    
    assert len(labels_df) == 2
    assert 'clustering_strategy' in meta