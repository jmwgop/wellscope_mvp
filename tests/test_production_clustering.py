# tests/test_production_clustering.py

import pytest
import numpy as np
import pandas as pd
from wellscope_mvp.pipeline.production_clustering import (
    detect_production_data,
    get_production_optimized_config,
    get_production_clustering_recommendation,
    calculate_production_similarity_stats
)


@pytest.fixture
def high_similarity_production_data():
    """Create synthetic high-similarity production data like Eagleford."""
    np.random.seed(42)
    n_wells = 50
    n_months = 12
    
    # Create decline curves with high similarity (90%+)
    # All wells start high and decline exponentially with similar patterns
    base_curve = np.exp(-np.linspace(0, 2, n_months))  # Exponential decline
    
    vectors_df = pd.DataFrame()
    vectors_df['API_UWI'] = [f"WELL_{i:03d}" for i in range(n_wells)]
    
    # Create vector columns first
    for j in range(n_months):
        vectors_df[f'v{j:02d}'] = 0.0
        
    # Add small random variations to create 90%+ similarity
    for i in range(n_wells):
        # Small random variations (5-10%) around base curve
        noise_factor = 0.95 + np.random.normal(0, 0.05, n_months)
        curve = base_curve * noise_factor * (0.8 + np.random.random() * 0.4)  # Scale variation
        
        for j in range(n_months):
            vectors_df.loc[i, f'v{j:02d}'] = curve[j]
    
    return vectors_df


@pytest.fixture 
def medium_similarity_production_data():
    """Create synthetic medium-similarity production data."""
    np.random.seed(123)
    n_wells = 30
    n_months = 12
    
    vectors_df = pd.DataFrame()
    vectors_df['API_UWI'] = [f"WELL_{i:03d}" for i in range(n_wells)]
    
    # Create more diverse decline patterns (80% similarity)
    for i in range(n_wells):
        # Three different decline patterns
        if i % 3 == 0:
            # Steep decline
            curve = np.exp(-np.linspace(0, 3, n_months))
        elif i % 3 == 1:
            # Gradual decline  
            curve = np.exp(-np.linspace(0, 1.5, n_months))
        else:
            # Mixed decline
            curve = np.exp(-np.linspace(0, 2.2, n_months))
            
        # Add moderate noise
        noise_factor = 0.85 + np.random.normal(0, 0.1, n_months)
        curve = curve * noise_factor * (0.5 + np.random.random() * 0.8)
        
        for j in range(n_months):
            vectors_df.loc[i, f'v{j:02d}'] = curve[j]
    
    return vectors_df


@pytest.fixture
def non_production_data():
    """Create synthetic non-production data (random, low similarity)."""
    np.random.seed(456)
    n_wells = 25
    n_features = 12
    
    vectors_df = pd.DataFrame()
    vectors_df['ID'] = [f"SAMPLE_{i:03d}" for i in range(n_wells)]
    
    # Random data with low similarity
    for j in range(n_features):
        vectors_df[f'v{j:02d}'] = np.random.normal(0, 1, n_wells)
    
    return vectors_df


class TestProductionDataDetection:
    """Test production data detection functionality."""
    
    def test_detect_high_similarity_production_data(self, high_similarity_production_data):
        """Test detection of high-similarity production data."""
        result = detect_production_data(high_similarity_production_data)
        
        assert result['is_production_data'] is True
        assert result['confidence'] > 0.85
        assert result['cosine_similarity_mean'] > 0.85
        assert 'high_similarity' in result['reason']
    
    def test_detect_medium_similarity_production_data(self, medium_similarity_production_data):
        """Test detection of medium-similarity production data.""" 
        result = detect_production_data(medium_similarity_production_data)
        
        assert result['is_production_data'] is True
        assert result['confidence'] > 0.7
        assert result['cosine_similarity_mean'] > 0.75  # May be higher than expected due to test data
    
    def test_detect_non_production_data(self, non_production_data):
        """Test that non-production data is not detected as production data."""
        result = detect_production_data(non_production_data)
        
        assert result['is_production_data'] is False
        assert result['confidence'] < 0.5
        assert result['cosine_similarity_mean'] < 0.75
    
    def test_insufficient_data_handling(self):
        """Test handling of insufficient data."""
        # Too few wells
        small_df = pd.DataFrame({
            'API': ['W1', 'W2'],
            'v01': [1.0, 0.9],
            'v02': [0.8, 0.7],
            'v03': [0.6, 0.5],
            'v04': [0.4, 0.3],
            'v05': [0.2, 0.1],
            'v06': [0.1, 0.05]
        })
        
        result = detect_production_data(small_df)
        assert result['is_production_data'] is False
        assert result['reason'] == 'insufficient_data_points'
    
    def test_insufficient_vector_length(self):
        """Test handling of insufficient vector length."""
        short_df = pd.DataFrame({
            'API': [f'W{i}' for i in range(20)],
            'v01': np.random.random(20),
            'v02': np.random.random(20)
        })
        
        result = detect_production_data(short_df)
        assert result['is_production_data'] is False
        assert result['reason'] == 'insufficient_vector_length'


class TestProductionOptimizedConfig:
    """Test production-optimized configuration generation."""
    
    def test_high_similarity_config(self, high_similarity_production_data):
        """Test configuration for high-similarity data."""
        similarity_stats = {'cosine_similarity_mean': 0.90}
        
        config = get_production_optimized_config(
            high_similarity_production_data, 
            similarity_stats
        )
        
        assert config.use_hdbscan is False  # Should use DBSCAN for high similarity
        assert config.eps == 0.05  # Aggressive epsilon from experiments
        assert config.min_samples == 2  # Minimal samples
        assert config.metric == "euclidean"  # Optimal for small eps
    
    def test_medium_similarity_config(self, medium_similarity_production_data):
        """Test configuration for medium-similarity data.""" 
        similarity_stats = {'cosine_similarity_mean': 0.82}
        
        config = get_production_optimized_config(
            medium_similarity_production_data,
            similarity_stats
        )
        
        assert config.use_hdbscan is False  # Still use DBSCAN
        assert config.eps == 0.08  # Less aggressive
        assert config.min_samples == 2
    
    def test_moderate_similarity_config(self, medium_similarity_production_data):
        """Test configuration for moderate-similarity data."""
        similarity_stats = {'cosine_similarity_mean': 0.68}
        
        config = get_production_optimized_config(
            medium_similarity_production_data,
            similarity_stats  
        )
        
        # Should use HDBSCAN with aggressive params for moderate similarity
        assert config.use_hdbscan is True
        assert config.metric == "cosine"  # Better for shape-based
        assert config.min_cluster_size <= 5  # Small clusters


class TestProductionClusteringRecommendation:
    """Test comprehensive production clustering recommendations."""
    
    def test_high_similarity_recommendation(self, high_similarity_production_data):
        """Test recommendation for high-similarity production data."""
        recommendation = get_production_clustering_recommendation(high_similarity_production_data)
        
        assert recommendation['is_production_data'] is True
        assert recommendation['confidence'] > 0.85
        assert recommendation['algorithm_choice'] == 'DBSCAN'
        assert recommendation['expected_clusters'] >= 2
        assert 'Production Data Detected' in recommendation['user_message']
        
        config = recommendation['recommended_config']
        assert config.eps == 0.05  # Optimal from experiments
    
    def test_non_production_recommendation(self, non_production_data):
        """Test recommendation for non-production data."""
        recommendation = get_production_clustering_recommendation(non_production_data)
        
        assert recommendation['is_production_data'] is False
        assert recommendation['confidence'] < 0.5
        assert 'Standard clustering parameters' in recommendation['user_message']


class TestProductionSimilarityStats:
    """Test production similarity statistics calculation."""
    
    def test_similarity_stats_calculation(self, high_similarity_production_data):
        """Test comprehensive similarity statistics."""
        stats = calculate_production_similarity_stats(high_similarity_production_data)
        
        assert stats['n_wells'] == 50
        assert stats['n_features'] == 12
        assert stats['cosine_similarity_mean'] > 0.8
        assert 'p50' in stats  # Median percentile
        assert 'p95' in stats  # 95th percentile 
        assert stats['high_similarity_95_pct'] > 0.3  # Many high-similarity pairs
    
    def test_empty_data_handling(self):
        """Test handling of empty or invalid data."""
        empty_df = pd.DataFrame()
        stats = calculate_production_similarity_stats(empty_df)
        
        assert 'error' in stats
        assert stats['error'] == 'no_vector_columns'
    
    def test_single_well_handling(self):
        """Test handling of single well."""
        single_df = pd.DataFrame({
            'API': ['W1'],
            'v01': [1.0],
            'v02': [0.8]
        })
        
        stats = calculate_production_similarity_stats(single_df)
        assert 'error' in stats
        assert stats['error'] == 'insufficient_wells'


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_missing_vector_columns(self):
        """Test handling of data without vector columns."""
        no_vectors_df = pd.DataFrame({
            'API': ['W1', 'W2'],
            'other_col': [1, 2]
        })
        
        result = detect_production_data(no_vectors_df)
        assert result['is_production_data'] is False
        assert 'insufficient_vector_length' in result['reason']
    
    def test_nan_handling(self):
        """Test handling of NaN values in vectors."""
        nan_df = pd.DataFrame({
            'API': [f'W{i}' for i in range(10)],
            'v01': [np.nan] * 5 + [1.0] * 5,
            'v02': [0.8] * 10
        })
        
        # Should handle NaNs gracefully 
        result = detect_production_data(nan_df)
        assert 'error' not in result  # Should not crash
    
    def test_all_zeros_handling(self):
        """Test handling of all-zero vectors."""
        zeros_df = pd.DataFrame({
            'API': [f'W{i}' for i in range(10)],
            **{f'v{j:02d}': [0.0] * 10 for j in range(12)}
        })
        
        result = detect_production_data(zeros_df)
        assert result['is_production_data'] is False  # All zeros shouldn't be production data


if __name__ == '__main__':
    pytest.main([__file__])