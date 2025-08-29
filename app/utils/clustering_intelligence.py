# app/utils/clustering_intelligence.py

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
from wellscope_mvp.pipeline.filter_inputs import compute_months_produced


def calculate_data_characteristics(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze dataset characteristics to inform intelligent clustering.
    
    Args:
        filtered_df: Filtered dataset from Step 2
        
    Returns:
        Dictionary with data characteristics for intelligent parameter selection
    """
    if filtered_df is None or len(filtered_df) == 0:
        return {'n_wells': 0, 'diversity_score': 1.0, 'formation_count': 0}
    
    n_wells = len(filtered_df['API14'].unique()) if 'API14' in filtered_df.columns else len(filtered_df)
    
    # Calculate formation diversity
    formation_count = 1
    if 'Target Formation' in filtered_df.columns:
        formation_count = filtered_df['Target Formation'].nunique()
    
    # Calculate completion year spread (proxy for technology/technique diversity)
    year_spread = 1
    if 'Completion Date' in filtered_df.columns:
        completion_dates = pd.to_datetime(filtered_df['Completion Date'], errors='coerce')
        years = completion_dates.dt.year.dropna()
        if len(years) > 0:
            year_spread = max(1, years.max() - years.min())
    
    # Calculate lateral length diversity
    lateral_cv = 0.3  # Default coefficient of variation
    lateral_cols = ['DI Lateral Length', 'Horizontal Length']
    for col in lateral_cols:
        if col in filtered_df.columns:
            lateral_data = pd.to_numeric(filtered_df[col], errors='coerce').dropna()
            if len(lateral_data) > 0 and lateral_data.mean() > 0:
                lateral_cv = lateral_data.std() / lateral_data.mean()
                break
    
    # Combine factors into diversity score (0.5 = low diversity, 1.5 = high diversity)
    diversity_score = (
        0.5 +  # Base diversity
        (formation_count - 1) * 0.1 +  # Formation diversity
        min(year_spread / 10.0, 0.3) +  # Year spread diversity (capped)
        min(lateral_cv, 0.7)  # Lateral length diversity (capped)
    )
    diversity_score = max(0.5, min(2.0, diversity_score))  # Clamp to reasonable range
    
    return {
        'n_wells': n_wells,
        'diversity_score': diversity_score,
        'formation_count': formation_count,
        'year_spread': year_spread,
        'lateral_cv': lateral_cv
    }


def calculate_intelligent_cluster_params(
    data_chars: Dict[str, Any],
    expected_clusters: Optional[int] = None,
    group_size_preference: str = "medium",  # "small", "medium", "large"
    sensitivity: str = "balanced"  # "loose", "balanced", "strict"
) -> Dict[str, Any]:
    """
    Calculate intelligent clustering parameters based on data characteristics and user preferences.
    
    Args:
        data_chars: Output from calculate_data_characteristics()
        expected_clusters: User's expected number of clusters (optional)
        group_size_preference: "small", "medium", or "large" groups
        sensitivity: "loose", "balanced", or "strict" clustering
        
    Returns:
        Dictionary with intelligent cluster parameters
    """
    n_wells = data_chars['n_wells']
    diversity_score = data_chars['diversity_score']
    
    if n_wells == 0:
        return {
            'min_cluster_size': 2,
            'min_samples': 2,
            'use_hdbscan': False,
            'algorithm_recommendation': 'insufficient_data'
        }
    
    # Calculate base cluster size percentage based on dataset size
    if n_wells <= 50:
        base_pct = 0.12  # 12% for small datasets
        algorithm = 'DBSCAN'  # HDBSCAN may be too sophisticated for small datasets
    elif n_wells <= 200:
        base_pct = 0.08  # 8% for small-medium datasets
        algorithm = 'HDBSCAN'
    elif n_wells <= 1000:
        base_pct = 0.04  # 4% for medium datasets
        algorithm = 'HDBSCAN'
    elif n_wells <= 5000:
        base_pct = 0.025  # 2.5% for large datasets
        algorithm = 'HDBSCAN'
    else:
        base_pct = 0.015  # 1.5% for massive datasets
        algorithm = 'HDBSCAN'
    
    # Adjust for group size preference
    size_multipliers = {
        'small': 0.6,    # Smaller clusters
        'medium': 1.0,   # Default size
        'large': 1.6     # Larger clusters
    }
    size_multiplier = size_multipliers.get(group_size_preference, 1.0)
    
    # Adjust for sensitivity
    sensitivity_multipliers = {
        'loose': 0.7,     # More permissive clustering
        'balanced': 1.0,  # Default sensitivity
        'strict': 1.4     # More strict clustering
    }
    sensitivity_multiplier = sensitivity_multipliers.get(sensitivity, 1.0)
    
    # Apply diversity factor (higher diversity needs larger clusters for stability)
    diversity_multiplier = max(0.8, min(1.3, diversity_score))
    
    # Calculate final min_cluster_size
    final_pct = base_pct * size_multiplier * sensitivity_multiplier * diversity_multiplier
    min_cluster_size = max(2, int(n_wells * final_pct))
    
    # Ensure reasonable bounds
    min_cluster_size = min(min_cluster_size, max(2, n_wells // 3))  # Never more than 1/3 of data
    
    # Calculate min_samples (typically 50-80% of min_cluster_size)
    min_samples_ratio = {
        'loose': 0.4,
        'balanced': 0.6,
        'strict': 0.8
    }
    min_samples = max(2, int(min_cluster_size * min_samples_ratio.get(sensitivity, 0.6)))
    
    # Estimate expected clusters if not provided
    if expected_clusters is None:
        if n_wells <= 20:
            expected_clusters = max(2, n_wells // 8)
        elif n_wells <= 100:
            expected_clusters = max(2, n_wells // 15)
        elif n_wells <= 1000:
            expected_clusters = max(3, n_wells // 25)
        else:
            expected_clusters = max(5, n_wells // 50)
    
    return {
        'min_cluster_size': min_cluster_size,
        'min_samples': min_samples,
        'use_hdbscan': algorithm == 'HDBSCAN',
        'expected_clusters': expected_clusters,
        'algorithm_recommendation': algorithm,
        'confidence_score': _calculate_confidence_score(n_wells, diversity_score)
    }


def _calculate_confidence_score(n_wells: int, diversity_score: float) -> float:
    """Calculate confidence in clustering success (0.0 to 1.0)."""
    # Base confidence on dataset size
    if n_wells < 10:
        size_confidence = 0.3
    elif n_wells < 50:
        size_confidence = 0.6
    elif n_wells < 200:
        size_confidence = 0.8
    else:
        size_confidence = 0.9
    
    # Adjust for diversity (moderate diversity is ideal)
    if 0.8 <= diversity_score <= 1.2:
        diversity_confidence = 1.0  # Ideal diversity
    elif 0.6 <= diversity_score <= 1.4:
        diversity_confidence = 0.9  # Good diversity
    elif 0.4 <= diversity_score <= 1.6:
        diversity_confidence = 0.7  # Acceptable diversity
    else:
        diversity_confidence = 0.5  # Challenging diversity
    
    return (size_confidence + diversity_confidence) / 2


def get_user_friendly_suggestions(data_chars: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate user-friendly suggestions and warnings based on data characteristics.
    
    Returns:
        Dictionary with suggestions, warnings, and recommendations
    """
    n_wells = data_chars['n_wells']
    diversity_score = data_chars['diversity_score']
    
    suggestions = []
    warnings = []
    
    # Dataset size suggestions
    if n_wells < 20:
        warnings.append(f"Small dataset ({n_wells} wells) - clustering may be challenging")
        suggestions.append("Consider using 'loose' sensitivity for better results")
    elif n_wells > 2000:
        suggestions.append(f"Large dataset ({n_wells:,} wells) - may take longer to process")
    
    # Diversity suggestions
    if diversity_score > 1.4:
        suggestions.append("High data diversity detected - consider larger group sizes")
    elif diversity_score < 0.7:
        suggestions.append("Low data diversity detected - smaller group sizes may work well")
    
    # Formation-specific suggestions
    formation_count = data_chars.get('formation_count', 1)
    if formation_count > 3:
        suggestions.append(f"Multiple formations ({formation_count}) detected - expect diverse clustering")
    
    # Recommended parameters
    intelligent_params = calculate_intelligent_cluster_params(data_chars)
    
    return {
        'suggestions': suggestions,
        'warnings': warnings,
        'recommended_clusters': intelligent_params.get('expected_clusters', 3),
        'confidence': intelligent_params.get('confidence_score', 0.5),
        'algorithm': intelligent_params.get('algorithm_recommendation', 'HDBSCAN')
    }


def convert_to_cluster_config(
    intelligent_params: Dict[str, Any], 
    metric: str = "cosine"  # BEST default for production curve shape clustering
) -> ClusterConfig:
    """
    Convert intelligent parameters to ClusterConfig object.
    
    Args:
        intelligent_params: Output from calculate_intelligent_cluster_params()
        metric: Distance metric to use (cosine optimal for production shapes)
        
    Returns:
        ClusterConfig object ready for pipeline
    """
    return ClusterConfig(
        use_hdbscan=intelligent_params['use_hdbscan'],
        min_cluster_size=intelligent_params['min_cluster_size'],
        min_samples=intelligent_params.get('min_samples'),
        metric=metric,
        eps=0.5  # Default DBSCAN fallback eps
    )


def calculate_optimal_vector_length(filtered_df: pd.DataFrame, 
                                   min_months_filter: int = 0,
                                   api_col: str = "API14") -> Dict[str, Any]:
    """
    Calculate optimal vector length based on data maturity and filter settings.
    
    Args:
        filtered_df: Filtered dataset 
        min_months_filter: Minimum months from filter settings
        api_col: API column name
        
    Returns:
        Dictionary with vector length recommendations
    """
    if filtered_df is None or len(filtered_df) == 0:
        return {
            'recommended_length': 12,
            'max_safe_length': 12,
            'data_completeness': {},
            'warnings': ['No data available for vector length analysis']
        }
    
    # Calculate months produced for each well
    try:
        months_produced = compute_months_produced(filtered_df, api_col)
        well_months = filtered_df.groupby(api_col)['API14'].first().to_frame()
        well_months['months_available'] = months_produced.groupby(filtered_df[api_col]).first()
    except Exception:
        # Fallback if months calculation fails
        well_months = pd.DataFrame({
            'months_available': [min_months_filter] * len(filtered_df[api_col].unique())
        })
    
    months_available = well_months['months_available'].values
    n_wells = len(months_available)
    
    # Calculate data completeness at different vector lengths
    lengths_to_test = [6, 12, 18, 24, 30, 36]
    completeness = {}
    
    for length in lengths_to_test:
        wells_with_complete_data = (months_available >= length).sum()
        completeness[length] = {
            'wells_count': wells_with_complete_data,
            'percentage': wells_with_complete_data / n_wells * 100 if n_wells > 0 else 0
        }
    
    # Determine recommendations
    warnings = []
    
    # Rule 1: Vector length should not exceed min_months_filter significantly
    if min_months_filter > 0:
        max_safe_from_filter = min_months_filter
        if min_months_filter < 6:
            warnings.append(f"Min months filter ({min_months_filter}) is very low - consider increasing to 6+ for better clustering")
    else:
        max_safe_from_filter = 24
    
    # Rule 2: At least 70% of wells should have complete data for the chosen length
    min_completeness = 70.0
    recommended_length = 6  # Conservative default
    
    for length in sorted(lengths_to_test):
        if completeness[length]['percentage'] >= min_completeness:
            recommended_length = length
        else:
            break
    
    # Rule 3: Don't exceed filter constraint
    recommended_length = min(recommended_length, max_safe_from_filter)
    
    # Rule 4: Ensure minimum useful length
    recommended_length = max(recommended_length, 6)
    
    # Generate warnings and suggestions
    if min_months_filter > 0 and recommended_length < min_months_filter:
        warnings.append(f"Recommended vector length ({recommended_length}) is less than min months filter ({min_months_filter})")
        warnings.append("Consider reducing min months filter or accepting shorter vectors")
    
    if completeness[recommended_length]['percentage'] < 90:
        pct = completeness[recommended_length]['percentage']
        warnings.append(f"Only {pct:.1f}% of wells have complete {recommended_length}-month data")
    
    # Calculate max safe length (50% completeness threshold)
    max_safe_length = 6
    for length in sorted(lengths_to_test):
        if completeness[length]['percentage'] >= 50.0:
            max_safe_length = length
        else:
            break
    
    return {
        'recommended_length': recommended_length,
        'max_safe_length': max_safe_length,
        'data_completeness': completeness,
        'warnings': warnings,
        'min_months_filter': min_months_filter,
        'wells_analyzed': n_wells,
        'length_options': _generate_length_options(completeness, min_months_filter)
    }


def _generate_length_options(completeness: Dict[int, Dict], min_months_filter: int) -> List[Dict[str, Any]]:
    """Generate user-friendly vector length options with explanations."""
    options = []
    
    for length, stats in completeness.items():
        if stats['percentage'] >= 30:  # Only show viable options
            quality = "Excellent" if stats['percentage'] >= 90 else \
                     "Good" if stats['percentage'] >= 70 else \
                     "Fair" if stats['percentage'] >= 50 else "Poor"
            
            # Check filter compatibility
            filter_compatible = length <= min_months_filter if min_months_filter > 0 else True
            
            recommendation = ""
            if length == 6:
                recommendation = "Conservative - good for new wells"
            elif length == 12:
                recommendation = "Balanced - captures early decline"  
            elif length == 24:
                recommendation = "Comprehensive - full decline curve"
            elif length >= 36:
                recommendation = "Extended - mature wells only"
            
            options.append({
                'length': length,
                'completeness_pct': stats['percentage'],
                'wells_count': stats['wells_count'],
                'quality': quality,
                'filter_compatible': filter_compatible,
                'recommendation': recommendation
            })
    
    return sorted(options, key=lambda x: x['length'])


def get_vector_length_suggestions(filtered_df: pd.DataFrame,
                                 filters_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get user-friendly suggestions for vector length based on data and filters.
    
    Args:
        filtered_df: Filtered dataset
        filters_cfg: Filter configuration dictionary
        
    Returns:
        Dictionary with suggestions and recommendations
    """
    min_months_filter = filters_cfg.get('min_months_produced', 0)
    
    vector_analysis = calculate_optimal_vector_length(filtered_df, min_months_filter)
    
    suggestions = []
    warnings = vector_analysis['warnings'].copy()
    
    recommended = vector_analysis['recommended_length']
    max_safe = vector_analysis['max_safe_length']
    
    # Generate user-friendly suggestions
    if recommended == 6:
        suggestions.append("6-month vectors recommended - good for datasets with many new wells")
    elif recommended == 12:
        suggestions.append("12-month vectors recommended - captures initial decline phase")
    elif recommended >= 24:
        suggestions.append(f"{recommended}-month vectors recommended - comprehensive decline analysis")
    
    if max_safe > recommended:
        suggestions.append(f"Up to {max_safe} months possible with acceptable data quality")
    
    # Filter-specific suggestions
    if min_months_filter > 0:
        if min_months_filter == recommended:
            suggestions.append("Vector length matches your minimum months filter - optimal setup")
        elif min_months_filter > recommended:
            suggestions.append(f"Consider reducing min months filter to {recommended} for better vector quality")
        else:
            suggestions.append(f"Min months filter ({min_months_filter}) allows longer vectors up to {recommended} months")
    
    return {
        'recommendations': suggestions,
        'warnings': warnings,
        'optimal_length': recommended,
        'max_safe_length': max_safe,
        'vector_analysis': vector_analysis
    }