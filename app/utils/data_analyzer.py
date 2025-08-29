# app/utils/data_analyzer.py

from __future__ import annotations
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

# Import production clustering for early detection
try:
    from wellscope_mvp.pipeline.production_clustering import detect_production_data, calculate_production_similarity_stats
    PRODUCTION_DETECTION_AVAILABLE = True
except ImportError:
    PRODUCTION_DETECTION_AVAILABLE = False


def analyze_filtered_data(filtered_df: pd.DataFrame, 
                         filters_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze filtered data to provide immediate insights for smart recommendations.
    
    This function detects data characteristics that inform user-friendly recommendations
    without requiring production vectors to be built first.
    
    Args:
        filtered_df: Filtered well data from Step 2
        filters_cfg: Filter configuration for context
        
    Returns:
        Dictionary with comprehensive data analysis for smart recommendations
    """
    
    analysis_results = {
        'n_wells': 0,
        'production_type': 'Unknown',
        'similarity_mean': 0.0,
        'confidence': 0.0,
        'data_quality': 'Unknown',
        'formation_diversity': 0,
        'completion_year_span': 0,
        'avg_lateral_length': 0.0,
        'maturity_distribution': {},
        'regional_spread': 0,
        'analysis_timestamp': pd.Timestamp.now().isoformat()
    }
    
    if filtered_df is None or len(filtered_df) == 0:
        analysis_results['confidence'] = 0.0
        analysis_results['data_quality'] = 'No Data'
        return analysis_results
    
    # Basic data characteristics
    n_wells = len(filtered_df['API14'].unique()) if 'API14' in filtered_df.columns else len(filtered_df)
    analysis_results['n_wells'] = n_wells
    
    # Production stream detection
    production_stream = _detect_primary_production_stream(filtered_df, filters_cfg)
    analysis_results['production_type'] = production_stream
    
    # Formation diversity analysis
    formation_analysis = _analyze_formation_diversity(filtered_df)
    analysis_results.update(formation_analysis)
    
    # Completion timeline analysis
    completion_analysis = _analyze_completion_timeline(filtered_df)
    analysis_results.update(completion_analysis)
    
    # Lateral length analysis
    lateral_analysis = _analyze_lateral_lengths(filtered_df)
    analysis_results.update(lateral_analysis)
    
    # Well maturity analysis
    maturity_analysis = _analyze_well_maturity(filtered_df, filters_cfg)
    analysis_results.update(maturity_analysis)
    
    # Geographic spread analysis  
    geographic_analysis = _analyze_geographic_spread(filtered_df)
    analysis_results.update(geographic_analysis)
    
    # Early production similarity estimation (without full vectors)
    similarity_analysis = _estimate_production_similarity(filtered_df, filters_cfg)
    analysis_results.update(similarity_analysis)
    
    # Overall data quality and confidence assessment
    quality_assessment = _assess_data_quality(analysis_results)
    analysis_results.update(quality_assessment)
    
    return analysis_results


def _detect_primary_production_stream(filtered_df: pd.DataFrame, 
                                     filters_cfg: Dict[str, Any]) -> str:
    """Detect primary production stream from data and filters."""
    
    # Check filter preferences first
    if 'production_stream_focus' in filters_cfg:
        return filters_cfg['production_stream_focus']
    
    # Look at formation names for hints
    if 'Target Formation' in filtered_df.columns:
        formations = filtered_df['Target Formation'].str.upper().fillna('')
        
        # Oil-heavy formations
        oil_indicators = formations.str.contains('EAGLEFORD|EAGLE FORD|BAKKEN|PERMIAN|NIOBRARA|AUSTIN|BUDA', na=False)
        if oil_indicators.any():
            oil_pct = oil_indicators.sum() / len(formations)
            if oil_pct > 0.7:
                return 'Oil'
        
        # Gas-heavy formations  
        gas_indicators = formations.str.contains('MARCELLUS|UTICA|HAYNESVILLE|FAYETTEVILLE|BARNETT', na=False)
        if gas_indicators.any():
            gas_pct = gas_indicators.sum() / len(formations)
            if gas_pct > 0.7:
                return 'Gas'
    
    # Default based on dataset size and other factors
    if len(filtered_df) > 200:
        return 'Oil'  # Larger datasets often oil-focused
    else:
        return 'Mixed'


def _analyze_formation_diversity(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze formation diversity in the dataset."""
    
    if 'Target Formation' not in filtered_df.columns:
        return {
            'formation_diversity': 1,
            'formations_count': 1,
            'primary_formation': 'Unknown'
        }
    
    formations = filtered_df['Target Formation'].fillna('Unknown')
    formation_counts = formations.value_counts()
    
    diversity_score = len(formation_counts) / max(1, len(formations)) * 10  # Scale to 0-10
    
    return {
        'formation_diversity': min(10, diversity_score),
        'formations_count': len(formation_counts),
        'primary_formation': formation_counts.index[0] if len(formation_counts) > 0 else 'Unknown',
        'formation_distribution': formation_counts.to_dict()
    }


def _analyze_completion_timeline(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze completion date timeline and technology evolution."""
    
    if 'Completion Date' not in filtered_df.columns:
        return {
            'completion_year_span': 1,
            'avg_completion_year': 2020,
            'technology_diversity': 0.5
        }
    
    completion_dates = pd.to_datetime(filtered_df['Completion Date'], errors='coerce').dropna()
    
    if len(completion_dates) == 0:
        return {
            'completion_year_span': 1,
            'avg_completion_year': 2020,
            'technology_diversity': 0.5
        }
    
    years = completion_dates.dt.year
    year_span = years.max() - years.min() + 1
    avg_year = int(years.mean())
    
    # Technology diversity proxy (more years = more diverse completion techniques)
    tech_diversity = min(1.0, year_span / 10.0)  # Scale to 0-1
    
    return {
        'completion_year_span': year_span,
        'avg_completion_year': avg_year,
        'technology_diversity': tech_diversity,
        'earliest_completion': int(years.min()),
        'latest_completion': int(years.max())
    }


def _analyze_lateral_lengths(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze lateral length distribution."""
    
    # Try common lateral length column names
    lateral_cols = ['DI Lateral Length', 'Horizontal Length', 'Lateral Length']
    lateral_col = None
    
    for col in lateral_cols:
        if col in filtered_df.columns:
            lateral_col = col
            break
    
    if lateral_col is None:
        return {
            'avg_lateral_length': 10000.0,  # Reasonable default
            'lateral_diversity': 0.5
        }
    
    lateral_data = pd.to_numeric(filtered_df[lateral_col], errors='coerce').dropna()
    
    if len(lateral_data) == 0:
        return {
            'avg_lateral_length': 10000.0,
            'lateral_diversity': 0.5
        }
    
    avg_lateral = float(lateral_data.mean())
    
    # Diversity based on coefficient of variation
    if avg_lateral > 0:
        lateral_cv = lateral_data.std() / avg_lateral
        diversity = min(1.0, lateral_cv)  # Scale to 0-1
    else:
        diversity = 0.5
    
    return {
        'avg_lateral_length': avg_lateral,
        'lateral_diversity': diversity,
        'lateral_std': float(lateral_data.std()),
        'lateral_min': float(lateral_data.min()),
        'lateral_max': float(lateral_data.max())
    }


def _analyze_well_maturity(filtered_df: pd.DataFrame, 
                          filters_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze well maturity distribution for optimal vector lengths."""
    
    from wellscope_mvp.pipeline.filter_inputs import compute_months_produced
    
    try:
        api_col = 'API14' if 'API14' in filtered_df.columns else filtered_df.columns[0]
        months_produced = compute_months_produced(filtered_df, api_col)
        
        if len(months_produced) == 0:
            return {'maturity_distribution': {'6-12': 0.5, '12-24': 0.3, '24+': 0.2}}
        
        # Categorize maturity
        maturity_buckets = {
            '0-6': (months_produced < 6).sum(),
            '6-12': ((months_produced >= 6) & (months_produced < 12)).sum(),
            '12-18': ((months_produced >= 12) & (months_produced < 18)).sum(),
            '18-24': ((months_produced >= 18) & (months_produced < 24)).sum(),
            '24-36': ((months_produced >= 24) & (months_produced < 36)).sum(),
            '36+': (months_produced >= 36).sum()
        }
        
        total_wells = len(months_produced)
        maturity_distribution = {k: v/total_wells for k, v in maturity_buckets.items()}
        
        # Calculate optimal vector length based on distribution
        if maturity_distribution['24+'] > 0.7:
            optimal_months = 24
        elif maturity_distribution['18-24'] + maturity_distribution['24+'] > 0.6:
            optimal_months = 18
        elif maturity_distribution['12-18'] + maturity_distribution['18-24'] + maturity_distribution['24+'] > 0.7:
            optimal_months = 12
        else:
            optimal_months = 6
        
        return {
            'maturity_distribution': maturity_distribution,
            'optimal_vector_length': optimal_months,
            'avg_months_produced': float(months_produced.mean()),
            'mature_well_fraction': maturity_distribution.get('24+', 0.0)
        }
        
    except Exception:
        # Fallback if months calculation fails
        return {
            'maturity_distribution': {'6-12': 0.4, '12-24': 0.4, '24+': 0.2},
            'optimal_vector_length': 12,
            'avg_months_produced': 15.0,
            'mature_well_fraction': 0.2
        }


def _analyze_geographic_spread(filtered_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze geographic spread of wells."""
    
    lat_col = None
    lon_col = None
    
    # Find latitude/longitude columns
    for col in filtered_df.columns:
        if 'lat' in col.lower():
            lat_col = col
        elif 'lon' in col.lower() or 'lng' in col.lower():
            lon_col = col
    
    if lat_col is None or lon_col is None:
        return {'regional_spread': 0.5}  # Medium spread assumption
    
    try:
        lats = pd.to_numeric(filtered_df[lat_col], errors='coerce').dropna()
        lons = pd.to_numeric(filtered_df[lon_col], errors='coerce').dropna()
        
        if len(lats) == 0 or len(lons) == 0:
            return {'regional_spread': 0.5}
        
        # Calculate geographic spread (rough approximation)
        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()
        
        # Convert to approximate distance spread (very rough)
        spread_score = min(1.0, (lat_range + lon_range) / 2.0)  # Scale to 0-1
        
        return {
            'regional_spread': spread_score,
            'lat_range': float(lat_range),
            'lon_range': float(lon_range)
        }
        
    except Exception:
        return {'regional_spread': 0.5}


def _estimate_production_similarity(filtered_df: pd.DataFrame, 
                                   filters_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Estimate production similarity without building full vectors."""
    
    # Default estimates based on data characteristics
    base_similarity = 0.6  # Conservative default
    
    # Adjust based on formation homogeneity
    if 'Target Formation' in filtered_df.columns:
        formations = filtered_df['Target Formation'].fillna('Mixed')
        formation_counts = formations.value_counts()
        
        if len(formation_counts) == 1:
            # Single formation - higher similarity expected
            base_similarity = 0.85
        elif len(formation_counts) <= 3 and formation_counts.iloc[0] / len(formations) > 0.7:
            # Dominant formation - moderate-high similarity
            base_similarity = 0.75
        else:
            # Mixed formations - moderate similarity
            base_similarity = 0.65
    
    # Adjust based on completion timeline (get from filtered_df since we don't have full analysis here)
    if 'Completion Date' in filtered_df.columns:
        completion_dates = pd.to_datetime(filtered_df['Completion Date'], errors='coerce').dropna()
        if len(completion_dates) > 0:
            years = completion_dates.dt.year
            completion_span = years.max() - years.min() + 1
        else:
            completion_span = 5  # Default
    else:
        completion_span = 5  # Default
    if completion_span <= 2:
        # Similar vintage wells - higher similarity
        base_similarity += 0.05
    elif completion_span > 8:
        # Wide vintage range - lower similarity  
        base_similarity -= 0.1
    
    # Adjust based on geographic spread (calculate inline since we don't have full analysis here)
    regional_spread = _calculate_simple_geographic_spread(filtered_df)
    if regional_spread < 0.3:
        # Tight geographic area - higher similarity
        base_similarity += 0.05
    elif regional_spread > 0.7:
        # Wide geographic spread - lower similarity
        base_similarity -= 0.05
    
    # Clamp to reasonable range
    similarity_estimate = max(0.3, min(0.95, base_similarity))
    
    return {
        'similarity_mean': similarity_estimate,
        'similarity_confidence': 0.7,  # Moderate confidence in estimate
        'similarity_source': 'estimated_from_metadata'
    }


def _assess_data_quality(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Assess overall data quality and analysis confidence."""
    
    n_wells = analysis_results['n_wells']
    
    # Size-based confidence
    if n_wells < 20:
        size_confidence = 0.4
        quality_level = 'Limited'
    elif n_wells < 100:
        size_confidence = 0.7
        quality_level = 'Good'
    elif n_wells < 500:
        size_confidence = 0.9
        quality_level = 'Excellent'
    else:
        size_confidence = 0.95
        quality_level = 'Excellent'
    
    # Formation diversity confidence
    formation_diversity = analysis_results.get('formation_diversity', 5)
    if formation_diversity < 2:
        diversity_confidence = 0.9  # Homogeneous is good for clustering
    elif formation_diversity < 5:
        diversity_confidence = 0.8  # Moderate diversity is fine
    else:
        diversity_confidence = 0.6  # High diversity is challenging
    
    # Maturity confidence
    mature_fraction = analysis_results.get('mature_well_fraction', 0.2)
    if mature_fraction > 0.5:
        maturity_confidence = 0.9  # Lots of mature wells is good
    elif mature_fraction > 0.2:
        maturity_confidence = 0.7  # Some mature wells is okay
    else:
        maturity_confidence = 0.5  # Few mature wells is challenging
    
    # Overall confidence
    overall_confidence = (size_confidence + diversity_confidence + maturity_confidence) / 3
    
    return {
        'confidence': overall_confidence,
        'data_quality': quality_level,
        'size_confidence': size_confidence,
        'diversity_confidence': diversity_confidence,
        'maturity_confidence': maturity_confidence
    }


def _calculate_simple_geographic_spread(filtered_df: pd.DataFrame) -> float:
    """Calculate simple geographic spread metric."""
    lat_col = None
    lon_col = None
    
    # Find latitude/longitude columns
    for col in filtered_df.columns:
        if 'lat' in col.lower():
            lat_col = col
        elif 'lon' in col.lower() or 'lng' in col.lower():
            lon_col = col
    
    if lat_col is None or lon_col is None:
        return 0.5  # Medium spread assumption
    
    try:
        lats = pd.to_numeric(filtered_df[lat_col], errors='coerce').dropna()
        lons = pd.to_numeric(filtered_df[lon_col], errors='coerce').dropna()
        
        if len(lats) == 0 or len(lons) == 0:
            return 0.5
        
        # Calculate geographic spread (rough approximation)
        lat_range = lats.max() - lats.min()
        lon_range = lons.max() - lons.min()
        
        # Convert to approximate distance spread (very rough)
        spread_score = min(1.0, (lat_range + lon_range) / 2.0)  # Scale to 0-1
        
        return spread_score
        
    except Exception:
        return 0.5


def estimate_processing_time(n_wells: int, vector_months: int) -> str:
    """Estimate processing time for the analysis."""
    
    # Rough processing time estimates
    if n_wells < 100:
        base_time = 30  # seconds
    elif n_wells < 500:
        base_time = 90
    elif n_wells < 1000:
        base_time = 180
    else:
        base_time = 300
    
    # Adjust for vector complexity
    if vector_months > 24:
        base_time *= 1.2
    elif vector_months > 18:
        base_time *= 1.1
    
    # Convert to user-friendly format
    if base_time < 60:
        return f"{int(base_time)} seconds"
    elif base_time < 300:
        return f"{int(base_time/60)} minutes"
    else:
        return f"{int(base_time/60)}-{int(base_time/60)+1} minutes"