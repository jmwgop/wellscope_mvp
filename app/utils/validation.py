# app/utils/validation.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
from pathlib import Path

from wellscope_mvp.pipeline.filter_inputs import FilterConfig
from wellscope_mvp.pipeline.vector_builder import VectorConfig
from wellscope_mvp.pipeline.cluster_runner import ClusterConfig
from wellscope_mvp.pipeline.umap_projector import ProjectionConfig

def validate_filter_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate filter configuration and return list of error messages.
    
    Returns empty list if all validations pass.
    """
    errors = []
    
    # Completion year range validation
    if 'completion_year_range' in config and config['completion_year_range']:
        start, end = config['completion_year_range']
        if start is not None and end is not None:
            if start > end:
                errors.append("Completion year start cannot be after end year")
            if start < 1900:
                errors.append("Completion year start seems too early (before 1900)")
            if end > 2030:
                errors.append("Completion year end seems too far in future (after 2030)")
        elif start is not None and start < 1900:
            errors.append("Completion year start seems too early (before 1900)")
        elif end is not None and end > 2030:
            errors.append("Completion year end seems too far in future (after 2030)")
    
    # Lateral length range validation
    if 'lateral_ft_range' in config and config['lateral_ft_range']:
        min_ft, max_ft = config['lateral_ft_range']
        if min_ft is not None and max_ft is not None:
            if min_ft > max_ft:
                errors.append("Minimum lateral length cannot be greater than maximum")
            if min_ft < 0:
                errors.append("Lateral length cannot be negative")
            if max_ft > 50000:
                errors.append("Maximum lateral length seems unusually high (>50,000 ft)")
        elif min_ft is not None and min_ft < 0:
            errors.append("Minimum lateral length cannot be negative")
        elif max_ft is not None and max_ft > 50000:
            errors.append("Maximum lateral length seems unusually high (>50,000 ft)")
    
    # Minimum months produced validation
    if 'min_months_produced' in config:
        months = config['min_months_produced']
        if months is not None:
            if months < 0:
                errors.append("Minimum months produced cannot be negative")
            elif months > 120:
                errors.append("Minimum months produced seems unusually high (>10 years)")
    
    # Formation validation (basic check)
    if 'formations' in config and config['formations']:
        formations = config['formations']
        if not isinstance(formations, (list, tuple)):
            errors.append("Formations must be a list or tuple")
        elif len(formations) == 0:
            errors.append("At least one formation must be selected")
    
    # Operator validation (basic check)  
    if 'operators' in config and config['operators']:
        operators = config['operators']
        if not isinstance(operators, (list, tuple)):
            errors.append("Operators must be a list or tuple")
        elif len(operators) == 0:
            errors.append("At least one operator must be selected")
    
    return errors

def validate_vector_config(config: Dict[str, Any]) -> List[str]:
    """Validate vector configuration parameters."""
    errors = []
    
    # Months validation
    if 'months' in config:
        months = config['months']
        if not isinstance(months, int) or months < 1:
            errors.append("Vector months must be a positive integer")
        elif months > 120:
            errors.append("Vector months seems unusually high (>10 years)")
        elif months < 6:
            errors.append("Vector months should be at least 6 for meaningful analysis")
    
    # Stream validation
    if 'stream' in config:
        stream = config['stream']
        valid_streams = ['oil', 'gas', 'water', 'boe']
        if stream not in valid_streams:
            errors.append(f"Stream must be one of: {', '.join(valid_streams)}")
    
    # Normalization validation
    if 'normalize' in config:
        normalize = config['normalize']
        valid_modes = ['q_over_qmax', 'pct_decline']
        if normalize not in valid_modes:
            errors.append(f"Normalization must be one of: {', '.join(valid_modes)}")
    
    # BOE gas factor validation
    if 'boe_gas_factor' in config:
        factor = config['boe_gas_factor']
        if not isinstance(factor, (int, float)) or factor <= 0:
            errors.append("BOE gas factor must be a positive number")
        elif factor < 1 or factor > 10:
            errors.append("BOE gas factor should typically be between 1-10 (common: 6.0)")
    
    return errors

def validate_cluster_config(config: Any) -> List[str]:
    """
    Validate clustering configuration parameters.
    
    Args:
        config: Either a ClusterConfig object or a dictionary with config parameters
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    # Handle both ClusterConfig objects and dictionaries
    if hasattr(config, '__dict__'):
        # ClusterConfig object
        config_dict = {
            'min_cluster_size': config.min_cluster_size,
            'min_samples': config.min_samples,
            'metric': config.metric,
            'eps': config.eps if hasattr(config, 'eps') else 0.5,
            'use_hdbscan': config.use_hdbscan
        }
    elif isinstance(config, dict):
        # Dictionary format
        config_dict = config
    else:
        errors.append("Config must be a ClusterConfig object or dictionary")
        return errors
    
    # Min cluster size validation
    if 'min_cluster_size' in config_dict:
        size = config_dict['min_cluster_size']
        if not isinstance(size, int) or size < 2:
            errors.append("Minimum cluster size must be at least 2")
        elif size > 1000:
            errors.append("Minimum cluster size seems unusually high (>1000)")
    
    # DBSCAN eps validation
    if 'eps' in config_dict:
        eps = config_dict['eps']
        if not isinstance(eps, (int, float)) or eps <= 0:
            errors.append("DBSCAN eps must be a positive number")
        elif eps > 10:
            errors.append("DBSCAN eps seems unusually high (>10)")
    
    # Min samples validation
    if 'min_samples' in config_dict and config_dict['min_samples'] is not None:
        samples = config_dict['min_samples']
        if not isinstance(samples, int) or samples < 1:
            errors.append("Minimum samples must be a positive integer")
        elif samples > 100:
            errors.append("Minimum samples seems unusually high (>100)")
    
    # Metric validation
    if 'metric' in config_dict:
        metric = config_dict['metric']
        valid_metrics = ['euclidean', 'manhattan', 'cosine', 'hamming']
        if metric not in valid_metrics:
            errors.append(f"Metric must be one of: {', '.join(valid_metrics)}")
    
    return errors

def validate_projection_config(config: Dict[str, Any]) -> List[str]:
    """Validate UMAP projection configuration parameters."""
    errors = []
    
    # N components validation
    if 'n_components' in config:
        n_comp = config['n_components']
        if not isinstance(n_comp, int) or n_comp < 2:
            errors.append("Number of components must be at least 2")
        elif n_comp > 10:
            errors.append("Number of components seems unusually high (>10)")
    
    # N neighbors validation
    if 'n_neighbors' in config:
        n_neigh = config['n_neighbors']
        if not isinstance(n_neigh, int) or n_neigh < 2:
            errors.append("Number of neighbors must be at least 2")
        elif n_neigh > 200:
            errors.append("Number of neighbors seems unusually high (>200)")
    
    # Min distance validation
    if 'min_dist' in config:
        min_dist = config['min_dist']
        if not isinstance(min_dist, (int, float)) or min_dist < 0:
            errors.append("Minimum distance must be non-negative")
        elif min_dist > 1:
            errors.append("Minimum distance should typically be ≤ 1.0")
    
    # Random state validation
    if 'random_state' in config and config['random_state'] is not None:
        rs = config['random_state']
        if not isinstance(rs, int) or rs < 0:
            errors.append("Random state must be a non-negative integer")
    
    return errors

def validate_file_upload(file_path: Union[str, Path], max_size_mb: float = 100) -> List[str]:
    """Validate uploaded file properties."""
    errors = []
    
    try:
        path = Path(file_path)
        
        # Check if file exists
        if not path.exists():
            errors.append("File does not exist")
            return errors
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            errors.append(f"File size ({size_mb:.1f} MB) exceeds limit ({max_size_mb} MB)")
        
        # Check file extension
        if path.suffix.lower() not in ['.csv', '.tsv']:
            errors.append("File must be a CSV or TSV file")
        
        # Basic readability test
        try:
            pd.read_csv(path, nrows=5)
        except Exception as e:
            errors.append(f"Cannot read file as CSV: {str(e)}")
    
    except Exception as e:
        errors.append(f"File validation error: {str(e)}")
    
    return errors

def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str], 
                                df_name: str = "DataFrame") -> List[str]:
    """Validate DataFrame has required structure."""
    errors = []
    
    if df is None:
        errors.append(f"{df_name} is None")
        return errors
    
    if not isinstance(df, pd.DataFrame):
        errors.append(f"{df_name} is not a DataFrame")
        return errors
    
    if len(df) == 0:
        errors.append(f"{df_name} is empty")
    
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"{df_name} missing required columns: {', '.join(missing_columns)}")
    
    return errors

def validate_pipeline_inputs(headers_df: Optional[pd.DataFrame], 
                           monthly_df: Optional[pd.DataFrame]) -> List[str]:
    """Validate inputs are ready for pipeline execution."""
    errors = []
    
    # Headers DataFrame validation
    headers_errors = validate_dataframe_structure(
        headers_df, 
        ['API14'], 
        "Headers DataFrame"
    )
    errors.extend(headers_errors)
    
    # Monthly DataFrame validation  
    monthly_errors = validate_dataframe_structure(
        monthly_df,
        ['API/UWI', 'Monthly Production Date', 'Monthly Oil'],
        "Monthly DataFrame"
    )
    errors.extend(monthly_errors)
    
    # Cross-validation if both DataFrames are valid
    if not headers_errors and not monthly_errors:
        # Check for overlapping APIs
        if headers_df is not None and monthly_df is not None:
            headers_apis = set(headers_df['API14'].astype(str))
            monthly_apis = set(monthly_df['API/UWI'].astype(str))
            
            if len(headers_apis.intersection(monthly_apis)) == 0:
                errors.append("No matching APIs found between headers and monthly data")
    
    return errors

def validate_session_state(session: Dict[str, Any], required_keys: List[str]) -> List[str]:
    """Validate session state has required data."""
    errors = []
    
    for key in required_keys:
        if key not in session:
            errors.append(f"Session missing required key: {key}")
        elif session[key] is None:
            errors.append(f"Session key '{key}' is None")
        elif isinstance(session[key], pd.DataFrame) and len(session[key]) == 0:
            errors.append(f"Session key '{key}' contains empty DataFrame")
    
    return errors

def format_validation_errors(errors: List[str], title: str = "Validation Errors") -> str:
    """Format validation errors for user display."""
    if not errors:
        return ""
    
    if len(errors) == 1:
        return f"**{title}:** {errors[0]}"
    
    formatted = f"**{title}:**\n"
    for i, error in enumerate(errors, 1):
        formatted += f"{i}. {error}\n"
    
    return formatted.strip()

def get_validation_summary(all_errors: Dict[str, List[str]]) -> Tuple[bool, str]:
    """
    Get overall validation summary from multiple validation results.
    
    Returns:
        (is_valid, formatted_message)
    """
    total_errors = sum(len(errors) for errors in all_errors.values())
    
    if total_errors == 0:
        return True, "✅ All validations passed"
    
    summary = f"❌ Found {total_errors} validation error{'s' if total_errors != 1 else ''}\n\n"
    
    for section, errors in all_errors.items():
        if errors:
            summary += format_validation_errors(errors, section) + "\n\n"
    
    return False, summary.strip()

def validate_uploaded_data(headers_df: Optional[pd.DataFrame], monthly_df: Optional[pd.DataFrame]) -> List[str]:
    """
    Validate uploaded data files are suitable for analysis.
    
    Args:
        headers_df: Well headers DataFrame
        monthly_df: Monthly production DataFrame
        
    Returns:
        List of validation error messages (empty if valid)
    """
    return validate_pipeline_inputs(headers_df, monthly_df)