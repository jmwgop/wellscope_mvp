# app/utils/guards.py

from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

from app.state.session import SessionKeys

def ensure_uploaded_data(session: Dict[str, Any]) -> None:
    """
    Ensure both headers and monthly data are loaded in session.
    
    Raises:
        ValueError: If required data is missing
    """
    if SessionKeys.HEADERS_DF not in session or session[SessionKeys.HEADERS_DF] is None:
        raise ValueError("Headers data not uploaded. Please upload Well Headers CSV first.")
    
    if SessionKeys.MONTHLY_DF not in session or session[SessionKeys.MONTHLY_DF] is None:
        raise ValueError("Monthly data not uploaded. Please upload Monthly Production CSV first.")
    
    # Validate they're DataFrames
    headers_df = session[SessionKeys.HEADERS_DF]
    monthly_df = session[SessionKeys.MONTHLY_DF]
    
    if not isinstance(headers_df, pd.DataFrame) or len(headers_df) == 0:
        raise ValueError("Headers data is empty or invalid.")
    
    if not isinstance(monthly_df, pd.DataFrame) or len(monthly_df) == 0:
        raise ValueError("Monthly data is empty or invalid.")

def ensure_filtered_data(session: Dict[str, Any]) -> None:
    """
    Ensure filtered and joined data exists.
    
    Raises:
        ValueError: If filtered data is missing
    """
    ensure_uploaded_data(session)
    
    if SessionKeys.JOINED_DF not in session or session[SessionKeys.JOINED_DF] is None:
        raise ValueError("Data not joined yet. Please run analysis first.")
    
    joined_df = session[SessionKeys.JOINED_DF]
    if not isinstance(joined_df, pd.DataFrame) or len(joined_df) == 0:
        raise ValueError("No wells remain after filtering. Please adjust filter criteria.")

def ensure_analysis_complete(session: Dict[str, Any]) -> None:
    """
    Ensure complete pipeline analysis has been run.
    
    Raises:
        ValueError: If analysis is incomplete
    """
    ensure_filtered_data(session)
    
    required_keys = [
        SessionKeys.VECTORS_DF,
        SessionKeys.LABELS_DF, 
        SessionKeys.COORDS_DF,
        SessionKeys.SCORES_DF
    ]
    
    missing_keys = []
    for key in required_keys:
        if key not in session or session[key] is None:
            missing_keys.append(key)
        elif isinstance(session[key], pd.DataFrame) and len(session[key]) == 0:
            missing_keys.append(f"{key} (empty)")
    
    if missing_keys:
        raise ValueError(f"Analysis incomplete. Missing: {', '.join(missing_keys)}")

def validate_filter_config(filters: Dict[str, Any]) -> List[str]:
    """
    Validate filter configuration and return warnings.
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    if 'completion_year_range' in filters and filters['completion_year_range']:
        start, end = filters['completion_year_range']
        if start and end and start > end:
            warnings.append("Completion year range: start year is after end year")
        if start and start > 2030:
            warnings.append("Completion year range: start year seems too high")
        if end and end < 1990:
            warnings.append("Completion year range: end year seems too low")
    
    if 'lateral_ft_range' in filters and filters['lateral_ft_range']:
        min_ft, max_ft = filters['lateral_ft_range']
        if min_ft and max_ft and min_ft > max_ft:
            warnings.append("Lateral length range: minimum is greater than maximum")
        if min_ft and min_ft < 0:
            warnings.append("Lateral length range: minimum cannot be negative")
        if max_ft and max_ft > 50000:
            warnings.append("Lateral length range: maximum seems unusually high")
    
    if 'min_months_produced' in filters:
        months = filters['min_months_produced']
        if months and months < 0:
            warnings.append("Minimum months produced cannot be negative")
        if months and months > 120:
            warnings.append("Minimum months produced seems unusually high (>10 years)")
    
    return warnings