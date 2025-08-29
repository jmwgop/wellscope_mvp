# app/state/session.py

from __future__ import annotations
from typing import Any, Optional, Dict, TypeVar
import pandas as pd

# Session state keys - typed for IDE support
class SessionKeys:
    HEADERS_DF = "headers_df"
    MONTHLY_DF = "monthly_df" 
    JOINED_DF = "joined_df"
    FILTERS_CFG = "filters_cfg"
    VECTORS_DF = "vectors_df"
    LABELS_DF = "labels_df"
    COORDS_DF = "coords_df"
    SCORES_DF = "scores_df"
    
    # Metadata
    HEADERS_META = "headers_meta"
    MONTHLY_META = "monthly_meta"
    JOIN_STATS = "join_stats"
    FILTER_STATS = "filter_stats"
    VECTOR_META = "vector_meta"
    CLUSTER_META = "cluster_meta"
    PROJECTION_META = "projection_meta"
    SIMILARITY_META = "similarity_meta"

T = TypeVar('T')

def get_session_value(session_dict: Dict[str, Any], key: str, default: Optional[T] = None) -> Optional[T]:
    """Get value from session dict with optional default."""
    return session_dict.get(key, default)

def set_session_value(session_dict: Dict[str, Any], key: str, value: Any) -> None:
    """Set value in session dict."""
    session_dict[key] = value

def clear_session_keys(session_dict: Dict[str, Any], keys_to_clear: list[str]) -> None:
    """Clear specified keys from session dict."""
    for key in keys_to_clear:
        if key in session_dict:
            del session_dict[key]

def has_required_data(session_dict: Dict[str, Any], required_keys: list[str]) -> bool:
    """Check if session contains all required keys with non-None values."""
    return all(
        key in session_dict and session_dict[key] is not None 
        for key in required_keys
    )

def get_dataframe(session_dict: Dict[str, Any], key: str) -> Optional[pd.DataFrame]:
    """Get DataFrame from session, return None if not present or not a DataFrame."""
    value = session_dict.get(key)
    return value if isinstance(value, pd.DataFrame) else None

def set_dataframe(session_dict: Dict[str, Any], key: str, df: pd.DataFrame) -> None:
    """Set DataFrame in session with validation."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected DataFrame, got {type(df)}")
    session_dict[key] = df

# Convenience functions for common session operations
def clear_pipeline_outputs(session_dict: Dict[str, Any]) -> None:
    """Clear all pipeline output data (preserves uploaded files)."""
    pipeline_keys = [
        SessionKeys.JOINED_DF, SessionKeys.VECTORS_DF, SessionKeys.LABELS_DF,
        SessionKeys.COORDS_DF, SessionKeys.SCORES_DF,
        SessionKeys.JOIN_STATS, SessionKeys.FILTER_STATS, SessionKeys.VECTOR_META,
        SessionKeys.CLUSTER_META, SessionKeys.PROJECTION_META, SessionKeys.SIMILARITY_META
    ]
    clear_session_keys(session_dict, pipeline_keys)

def has_uploaded_data(session_dict: Dict[str, Any]) -> bool:
    """Check if both CSV files have been uploaded and loaded."""
    return has_required_data(session_dict, [SessionKeys.HEADERS_DF, SessionKeys.MONTHLY_DF])

def has_analysis_results(session_dict: Dict[str, Any]) -> bool:
    """Check if pipeline analysis has been completed."""
    return has_required_data(session_dict, [
        SessionKeys.VECTORS_DF, SessionKeys.LABELS_DF, 
        SessionKeys.COORDS_DF, SessionKeys.SCORES_DF
    ])