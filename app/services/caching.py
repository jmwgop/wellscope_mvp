# app/services/caching.py

from __future__ import annotations
import hashlib
import copy
from typing import Dict, Any, Optional, Callable
import pandas as pd

from app.services.pipeline_driver import run_complete_pipeline

# Global cache for fallback when Streamlit unavailable
_fallback_cache: Dict[str, Dict[str, Any]] = {}

def _hash_dataframe(df: pd.DataFrame) -> str:
    """Generate stable hash for DataFrame cache key."""
    if df is None or len(df) == 0:
        return "empty_df"
    
    # Use pandas util hash for consistent results
    try:
        # Include shape and dtypes in hash for robustness
        content = f"{df.shape}_{df.dtypes.to_dict()}_{pd.util.hash_pandas_object(df).sum()}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        # Fallback: use string representation (less efficient but reliable)
        content = f"{df.shape}_{str(df.dtypes)}_{str(df.values.tobytes())}"
        return hashlib.md5(content.encode()).hexdigest()[:16]

def _generate_cache_key(headers_df: pd.DataFrame, monthly_df: pd.DataFrame, filters_cfg: Optional[Dict[str, Any]]) -> str:
    """Generate cache key from pipeline inputs."""
    headers_hash = _hash_dataframe(headers_df)
    monthly_hash = _hash_dataframe(monthly_df)
    filters_hash = str(hash(str(sorted((filters_cfg or {}).items()))))
    
    cache_key = f"pipeline_{headers_hash}_{monthly_hash}_{filters_hash}"
    return cache_key

def cached_run_pipeline(
    headers_df: pd.DataFrame, 
    monthly_df: pd.DataFrame, 
    filters_cfg: Optional[Dict[str, Any]] = None,
    force_fallback: bool = False
) -> Dict[str, Any]:
    """
    Cached version of run_complete_pipeline.
    
    Uses Streamlit's @st.cache_data if available, otherwise uses simple fallback cache.
    Cache key is based on DataFrame contents and filter configuration.
    
    Args:
        force_fallback: If True, skip Streamlit caching and use fallback (for testing)
    """
    # Use fallback cache if forced or Streamlit not available
    if force_fallback:
        use_streamlit = False
    else:
        try:
            import streamlit as st
            use_streamlit = True
        except ImportError:
            use_streamlit = False
    
    if use_streamlit:
        # Use Streamlit caching
        import streamlit as st
        
        @st.cache_data(show_spinner="Running ML pipeline...")
        def _cached_pipeline(headers_hash: str, monthly_hash: str, filters_str: str) -> Dict[str, Any]:
            # Note: This captures the DataFrames from the outer scope
            return run_complete_pipeline(headers_df, monthly_df, filters_cfg)
        
        # Generate cache-friendly inputs
        headers_hash = _hash_dataframe(headers_df)
        monthly_hash = _hash_dataframe(monthly_df)  
        filters_str = str(sorted((filters_cfg or {}).items()))
        
        return _cached_pipeline(headers_hash, monthly_hash, filters_str)
        
    else:
        # Use fallback cache
        cache_key = _generate_cache_key(headers_df, monthly_df, filters_cfg)
        
        if cache_key in _fallback_cache:
            return copy.deepcopy(_fallback_cache[cache_key])  # Return deep copy to prevent mutation
        
        # Cache miss - run pipeline and store result
        result = run_complete_pipeline(headers_df, monthly_df, filters_cfg)
        _fallback_cache[cache_key] = copy.deepcopy(result)  # Store deep copy
        
        return result

def clear_pipeline_cache() -> None:
    """Clear pipeline cache for development/debugging."""
    global _fallback_cache
    
    # Clear Streamlit cache if available
    try:
        import streamlit as st
        st.cache_data.clear()
    except ImportError:
        pass
    
    # Clear fallback cache
    _fallback_cache.clear()

def get_cache_info() -> Dict[str, Any]:
    """Get cache statistics for monitoring."""
    cache_info = {
        'fallback_cache_size': len(_fallback_cache),
        'fallback_cache_keys': list(_fallback_cache.keys()),
        'streamlit_available': False
    }
    
    try:
        import streamlit as st
        cache_info['streamlit_available'] = True
    except ImportError:
        pass
    
    return cache_info

# Convenience alias for the main caching function
run_cached_pipeline = cached_run_pipeline