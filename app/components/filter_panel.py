# app/components/filter_panel.py

from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from app.utils.formatting import format_filter_summary, format_number
from app.utils.validation import validate_filter_config, format_validation_errors
from app.config.ui_defaults import DEFAULTS

def render_filter_panel(joined_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """
    Render filter configuration panel.
    
    Args:
        joined_df: Optional joined dataset to extract filter options from
        
    Returns:
        Dictionary containing filter configuration
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_filter_panel()
    
    st.subheader("🔍 Data Filters")
    
    # Initialize filter config with defaults
    config = {}
    
    # Formation filter
    formations_available = []
    if joined_df is not None and 'Target Formation' in joined_df.columns:
        formations_available = sorted(joined_df['Target Formation'].dropna().unique().tolist())
    
    if formations_available:
        st.markdown("**Formation**")
        selected_formations = st.multiselect(
            "Select formations to include",
            options=formations_available,
            default=None,
            key="filter_formations",
            help="Leave empty to include all formations"
        )
        if selected_formations:
            config['formations'] = selected_formations
    else:
        st.info("💡 Formation filter will be available after data upload")
    
    # Operator filter
    operators_available = []
    if joined_df is not None and 'Operator (Reported)' in joined_df.columns:
        operators_available = sorted(joined_df['Operator (Reported)'].dropna().unique().tolist())
    
    if operators_available and len(operators_available) > 1:
        st.markdown("**Operator**")
        selected_operators = st.multiselect(
            "Select operators to include",
            options=operators_available,
            default=None,
            key="filter_operators",
            help="Leave empty to include all operators"
        )
        if selected_operators:
            config['operators'] = selected_operators
    
    # Completion year range
    st.markdown("**Completion Year**")
    
    # Determine available year range from data
    min_year, max_year = 2010, 2024
    if joined_df is not None and 'Completion Date' in joined_df.columns:
        completion_dates = pd.to_datetime(joined_df['Completion Date'], errors='coerce')
        valid_years = completion_dates.dt.year.dropna()
        if len(valid_years) > 0:
            min_year = int(valid_years.min())
            max_year = int(valid_years.max())
    
    year_range = st.slider(
        "Completion year range",
        min_value=min_year,
        max_value=max_year,
        value=(max(min_year, DEFAULTS.default_completion_year_range[0] or min_year),
               min(max_year, DEFAULTS.default_completion_year_range[1] or max_year)),
        step=1,
        key="filter_year_range",
        help="Wells completed within this date range"
    )
    config['completion_year_range'] = year_range
    
    # Lateral length range
    st.markdown("**Lateral Length**")
    
    # Determine available lateral length range from data
    min_lateral, max_lateral = 0, 25000
    if joined_df is not None:
        lateral_cols = ['DI Lateral Length', 'Horizontal Length']
        available_lateral_cols = [col for col in lateral_cols if col in joined_df.columns]
        
        if available_lateral_cols:
            lateral_data = joined_df[available_lateral_cols[0]].dropna()
            if len(lateral_data) > 0:
                min_lateral = max(0, int(lateral_data.min()))
                max_lateral = int(lateral_data.max())
    
    col1, col2 = st.columns(2)
    with col1:
        min_lateral_input = st.number_input(
            "Minimum lateral length (ft)",
            min_value=0,
            max_value=50000,
            value=int(DEFAULTS.default_lateral_range[0] or min_lateral),
            step=500,
            key="filter_min_lateral",
            help="Minimum lateral length in feet"
        )
    
    with col2:
        max_lateral_input = st.number_input(
            "Maximum lateral length (ft)",
            min_value=0,
            max_value=50000,
            value=int(DEFAULTS.default_lateral_range[1] or max_lateral),
            step=500,
            key="filter_max_lateral",
            help="Maximum lateral length in feet"
        )
    
    config['lateral_ft_range'] = (min_lateral_input, max_lateral_input)
    
    # Minimum months produced
    st.markdown("**Production History**")
    min_months = st.slider(
        "Minimum months with production",
        min_value=0,
        max_value=120,
        value=DEFAULTS.default_min_months_produced,
        step=1,
        key="filter_min_months",
        help="Wells must have production data for at least this many months"
    )
    config['min_months_produced'] = min_months
    
    # Validate configuration
    validation_errors = validate_filter_config(config)
    
    # Show filter summary with live preview
    if joined_df is not None:
        # Calculate total unique wells
        total_wells = len(joined_df['API14'].unique()) if 'API14' in joined_df.columns else len(joined_df)
        
        # Apply current filters to estimate matching wells
        try:
            filtered_preview = _apply_basic_filters(joined_df, config)
            matching_wells = len(filtered_preview['API14'].unique()) if 'API14' in filtered_preview.columns else len(filtered_preview)
        except:
            # Fallback to total if filtering fails
            matching_wells = total_wells
        
        summary = format_filter_summary(config, matching_wells)
        
        with st.expander("📋 Filter Summary", expanded=True):
            # Show live count
            st.metric("Wells Matching Filters", f"{matching_wells:,}", f"{matching_wells/total_wells*100:.1f}% of {total_wells:,}")
            st.divider()
            for item in summary:
                st.write(f"• {item}")
        
        if validation_errors:
            st.error(format_validation_errors(validation_errors, "Filter Validation"))
    
    # Return just the config dict directly for FilterConfig()
    return config

def _apply_basic_filters(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply basic filters for live preview (simplified version)."""
    filtered_df = df.copy()
    
    # Formation filter
    if 'formations' in config and config['formations']:
        if 'Target Formation' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Target Formation'].isin(config['formations'])]
    
    # Completion year filter
    if 'completion_year_range' in config and config['completion_year_range']:
        if 'Completion Date' in filtered_df.columns:
            start_year, end_year = config['completion_year_range']
            completion_dates = pd.to_datetime(filtered_df['Completion Date'], errors='coerce')
            years = completion_dates.dt.year
            filtered_df = filtered_df[(years >= start_year) & (years <= end_year)]
    
    # Lateral length filter (simplified - just check first available column)
    if 'lateral_ft_range' in config and config['lateral_ft_range']:
        lateral_cols = ['DI Lateral Length', 'Horizontal Length']
        available_col = None
        for col in lateral_cols:
            if col in filtered_df.columns:
                available_col = col
                break
        
        if available_col:
            min_lateral, max_lateral = config['lateral_ft_range']
            lateral_data = pd.to_numeric(filtered_df[available_col], errors='coerce')
            filtered_df = filtered_df[(lateral_data >= min_lateral) & (lateral_data <= max_lateral)]
    
    return filtered_df

def render_advanced_filters(joined_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    """Render advanced filter options."""
    if not STREAMLIT_AVAILABLE:
        return {}
    
    with st.expander("⚙️ Advanced Filters", expanded=False):
        config = {}
        
        # Well status filter
        if joined_df is not None and 'Well Status' in joined_df.columns:
            status_options = sorted(joined_df['Well Status'].dropna().unique().tolist())
            if status_options:
                selected_statuses = st.multiselect(
                    "Well Status",
                    options=status_options,
                    default=None,
                    key="filter_well_status",
                    help="Filter by well operational status"
                )
                if selected_statuses:
                    config['well_status_in'] = selected_statuses
        
        # Subplay filter (if available)
        if joined_df is not None and 'DI Subplay' in joined_df.columns:
            subplay_options = sorted(joined_df['DI Subplay'].dropna().unique().tolist())
            if subplay_options:
                selected_subplays = st.multiselect(
                    "Subplay",
                    options=subplay_options,
                    default=None,
                    key="filter_subplay",
                    help="Filter by geological subplay"
                )
                if selected_subplays:
                    config['subplays'] = selected_subplays
        
        return config

def get_filter_options_from_data(df: pd.DataFrame) -> Dict[str, List[Any]]:
    """Extract available filter options from dataset."""
    options = {}
    
    # Formation options
    if 'Target Formation' in df.columns:
        options['formations'] = sorted(df['Target Formation'].dropna().unique().tolist())
    
    # Operator options
    if 'Operator (Reported)' in df.columns:
        options['operators'] = sorted(df['Operator (Reported)'].dropna().unique().tolist())
    
    # Well status options
    if 'Well Status' in df.columns:
        options['well_statuses'] = sorted(df['Well Status'].dropna().unique().tolist())
    
    # Subplay options
    if 'DI Subplay' in df.columns:
        options['subplays'] = sorted(df['DI Subplay'].dropna().unique().tolist())
    
    # Year range from completion dates
    if 'Completion Date' in df.columns:
        completion_dates = pd.to_datetime(df['Completion Date'], errors='coerce')
        valid_years = completion_dates.dt.year.dropna()
        if len(valid_years) > 0:
            options['year_range'] = (int(valid_years.min()), int(valid_years.max()))
    
    # Lateral length range
    lateral_cols = ['DI Lateral Length', 'Horizontal Length']
    available_lateral_cols = [col for col in lateral_cols if col in df.columns]
    if available_lateral_cols:
        lateral_data = df[available_lateral_cols[0]].dropna()
        if len(lateral_data) > 0:
            options['lateral_range'] = (float(lateral_data.min()), float(lateral_data.max()))
    
    return options

def apply_filters_preview(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Apply filters to dataframe and return preview with statistics.
    
    Returns:
        (filtered_df, filter_stats)
    """
    if df is None or len(df) == 0:
        return df, {'input_rows': 0, 'output_rows': 0}
    
    original_count = len(df)
    filtered_df = df.copy()
    stats = {'input_rows': original_count}
    
    # Formation filter
    if 'formations' in config and config['formations']:
        if 'Target Formation' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Target Formation'].isin(config['formations'])]
            stats['after_formation_filter'] = len(filtered_df)
    
    # Operator filter
    if 'operators' in config and config['operators']:
        if 'Operator (Reported)' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['Operator (Reported)'].isin(config['operators'])]
            stats['after_operator_filter'] = len(filtered_df)
    
    # Completion year filter
    if 'completion_year_range' in config and config['completion_year_range']:
        if 'Completion Date' in filtered_df.columns:
            start_year, end_year = config['completion_year_range']
            completion_dates = pd.to_datetime(filtered_df['Completion Date'], errors='coerce')
            years = completion_dates.dt.year
            year_mask = (years >= start_year) & (years <= end_year)
            filtered_df = filtered_df[year_mask]
            stats['after_year_filter'] = len(filtered_df)
    
    # Lateral length filter
    if 'lateral_ft_range' in config and config['lateral_ft_range']:
        lateral_cols = ['DI Lateral Length', 'Horizontal Length']
        available_lateral_col = None
        for col in lateral_cols:
            if col in filtered_df.columns:
                available_lateral_col = col
                break
        
        if available_lateral_col:
            min_ft, max_ft = config['lateral_ft_range']
            lateral_data = pd.to_numeric(filtered_df[available_lateral_col], errors='coerce')
            lateral_mask = (lateral_data >= min_ft) & (lateral_data <= max_ft)
            filtered_df = filtered_df[lateral_mask]
            stats['after_lateral_filter'] = len(filtered_df)
    
    stats['output_rows'] = len(filtered_df)
    stats['kept_fraction'] = stats['output_rows'] / stats['input_rows'] if stats['input_rows'] > 0 else 0
    
    return filtered_df, stats

def _mock_filter_panel() -> Dict[str, Any]:
    """Mock filter panel for testing when Streamlit not available."""
    return {
        'completion_year_range': (2018, 2024),
        'lateral_ft_range': (5000, 20000),
        'min_months_produced': 6
    }