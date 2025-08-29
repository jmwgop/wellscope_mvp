# app/components/upload_panel.py

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import pandas as pd
from io import BytesIO

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from app.services.io import load_headers_from_file, load_monthly_from_file, get_dataframe_summary
from app.utils.formatting import format_dataframe_summary, format_file_size
from app.utils.validation import validate_file_upload, format_validation_errors
from app.config.ui_defaults import DEFAULTS

def render_upload_panel() -> Dict[str, Any]:
    """
    Render file upload panel for CSV files.
    
    Returns:
        Dictionary containing upload results:
        {
            'headers_df': pd.DataFrame or None,
            'monthly_df': pd.DataFrame or None, 
            'headers_meta': dict or None,
            'monthly_meta': dict or None,
            'upload_status': str,
            'errors': list
        }
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_upload_panel()
    
    st.subheader("📁 Data Upload")
    
    # Initialize result structure
    result = {
        'headers_df': None,
        'monthly_df': None,
        'headers_meta': None,
        'monthly_meta': None,
        'upload_status': 'pending',
        'files_uploaded': False,
        'errors': []
    }
    
    col1, col2 = st.columns(2)
    
    # Headers file upload
    with col1:
        st.markdown("**Well Headers CSV**")
        headers_file = st.file_uploader(
            "Choose headers file",
            type=['csv'],
            key="headers_upload",
            help="CSV file containing well header information with API14 column"
        )
        
        if headers_file is not None:
            try:
                # Load and validate headers
                headers_df, headers_meta = load_headers_from_file(headers_file)
                result['headers_df'] = headers_df
                result['headers_meta'] = headers_meta
                
                # Display summary
                summary = format_dataframe_summary(headers_df)
                st.success(f"✅ Loaded {summary['rows']} wells, {summary['columns']} columns")
                
                # Show preview
                with st.expander("Preview Headers Data"):
                    st.dataframe(headers_df.head(10), use_container_width=True)
                
            except Exception as e:
                result['errors'].append(f"Headers upload error: {str(e)}")
                st.error(f"❌ Headers upload failed: {str(e)}")
    
    # Monthly data file upload
    with col2:
        st.markdown("**Monthly Production CSV**")
        monthly_file = st.file_uploader(
            "Choose monthly production file",
            type=['csv'],
            key="monthly_upload", 
            help="CSV file containing monthly production data with API/UWI column"
        )
        
        if monthly_file is not None:
            try:
                # Load and validate monthly data
                monthly_df, monthly_meta = load_monthly_from_file(monthly_file)
                result['monthly_df'] = monthly_df
                result['monthly_meta'] = monthly_meta
                
                # Display summary
                summary = format_dataframe_summary(monthly_df)
                st.success(f"✅ Loaded {summary['rows']} records, {summary['columns']} columns")
                
                # Show preview
                with st.expander("Preview Monthly Data"):
                    st.dataframe(monthly_df.head(10), use_container_width=True)
                
            except Exception as e:
                result['errors'].append(f"Monthly upload error: {str(e)}")
                st.error(f"❌ Monthly upload failed: {str(e)}")
    
    # Overall status
    if result['headers_df'] is not None and result['monthly_df'] is not None:
        result['upload_status'] = 'complete'
        result['files_uploaded'] = True  # This enables workflow progression to Step 2+
        
        # Show combined summary
        st.success("🎉 Both files uploaded successfully!")
        
        with st.expander("📊 Data Summary", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Wells", format_dataframe_summary(result['headers_df'])['rows'])
            
            with col2:
                total_records = len(result['monthly_df'])
                st.metric("Production Records", f"{total_records:,}")
            
            with col3:
                total_memory = (
                    result['headers_df'].memory_usage(deep=True).sum() + 
                    result['monthly_df'].memory_usage(deep=True).sum()
                )
                st.metric("Memory Usage", format_file_size(total_memory))
        
    elif result['headers_df'] is not None or result['monthly_df'] is not None:
        result['upload_status'] = 'partial'
        st.warning("⚠️ Please upload both files to proceed with analysis")
    
    else:
        result['upload_status'] = 'pending'
        st.info("📤 Upload both CSV files to get started")
    
    # Show errors if any
    if result['errors']:
        st.error("❌ Upload Errors:\n" + "\n".join(f"• {error}" for error in result['errors']))
    
    return result

def render_upload_instructions() -> None:
    """Render upload instructions and requirements."""
    if not STREAMLIT_AVAILABLE:
        return
    
    with st.expander("📋 Upload Instructions", expanded=False):
        st.markdown("""
        **Required Files:**
        
        1. **Well Headers CSV** - Must contain:
           - `API14` column (14-digit well identifier)
           - Well location and completion data
           - Formation and operator information
        
        2. **Monthly Production CSV** - Must contain:
           - `API/UWI` column (well identifier matching headers)
           - `Monthly Production Date` column
           - `Monthly Oil (bbl)` column 
           - Optional: Monthly Gas, Water production columns
        
        **File Requirements:**
        - Format: CSV files only
        - Size: Max 100MB per file
        - Encoding: UTF-8 recommended
        
        **Tips:**
        - Ensure API identifiers match between files
        - Remove any leading/trailing spaces in column names
        - Use consistent date formats (YYYY-MM-DD recommended)
        """)

def validate_uploaded_files(headers_file, monthly_file) -> Tuple[bool, List[str]]:
    """
    Validate uploaded files before processing.
    
    Returns:
        (is_valid, error_messages)
    """
    errors = []
    
    if headers_file is None:
        errors.append("Headers file is required")
    else:
        # Basic file validation for headers
        if hasattr(headers_file, 'size') and headers_file.size > DEFAULTS.max_file_size_mb * 1024 * 1024:
            errors.append(f"Headers file too large (max {DEFAULTS.max_file_size_mb}MB)")
    
    if monthly_file is None:
        errors.append("Monthly production file is required")
    else:
        # Basic file validation for monthly
        if hasattr(monthly_file, 'size') and monthly_file.size > DEFAULTS.max_file_size_mb * 1024 * 1024:
            errors.append(f"Monthly file too large (max {DEFAULTS.max_file_size_mb}MB)")
    
    return len(errors) == 0, errors

def _mock_upload_panel() -> Dict[str, Any]:
    """Mock upload panel for testing when Streamlit not available."""
    return {
        'headers_df': None,
        'monthly_df': None,
        'headers_meta': None,
        'monthly_meta': None,
        'upload_status': 'pending',
        'files_uploaded': False,
        'errors': ['Streamlit not available - using mock upload panel']
    }

def get_upload_progress(headers_df: Optional[pd.DataFrame], 
                       monthly_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    Get upload progress information for status tracking.
    
    Returns:
        Dictionary with progress metrics
    """
    progress = {
        'headers_uploaded': headers_df is not None,
        'monthly_uploaded': monthly_df is not None,
        'progress_pct': 0.0,
        'status_text': 'No files uploaded',
        'next_step': 'Upload well headers CSV file'
    }
    
    if headers_df is not None and monthly_df is not None:
        progress.update({
            'progress_pct': 1.0,
            'status_text': 'Both files uploaded successfully',
            'next_step': 'Configure filters and run analysis'
        })
    elif headers_df is not None:
        progress.update({
            'progress_pct': 0.5,
            'status_text': 'Headers uploaded',
            'next_step': 'Upload monthly production CSV file'
        })
    elif monthly_df is not None:
        progress.update({
            'progress_pct': 0.5,
            'status_text': 'Monthly data uploaded',
            'next_step': 'Upload well headers CSV file'
        })
    
    return progress