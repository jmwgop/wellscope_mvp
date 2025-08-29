# app/components/session_manager.py

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime
import pandas as pd

try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

from app.services.session_persistence import get_session_manager, get_auto_save_manager
from app.state.session import get_session_value, set_session_value


def render_session_recovery_panel() -> Dict[str, Any]:
    """
    Render session recovery panel with options to restore previous sessions.
    
    Returns:
        Dictionary with recovery status and loaded session data
    """
    if not STREAMLIT_AVAILABLE:
        return _mock_session_recovery()
    
    session_manager = get_session_manager()
    sessions = session_manager.list_sessions(limit=10)
    
    if not sessions:
        st.info("💾 No previous sessions found")
        return {'recovery_attempted': False, 'session_data': None}
    
    st.subheader("📂 Session Recovery")
    st.markdown("Found previous analysis sessions. Choose one to restore:")
    
    # Show session list
    session_options = {}
    for session_meta in sessions:
        session_id = session_meta['session_id']
        created_at = datetime.fromisoformat(session_meta['created_at'])
        
        # Build description
        status_parts = []
        if session_meta.get('has_uploaded_data'):
            status_parts.append("📁 Data")
        if session_meta.get('has_analysis_results'):
            status_parts.append("🎯 Analysis")
        
        status_str = " + ".join(status_parts) if status_parts else "⚠️ Incomplete"
        size_mb = session_meta.get('size_mb', 0)
        
        display_name = f"{created_at.strftime('%Y-%m-%d %H:%M')} - {status_str} ({size_mb:.1f}MB)"
        session_options[display_name] = session_id
    
    # Add option for no recovery
    session_options["🆕 Start fresh (don't restore)"] = None
    
    # Session selection
    selected_display = st.selectbox(
        "Select session to restore:",
        options=list(session_options.keys()),
        index=0,  # Default to most recent
        key="session_recovery_selection"
    )
    
    selected_session_id = session_options[selected_display]
    
    # Recovery action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Restore Selected Session", type="primary", disabled=(selected_session_id is None)):
            if selected_session_id:
                return _perform_session_recovery(selected_session_id)
    
    with col2:
        if st.button("🗑️ Delete Selected Session", disabled=(selected_session_id is None)):
            if selected_session_id and session_manager.delete_session(selected_session_id):
                st.success("✅ Session deleted")
                st.rerun()
    
    return {'recovery_attempted': False, 'session_data': None}


def _perform_session_recovery(session_id: str) -> Dict[str, Any]:
    """Perform the actual session recovery."""
    if not STREAMLIT_AVAILABLE:
        return {'recovery_attempted': False, 'session_data': None}
    
    try:
        session_manager = get_session_manager()
        session_data = session_manager.load_session(session_id)
        
        if session_data is None:
            st.error("❌ Failed to load session data")
            return {'recovery_attempted': True, 'session_data': None}
        
        # Remove metadata from session data (it's internal)
        session_data.pop('_session_metadata', None)
        
        # Update current session state
        current_session = st.session_state
        
        # Clear existing session data
        keys_to_clear = [k for k in current_session.keys() if not k.startswith('_')]
        for key in keys_to_clear:
            del current_session[key]
        
        # Load recovered data
        current_session.update(session_data)
        
        # Set up auto-save manager with recovered session
        auto_save_manager = get_auto_save_manager()
        auto_save_manager.set_session_id(session_id)
        
        st.success(f"✅ Successfully restored session from {session_id}")
        st.rerun()  # Refresh the page to show restored data
        
        return {'recovery_attempted': True, 'session_data': session_data}
        
    except Exception as e:
        st.error(f"❌ Failed to restore session: {e}")
        return {'recovery_attempted': True, 'session_data': None}


def render_session_management_sidebar() -> None:
    """Render session management controls in the sidebar."""
    if not STREAMLIT_AVAILABLE:
        return
    
    with st.sidebar.expander("💾 Session Management", expanded=False):
        session_manager = get_session_manager()
        auto_save_manager = get_auto_save_manager()
        
        # Current session info
        current_session_id = auto_save_manager.current_session_id
        if current_session_id:
            st.info(f"📝 Current: `{current_session_id}`")
        else:
            st.info("📝 No active session")
        
        # Manual save
        if st.button("💾 Save Session Now", use_container_width=True):
            try:
                session_id = auto_save_manager.auto_save(st.session_state, "manual_save")
                if session_id:
                    st.success("✅ Session saved!")
                else:
                    st.warning("⚠️ Nothing to save")
            except Exception as e:
                st.error(f"❌ Save failed: {e}")
        
        # Auto-save toggle
        auto_save_enabled = getattr(auto_save_manager, 'auto_save_enabled', True)
        new_auto_save = st.checkbox("🔄 Auto-save enabled", value=auto_save_enabled)
        if new_auto_save != auto_save_enabled:
            auto_save_manager.enable_auto_save(new_auto_save)
        
        # Session list
        sessions = session_manager.list_sessions(limit=5)
        if sessions:
            st.markdown("**Recent Sessions:**")
            for session_meta in sessions[:3]:  # Show top 3
                session_id = session_meta['session_id']
                created_at = datetime.fromisoformat(session_meta['created_at'])
                size_mb = session_meta.get('size_mb', 0)
                
                # Short display
                display_time = created_at.strftime('%m/%d %H:%M')
                st.text(f"• {display_time} ({size_mb:.1f}MB)")
        
        # Cleanup button
        if st.button("🧹 Cleanup Old Sessions", help="Delete sessions older than 7 days"):
            try:
                deleted_count = session_manager.cleanup_old_sessions()
                if deleted_count > 0:
                    st.success(f"✅ Deleted {deleted_count} old sessions")
                else:
                    st.info("ℹ️ No old sessions to delete")
            except Exception as e:
                st.error(f"❌ Cleanup failed: {e}")


def render_auto_recovery_banner() -> bool:
    """
    Show a banner offering to recover the most recent session.
    
    Returns:
        True if recovery was attempted, False otherwise
    """
    if not STREAMLIT_AVAILABLE:
        return False
    
    session_manager = get_session_manager()
    latest_session = session_manager.get_latest_session()
    
    if not latest_session:
        return False
    
    # Check if current session is empty (just started)
    current_session = st.session_state
    has_current_data = any(
        key in current_session for key in ['headers_df', 'monthly_df', 'pipeline_results']
    )
    
    if has_current_data:
        return False  # Don't show banner if we already have data
    
    # Check if latest session is recent (within 24 hours)
    created_at = datetime.fromisoformat(latest_session['created_at'])
    hours_ago = (datetime.now() - created_at).total_seconds() / 3600
    
    if hours_ago > 24:
        return False  # Don't auto-suggest very old sessions
    
    # Show recovery banner
    session_id = latest_session['session_id']
    time_ago = f"{int(hours_ago)}h ago" if hours_ago >= 1 else f"{int(hours_ago * 60)}m ago"
    
    st.info(f"📂 **Session Recovery Available** - Found recent session from {time_ago}")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        if latest_session.get('has_analysis_results'):
            st.markdown("✅ Contains complete analysis results")
        elif latest_session.get('has_uploaded_data'):
            st.markdown("📁 Contains uploaded data")
        else:
            st.markdown("⚠️ Incomplete session")
    
    with col2:
        if st.button("🔄 Restore", key="auto_recovery_restore"):
            return _perform_session_recovery(session_id)['recovery_attempted']
    
    with col3:
        if st.button("🆕 Start Fresh", key="auto_recovery_dismiss"):
            # Mark that we don't want auto-recovery for this session
            st.session_state['_dismiss_auto_recovery'] = True
            st.rerun()
    
    return False


def _mock_session_recovery() -> Dict[str, Any]:
    """Mock session recovery for testing."""
    return {'recovery_attempted': False, 'session_data': None}


# Utility functions for session validation
def validate_session_data(session_data: Dict[str, Any]) -> List[str]:
    """Validate loaded session data and return any warnings."""
    warnings = []
    
    # Check for required DataFrames
    required_dfs = ['headers_df', 'monthly_df']
    for df_key in required_dfs:
        df = session_data.get(df_key)
        if df is None:
            warnings.append(f"Missing {df_key}")
        elif not isinstance(df, pd.DataFrame) or len(df) == 0:
            warnings.append(f"Empty or invalid {df_key}")
    
    # Check for analysis results
    analysis_keys = ['vectors_df', 'labels_df', 'coords_df', 'scores_df']
    has_analysis = any(session_data.get(key) is not None for key in analysis_keys)
    
    if not has_analysis and any(session_data.get(key) is not None for key in ['pipeline_results']):
        warnings.append("Analysis results may be incomplete")
    
    return warnings


def get_session_summary(session_data: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of session data contents."""
    summary = {
        'has_data': False,
        'has_analysis': False,
        'data_summary': {},
        'analysis_summary': {}
    }
    
    # Check data
    headers_df = session_data.get('headers_df')
    monthly_df = session_data.get('monthly_df')
    
    if isinstance(headers_df, pd.DataFrame) and isinstance(monthly_df, pd.DataFrame):
        summary['has_data'] = True
        summary['data_summary'] = {
            'wells': len(headers_df),
            'monthly_records': len(monthly_df)
        }
    
    # Check analysis
    labels_df = session_data.get('labels_df')
    if isinstance(labels_df, pd.DataFrame):
        summary['has_analysis'] = True
        n_clusters = len([c for c in labels_df['label'].unique() if c != -1])
        summary['analysis_summary'] = {
            'clusters': n_clusters,
            'wells_clustered': len(labels_df)
        }
    
    return summary