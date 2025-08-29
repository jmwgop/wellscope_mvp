# app/services/session_persistence.py

from __future__ import annotations
import os
import json
import pickle
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd

from app.state.session import SessionKeys


class SessionManager:
    """Manages persistent storage of analysis sessions."""
    
    def __init__(self, base_dir: str = ".wellscope_sessions"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
    def create_session_id(self) -> str:
        """Generate unique session ID with timestamp."""
        return f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def get_session_dir(self, session_id: str) -> Path:
        """Get directory path for a session."""
        return self.base_dir / session_id
    
    def save_session(self, session_data: Dict[str, Any], session_id: Optional[str] = None) -> str:
        """
        Save complete session state to disk.
        
        Args:
            session_data: Complete session state dictionary
            session_id: Optional session ID, generates new if None
            
        Returns:
            Session ID of saved session
        """
        if session_id is None:
            session_id = self.create_session_id()
        
        session_dir = self.get_session_dir(session_id)
        session_dir.mkdir(exist_ok=True)
        
        # Save metadata
        metadata = {
            'session_id': session_id,
            'created_at': datetime.now().isoformat(),
            'session_keys': list(session_data.keys()),
            'has_uploaded_data': SessionKeys.HEADERS_DF in session_data and SessionKeys.MONTHLY_DF in session_data,
            'has_analysis_results': all(key in session_data for key in [SessionKeys.VECTORS_DF, SessionKeys.LABELS_DF, SessionKeys.COORDS_DF, SessionKeys.SCORES_DF]),
            'wellscope_version': '0.1.0'
        }
        
        with open(session_dir / 'session_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save DataFrames as CSV files
        dataframe_keys = [
            SessionKeys.HEADERS_DF, SessionKeys.MONTHLY_DF, SessionKeys.JOINED_DF,
            SessionKeys.VECTORS_DF, SessionKeys.LABELS_DF, SessionKeys.COORDS_DF, SessionKeys.SCORES_DF
        ]
        
        for key in dataframe_keys:
            if key in session_data and isinstance(session_data[key], pd.DataFrame):
                csv_filename = f"{key}.csv"
                session_data[key].to_csv(session_dir / csv_filename, index=False)
        
        # Save other data as JSON/pickle
        other_data = {}
        for key, value in session_data.items():
            if key not in dataframe_keys:  # Include None values
                try:
                    # Try JSON serialization first
                    json.dumps(value)
                    other_data[key] = {'type': 'json', 'data': value}
                except (TypeError, ValueError):
                    # Fall back to pickle for complex objects
                    other_data[key] = {'type': 'pickle', 'data': value}
        
        # Save JSON-serializable data
        json_data = {k: v['data'] for k, v in other_data.items() if v['type'] == 'json'}
        if json_data:
            with open(session_dir / 'session_data.json', 'w') as f:
                json.dump(json_data, f, indent=2)
        
        # Save pickle data
        pickle_data = {k: v['data'] for k, v in other_data.items() if v['type'] == 'pickle'}
        if pickle_data:
            with open(session_dir / 'session_data.pkl', 'wb') as f:
                pickle.dump(pickle_data, f)
        
        return session_id
    
    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Load session state from disk.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            Complete session state dictionary or None if not found
        """
        session_dir = self.get_session_dir(session_id)
        
        if not session_dir.exists():
            return None
        
        try:
            session_data = {}
            
            # Load metadata
            metadata_file = session_dir / 'session_metadata.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                session_data['_session_metadata'] = metadata
            
            # Load DataFrames
            dataframe_keys = [
                SessionKeys.HEADERS_DF, SessionKeys.MONTHLY_DF, SessionKeys.JOINED_DF,
                SessionKeys.VECTORS_DF, SessionKeys.LABELS_DF, SessionKeys.COORDS_DF, SessionKeys.SCORES_DF
            ]
            
            for key in dataframe_keys:
                csv_file = session_dir / f"{key}.csv"
                if csv_file.exists():
                    session_data[key] = pd.read_csv(csv_file)
            
            # Load JSON data
            json_file = session_dir / 'session_data.json'
            if json_file.exists():
                with open(json_file) as f:
                    json_data = json.load(f)
                session_data.update(json_data)
            
            # Load pickle data
            pickle_file = session_dir / 'session_data.pkl'
            if pickle_file.exists():
                with open(pickle_file, 'rb') as f:
                    pickle_data = pickle.load(f)
                session_data.update(pickle_data)
            
            return session_data
            
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None
    
    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        List available sessions with metadata.
        
        Args:
            limit: Maximum number of sessions to return
            
        Returns:
            List of session metadata dictionaries
        """
        sessions = []
        
        if not self.base_dir.exists():
            return sessions
        
        for session_dir in sorted(self.base_dir.iterdir(), reverse=True):
            if not session_dir.is_dir():
                continue
            
            metadata_file = session_dir / 'session_metadata.json'
            if metadata_file.exists():
                try:
                    with open(metadata_file) as f:
                        metadata = json.load(f)
                    
                    # Add file system info
                    metadata['size_mb'] = sum(f.stat().st_size for f in session_dir.glob('*') if f.is_file()) / 1024 / 1024
                    metadata['last_modified'] = datetime.fromtimestamp(session_dir.stat().st_mtime).isoformat()
                    
                    sessions.append(metadata)
                    
                    if len(sessions) >= limit:
                        break
                        
                except Exception as e:
                    print(f"Error reading metadata for {session_dir}: {e}")
                    continue
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session and all its files.
        
        Args:
            session_id: Session ID to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        session_dir = self.get_session_dir(session_id)
        
        if not session_dir.exists():
            return False
        
        try:
            shutil.rmtree(session_dir)
            return True
        except Exception as e:
            print(f"Error deleting session {session_id}: {e}")
            return False
    
    def cleanup_old_sessions(self, keep_days: int = 7, keep_count: int = 10) -> int:
        """
        Clean up old sessions based on age and count.
        
        Args:
            keep_days: Keep sessions newer than this many days
            keep_count: Always keep at least this many most recent sessions
            
        Returns:
            Number of sessions deleted
        """
        sessions = self.list_sessions(limit=100)  # Get more for cleanup
        
        if len(sessions) <= keep_count:
            return 0  # Don't delete if we have fewer than keep_count
        
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        deleted_count = 0
        
        # Keep the most recent keep_count sessions, regardless of age
        sessions_to_check = sessions[keep_count:]
        
        for session_meta in sessions_to_check:
            try:
                created_at = datetime.fromisoformat(session_meta['created_at'])
                if created_at < cutoff_date:
                    if self.delete_session(session_meta['session_id']):
                        deleted_count += 1
            except Exception as e:
                print(f"Error during cleanup of session {session_meta.get('session_id', 'unknown')}: {e}")
                continue
        
        return deleted_count
    
    def get_latest_session(self) -> Optional[Dict[str, Any]]:
        """Get metadata for the most recent session."""
        sessions = self.list_sessions(limit=1)
        return sessions[0] if sessions else None
    
    def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self.get_session_dir(session_id).exists()


class AutoSaveManager:
    """Manages automatic saving at workflow checkpoints."""
    
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.current_session_id: Optional[str] = None
        self.auto_save_enabled = True
    
    def set_session_id(self, session_id: str) -> None:
        """Set the current session ID for auto-saving."""
        self.current_session_id = session_id
    
    def enable_auto_save(self, enabled: bool = True) -> None:
        """Enable or disable auto-saving."""
        self.auto_save_enabled = enabled
    
    def auto_save(self, session_data: Dict[str, Any], checkpoint: str = "auto") -> Optional[str]:
        """
        Automatically save session data if enabled.
        
        Args:
            session_data: Current session state
            checkpoint: Name of the checkpoint (for logging)
            
        Returns:
            Session ID if saved, None otherwise
        """
        if not self.auto_save_enabled:
            return None
        
        try:
            # Use existing session ID or create new one
            session_id = self.session_manager.save_session(
                session_data, 
                self.current_session_id
            )
            
            # Update current session ID if it was newly created
            if self.current_session_id is None:
                self.current_session_id = session_id
            
            print(f"Auto-saved session at checkpoint: {checkpoint}")
            return session_id
            
        except Exception as e:
            print(f"Auto-save failed at checkpoint {checkpoint}: {e}")
            return None


# Global session manager instance
_session_manager: Optional[SessionManager] = None
_auto_save_manager: Optional[AutoSaveManager] = None

def get_session_manager() -> SessionManager:
    """Get the global session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager

def get_auto_save_manager() -> AutoSaveManager:
    """Get the global auto-save manager instance."""
    global _auto_save_manager
    if _auto_save_manager is None:
        _auto_save_manager = AutoSaveManager(get_session_manager())
    return _auto_save_manager