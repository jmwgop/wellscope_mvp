# tests/app/test_session_persistence.py

import pytest
import tempfile
import shutil
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

from app.services.session_persistence import SessionManager, AutoSaveManager


@pytest.fixture
def temp_session_dir():
    """Create a temporary directory for session testing."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def session_manager(temp_session_dir):
    """Create a session manager with temporary directory."""
    return SessionManager(base_dir=temp_session_dir)


@pytest.fixture
def sample_session_data():
    """Create sample session data for testing."""
    headers_df = pd.DataFrame({
        'API14': ['42123456780000', '42123456790000'],
        'Operator': ['Test Operator 1', 'Test Operator 2'],
        'Target Formation': ['EAGLEFORD', 'EAGLEFORD']
    })
    
    monthly_df = pd.DataFrame({
        'API_UWI': ['42123456780000', '42123456780000', '42123456790000'],
        'Monthly Oil': [1000, 900, 800],
        'Monthly Gas': [2000, 1800, 1600],
        'Monthly Production Date': ['2023-01-01', '2023-02-01', '2023-01-01']
    })
    
    return {
        'headers_df': headers_df,
        'monthly_df': monthly_df,
        'filters_cfg': {'formations': ['EAGLEFORD'], 'min_months_produced': 6},
        'simple_data': {'test_key': 'test_value', 'number': 42}
    }


class TestSessionManager:
    """Test session manager functionality."""
    
    def test_create_session_id(self, session_manager):
        """Test session ID creation."""
        session_id = session_manager.create_session_id()
        assert session_id.startswith('session_')
        assert len(session_id) > len('session_')
    
    def test_save_and_load_session(self, session_manager, sample_session_data):
        """Test saving and loading a complete session."""
        # Save session
        session_id = session_manager.save_session(sample_session_data)
        assert session_id is not None
        assert session_manager.session_exists(session_id)
        
        # Load session
        loaded_data = session_manager.load_session(session_id)
        assert loaded_data is not None
        
        # Check DataFrames
        assert isinstance(loaded_data['headers_df'], pd.DataFrame)
        assert isinstance(loaded_data['monthly_df'], pd.DataFrame)
        assert len(loaded_data['headers_df']) == 2
        assert len(loaded_data['monthly_df']) == 3
        
        # Check other data
        assert loaded_data['filters_cfg'] == sample_session_data['filters_cfg']
        assert loaded_data['simple_data'] == sample_session_data['simple_data']
        
        # Check metadata
        assert '_session_metadata' in loaded_data
        metadata = loaded_data['_session_metadata']
        assert metadata['session_id'] == session_id
        assert metadata['has_uploaded_data'] is True
        assert 'created_at' in metadata
    
    def test_save_with_custom_session_id(self, session_manager, sample_session_data):
        """Test saving with a custom session ID."""
        custom_id = "custom_test_session"
        
        session_id = session_manager.save_session(sample_session_data, custom_id)
        assert session_id == custom_id
        
        loaded_data = session_manager.load_session(custom_id)
        assert loaded_data is not None
    
    def test_list_sessions(self, session_manager, sample_session_data):
        """Test listing sessions."""
        # Initially empty
        sessions = session_manager.list_sessions()
        assert len(sessions) == 0
        
        # Save a few sessions with explicit IDs
        ids = []
        for i in range(3):
            test_data = sample_session_data.copy()
            test_data['session_num'] = i
            session_id = session_manager.save_session(test_data, f"test_session_{i}")
            ids.append(session_id)
        
        # List sessions
        sessions = session_manager.list_sessions()
        assert len(sessions) == 3
        
        # Check that all sessions are present
        session_ids = [s['session_id'] for s in sessions]
        assert set(session_ids) == set(ids)  # All IDs present
        
        # Check metadata
        for session_meta in sessions:
            assert 'session_id' in session_meta
            assert 'created_at' in session_meta
            assert 'has_uploaded_data' in session_meta
            assert 'size_mb' in session_meta
    
    def test_delete_session(self, session_manager, sample_session_data):
        """Test session deletion."""
        # Save session
        session_id = session_manager.save_session(sample_session_data)
        assert session_manager.session_exists(session_id)
        
        # Delete session
        success = session_manager.delete_session(session_id)
        assert success is True
        assert not session_manager.session_exists(session_id)
        
        # Try to load deleted session
        loaded_data = session_manager.load_session(session_id)
        assert loaded_data is None
    
    def test_cleanup_old_sessions(self, session_manager, sample_session_data):
        """Test cleanup functionality."""
        # Save multiple sessions with explicit IDs
        ids = []
        for i in range(15):
            test_data = sample_session_data.copy()
            test_data['session_num'] = i
            session_id = session_manager.save_session(test_data, f"cleanup_test_session_{i:02d}")
            ids.append(session_id)
        
        # Manually modify creation times to make some appear old
        import time
        from datetime import timedelta
        for i in range(5):  # Make first 5 sessions appear old
            session_dir = session_manager.get_session_dir(ids[i])
            metadata_file = session_dir / 'session_metadata.json'
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
                # Make it appear 10 days old
                old_date = datetime.now() - timedelta(days=10)
                metadata['created_at'] = old_date.isoformat()
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2)
        
        # Cleanup (keep 10, delete older than 7 days - should delete 5)
        deleted_count = session_manager.cleanup_old_sessions(keep_days=7, keep_count=10)
        assert deleted_count == 5
        
        # Check remaining sessions
        sessions = session_manager.list_sessions()
        assert len(sessions) == 10
        
        # Should keep the 10 most recent
        remaining_ids = [s['session_id'] for s in sessions]
        assert set(remaining_ids) == set(ids[-10:])
    
    def test_load_nonexistent_session(self, session_manager):
        """Test loading a session that doesn't exist."""
        loaded_data = session_manager.load_session("nonexistent_session")
        assert loaded_data is None
        
        assert not session_manager.session_exists("nonexistent_session")


class TestAutoSaveManager:
    """Test auto-save functionality."""
    
    def test_auto_save_enabled_disabled(self, session_manager):
        """Test enabling/disabling auto-save."""
        auto_save = AutoSaveManager(session_manager)
        
        # Default should be enabled
        assert auto_save.auto_save_enabled is True
        
        # Disable
        auto_save.enable_auto_save(False)
        assert auto_save.auto_save_enabled is False
        
        # Re-enable
        auto_save.enable_auto_save(True)
        assert auto_save.auto_save_enabled is True
    
    def test_auto_save_with_session_id(self, session_manager, sample_session_data):
        """Test auto-save with existing session ID."""
        auto_save = AutoSaveManager(session_manager)
        
        # Set session ID
        test_id = "test_auto_save_session"
        auto_save.set_session_id(test_id)
        
        # Auto-save should use the set ID
        result_id = auto_save.auto_save(sample_session_data)
        assert result_id == test_id
        
        # Session should exist
        assert session_manager.session_exists(test_id)
        loaded_data = session_manager.load_session(test_id)
        assert loaded_data is not None
    
    def test_auto_save_disabled(self, session_manager, sample_session_data):
        """Test that auto-save respects enabled/disabled state."""
        auto_save = AutoSaveManager(session_manager)
        
        # Disable auto-save
        auto_save.enable_auto_save(False)
        
        # Try to auto-save - should return None
        result_id = auto_save.auto_save(sample_session_data)
        assert result_id is None
    
    def test_auto_save_creates_new_session(self, session_manager, sample_session_data):
        """Test auto-save creating new session when no ID is set."""
        auto_save = AutoSaveManager(session_manager)
        
        # No session ID set - should create new one
        result_id = auto_save.auto_save(sample_session_data)
        assert result_id is not None
        assert result_id.startswith('session_')
        
        # Should update current session ID
        assert auto_save.current_session_id == result_id
        
        # Second save should use same ID
        result_id_2 = auto_save.auto_save(sample_session_data, "second_checkpoint")
        assert result_id_2 == result_id


class TestSessionDataTypes:
    """Test handling of different data types in sessions."""
    
    def test_dataframe_serialization(self, session_manager):
        """Test that DataFrames are properly serialized to CSV."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z'],
            'C': [1.1, 2.2, 3.3]
        })
        
        session_data = {'test_df': df}
        session_id = session_manager.save_session(session_data)
        
        loaded_data = session_manager.load_session(session_id)
        loaded_df = loaded_data['test_df']
        
        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ['A', 'B', 'C']
        assert loaded_df['A'].tolist() == [1, 2, 3]
        assert loaded_df['B'].tolist() == ['x', 'y', 'z']
    
    def test_json_serializable_data(self, session_manager):
        """Test JSON-serializable data types."""
        session_data = {
            'string': 'test',
            'number': 42,
            'float': 3.14,
            'boolean': True,
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'none': None
        }
        
        session_id = session_manager.save_session(session_data)
        loaded_data = session_manager.load_session(session_id)
        
        for key, expected_value in session_data.items():
            assert loaded_data[key] == expected_value
    
    def test_empty_session(self, session_manager):
        """Test saving and loading empty session."""
        session_data = {}
        session_id = session_manager.save_session(session_data)
        
        loaded_data = session_manager.load_session(session_id)
        assert loaded_data is not None
        
        # Should only have metadata
        non_meta_keys = [k for k in loaded_data.keys() if not k.startswith('_')]
        assert len(non_meta_keys) == 0


def test_session_manager_singleton():
    """Test that get_session_manager returns consistent instance."""
    from app.services.session_persistence import get_session_manager, get_auto_save_manager
    
    manager1 = get_session_manager()
    manager2 = get_session_manager()
    assert manager1 is manager2
    
    auto_save1 = get_auto_save_manager()
    auto_save2 = get_auto_save_manager()
    assert auto_save1 is auto_save2