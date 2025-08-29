# tests/test_main.py

import pytest
from pathlib import Path


def test_main_imports():
    """Test that main.py imports without error."""
    # This will test the import path resolution
    import main
    
    # Should have access to render_page function
    assert hasattr(main, 'render_page')
    assert callable(main.render_page)


def test_main_execution():
    """Test that main can be executed."""
    import main
    
    # Should be able to call render_page without error
    result = main.render_page()
    
    # Should return some result (either mock or real Streamlit result)
    assert result is not None
    assert isinstance(result, dict)
    assert result.get('page_rendered') == True


def test_project_structure():
    """Test that the project structure is correct for main.py."""
    project_root = Path(__file__).parent.parent
    main_py = project_root / "main.py"
    
    assert main_py.exists()
    assert main_py.is_file()
    
    # Check that the app directory exists
    app_dir = project_root / "app"
    assert app_dir.exists()
    assert app_dir.is_dir()
    
    # Check that the pages directory exists
    pages_dir = app_dir / "pages"
    assert pages_dir.exists()
    assert pages_dir.is_dir()
    
    # Check that the main page exists
    main_page = pages_dir / "page_upload_analyze.py"
    assert main_page.exists()
    assert main_page.is_file()


def test_main_module_docstring():
    """Test that main.py has proper documentation."""
    import main
    
    assert main.__doc__ is not None
    assert "WellScope MVP" in main.__doc__
    assert "streamlit run main.py" in main.__doc__