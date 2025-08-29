#!/usr/bin/env python3
"""
WellScope MVP - Main Entry Point

Run with: streamlit run main.py
"""

import sys
from pathlib import Path

# Add the project root to the path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now we can import our page
from app.pages.page_upload_analyze import render_page

if __name__ == "__main__":
    # Entry point for the Streamlit application
    render_page()