#!/usr/bin/env python3
"""
Clean Python Cache
==================
Recursively removes all __pycache__ directories and .pyc files from the project.

Usage:
    python scripts/clean_pycache.py
    python scripts/clean_pycache.py --dry-run  # See what would be deleted
"""

import argparse
import shutil
from pathlib import Path

def clean_pycache(root_dir: Path, dry_run: bool = False) -> None:
    """Remove all __pycache__ directories and .pyc files recursively."""
    
    removed_dirs = 0
    removed_files = 0
    
    # Find all __pycache__ directories
    for pycache_dir in root_dir.rglob("__pycache__"):
        if pycache_dir.is_dir():
            if dry_run:
                print(f"Would remove directory: {pycache_dir}")
            else:
                shutil.rmtree(pycache_dir)
                print(f"Removed directory: {pycache_dir}")
            removed_dirs += 1
    
    # Find all .pyc files
    for pyc_file in root_dir.rglob("*.pyc"):
        if pyc_file.is_file():
            if dry_run:
                print(f"Would remove file: {pyc_file}")
            else:
                pyc_file.unlink()
                print(f"Removed file: {pyc_file}")
            removed_files += 1
    
    # Summary
    action = "Would remove" if dry_run else "Removed"
    print(f"\n{action}: {removed_dirs} __pycache__ directories, {removed_files} .pyc files")

def main():
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Clean Python cache files and directories"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    
    print(f"Cleaning Python cache from: {project_root}")
    if args.dry_run:
        print("DRY RUN - No files will be deleted")
    print("-" * 50)
    
    clean_pycache(project_root, dry_run=args.dry_run)

if __name__ == "__main__":
    main()