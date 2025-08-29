# app/utils/session_utils.py

from __future__ import annotations
import argparse
import sys
from datetime import datetime, timedelta
from typing import Optional, List
from pathlib import Path

from app.services.session_persistence import get_session_manager


def list_sessions(limit: int = 20, show_details: bool = False) -> None:
    """List all available sessions."""
    session_manager = get_session_manager()
    sessions = session_manager.list_sessions(limit=limit)
    
    if not sessions:
        print("No sessions found.")
        return
    
    print(f"Found {len(sessions)} session(s):\n")
    
    for i, session_meta in enumerate(sessions, 1):
        session_id = session_meta['session_id']
        created_at = datetime.fromisoformat(session_meta['created_at'])
        size_mb = session_meta.get('size_mb', 0)
        
        # Status indicators
        status_parts = []
        if session_meta.get('has_uploaded_data'):
            status_parts.append("📁 Data")
        if session_meta.get('has_analysis_results'):
            status_parts.append("🎯 Analysis")
        
        status_str = " + ".join(status_parts) if status_parts else "⚠️ Incomplete"
        
        print(f"{i:2d}. {session_id}")
        print(f"    Created: {created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Status:  {status_str}")
        print(f"    Size:    {size_mb:.1f} MB")
        
        if show_details:
            session_keys = session_meta.get('session_keys', [])
            print(f"    Keys:    {', '.join(session_keys[:5])}")
            if len(session_keys) > 5:
                print(f"             ... and {len(session_keys) - 5} more")
        
        print()


def cleanup_sessions(
    dry_run: bool = False, 
    keep_days: int = 7, 
    keep_count: int = 10,
    force: bool = False
) -> None:
    """Clean up old sessions."""
    session_manager = get_session_manager()
    
    # Get sessions for analysis
    all_sessions = session_manager.list_sessions(limit=100)
    
    if len(all_sessions) <= keep_count:
        print(f"Only {len(all_sessions)} sessions found. Keeping all (minimum: {keep_count})")
        return
    
    cutoff_date = datetime.now() - timedelta(days=keep_days)
    sessions_to_delete = []
    
    # Sessions beyond keep_count that are older than keep_days
    for session_meta in all_sessions[keep_count:]:
        created_at = datetime.fromisoformat(session_meta['created_at'])
        if created_at < cutoff_date:
            sessions_to_delete.append(session_meta)
    
    if not sessions_to_delete:
        print(f"No sessions older than {keep_days} days to cleanup.")
        return
    
    print(f"Sessions to {'DELETE' if not dry_run else 'delete'} (older than {keep_days} days):")
    total_size = 0
    
    for session_meta in sessions_to_delete:
        session_id = session_meta['session_id']
        created_at = datetime.fromisoformat(session_meta['created_at'])
        size_mb = session_meta.get('size_mb', 0)
        total_size += size_mb
        
        age_days = (datetime.now() - created_at).days
        print(f"  • {session_id} ({age_days}d old, {size_mb:.1f}MB)")
    
    print(f"\nTotal: {len(sessions_to_delete)} sessions, {total_size:.1f} MB")
    
    if dry_run:
        print("\n[DRY RUN] No files were actually deleted.")
        return
    
    if not force:
        response = input(f"\nDelete {len(sessions_to_delete)} sessions? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Cleanup cancelled.")
            return
    
    # Perform deletion
    deleted_count = 0
    for session_meta in sessions_to_delete:
        session_id = session_meta['session_id']
        if session_manager.delete_session(session_id):
            deleted_count += 1
            print(f"✅ Deleted {session_id}")
        else:
            print(f"❌ Failed to delete {session_id}")
    
    print(f"\n🎉 Cleanup complete: {deleted_count}/{len(sessions_to_delete)} sessions deleted")


def delete_session(session_id: str, force: bool = False) -> None:
    """Delete a specific session."""
    session_manager = get_session_manager()
    
    if not session_manager.session_exists(session_id):
        print(f"❌ Session '{session_id}' not found")
        return
    
    if not force:
        response = input(f"Delete session '{session_id}'? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Delete cancelled.")
            return
    
    if session_manager.delete_session(session_id):
        print(f"✅ Deleted session '{session_id}'")
    else:
        print(f"❌ Failed to delete session '{session_id}'")


def export_session(session_id: str, output_dir: Optional[str] = None) -> None:
    """Export a session to a specified directory."""
    session_manager = get_session_manager()
    
    # Load session
    session_data = session_manager.load_session(session_id)
    if not session_data:
        print(f"❌ Failed to load session '{session_id}'")
        return
    
    # Create output directory
    if output_dir is None:
        output_dir = f"exported_{session_id}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Exporting session '{session_id}' to '{output_path}'...")
    
    # Copy session files
    import shutil
    session_dir = session_manager.get_session_dir(session_id)
    
    try:
        for file_path in session_dir.glob('*'):
            if file_path.is_file():
                shutil.copy2(file_path, output_path / file_path.name)
        
        print(f"✅ Session exported to '{output_path}'")
        
        # List exported files
        exported_files = list(output_path.glob('*'))
        print(f"Exported {len(exported_files)} files:")
        for file_path in exported_files:
            size_mb = file_path.stat().st_size / 1024 / 1024
            print(f"  • {file_path.name} ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"❌ Export failed: {e}")


def get_session_info(session_id: str) -> None:
    """Get detailed information about a session."""
    session_manager = get_session_manager()
    
    if not session_manager.session_exists(session_id):
        print(f"❌ Session '{session_id}' not found")
        return
    
    session_data = session_manager.load_session(session_id)
    if not session_data:
        print(f"❌ Failed to load session '{session_id}'")
        return
    
    # Remove metadata for cleaner display
    metadata = session_data.pop('_session_metadata', {})
    
    print(f"📊 Session Information: {session_id}")
    print(f"Created: {metadata.get('created_at', 'Unknown')}")
    print(f"Version: {metadata.get('wellscope_version', 'Unknown')}")
    print()
    
    # Data summary
    from app.components.session_manager import get_session_summary
    summary = get_session_summary(session_data)
    
    if summary['has_data']:
        print("📁 Data:")
        data_sum = summary['data_summary']
        print(f"  • Wells: {data_sum.get('wells', 0):,}")
        print(f"  • Monthly Records: {data_sum.get('monthly_records', 0):,}")
        print()
    
    if summary['has_analysis']:
        print("🎯 Analysis:")
        analysis_sum = summary['analysis_summary']
        print(f"  • Clusters: {analysis_sum.get('clusters', 0)}")
        print(f"  • Wells Clustered: {analysis_sum.get('wells_clustered', 0):,}")
        print()
    
    # Session keys
    print("🔑 Session Keys:")
    keys = [k for k in session_data.keys() if not k.startswith('_')]
    for key in sorted(keys):
        value = session_data[key]
        if hasattr(value, '__len__'):
            print(f"  • {key}: {type(value).__name__} (length: {len(value)})")
        else:
            print(f"  • {key}: {type(value).__name__}")


def main():
    """Command-line interface for session management."""
    parser = argparse.ArgumentParser(description='WellScope Session Management Utility')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all sessions')
    list_parser.add_argument('--limit', type=int, default=20, help='Maximum sessions to show')
    list_parser.add_argument('--details', action='store_true', help='Show detailed information')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old sessions')
    cleanup_parser.add_argument('--dry-run', action='store_true', help='Show what would be deleted')
    cleanup_parser.add_argument('--keep-days', type=int, default=7, help='Keep sessions newer than N days')
    cleanup_parser.add_argument('--keep-count', type=int, default=10, help='Always keep N most recent sessions')
    cleanup_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    # Delete command
    delete_parser = subparsers.add_parser('delete', help='Delete a specific session')
    delete_parser.add_argument('session_id', help='Session ID to delete')
    delete_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export a session')
    export_parser.add_argument('session_id', help='Session ID to export')
    export_parser.add_argument('--output-dir', help='Output directory (default: exported_<session_id>)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show session information')
    info_parser.add_argument('session_id', help='Session ID to inspect')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_sessions(limit=args.limit, show_details=args.details)
    elif args.command == 'cleanup':
        cleanup_sessions(
            dry_run=args.dry_run,
            keep_days=args.keep_days,
            keep_count=args.keep_count,
            force=args.force
        )
    elif args.command == 'delete':
        delete_session(args.session_id, force=args.force)
    elif args.command == 'export':
        export_session(args.session_id, args.output_dir)
    elif args.command == 'info':
        get_session_info(args.session_id)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()