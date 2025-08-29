# app/utils/formatting.py

from __future__ import annotations
from typing import Union, Optional, Dict, Any, List
import pandas as pd
import numpy as np

def format_number(value: Union[int, float, str, None], precision: int = 1) -> str:
    """Format numbers with appropriate units (K, M, MM)."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        num = float(value)
    except (ValueError, TypeError):
        return str(value)
    
    if abs(num) >= 1_000_000_000:
        return f"{num / 1_000_000_000:.{precision}f}B"
    elif abs(num) >= 1_000_000:
        return f"{num / 1_000_000:.{precision}f}M"
    elif abs(num) >= 1_000:
        return f"{num / 1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"

def format_percentage(value: Union[float, str, None], precision: int = 1) -> str:
    """Format values as percentages."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        num = float(value)
        return f"{num * 100:.{precision}f}%"
    except (ValueError, TypeError):
        return str(value)

def format_date(value: Union[str, pd.Timestamp, None]) -> str:
    """Format dates consistently."""
    if value is None or pd.isna(value):
        return "N/A"
    
    try:
        if isinstance(value, str):
            date = pd.to_datetime(value)
        else:
            date = value
        return date.strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(value)

def format_file_size(size_bytes: Union[int, float]) -> str:
    """Format file size in human readable format."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes / (1024**2):.1f} MB"
    else:
        return f"{size_bytes / (1024**3):.1f} GB"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    else:
        return f"{seconds / 3600:.1f}h"

def get_cluster_colors(n_clusters: int, include_noise: bool = True) -> List[str]:
    """Generate consistent colors for cluster visualization."""
    # Standard color palette for clusters
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7",
        "#DDA0DD", "#98D8C8", "#F7DC6F", "#BB8FCE", "#85C1E9",
        "#F8C471", "#82E0AA", "#F1948A", "#85C1E9", "#D7BDE2",
        "#A9DFBF", "#F9E79F", "#D5A6BD", "#7FB3D3", "#C39BD3"
    ]
    
    # Extend with more colors if needed
    while len(colors) < n_clusters:
        colors.extend(colors)
    
    cluster_colors = colors[:n_clusters]
    
    if include_noise:
        # Add gray for noise points (cluster -1)
        cluster_colors = ["#808080"] + cluster_colors
    
    return cluster_colors

def format_cluster_label(cluster_id: int) -> str:
    """Format cluster labels consistently."""
    if cluster_id == -1:
        return "Noise"
    else:
        return f"Cluster {cluster_id}"

def format_similarity_score(score: float) -> str:
    """Format similarity scores with color coding."""
    if pd.isna(score):
        return "N/A"
    
    return f"{score:.3f}"

def get_similarity_color(score: float, threshold: float = 0.7) -> str:
    """Get color for similarity score based on threshold."""
    if pd.isna(score):
        return "#CCCCCC"  # Gray for N/A
    elif score >= threshold:
        return "#2ECC40"  # Green for high similarity
    elif score >= threshold * 0.5:
        return "#FF851B"  # Orange for medium similarity
    else:
        return "#FF4136"  # Red for low similarity

def format_dataframe_summary(df: pd.DataFrame) -> Dict[str, str]:
    """Generate formatted summary statistics for a DataFrame."""
    if df is None or len(df) == 0:
        return {
            "rows": "0",
            "columns": "0",
            "memory": "0 B",
            "completeness": "N/A"
        }
    
    memory_usage = df.memory_usage(deep=True).sum()
    non_null_pct = (df.count().sum() / (len(df) * len(df.columns))) if len(df) > 0 else 0
    
    return {
        "rows": format_number(len(df), 0),
        "columns": format_number(len(df.columns), 0),
        "memory": format_file_size(memory_usage),
        "completeness": format_percentage(non_null_pct)
    }

def format_filter_summary(filters: Dict[str, Any], well_count: Optional[int] = None) -> List[str]:
    """Generate human-readable filter summary."""
    summary = []
    
    if filters.get('formations'):
        formations = filters['formations']
        if len(formations) == 1:
            summary.append(f"Formation: {formations[0]}")
        else:
            summary.append(f"Formations: {', '.join(formations[:2])}{'+' + str(len(formations)-2) + ' more' if len(formations) > 2 else ''}")
    
    if filters.get('completion_year_range'):
        start, end = filters['completion_year_range']
        if start and end:
            summary.append(f"Completion: {start}-{end}")
        elif start:
            summary.append(f"Completion: {start}+")
        elif end:
            summary.append(f"Completion: <{end}")
    
    if filters.get('lateral_ft_range'):
        min_ft, max_ft = filters['lateral_ft_range']
        if min_ft is not None and max_ft is not None:
            summary.append(f"Lateral: {format_number(min_ft, 0)}-{format_number(max_ft, 0)} ft")
        elif min_ft is not None:
            summary.append(f"Lateral: >{format_number(min_ft, 0)} ft")
        elif max_ft is not None:
            summary.append(f"Lateral: <{format_number(max_ft, 0)} ft")
    
    if filters.get('min_months_produced'):
        months = filters['min_months_produced']
        if months > 0:
            summary.append(f"Min production: {months} months")
    
    if well_count is not None:
        summary.append(f"Wells: {format_number(well_count, 0)}")
    
    return summary if summary else ["No filters applied"]

def format_pipeline_stats(stats: Dict[str, Any]) -> Dict[str, str]:
    """Format pipeline execution statistics."""
    formatted = {}
    
    for key, value in stats.items():
        if isinstance(value, dict):
            # Nested dict - format recursively
            nested = format_pipeline_stats(value)
            for nested_key, nested_value in nested.items():
                formatted[f"{key}_{nested_key}"] = nested_value
        elif key.endswith('_rows') or key.endswith('_count') or key.endswith('_size'):
            formatted[key] = format_number(value, 0)
        elif key.endswith('_fraction') or key.endswith('_pct') or key.endswith('_ratio'):
            formatted[key] = format_percentage(value)
        elif key.endswith('_time') or key.endswith('_duration'):
            formatted[key] = format_duration(value)
        elif key.endswith('_bytes') or key.endswith('_memory'):
            formatted[key] = format_file_size(value)
        else:
            formatted[key] = str(value)
    
    return formatted

def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """Truncate long text with ellipsis."""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix

def format_api_display(api: str) -> str:
    """Format API numbers for display (add hyphens for readability)."""
    if not api or len(api) != 14:
        return api
    
    # Format as XX-XXX-XXXXX-XX-XX
    return f"{api[:2]}-{api[2:5]}-{api[5:10]}-{api[10:12]}-{api[12:14]}"