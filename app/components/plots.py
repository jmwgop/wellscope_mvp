# app/components/plots.py

from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np

try:
    import streamlit as st
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from app.utils.formatting import get_cluster_colors, format_cluster_label, format_number

def render_cluster_scatter(coords_df: pd.DataFrame, labels_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Render 2D scatter plot of clustered wells.
    
    Args:
        coords_df: DataFrame with x, y coordinates from UMAP
        labels_df: DataFrame with cluster labels
        
    Returns:
        Plotly figure or None if plotting unavailable
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    if coords_df is None or len(coords_df) == 0:
        st.warning("No coordinate data available for plotting")
        return None
    
    # Merge coordinates with cluster labels
    api_col = [col for col in coords_df.columns if not col.startswith(('x', 'y', 'z'))][0]
    plot_df = coords_df.merge(labels_df[[api_col, 'label']], on=api_col, how='left')
    
    # Get unique clusters and colors
    unique_clusters = sorted(plot_df['label'].unique())
    cluster_colors = get_cluster_colors(len(unique_clusters), include_noise=True)
    
    # Create color mapping
    color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
    plot_df['color'] = plot_df['label'].map(color_map)
    plot_df['cluster_name'] = plot_df['label'].apply(format_cluster_label)
    
    # Create scatter plot
    fig = px.scatter(
        plot_df,
        x='x', y='y',
        color='cluster_name',
        color_discrete_map={format_cluster_label(k): v for k, v in color_map.items()},
        title="Well Clustering - 2D Projection",
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        hover_data={api_col: True, 'label': True}
    )
    
    # Update layout
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left", 
            x=1.01
        )
    )
    
    return fig

def render_production_curves(vectors_df: pd.DataFrame, labels_df: pd.DataFrame, 
                           selected_cluster: Optional[int] = None, 
                           max_curves: int = 50) -> Optional[go.Figure]:
    """
    Render production curves for wells in selected cluster.
    
    Args:
        vectors_df: DataFrame with production vectors
        labels_df: DataFrame with cluster labels
        selected_cluster: Cluster ID to show curves for
        max_curves: Maximum number of curves to display
        
    Returns:
        Plotly figure or None
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    if vectors_df is None or len(vectors_df) == 0:
        st.warning("No production vector data available")
        return None
    
    # Get API column
    api_col = [col for col in vectors_df.columns if not col.startswith('v')][0]
    
    # Merge with cluster labels
    plot_df = vectors_df.merge(labels_df[[api_col, 'label']], on=api_col, how='left')
    
    # Filter by selected cluster if specified
    if selected_cluster is not None:
        plot_df = plot_df[plot_df['label'] == selected_cluster]
        title = f"Production Curves - {format_cluster_label(selected_cluster)}"
    else:
        title = "Production Curves - All Wells"
    
    if len(plot_df) == 0:
        st.warning("No wells found for the selected cluster")
        return None
    
    # Limit number of curves for performance
    if len(plot_df) > max_curves:
        plot_df = plot_df.sample(n=max_curves, random_state=42)
    
    # Extract vector columns
    vector_cols = [col for col in plot_df.columns if col.startswith('v')]
    months = list(range(1, len(vector_cols) + 1))
    
    # Create figure
    fig = go.Figure()
    
    # Get cluster colors
    unique_clusters = sorted(plot_df['label'].unique())
    cluster_colors = get_cluster_colors(len(unique_clusters))
    color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
    
    # Add curves
    for _, row in plot_df.iterrows():
        values = [row[col] for col in vector_cols]
        cluster_label = format_cluster_label(row['label'])
        
        fig.add_trace(go.Scatter(
            x=months,
            y=values,
            mode='lines',
            name=row[api_col],
            line=dict(
                color=color_map.get(row['label'], '#CCCCCC'),
                width=1.5
            ),
            legendgroup=cluster_label,
            showlegend=False,  # Too many individual wells to show in legend
            hovertemplate=f"Well: {row[api_col]}<br>" +
                         f"Cluster: {cluster_label}<br>" +
                         "Month: %{x}<br>" +
                         "Normalized Production: %{y:.3f}<extra></extra>"
        ))
    
    # Update layout
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title="Normalized Production",
        width=800,
        height=500,
        hovermode='closest'
    )
    
    return fig

def render_cluster_summary_chart(labels_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Render cluster size summary chart.
    
    Args:
        labels_df: DataFrame with cluster labels and sizes
        
    Returns:
        Plotly figure or None
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    if labels_df is None or len(labels_df) == 0:
        return None
    
    # Calculate cluster sizes
    cluster_counts = labels_df['label'].value_counts().sort_index()
    
    # Prepare data
    cluster_names = [format_cluster_label(idx) for idx in cluster_counts.index]
    colors = get_cluster_colors(len(cluster_counts))
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=cluster_names,
            y=cluster_counts.values,
            marker_color=colors,
            text=cluster_counts.values,
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Cluster Sizes",
        xaxis_title="Cluster",
        yaxis_title="Number of Wells",
        width=600,
        height=400
    )
    
    return fig

def render_similarity_histogram(scores_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Render histogram of similarity scores.
    
    Args:
        scores_df: DataFrame with similarity scores
        
    Returns:
        Plotly figure or None
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    if scores_df is None or len(scores_df) == 0:
        return None
    
    # Create histogram
    fig = px.histogram(
        scores_df,
        x='similarity',
        nbins=30,
        title="Distribution of Similarity Scores",
        labels={'similarity': 'Cosine Similarity', 'count': 'Number of Wells'},
        color_discrete_sequence=['#1f77b4']
    )
    
    # Add mean line
    mean_similarity = scores_df['similarity'].mean()
    fig.add_vline(
        x=mean_similarity,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Mean: {mean_similarity:.3f}"
    )
    
    fig.update_layout(
        width=600,
        height=400
    )
    
    return fig

def render_interactive_plots(pipeline_results: Dict[str, Any]) -> None:
    """
    Render all interactive plots based on pipeline results.
    
    Args:
        pipeline_results: Complete results from pipeline execution
    """
    if not PLOTTING_AVAILABLE:
        st.error("Plotting libraries not available. Install plotly to enable visualizations.")
        return
    
    coords_df = pipeline_results.get('coords_df')
    labels_df = pipeline_results.get('labels_df') 
    vectors_df = pipeline_results.get('vectors_df')
    scores_df = pipeline_results.get('scores_df')
    
    if coords_df is None or labels_df is None:
        st.warning("Insufficient data for plotting. Please run analysis first.")
        return
    
    # Main cluster scatter plot
    st.subheader("🎯 Well Clustering Visualization")
    
    scatter_fig = render_cluster_scatter(coords_df, labels_df)
    if scatter_fig:
        st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Cluster summary and production curves in columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Cluster Summary")
        summary_fig = render_cluster_summary_chart(labels_df)
        if summary_fig:
            st.plotly_chart(summary_fig, use_container_width=True)
    
    with col2:
        st.subheader("📈 Production Curves")
        
        # Cluster selection for production curves
        unique_clusters = sorted(labels_df['label'].unique())
        cluster_options = [None] + unique_clusters
        cluster_labels = ["All Clusters"] + [format_cluster_label(c) for c in unique_clusters]
        
        selected_idx = st.selectbox(
            "Select cluster to view curves:",
            range(len(cluster_options)),
            format_func=lambda i: cluster_labels[i],
            key="curve_cluster_select"
        )
        
        selected_cluster = cluster_options[selected_idx]
        
        if vectors_df is not None:
            curves_fig = render_production_curves(vectors_df, labels_df, selected_cluster)
            if curves_fig:
                st.plotly_chart(curves_fig, use_container_width=True)
    
    # Similarity scores histogram
    if scores_df is not None and len(scores_df) > 0:
        st.subheader("📏 Similarity Distribution")
        similarity_fig = render_similarity_histogram(scores_df)
        if similarity_fig:
            st.plotly_chart(similarity_fig, use_container_width=True)

def render_3d_scatter(coords_df: pd.DataFrame, labels_df: pd.DataFrame) -> Optional[go.Figure]:
    """
    Render 3D scatter plot if available.
    
    Args:
        coords_df: DataFrame with x, y, z coordinates
        labels_df: DataFrame with cluster labels
        
    Returns:
        Plotly figure or None
    """
    if not PLOTTING_AVAILABLE:
        return None
    
    # Check if we have 3D coordinates
    if 'z' not in coords_df.columns:
        return None
    
    api_col = [col for col in coords_df.columns if not col.startswith(('x', 'y', 'z'))][0]
    plot_df = coords_df.merge(labels_df[[api_col, 'label']], on=api_col, how='left')
    
    # Get colors
    unique_clusters = sorted(plot_df['label'].unique())
    cluster_colors = get_cluster_colors(len(unique_clusters))
    color_map = {cluster: color for cluster, color in zip(unique_clusters, cluster_colors)}
    
    plot_df['color'] = plot_df['label'].map(color_map)
    plot_df['cluster_name'] = plot_df['label'].apply(format_cluster_label)
    
    # Create 3D scatter
    fig = px.scatter_3d(
        plot_df,
        x='x', y='y', z='z',
        color='cluster_name',
        color_discrete_map={format_cluster_label(k): v for k, v in color_map.items()},
        title="Well Clustering - 3D Projection",
        labels={'x': 'UMAP Dim 1', 'y': 'UMAP Dim 2', 'z': 'UMAP Dim 3'}
    )
    
    fig.update_layout(
        width=800,
        height=600
    )
    
    return fig

def get_plot_download_data(fig: go.Figure, format: str = 'png') -> bytes:
    """
    Generate plot download data.
    
    Args:
        fig: Plotly figure
        format: Export format ('png', 'html', 'svg')
        
    Returns:
        Bytes data for download
    """
    if not PLOTTING_AVAILABLE or fig is None:
        return b""
    
    if format == 'html':
        return fig.to_html().encode('utf-8')
    elif format == 'png':
        return fig.to_image(format='png', engine='kaleido')
    elif format == 'svg':
        return fig.to_image(format='svg', engine='kaleido').encode('utf-8')
    else:
        return b""