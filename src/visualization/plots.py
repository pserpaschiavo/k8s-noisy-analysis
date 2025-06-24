"""
Module: src.visualization.plots
Description: Centralized plotting functions for the analysis pipeline, using a unified configuration.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import networkx as nx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from typing import Dict, List, Optional

from src.visualization_config import PUBLICATION_CONFIG

logger = logging.getLogger(__name__)

# Apply global style settings from the configuration
plt.rcParams.update(PUBLICATION_CONFIG.get('figure_style', {}))


def save_plot(fig, out_path: str):
    """Saves a matplotlib figure to a file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        dpi = PUBLICATION_CONFIG.get('figure_style', {}).get('figure.dpi', 300)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {out_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to save plot to {out_path}: {e}")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, method: str = 'pearson') -> str | None:
    """Plots a correlation heatmap using the centralized configuration."""
    if corr_matrix.empty:
        logger.warning(f"Empty correlation matrix for {metric}, {phase}, {round_id}")
        return None

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric})
    metric_name = metric_info['name']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    cmap = PUBLICATION_CONFIG['heatmap_colormaps'].get('correlation', 'vlag')

    fig, ax = plt.subplots()
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, 1)] = True

    sns.heatmap(
        corr_matrix, annot=True, cmap=cmap, vmin=-1, vmax=1, center=0, square=True,
        linewidths=0.5, mask=mask, cbar_kws={"label": f"{method.title()} Correlation", "shrink": 0.8}, ax=ax
    )

    title = f'{method.title()} Correlation: {metric_name}\n{phase_display} - Round {round_id}'
    ax.set_title(title, fontweight='bold')
    out_path = os.path.join(out_dir, f"correlation_heatmap_{method}_{metric}_{phase}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


def plot_covariance_heatmap(cov_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str) -> str | None:
    """
    Plots a heatmap of the covariance matrix between tenants for a given metric, phase, and round.
    """
    if cov_matrix.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, cmap='crest', center=0, square=True, linewidths=0.5, cbar_kws={"label": "Covariance"}, ax=ax)
    ax.set_title(f'Covariance between tenants\n{metric} - {phase} - {round_id}')
    
    out_path = os.path.join(out_dir, f"covariance_heatmap_{metric}_{phase}_{round_id}.png")
    save_plot(fig, out_path)

    return out_path


def plot_ccf(ccf_dict: dict, metric: str, phase: str, round_id: str, out_dir: str, max_lag: int = 20) -> list:
    """
    Plots cross-correlation (CCF) for pairs of tenants.
    
    Args:
        ccf_dict: Dictionary with CCF results
        metric: Name of the metric
        phase: Experimental phase
        round_id: Round ID
        out_dir: Output directory
        max_lag: Maximum number of lags used in the calculation
        
    Returns:
        List of paths to the generated plots
    """
    if not ccf_dict:
        logger.warning(f"No CCF data to plot: {metric}, {phase}, {round_id}")
        return []
    
    out_paths = []
    for (tenant1, tenant2), ccf_vals in ccf_dict.items():
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Adjust lags to center around zero
        lags = np.arange(-max_lag, max_lag + 1)
        ax.stem(lags, ccf_vals, linefmt='b-', markerfmt='bo', basefmt='r-')
        
        # Add reference lines
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.6)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.6)
        
        # Find the lag with the highest correlation (in absolute value)
        max_idx = np.argmax(np.abs(ccf_vals))
        max_lag_val = lags[max_idx]
        max_corr = ccf_vals[max_idx]
        
        # Mark the correlation peak
        ax.plot(max_lag_val, max_corr, 'ro', markersize=10)
        ax.annotate(f'Max: {max_corr:.3f} @ lag {max_lag_val}', 
                   xy=(max_lag_val, max_corr),
                   xytext=(max_lag_val + 1, max_corr + 0.05),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=10)
        
        # Add directional interpretation
        direction = ""
        if max_lag_val > 0:
            direction = f"{tenant1} → {tenant2}"
        elif max_lag_val < 0:
            direction = f"{tenant2} → {tenant1}"
        else:
            direction = "Contemporaneous"
        
        ax.set_title(f'Cross-correlation between {tenant1} and {tenant2}\n{metric} - {phase} - {round_id}\nRelationship: {direction}', 
                 fontsize=12, fontweight='bold')
        ax.set_xlabel('Lag')
        ax.set_ylabel('Cross-correlation')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add confidence bands (approximation with 1.96/sqrt(N))
        n_samples = 2 * max_lag + 1  # Approximation
        conf_interval = 1.96 / np.sqrt(n_samples)
        ax.axhspan(-conf_interval, conf_interval, alpha=0.2, color='gray')
        
        out_path = os.path.join(out_dir, f"ccf_{tenant1}_{tenant2}_{metric}_{phase}_{round_id}.png")
        save_plot(fig, out_path)
        out_paths.append(out_path)
    
    return out_paths


def plot_lag_scatter(df: pd.DataFrame, metric: str, phase: str, round_id: str, tenant1: str, tenant2: str, lag: int, out_dir: str) -> str | None:
    """
    Plots a scatter diagram with lag between two tenants.
    
    Args:
        df: DataFrame in long format
        metric: Name of the metric
        phase: Experimental phase
        round_id: Round ID
        tenant1: First tenant (x-axis)
        tenant2: Second tenant (y-axis)
        lag: Lag in periods (>0: tenant1 leads, <0: tenant2 leads)
        out_dir: Output directory
        
    Returns:
        Path to the generated plot or None
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for lag scatter: {metric}, {phase}, {round_id}")
        return None
    
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    if tenant1 not in wide.columns or tenant2 not in wide.columns:
        logger.warning(f"Tenants {tenant1} or {tenant2} not found in the data")
        return None
    
    # Handle missing data
    if wide.isna().any().any():
        wide = wide.interpolate(method='linear')
    
    ts1 = wide[tenant1]
    ts2 = wide[tenant2]
    
    if lag > 0:
        # tenant1 leads (tenant1 at t-lag influences tenant2 at t)
        paired_data = pd.DataFrame({
            tenant1: ts1[:-lag].values,
            tenant2: ts2[lag:].values
        })
        title = f'Lag scatter ({lag} periods): {tenant1} → {tenant2}'
    elif lag < 0:
        # tenant2 leads (tenant2 at t+lag influences tenant1 at t)
        lag_abs = abs(lag)
        paired_data = pd.DataFrame({
            tenant1: ts1[lag_abs:].values,
            tenant2: ts2[:-lag_abs].values
        })
        title = f'Lag scatter ({abs(lag)} periods): {tenant2} → {tenant1}'
    else:
        # Contemporary (lag = 0)
        paired_data = pd.DataFrame({
            tenant1: ts1.values,
            tenant2: ts2.values
        })
        title = f'Contemporary scatter: {tenant1} vs {tenant2}'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Calculate correlation on the paired data
    correlation = paired_data.corr().iloc[0, 1]
    
    # Scatter plot with trend line
    sns.regplot(x=tenant1, y=tenant2, data=paired_data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    
    ax.set_title(f'{title}\n{metric} - {phase} - {round_id}\nCorrelation: {correlation:.3f}', fontsize=12, fontweight='bold')
    ax.set_xlabel(f'{tenant1} (t{"-"+str(lag) if lag > 0 else ""})')
    ax.set_ylabel(f'{tenant2} (t{"+"+str(abs(lag)) if lag < 0 else ""})')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    out_path = os.path.join(out_dir, f"lag_scatter_{tenant1}_{tenant2}_{lag}_{metric}_{phase}_{round_id}.png")
    save_plot(fig, out_path)
    
    return out_path

# Causality Plots

def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', threshold_mode: str = 'less'):
    """Plots a causality graph using the centralized configuration."""
    if causality_matrix.empty:
        logger.warning(f"Causality matrix is empty for {metric}. Skipping plot.")
        return

    G = nx.DiGraph() if directed else nx.Graph()
    tenants = causality_matrix.index.tolist()
    G.add_nodes_from(tenants)

    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    node_colors: list[str] = [PUBLICATION_CONFIG['tenant_colors'].get(t, '#8c564b') for t in tenants]
    node_labels = {t: tenant_display_names.get(t, t) for t in tenants}

    edges, edge_weights, edge_labels = [], [], {}
    edge_weights: list[float] = []
    for src in tenants:
        for tgt in tenants:
            if src == tgt: continue
            val = causality_matrix.at[src, tgt]
            is_significant, weight = False, 0
            if threshold_mode == 'less' and not pd.isna(val) and float(val) < threshold:
                is_significant, weight = True, 1 - float(val)
            elif threshold_mode == 'greater' and not pd.isna(val) and float(val) > threshold:
                is_significant, weight = True, float(val)
            
            if is_significant:
                G.add_edge(src, tgt, weight=weight)
                edges.append((src, tgt))
                edge_weights.append(weight * 5 + 1)
                edge_labels[(src, tgt)] = f"{weight:.2f}"

    if not G.nodes:
        logger.warning(f"No nodes in graph for {metric}. Skipping plot.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=50)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#f8f8f8')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, node_shape='o', edgecolors='darkblue', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight='bold', font_color='white', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->' if directed else '-', width=edge_weights, edge_color='gray', alpha=0.8, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7), ax=ax)

    metric_name = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric})['name']
    title = f'Causality: {metric_name} (Edges: {threshold_mode.replace("less", "p <").replace("greater", "TE >")} {threshold:.2g})'
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

    legend_patches = [mpatches.Patch(color=c, label=l) for t, l, c in zip(tenants, node_labels.values(), node_colors)]
    ax.legend(handles=legend_patches, title="Tenants", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    save_plot(fig, out_path)
    return out_path


def plot_causality_heatmap(causality_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, method: str, value_type: str = 'p-value') -> str | None:
    """Plots a causality heatmap using the centralized configuration."""
    if causality_matrix.empty:
        logger.warning(f"Empty causality matrix for {metric}, {phase}, {round_id}")
        return None

    metric_display = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric})['name']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    cmap = PUBLICATION_CONFIG['heatmap_colormaps'].get(value_type, 'viridis')
    cbar_label = f"{method} {value_type.replace('_', ' ').title()}"

    fig, ax = plt.subplots()
    sns.heatmap(causality_matrix, annot=True, cmap=cmap, fmt=".3f", linewidths=0.5, cbar_kws={"label": cbar_label}, ax=ax)

    title = f'{method} Causality: {metric_display}\n{phase_display} - Round {round_id}'
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Caused by (Source Tenant)")
    ax.set_ylabel("Affected (Target Tenant)")
    
    out_path = os.path.join(out_dir, f"{method.lower()}_causality_heatmap_{metric}_{phase}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


# Descriptive Analysis Plots

def plot_metric_timeseries_multi_tenant(df: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """Plots multi-tenant time series using the centralized configuration."""
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for time series: {metric}, {phase}, {round_id}")
        return None

    subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_markers = PUBLICATION_CONFIG['tenant_markers']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']

    fig, ax = plt.subplots()
    phase_start = subset['timestamp'].min()

    for tenant_id in sorted(subset['tenant_id'].unique()):
        group = subset[subset['tenant_id'] == tenant_id].sort_values('timestamp')
        if group.empty: continue
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
        ax.plot(elapsed, group['metric_value'], 
                marker=tenant_markers.get(tenant_id, 'x'), 
                linestyle='-', 
                color=tenant_colors.get(tenant_id, '#7f7f7f'),
                label=tenant_display_names.get(tenant_id, tenant_id))

    ax.set_title(f'Time Series: {metric_info["name"]} - {phase_display} (Round {round_id})', fontweight='bold')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(f'{metric_info["name"]} ({metric_info["unit"]})' if metric_info["unit"] else metric_info["name"])
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


def plot_metric_timeseries_multi_tenant_all_phases(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """Plots a single time series for a metric, with phases indicated by shaded regions."""
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for multi-phase plot: {metric}, {round_id}.")
        return None

    subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    subset = subset.sort_values('timestamp')

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_markers = PUBLICATION_CONFIG['tenant_markers']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_colors = PUBLICATION_CONFIG['phase_colors']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    fig, ax = plt.subplots(figsize=(15, 7))
    round_start_time = subset['timestamp'].min()

    # Plot tenant data first
    legend_handles = []
    tenants = sorted(subset['tenant_id'].unique())
    for tenant_id in tenants:
        group = subset[subset['tenant_id'] == tenant_id]
        if group.empty: continue
        
        elapsed = (group['timestamp'] - round_start_time).dt.total_seconds()
        color = tenant_colors.get(tenant_id, '#7f7f7f')
        marker = tenant_markers.get(tenant_id, 'x')
        label = tenant_display_names.get(tenant_id, tenant_id)
        
        ax.plot(elapsed, group['metric_value'], marker=marker, linestyle='-', color=color, label=label)

    # Create shaded regions for phases
    phase_patches = []
    # Ensure phases are sorted chronologically for correct plotting
    phases = sorted(subset['experimental_phase'].unique(), 
                    key=lambda p: subset[subset['experimental_phase'] == p]['timestamp'].min())

    for phase in phases:
        phase_df = subset[subset['experimental_phase'] == phase]
        if phase_df.empty:
            continue
        start_time = (phase_df['timestamp'].min() - round_start_time).total_seconds()
        end_time = (phase_df['timestamp'].max() - round_start_time).total_seconds()
        
        # Clean up phase name to handle potential inconsistencies.
        clean_phase_name = phase.strip()
        
        # The config keys might be the base name (e.g., "Baseline") 
        # while the data has a numeric prefix (e.g., "1 - Baseline").
        # We try to match the full name first, then the base name.
        base_phase_name = clean_phase_name.split(' - ', 1)[-1]
        
        # Normalize the base name to match config keys (e.g., "CPU Noise" -> "cpu-noise")
        normalized_base_name = base_phase_name.lower().replace(' ', '-')

        # Determine color by trying the full name, then the normalized base name.
        color = phase_colors.get(clean_phase_name)
        if not color:
            color = phase_colors.get(normalized_base_name)
        if not color:
            logger.warning(f"No color found for phase '{clean_phase_name}' or base name '{normalized_base_name}'. Defaulting.")
            color = '#dddddd'

        # Determine display name for the legend, trying full, base, and normalized names.
        display_name = phase_display_names.get(clean_phase_name, 
                                               phase_display_names.get(base_phase_name, 
                                                                       phase_display_names.get(normalized_base_name, clean_phase_name)))
        
        # Draw the shaded region with slightly higher alpha for better visibility.
        ax.axvspan(start_time, end_time, color=color, alpha=0.35, lw=0)
        phase_patches.append(mpatches.Patch(color=color, alpha=0.35, label=display_name))

    # Combine legends
    tenant_legend = ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(tenant_legend)
    ax.legend(handles=phase_patches, title="Phase", bbox_to_anchor=(1.05, 0), loc='lower left')

    ax.set_title(f'Time Series: {metric_info["name"]} - All Phases (Round {round_id})', fontweight='bold')
    ax.set_xlabel("Time (seconds from round start)")
    ax.set_ylabel(f'{metric_info["name"]} ({metric_info["unit"]})' if metric_info["unit"] else metric_info["name"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    out_path = os.path.join(out_dir, f"timeseries_all_phases_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


def plot_metric_barplot_by_phase(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """Generates a bar plot using the centralized configuration."""
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for bar plot: {metric}, {round_id}. Check input dataframe.")
        return None

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    color_palette = {tenant_display_names.get(k, k): v for k, v in PUBLICATION_CONFIG['tenant_colors'].items()}

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x='experimental_phase', y='metric_value', hue='tenant_id', palette=color_palette, errorbar='sd', capsize=0.1, ax=ax)

    ax.set_title(f'Mean {metric_info["name"]} by Phase (Round {round_id})', fontweight='bold')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'Mean {metric_info["name"]} ({metric_info["unit"]})' if metric_info["unit"] else f'Mean {metric_info["name"]}')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    plt.subplots_adjust(bottom=0.2)
    out_path = os.path.join(out_dir, f"barplot_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


def plot_metric_boxplot(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """Generates a box plot using the centralized configuration."""
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for box plot: {metric}, {round_id}. Check input dataframe.")
        return None

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    color_palette = {tenant_display_names.get(k, k): v for k, v in PUBLICATION_CONFIG['tenant_colors'].items()}

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=plot_df, x='experimental_phase', y='metric_value', hue='tenant_id', palette=color_palette, showfliers=False, ax=ax)

    ax.set_title(f'Distribution of {metric_info["name"]} by Phase (Round {round_id})', fontweight='bold')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'{metric_info["name"]} ({metric_info["unit"]})' if metric_info["unit"] else metric_info["name"])
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--' , linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    plt.subplots_adjust(bottom=0.2)
    out_path = os.path.join(out_dir, f"boxplot_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str) -> str | None:
    """Plots time series with highlighted anomalies using the centralized configuration."""
    if anomalies.empty:
        logger.info(f"No anomalies to plot for: {metric}, {phase}, {round_id}")
        return None

    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data to plot anomalies for: {metric}, {phase}, {round_id}")
        return None

    subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'], errors='coerce')

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']

    fig, ax = plt.subplots()
    phase_start = subset['timestamp'].min()

    for tenant_id in sorted(subset['tenant_id'].unique()):
        group = subset[subset['tenant_id'] == tenant_id].sort_values('timestamp')
        if group.empty: continue
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
        ax.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', 
                color=tenant_colors.get(tenant_id, '#7f7f7f'), 
                label=tenant_display_names.get(tenant_id, tenant_id), alpha=0.7)

    for tenant_id in sorted(anomalies['tenant_id'].unique()):
        anomaly_group = anomalies[anomalies['tenant_id'] == tenant_id]
        if anomaly_group.empty: continue
        elapsed = (anomaly_group['timestamp'] - phase_start).dt.total_seconds()
        ax.scatter(elapsed, anomaly_group['metric_value'], color='red', s=100, marker='X', 
                   edgecolors='black', linewidth=1, zorder=10)

    ax.scatter([], [], color='red', s=100, marker='X', edgecolors='black', linewidth=1, label='Anomaly')
    ax.set_title(f'Anomalies in {metric_info["name"]} - {phase_display} (Round {round_id})', fontweight='bold')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(f'{metric_info["name"]} ({metric_info["unit"]})' if metric_info["unit"] else metric_info["name"])
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Tenant / Event', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"anomalies_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path


def generate_consolidated_boxplot(
    df_long: pd.DataFrame,
    metric: str,
    output_dir: str
) -> Optional[str]:
    """
    Gera um boxplot consolidado para uma métrica, comparando fases experimentais
    entre todos os rounds.

    Args:
        df_long: DataFrame em formato long com dados de todos os rounds.
        metric: A métrica a ser plotada.
        output_dir: Diretório para salvar o gráfico.

    Returns:
        Caminho do arquivo de imagem do gráfico gerado ou None em caso de erro.
    """
    logger.info(f"Gerando boxplot consolidado para a métrica: {metric}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_df = df_long[df_long['metric_name'] == metric]
    
    if metric_df.empty:
        logger.warning(f"Não há dados para a métrica '{metric}'. O boxplot não será gerado.")
        plt.close(fig)
        return None

    sns.boxplot(data=metric_df, x='experimental_phase', y='metric_value', hue='tenant_id', ax=ax)
    
    ax.set_title(f"Boxplot Consolidado para {metric} (Todos os Rounds)")
    ax.set_xlabel("Fase Experimental")
    ax.set_ylabel("Valor da Métrica")
    plt.xticks(rotation=45)
    
    output_path = os.path.join(output_dir, f"consolidated_boxplot_{metric}.png")
    save_plot(fig, output_path)
    return output_path

def generate_consolidated_heatmap(
    aggregated_matrix: pd.DataFrame,
    output_dir: str,
    title: str,
    filename: str
) -> Optional[str]:
    """
    Gera um heatmap consolidado a partir de uma matriz agregada.

    Args:
        aggregated_matrix: Matriz de dados agregados (e.g., Jaccard, Spearman).
        output_dir: Diretório para salvar o gráfico.
        title: Título do gráfico.
        filename: Nome do arquivo de saída.

    Returns:
        Caminho do arquivo de imagem do gráfico gerado ou None em caso de erro.
    """
    logger.info(f"Gerando heatmap consolidado para: {title}")
    
    if aggregated_matrix.empty:
        logger.warning(f"Matriz agregada vazia para '{title}'. O heatmap não será gerado.")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(aggregated_matrix, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel("Par de Rounds")
    ax.set_ylabel("Métrica")
    
    output_path = os.path.join(output_dir, filename)
    save_plot(fig, output_path)
    return output_path
