"""
Module: src.visualization.plots
Description: Centralized plotting functions for the analysis pipeline.
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
import matplotlib.cm as cm

from src.visualization_config import PUBLICATION_CONFIG

logger = logging.getLogger(__name__)

# Global plotting settings can be defined here
sns.set_theme(style="whitegrid")
plt.style.use('tableau-colorblind10')


def save_plot(fig, out_path: str, dpi: int = 300):
    """Saves a matplotlib figure to a file."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {out_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to save plot to {out_path}: {e}")


def plot_correlation_heatmap(corr_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, method: str = 'pearson') -> str | None:
    """
    Plots a heatmap of the correlation matrix between tenants for a given metric, phase, and round.
    
    Args:
        corr_matrix: Calculated correlation matrix
        metric: Metric name
        phase: Experimental phase 
        round_id: Round ID
        out_dir: Output directory
        method: Correlation method used ('pearson', 'kendall', or 'spearman')
        
    Returns:
        Path to the generated plot or None if there is no data
    """
    if corr_matrix.empty:
        logger.warning(f"Empty correlation matrix. Cannot generate heatmap for {metric}, {phase}, {round_id}")
        return None
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Improve visualization with masks for the upper triangle
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, 1)] = True
    
    # Enhance the aesthetics of the heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='vlag', 
        vmin=-1, 
        vmax=1, 
        center=0, 
        square=True, 
        linewidths=0.5, 
        mask=mask,
        cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.8},
        ax=ax
    )
    
    ax.set_title(f'{method.title()} correlation between tenants\n{metric} - {phase} - {round_id}', fontsize=12, fontweight='bold')
    
    out_path = os.path.join(out_dir, f"correlation_heatmap_{metric}_{phase}_{round_id}.png")
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

def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', metric_color: str = '', threshold_mode: str = 'less'):
    """
    Plots a causality graph from a causality matrix (e.g., Granger p-values or scores).
    Edges are drawn based on the threshold and threshold_mode.
    - threshold_mode='less': Edge where value < threshold (e.g., for p-values)
    - threshold_mode='greater': Edge where value > threshold (e.g., for TE scores)
    """
    if causality_matrix.empty:
        logger.warning(f"Causality matrix is empty for {metric}. Skipping plot.")
        return None

    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    
    tenants = causality_matrix.index.tolist()
    G.add_nodes_from(tenants)
    edge_labels = {}
    edges = []
    edge_weights = []
    edge_colors = []
    
    metric_palette = {
        'cpu_usage': 'tab:blue',
        'memory_usage': 'tab:orange',
        'disk_io': 'tab:green',
        'network_io': 'tab:red',
    }
    color = metric_color if metric_color else metric_palette.get(metric, 'tab:blue')

    for src in tenants:
        for tgt in tenants:
            if src != tgt:
                val = causality_matrix.at[src, tgt]
                is_significant = False
                weight = 0

                if threshold_mode == 'less':
                    if not pd.isna(val) and float(val) < threshold:
                        is_significant = True
                        weight = 1 - float(val)
                elif threshold_mode == 'greater':
                    if not pd.isna(val) and float(val) > threshold:
                        is_significant = True
                        weight = float(val)

                if is_significant:
                    G.add_edge(src, tgt, weight=weight)
                    edges.append((src, tgt))
                    edge_weights.append(weight * 6 + 1)
                    edge_colors.append(color)
                    edge_labels[(src, tgt)] = f"{weight:.2f}"

    sorted_tenants = sorted(tenants)
    pos = nx.circular_layout(G)

    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_facecolor('#f8f8f8')

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1600, edgecolors='darkblue', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=13, font_weight='bold', font_color='black', ax=ax)

    for edge, w, c in zip(edges, edge_weights, edge_colors):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], arrowstyle='->' if directed else '-', 
                             arrows=directed, width=w, edge_color=c, alpha=0.9,
                             connectionstyle='arc3,rad=0.2', ax=ax)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=11, font_weight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7), ax=ax)

    title = f'Causality Graph ({"Directed" if directed else "Undirected"})\nMetric: {metric} | Edges: {threshold_mode.replace("less", "p <").replace("greater", "TE >")} {threshold:.2g}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axis('off')

    legend_elements = [mlines.Line2D([0], [0], color=color, lw=3, label=f'{metric if metric else "Metric"}')]
    ax.legend(handles=legend_elements, loc='lower left')

    save_plot(fig, out_path)
    return out_path


# Descriptive Analysis Plots

def plot_metric_timeseries_multi_tenant(df: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """
    Plots time series for all tenants for a given metric, phase, and round,
    using the centralized academic publication configuration.
    X-axis: relative time in seconds; Y-axis: metric value; Colors: tenants.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for {metric}, {phase}, {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']

    phase_start = subset['timestamp'].min()
    
    fig, ax = plt.subplots(figsize=(14, 7))

    tenants = sorted(subset['tenant_id'].unique())

    for tenant_id in tenants:
        group = subset[subset['tenant_id'] == tenant_id].sort_values('timestamp')
        if group.empty:
            continue
        
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
        
        display_name = tenant_display_names.get(tenant_id, tenant_id)
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        
        ax.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', label=display_name, color=color)
    
    ax.set_title(f'Time Series - {metric_name} - {phase_display} (Round {round_id})')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    save_plot(fig, out_path)
    
    return out_path

def plot_metric_barplot_by_phase(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Generates a bar plot of a metric, grouped by phase and tenant,
    using the centralized academic publication configuration.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for bar plot of {metric} in {round_id}")
        return None

    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Create a color palette mapped to display names
    color_palette = {
        tenant_display_names.get(k, k): v 
        for k, v in tenant_colors.items()
    }

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    phase_order = [phase_display_names[p] for p in PUBLICATION_CONFIG['phase_display_names'] if phase_display_names.get(p) in plot_df['experimental_phase'].unique()]
    tenant_order = [tenant_display_names[t] for t in PUBLICATION_CONFIG['tenant_display_names'] if tenant_display_names.get(t) in plot_df['tenant_id'].unique()]

    fig, ax = plt.subplots(figsize=(14, 7))
    sns.barplot(
        data=plot_df,
        x='experimental_phase',
        y='metric_value',
        hue='tenant_id',
        palette=color_palette,
        order=phase_order,
        hue_order=tenant_order,
        errorbar='sd',
        capsize=0.1,
        ax=ax
    )

    ax.set_title(f'Mean {metric_name} by Phase and Tenant (Round {round_id})')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'Mean {metric_name} ({metric_unit})' if metric_unit else f'Mean {metric_name}')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"barplot_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    
    return out_path

def plot_metric_timeseries_multi_tenant_all_phases(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Plots time series for all tenants across all phases in a single plot,
    using the centralized academic publication configuration.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)].copy()
    if subset.empty:
        logger.warning(f"No data for all-phases time series of {metric} in {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')

    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_colors = PUBLICATION_CONFIG['phase_colors']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    phase_order = list(phase_display_names.keys())
    subset['phase_order'] = pd.Categorical(subset['experimental_phase'], categories=phase_order, ordered=True)
    subset = subset.sort_values(['phase_order', 'timestamp'])
    
    tenants = sorted(subset['tenant_id'].unique())
    
    fig, ax = plt.subplots(figsize=(18, 8))
    
    legend_handles = []
    time_offset = 0
    last_time = 0

    for tenant_id in tenants:
        display_name = tenant_display_names.get(tenant_id, tenant_id)
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        legend_handles.append(mpatches.Patch(color=color, label=display_name))

    for phase_key in phase_order:
        phase_data = subset[subset['experimental_phase'] == phase_key]
        if phase_data.empty:
            continue

        phase_start_time = phase_data['timestamp'].min()
        phase_end_time = phase_data['timestamp'].max()
        phase_duration = (phase_end_time - phase_start_time).total_seconds()

        for tenant_id in tenants:
            group = phase_data[phase_data['tenant_id'] == tenant_id].sort_values('timestamp')
            if group.empty:
                continue
            
            elapsed = (group['timestamp'] - phase_start_time).dt.total_seconds() + time_offset
            color = tenant_colors.get(tenant_id, tenant_colors['default'])
            ax.plot(elapsed, group['metric_value'], marker='o', markersize=2.5, linestyle='-', color=color, alpha=0.8)
        
        current_phase_end = time_offset + phase_duration
        ax.axvspan(time_offset, current_phase_end, color=phase_colors.get(phase_key, '#f0f0f0'), alpha=0.2, zorder=0)
        
        phase_display = phase_display_names.get(phase_key, phase_key)
        ax.annotate(
            phase_display,
            xy=((time_offset + current_phase_end) / 2, 1.01),
            xycoords=("data", "axes fraction"),
            ha='center', va='bottom', fontsize=11, color='black', fontweight='bold'
        )
        
        time_offset = current_phase_end
        last_time = max(last_time, time_offset)

    ax.set_xlim(0, last_time)
    ax.set_title(f'Time Series - {metric_name} (All Phases, Round {round_id})')
    ax.set_xlabel("Time (seconds, phases concatenated)")
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.legend(handles=legend_handles, title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_allphases_{round_id}.png")
    save_plot(fig, out_path)
    
    return out_path

def plot_metric_boxplot(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Generates a grouped box plot of a metric by phase and tenant,
    using the centralized academic publication configuration.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for box plot of {metric} in {round_id}")
        return None

    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Create a color palette mapped to display names
    color_palette = {
        tenant_display_names.get(k, k): v 
        for k, v in tenant_colors.items()
    }

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    # Robustly determine order from available data
    phase_order = [phase_display_names.get(p, p) for p in PUBLICATION_CONFIG.get('phase_order', []) if phase_display_names.get(p, p) in plot_df['experimental_phase'].unique()]
    if not phase_order:
        phase_order = sorted(plot_df['experimental_phase'].unique())

    tenant_order = [tenant_display_names.get(t, t) for t in PUBLICATION_CONFIG.get('tenant_order', []) if tenant_display_names.get(t, t) in plot_df['tenant_id'].unique()]
    if not tenant_order:
        tenant_order = sorted(plot_df['tenant_id'].unique())


    fig, ax = plt.subplots(figsize=(16, 8))
    sns.boxplot(
        data=plot_df,
        x='experimental_phase',
        y='metric_value',
        hue='tenant_id',
        palette=color_palette,
        order=phase_order,
        hue_order=tenant_order,
        showfliers=False,
        ax=ax
    )

    ax.set_title(f'Box Plot of {metric_name} by Phase and Tenant (Round {round_id})')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"boxplot_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    
    return out_path

def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str) -> str | None:
    """
    Plots time series with highlighted anomalies, using the centralized
    academic publication configuration.
    """
    if anomalies.empty:
        logger.info(f"No anomalies to plot for: {metric}, {phase}, {round_id}")
        return None

    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data to plot anomalies for: {metric}, {phase}, {round_id}")
        return None

    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(anomalies['timestamp']):
        anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'], errors='coerce')

    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']

    phase_start = subset['timestamp'].min()

    fig, ax = plt.subplots(figsize=(14, 7))

    tenants = sorted(subset['tenant_id'].unique())

    for tenant_id in tenants:
        group = subset[subset['tenant_id'] == tenant_id].sort_values('timestamp')
        if group.empty:
            continue
        
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
        display_name = tenant_display_names.get(tenant_id, tenant_id)
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        
        ax.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', 
                label=display_name, color=color, alpha=0.7)

    anomaly_handles = []
    for tenant_id in tenants:
        anomaly_group = anomalies[anomalies['tenant_id'] == tenant_id]
        if anomaly_group.empty:
            continue

        elapsed = (anomaly_group['timestamp'] - phase_start).dt.total_seconds()
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        
        ax.scatter(elapsed, anomaly_group['metric_value'], color='red', s=100, marker='X', 
                   edgecolors='black', linewidth=1, zorder=10)

    if not anomalies.empty:
        ax.scatter([], [], color='red', s=100, marker='X', edgecolors='black', 
                   linewidth=1, label='Anomaly')

    ax.set_title(f'Time Series with Anomalies - {metric_name} - {phase_display} (Round {round_id})')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Tenant / Event', bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"anomalies_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    save_plot(fig, out_path)
    
    return out_path
