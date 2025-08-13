"""
Module: src.visualization.descriptive_plots
Description: Plotting functions for descriptive analysis.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import matplotlib.patches as mpatches
import re
from typing import Dict, List, Optional

from src.config import PipelineConfig
from src.visualization_config import PUBLICATION_CONFIG
from src.visualization.plots import save_plot

logger = logging.getLogger(__name__)

# Apply global style settings from the configuration
plt.rcParams.update(PUBLICATION_CONFIG.get('figure_style', {}))

def plot_metric_timeseries_multi_tenant(df: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, config: PipelineConfig):
    """Plots multi-tenant time series using the centralized configuration."""
    subset_mask = (df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask].copy()
    if subset.empty:
        logger.warning(f"No data for time series: {metric}, {phase}, {round_id}")
        return None

    subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Get display name from global config
    metric_info = PUBLICATION_CONFIG.get('metric_display_names', {}).get(metric)
    if isinstance(metric_info, dict):
        metric_display_name = metric_info.get('name', metric)
    else:
        metric_display_name = metric_info or metric
    
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

    ax.set_title(f'Time Series: {metric_display_name} - {phase_display} (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(metric_display_name)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path

def plot_metric_timeseries_multi_tenant_all_phases(df: pd.DataFrame, metric: str, round_id: str, out_dir: str, config: PipelineConfig):
    """Plots a single time series for a metric, with phases indicated by shaded regions."""
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask].copy()
    if subset.empty:
        logger.warning(f"No data for multi-phase plot: {metric}, {round_id}.")
        return None

    subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    subset = subset.sort_values('timestamp')

    metric_info = PUBLICATION_CONFIG.get('metric_display_names', {}).get(metric)
    if isinstance(metric_info, dict):
        metric_display_name = metric_info.get('name', metric)
    else:
        metric_display_name = metric_info or metric
    
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_markers = PUBLICATION_CONFIG['tenant_markers']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_colors = PUBLICATION_CONFIG['phase_colors']
    phase_hatches = PUBLICATION_CONFIG.get('phase_hatches', {})
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

    # Helper: map various phase label formats to canonical config keys
    canonical_by_number = {
        1: '1 - Baseline',
        2: '2 - CPU Noise',
        3: '3 - Memory Noise',
        4: '4 - Network Noise',
        5: '5 - Disk Noise',
        6: '6 - Combined Noise',
        7: '7 - Recovery',
    }

    for phase in phases:
        phase_df = subset[subset['experimental_phase'] == phase]
        if phase_df.empty:
            continue
        start_time = (phase_df['timestamp'].min() - round_start_time).total_seconds()
        end_time = (phase_df['timestamp'].max() - round_start_time).total_seconds()

        # Guard against zero-width spans
        if not np.isfinite(start_time) or not np.isfinite(end_time):
            logger.warning(f"Invalid start/end time for phase '{phase}'. Skipping shading.")
            continue
        if end_time <= start_time:
            end_time = start_time + 1.0  # minimal visible width

        clean_phase_name = str(phase).strip()

        # Try to parse leading number and flexible separator variants like '1-Baseline', '1 - Baseline'
        m = re.match(r"^(\s*(?P<num>\d+)\s*[-–—]?\s*)?(?P<label>.*)$", clean_phase_name)
        phase_num = None
        phase_label = clean_phase_name
        if m:
            num_str = m.group('num')
            if num_str:
                try:
                    phase_num = int(num_str)
                except Exception:
                    phase_num = None
            lbl = m.group('label').strip()
            if lbl:
                phase_label = lbl

        # Build candidate keys for color/display mapping
        candidates: List[str] = []
        if phase_num in canonical_by_number:
            candidates.append(canonical_by_number[phase_num])  # canonical "X - Name"
        candidates.extend([
            clean_phase_name,
            clean_phase_name.replace(' - ', '-'),  # e.g., '1-Baseline'
            clean_phase_name.replace('-', ' - '),  # e.g., '1 - Baseline'
            phase_label,
            phase_label.title(),
            phase_label.replace('-', ' ').title(),
            phase_label.lower().replace(' ', '-'),  # normalized base
        ])

        # Deduplicate while keeping order
        seen = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        # Resolve color
        color = None
        for key in candidates:
            if key in phase_colors:
                color = phase_colors[key]
                break
        if not color and phase_num in canonical_by_number:
            # Last attempt: canonical key by number
            color = phase_colors.get(canonical_by_number[phase_num])
        if not color:
            logger.warning(f"No color found for phase variants {candidates}. Using default shade.")
            color = '#dddddd'

    # Resolve display name
        display_name = None
        for key in candidates:
            if key in phase_display_names:
                display_name = phase_display_names[key]
                break
        if not display_name and phase_num in canonical_by_number:
            display_name = phase_display_names.get(canonical_by_number[phase_num])
        if not display_name:
            # Fallback to cleaned label
            display_name = phase_label

        # Resolve hatch pattern (optional)
        hatch = None
        for key in candidates:
            if key in phase_hatches:
                hatch = phase_hatches[key]
                break
        if hatch is None and phase_num in canonical_by_number:
            hatch = phase_hatches.get(canonical_by_number[phase_num], '')

        # Draw shaded span behind the lines (use bar-like rectangle to support hatch)
        # Note: axvspan doesn't support hatch directly, so draw a Rectangle
        from matplotlib.patches import Rectangle
        rect = Rectangle((start_time, ax.get_ylim()[0]),
                         width=(end_time - start_time),
                         height=(ax.get_ylim()[1] - ax.get_ylim()[0]),
                         facecolor=color,
                         alpha=0.20,
                         edgecolor='none',
                         hatch=hatch or None,
                         zorder=0.4)
        ax.add_patch(rect)

        # Legend patch with same styling
        legend_patch = mpatches.Patch(facecolor=color, edgecolor='black', alpha=0.20, label=display_name, hatch=hatch or None)
        phase_patches.append(legend_patch)

    # Combine legends
    tenant_legend = ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(tenant_legend)
    ax.legend(handles=phase_patches, title="Phase", bbox_to_anchor=(1.05, 0), loc='lower left')

    ax.set_title(f'Time Series: {metric_display_name} - All Phases (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel("Time (seconds from round start)")
    ax.set_ylabel(metric_display_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.autoscale(enable=True, axis='x', tight=True)

    fig.tight_layout(rect=(0, 0, 0.88, 1))
    out_path = os.path.join(out_dir, f"timeseries_all_phases_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path

def plot_metric_barplot_by_phase(df: pd.DataFrame, metric: str, round_id: str, out_dir: str, config: PipelineConfig):
    """Generates a bar plot using the centralized configuration."""
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
    if subset.empty:
        logger.warning(f"No data for bar plot: {metric}, {round_id}. Check input dataframe.")
        return None

    metric_info = PUBLICATION_CONFIG.get('metric_display_names', {}).get(metric)
    if isinstance(metric_info, dict):
        metric_display_name = metric_info.get('name', metric)
    else:
        metric_display_name = metric_info or metric
    
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    color_palette = {tenant_display_names.get(k, k): v for k, v in PUBLICATION_CONFIG['tenant_colors'].items()}

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=plot_df, x='experimental_phase', y='metric_value', hue='tenant_id', palette=color_palette, errorbar='sd', capsize=0.1, ax=ax)

    ax.set_title(f'Mean {metric_display_name} by Phase (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'Mean {metric_display_name}')
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--' , linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    plt.subplots_adjust(bottom=0.2)
    out_path = os.path.join(out_dir, f"barplot_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path

def plot_metric_boxplot(df: pd.DataFrame, metric: str, round_id: str, out_dir: str, config: PipelineConfig):
    """Generates a box plot using the centralized configuration."""
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
    if subset.empty:
        logger.warning(f"No data for box plot: {metric}, {round_id}. Check input dataframe.")
        return None

    metric_info = PUBLICATION_CONFIG.get('metric_display_names', {}).get(metric)
    if isinstance(metric_info, dict):
        metric_display_name = metric_info.get('name', metric)
    else:
        metric_display_name = metric_info or metric
    
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    color_palette = {tenant_display_names.get(k, k): v for k, v in PUBLICATION_CONFIG['tenant_colors'].items()}

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.boxplot(data=plot_df, x='experimental_phase', y='metric_value', hue='tenant_id', palette=color_palette, showfliers=False, ax=ax)

    ax.set_title(f'Distribution of {metric_display_name} by Phase (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(metric_display_name)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--' , linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    plt.subplots_adjust(bottom=0.25)
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

    metric_info = PUBLICATION_CONFIG.get('metric_display_names', {}).get(metric)
    if isinstance(metric_info, dict):
        metric_display_name = metric_info.get('name', metric)
    else:
        metric_display_name = metric_info or metric
    
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
    ax.set_title(f'Anomalies in {metric_display_name} - {phase_display} (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(metric_display_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Tenant / Event', bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    out_path = os.path.join(out_dir, f"anomalies_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path

def plot_metric_distribution_by_phase(df: pd.DataFrame, metric: str, round_id: str, out_dir: str, config: PipelineConfig):
    """Generates a distribution plot (violin or boxen) using the centralized configuration."""
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
    if subset.empty:
        logger.warning(f"No data for distribution plot: {metric}, {round_id}. Check input dataframe.")
        return None

    metric_display_names = config.get_metric_display_names()
    metric_display_name = metric_display_names.get(metric, metric)

    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    color_palette = {tenant_display_names.get(k, k): v for k, v in PUBLICATION_CONFIG['tenant_colors'].items()}

    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.violinplot(data=plot_df, x='experimental_phase', y='metric_value', hue='tenant_id', palette=color_palette, inner='quartile', ax=ax)

    ax.set_title(f'Distribution of {metric_display_name} by Phase (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(metric_display_name)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--' , linewidth=0.5)
    fig.tight_layout(rect=(0, 0, 0.85, 1))
    plt.subplots_adjust(bottom=0.25)
    out_path = os.path.join(out_dir, f"distribution_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path

def plot_metric_heatmap(df: pd.DataFrame, metric: str, round_id: str, out_dir: str, config: PipelineConfig):
    """Generates a heatmap of metric values over time for all tenants."""
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
    if subset.empty:
        logger.warning(f"No data for heatmap: {metric}, {round_id}. Check dataframe.")
        return None

    metric_display_names = config.get_metric_display_names()
    metric_display_name = metric_display_names.get(metric, metric)

    # Pivot data
    pivot_df = subset.pivot_table(index='tenant_id', columns='timestamp', values='metric_value')
    if pivot_df.empty:
        logger.warning(f"Pivot table is empty for heatmap: {metric}, {round_id}.")
        return None

    fig, ax = plt.subplots(figsize=(18, 8))
    sns.heatmap(pivot_df, ax=ax, cmap="viridis", cbar_kws={'label': metric_display_name})

    ax.set_title(f'Heatmap of {metric_display_name} Over Time (round_id.capitalize())', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Tenant')
    ax.tick_params(axis='x', rotation=90)

    fig.tight_layout()
    out_path = os.path.join(out_dir, f"heatmap_{metric}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path
