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
from typing import Dict, List, Optional

from src.visualization_config import PUBLICATION_CONFIG
from src.visualization.plots import save_plot

logger = logging.getLogger(__name__)

# Apply global style settings from the configuration
plt.rcParams.update(PUBLICATION_CONFIG.get('figure_style', {}))

def plot_metric_timeseries_multi_tenant(df: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """Plots multi-tenant time series using the centralized configuration."""
    subset_mask = (df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask].copy()
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
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask].copy()
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
        
        clean_phase_name = phase.strip()
        base_phase_name = clean_phase_name.split(' - ', 1)[-1]
        normalized_base_name = base_phase_name.lower().replace(' ', '-')

        color = phase_colors.get(clean_phase_name)
        if not color:
            color = phase_colors.get(normalized_base_name)
        if not color:
            logger.warning(f"No color found for phase '{clean_phase_name}' or base name '{normalized_base_name}'. Defaulting.")
            color = '#dddddd'

        display_name = phase_display_names.get(clean_phase_name, 
                                               phase_display_names.get(base_phase_name, 
                                                                       phase_display_names.get(normalized_base_name, clean_phase_name)))
        
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
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
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
    subset_mask = (df['metric_name'] == metric) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
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
