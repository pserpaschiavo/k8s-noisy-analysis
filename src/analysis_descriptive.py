"""
Module: analysis_descriptive.py
Description: Descriptive statistics and plotting utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from functools import lru_cache
import logging

from src.visualization_config import PUBLICATION_CONFIG

# Setup logging
logger = logging.getLogger(__name__)

# Use the style from config
# plt.style.use('tableau-colorblind10') # Removed to use centralized config


def compute_descriptive_stats(df, groupby_cols=None) -> pd.DataFrame:
    """
    Compute descriptive statistics (count, mean, std, min, max, skewness, kurtosis) for metric_value,
    grouped by the specified columns.
    
    Args:
        df: DataFrame com os dados a serem analisados
        groupby_cols: List of columns to group by
        
    Returns:
        DataFrame with descriptive statistics
    """
    
    if groupby_cols is None:
        groupby_cols = ['tenant_id', 'metric_name', 'experimental_phase', 'round_id']
    
    # More comprehensive statistics including skewness and kurtosis
    stats = df.groupby(groupby_cols)['metric_value'].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('skewness', lambda x: x.skew()),
        ('kurtosis', lambda x: x.kurtosis())
    ]).reset_index()
    
    logger.info(f"Computed descriptive stats for {len(groupby_cols)} groups with {len(stats)} rows")
    return stats


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
    
    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    # Get standardized names, units, and colors
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']

    # Find the start timestamp of the phase
    phase_start = subset['timestamp'].min()
    
    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    tenants = sorted(subset['tenant_id'].unique())

    for tenant_id in tenants:
        group = subset[subset['tenant_id'] == tenant_id].sort_values('timestamp')
        if group.empty:
            continue
        
        # Calculate elapsed time in seconds from the phase start
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
        
        display_name = tenant_display_names.get(tenant_id, tenant_id)
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        
        ax.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', label=display_name, color=color)
    
    ax.set_title(f'Time Series - {metric_name} - {phase_display} (Round {round_id})')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust for legend
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Multi-tenant time series plot saved to {out_path}")
    return out_path


def plot_metric_barplot_by_phase(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Generates a bar plot of a metric, grouped by phase and tenant,
    using the centralized academic publication configuration.

    Args:
        df: DataFrame with the data to be analyzed.
        metric: Name of the metric to plot.
        round_id: ID of the round to analyze.
        out_dir: Directory to save the plot.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for bar plot of {metric} in {round_id}")
        return None

    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    # Get standardized names, units, and colors
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Map internal names to display names for plotting
    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    # Define order for plotting
    phase_order = [phase_display_names[p] for p in PUBLICATION_CONFIG['phase_display_names'] if phase_display_names[p] in plot_df['experimental_phase'].unique()]
    tenant_order = [tenant_display_names[t] for t in PUBLICATION_CONFIG['tenant_display_names'] if tenant_displayNames[t] in plot_df['tenant_id'].unique()]

    plt.figure(figsize=(14, 7))
    ax = sns.barplot(
        data=plot_df,
        x='experimental_phase',
        y='metric_value',
        hue='tenant_id',
        palette=tenant_colors,
        order=phase_order,
        hue_order=tenant_order,
        errorbar='sd',  # Show standard deviation
        capsize=0.1
    )

    ax.set_title(f'Mean {metric_name} by Phase and Tenant (Round {round_id})')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'Mean {metric_name} ({metric_unit})' if metric_unit else f'Mean {metric_name}')
    ax.tick_params(axis='x', rotation=45)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust for legend
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"barplot_{metric}_{round_id}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Bar plot saved to {out_path}")
    return out_path


def plot_metric_timeseries_multi_tenant_all_phases(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Plots time series for all tenants across all phases in a single plot,
    using the centralized academic publication configuration. Phases are
    distinguished by background colors and annotations.

    Args:
        df: DataFrame with the data.
        metric: Name of the metric to plot.
        round_id: ID of the round to analyze.
        out_dir: Directory to save the plot.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)].copy()
    if subset.empty:
        logger.warning(f"No data for all-phases time series of {metric} in {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')

    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    # Get standardized names, units, and colors
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_colors = PUBLICATION_CONFIG['phase_colors']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Sort by phase order and timestamp
    phase_order = list(phase_display_names.keys())
    subset['phase_order'] = pd.Categorical(subset['experimental_phase'], categories=phase_order, ordered=True)
    subset = subset.sort_values(['phase_order', 'timestamp'])
    
    tenants = sorted(subset['tenant_id'].unique())
    
    plt.figure(figsize=(18, 8))
    ax = plt.gca()
    
    legend_handles = []
    time_offset = 0
    last_time = 0

    # Create legend handles first
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
        
        # Add phase background shading and annotation
        current_phase_end = time_offset + phase_duration
        ax.axvspan(time_offset, current_phase_end, color=phase_colors.get(phase_key, '#f0f0f0'), alpha=0.2, zorder=0)
        
        # Annotate phase name
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

    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust for legend
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_allphases_{round_id}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"All-phases time series plot saved to {out_path}")
    return out_path


def plot_metric_boxplot(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Generates a grouped box plot of a metric by phase and tenant,
    using the centralized academic publication configuration.

    Args:
        df: DataFrame with the data.
        metric: Name of the metric to plot.
        round_id: ID of the round to analyze.
        out_dir: Directory to save the plot.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for box plot of {metric} in {round_id}")
        return None

    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    # Get standardized names, units, and colors
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Map internal names to display names for plotting
    plot_df = subset.copy()
    plot_df['tenant_id'] = plot_df['tenant_id'].map(tenant_display_names).fillna(plot_df['tenant_id'])
    plot_df['experimental_phase'] = plot_df['experimental_phase'].map(phase_display_names).fillna(plot_df['experimental_phase'])

    # Define order for plotting from the config, if available
    phase_order = PUBLICATION_CONFIG.get('phase_order', [])
    phase_order = [phase_display_names.get(p, p) for p in phase_order]
    
    tenant_order = PUBLICATION_CONFIG.get('tenant_order', [])
    tenant_order = [tenant_display_names.get(t, t) for t in tenant_order]

    plt.figure(figsize=(16, 8))
    ax = sns.boxplot(
        data=plot_df,
        x='experimental_phase',
        y='metric_value',
        hue='tenant_id',
        palette=tenant_colors,
        order=phase_order if phase_order else sorted(plot_df['experimental_phase'].unique()),
        hue_order=tenant_order if tenant_order else sorted(plot_df['tenant_id'].unique()),
        showfliers=False
    )

    ax.set_title(f'Box Plot of {metric_name} by Phase and Tenant (Round {round_id})')
    ax.set_xlabel('Experimental Phase')
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.tick_params(axis='x', rotation=45, ha="right")
    
    ax.legend(title='Tenant', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)

    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust for legend
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"boxplot_{metric}_{round_id}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Box plot saved to {out_path}")
    return out_path


def detect_anomalies(df: pd.DataFrame, metric: str, phase: str, round_id: str, window_size: int = 10, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detecta anomalias nas séries temporais usando rolling window e Z-score.
    
    Args:
        df: DataFrame em formato long
        metric: Nome da métrica para analisar
        phase: Fase experimental para filtrar
        round_id: ID do round para filtrar
        window_size: Tamanho da janela para médias móveis
        threshold: Limiar de Z-score para considerar um ponto como anomalia
        
    Returns:
        DataFrame com as anomalias detectadas
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)].copy()
    if subset.empty:
        logger.warning(f"Sem dados para detecção de anomalias: {metric}, {phase}, {round_id}")
        return pd.DataFrame()
    
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Ordenar por timestamp para rolling window
    subset = subset.sort_values(['tenant_id', 'timestamp'])
    
    # Inicializar DataFrame para resultados
    anomalies = pd.DataFrame()
    
    # Processar cada tenant separadamente
    for tenant, group in subset.groupby('tenant_id'):
        # Calcular estatísticas rolling (média e desvio padrão)
        rolling_mean = group['metric_value'].rolling(window=window_size, center=True).mean()
        rolling_std = group['metric_value'].rolling(window=window_size, center=True).std()
        
        # Substituir NaN no início/fim do rolling por médias globais
        rolling_mean = rolling_mean.fillna(group['metric_value'].mean())
        rolling_std = rolling_std.fillna(group['metric_value'].std())
        rolling_std = rolling_std.replace(0, group['metric_value'].std())  # Evitar divisão por zero
        
        # Calcular Z-score
        z_scores = np.abs((group['metric_value'] - rolling_mean) / rolling_std)
        
        # Identificar anomalias
        is_anomaly = z_scores > threshold
        
        # Filtrar anomalias e adicionar ao DataFrame de resultados
        tenant_anomalies = group[is_anomaly].copy()
        if not tenant_anomalies.empty:
            tenant_anomalies['z_score'] = z_scores[is_anomaly]
            anomalies = pd.concat([anomalies, tenant_anomalies])
    
    # Ordenar por gravidade (Z-score)
    if not anomalies.empty:
        anomalies = anomalies.sort_values('z_score', ascending=False)
        logger.info(f"Detectadas {len(anomalies)} anomalias para {metric} em {phase}, {round_id}")
    else:
        logger.info(f"Nenhuma anomalia detectada para {metric} em {phase}, {round_id}")
    
    return anomalies


def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str) -> str | None:
    """
    Plots time series with highlighted anomalies, using the centralized
    academic publication configuration.

    Args:
        df: Complete long-format DataFrame.
        anomalies: DataFrame with detected anomalies.
        metric: Name of the metric.
        phase: Experimental phase.
        round_id: ID of the round.
        out_dir: Output directory for the plot.

    Returns:
        Path to the generated plot or None if no data/anomalies.
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

    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    # Get standardized names, units, and colors
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    tenant_colors = PUBLICATION_CONFIG['tenant_colors']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']

    # Find the start timestamp of the phase
    phase_start = subset['timestamp'].min()

    plt.figure(figsize=(14, 7))
    ax = plt.gca()

    tenants = sorted(subset['tenant_id'].unique())

    # Plot normal time series
    for tenant_id in tenants:
        group = subset[subset['tenant_id'] == tenant_id].sort_values('timestamp')
        if group.empty:
            continue
        
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
        display_name = tenant_display_names.get(tenant_id, tenant_id)
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        
        ax.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', 
                label=display_name, color=color, alpha=0.7)

    # Highlight anomalies
    anomaly_handles = []
    for tenant_id in tenants:
        anomaly_group = anomalies[anomalies['tenant_id'] == tenant_id]
        if anomaly_group.empty:
            continue

        elapsed = (anomaly_group['timestamp'] - phase_start).dt.total_seconds()
        color = tenant_colors.get(tenant_id, tenant_colors['default'])
        
        ax.scatter(elapsed, anomaly_group['metric_value'], color='red', s=100, marker='X', 
                   edgecolors='black', linewidth=1, zorder=10)

    # Create a single legend entry for all anomalies
    if not anomalies.empty:
        ax.scatter([], [], color='red', s=100, marker='X', edgecolors='black', 
                   linewidth=1, label='Anomaly')

    ax.set_title(f'Time Series with Anomalies - {metric_name} - {phase_display} (Round {round_id})')
    ax.set_xlabel("Time (seconds from phase start)")
    ax.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(title='Tenant / Event', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust for legend
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"anomalies_{metric}_{phase.replace(' ', '_')}_{round_id}.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Anomaly plot saved to {out_path}")
    return out_path
