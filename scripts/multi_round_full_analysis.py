#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: multi_round_full_analysis.py
Description:
    Runs a consolidated multi-round analysis generating the following artifacts:

    - ECDF per Phase (cumulative distribution)
    - Boxplots per Phase (raw multi-round aggregated metrics)
    - Aggregated Impact per Phase (reused if present)
    - Correlation Heatmaps (mean across rounds per metric and phase)
    - Causality Heatmaps (mean across rounds per metric, phase and method)
    - Causality Consistency (Top-15 most frequent significant links)

    The script consumes per-round output directories (round-1, round-2, ...) of an
    already executed experiment. It does not re-run the pipeline; it only reads
    existing parquet/CSV artifacts and produces new aggregates.

Usage (example):
    python scripts/multi_round_full_analysis.py \\
        --experiment-dir outputs/sfi2-long/default_experiment \\
        --out-dir outputs/sfi2-long/default_experiment/multi_round_full_custom

Minimum per-round files (if available):
    data_export/processed_data_<round>.parquet     (for ECDF & boxplots)
    impact_analysis/csv/impact_analysis_summary_<round>.csv
    correlation_analysis/csv/correlation_matrix_<metric>_<phase>.csv
    causality_analysis/csv/granger_tidy_<round>.csv (optional)
    causality_analysis/csv/transfer_entropy_tidy_<round>.csv (optional)

Main outputs:
    <out-dir>/ecdf/ ecdf_{metric}.png + ecdf_{metric}.csv
    <out-dir>/boxplots_metric/ boxplot_metric_{metric}.png
    <out-dir>/impact/ (reuse or recomputed simple aggregate)
    <out-dir>/correlation/ correlation_heatmap_mean_{metric}_{phase}.png
    <out-dir>/causality/<method>/ causality_heatmap_{method}_{metric}_{phase}.png
    <out-dir>/causality/causality_consistency_top15.csv / causality_consistency_top15.png

Note: Depends only on pandas, numpy, seaborn, matplotlib, networkx.
"""

from __future__ import annotations

import argparse
import os
import glob
import logging
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import re

try:
    # Optional: centralized colors/names
    from src.visualization_config import PUBLICATION_CONFIG
except Exception:  # pragma: no cover - fallback mínimo
    PUBLICATION_CONFIG = {
        'tenant_colors': {},
        'tenant_display_names': {},
        'phase_display_names': {},
        'heatmap_colormaps': {'correlation': 'coolwarm'},
    }

logger = logging.getLogger("multi_round_full_analysis")

# Standardize global theme/palette to 'viridis'
sns.set_theme(style="whitegrid")
sns.set_palette('viridis')


# --------------------------------------------------------------------------------------
# Formatting helpers
# --------------------------------------------------------------------------------------

def format_metric_name(raw: str) -> str:
    """Format raw metric identifiers to nicer display names.

    Rules:
      - Lowercase
      - Replace consecutive non-alphanumeric ([_-]) with single space
      - Collapse multiple spaces
      - Map known substrings to concise names (e.g., 'cpu usage' -> 'CPU Usage').
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return s
    import re as _re
    s = s.replace('__', '_')
    s = _re.sub(r'[\-_]+', ' ', s.lower())
    s = _re.sub(r'\s+', ' ', s).strip()
    # Specific normalizations
    replacements = {
        'cpu_usage': 'CPU Usage',
        'memory_usage': 'Memory Usage',
        'disk_io_total': 'Disk I/O Total',
        'network_receive': 'Network Receive',
        'network_transmit': 'Network Transmit',
        'network_throughput': 'Network Throughput',
        # Mapeamentos de fallback para versões em minúsculas com espaços
        'cpu usage': 'CPU Usage',
        'memory usage': 'Memory Usage',
        'disk io total': 'Disk I/O Total',
        'network receive': 'Network Receive',
        'network transmit': 'Network Transmit',
        'io total': 'Disk I/O Total',
        'receive': 'Network Receive',
        'transmit': 'Network Transmit',
        'usage': 'Usage',
        'network throughput': 'Network Throughput',
    }
    return replacements.get(s, s.title())


def format_phase_name(raw: str) -> str:
    """Format experimental phase names to more readable display names.
    
    Rules:
      - Extract numeric prefix (e.g., '2-CPU-Noise' -> '2')
      - Format rest of the string with proper capitalization
      - Apply specific name replacements for known phases
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return s
    
    import re as _re
    # Extract numeric prefix (if any)
    prefix_match = _re.match(r'^(\d+)[-_](.+)$', s)
    
    if prefix_match:
        prefix = prefix_match.group(1)
        rest = prefix_match.group(2)
        
        # Replace hyphens/underscores with spaces and capitalize
        rest = _re.sub(r'[\-_]+', ' ', rest)
        
        # Specific phase name replacements
        replacements = {
            'cpu noise': 'CPU Noise',
            'memory noise': 'Memory Noise',
            'network noise': 'Network Noise',
            'disk noise': 'Disk Noise',
            'combined noise': 'Combined Noise',
            'baseline': 'Baseline',
            'recovery': 'Recovery'
        }
        
        rest_lower = rest.lower()
        if rest_lower in replacements:
            rest = replacements[rest_lower]
        else:
            rest = rest.title()
            
        return f"{prefix} - {rest}"
    else:
        # If no numeric prefix, just capitalize
        s = _re.sub(r'[\-_]+', ' ', s)
        return s.title()


# --------------------------------------------------------------------------------------
# Loading utilities
# --------------------------------------------------------------------------------------

def list_round_ids(experiment_dir: str) -> List[str]:
    rounds = sorted([d for d in os.listdir(experiment_dir) if d.startswith('round-') and os.path.isdir(os.path.join(experiment_dir, d))])
    return rounds


def load_all_processed_timeseries(experiment_dir: str, round_ids: List[str]) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []
    for r in round_ids:
        parquet_pattern = os.path.join(experiment_dir, r, 'data_export', f'processed_data_{r}.parquet')
        if os.path.exists(parquet_pattern):
            try:
                df = pd.read_parquet(parquet_pattern)
                if 'round_id' not in df.columns:
                    df['round_id'] = r
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {parquet_pattern}: {e}")
        else:
            logger.info(f"Parquet file not found for {r}, skipping ECDF/boxplots for this round.")
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    # Normalize expected column names
    rename_map = {
        'experimental_phase': 'experimental_phase',  # placeholder in case needed
        'metric': 'metric_name'
    }
    all_df.rename(columns=rename_map, inplace=True)
    return all_df


def load_impact_summaries(experiment_dir: str, round_ids: List[str]) -> pd.DataFrame:
    paths = []
    for r in round_ids:
        p = os.path.join(experiment_dir, r, 'impact_analysis', 'csv', f'impact_analysis_summary_{r}.csv')
        if os.path.exists(p):
            paths.append(p)
    if not paths:
        return pd.DataFrame()
    dfs = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading impact file: {p}: {e}")
    if not dfs:
        return pd.DataFrame()
    imp = pd.concat(dfs, ignore_index=True)
    return imp


def load_correlation_matrices(experiment_dir: str, round_ids: List[str]) -> pd.DataFrame:
    """Load correlation matrices exported per round into long format.

    Expected per-round path pattern:
        <experiment_dir>/round-X/correlation_analysis/csv/correlation_matrix_<metric>_<phase>.csv
    """
    records: List[pd.DataFrame] = []
    for r in round_ids:
        csv_dir = os.path.join(experiment_dir, r, 'correlation_analysis', 'csv')
        if not os.path.isdir(csv_dir):
            continue
        for path in glob.glob(os.path.join(csv_dir, 'correlation_matrix_*_*.csv')):
            fname = os.path.basename(path)
            try:
                parts = fname.replace('correlation_matrix_', '').rsplit('.csv', 1)[0]
                metric, phase = parts.split('_', 1)
                mat = pd.read_csv(path, index_col=0)
                mat.index.name = 'tenant1'
                mat.columns.name = 'tenant2'
                stacked = mat.stack().reset_index()
                stacked = stacked.rename(columns={0: 'correlation'})
                long = stacked
                long['metric'] = metric
                long['phase'] = phase
                long['round_id'] = r
                records.append(long)
            except Exception as e:
                logger.warning(f"Failed to process {fname}: {e}")
    if not records:
        return pd.DataFrame()
    return pd.concat(records, ignore_index=True)


def load_causality_tidy(experiment_dir: str, round_ids: List[str]) -> pd.DataFrame:
    """Load tidy files for granger & TE if they exist. Expected columns: target, source, metric, phase, round_id, p-value / score."""
    dfs = []
    for r in round_ids:
        cdir = os.path.join(experiment_dir, r, 'causality_analysis', 'csv')
        if not os.path.isdir(cdir):
            continue
        for prefix in ['granger_tidy', 'transfer_entropy_tidy']:
            fpath = os.path.join(cdir, f'{prefix}_{r}.csv')
            if os.path.exists(fpath):
                try:
                    df = pd.read_csv(fpath)
                    if 'round_id' not in df.columns:
                        df['round_id'] = r
                    # mark method
                    method = 'granger' if prefix.startswith('granger') else 'te'
                    df['method'] = method
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {fpath}: {e}")
    if not dfs:
        return pd.DataFrame()
    all_df = pd.concat(dfs, ignore_index=True)
    # Normalize names
    rename_map = {'metric_name': 'metric'}
    all_df.rename(columns=rename_map, inplace=True)
    return all_df


# --------------------------------------------------------------------------------------
# ECDF & Boxplots (raw multi-round metrics)
# --------------------------------------------------------------------------------------

def compute_ecdf(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    v = np.sort(values)
    n = v.size
    if n == 0:
        return v, np.array([])
    y = np.arange(1, n + 1) / n
    return v, y


def generate_multi_round_ecdf(ts_df: pd.DataFrame, out_dir: str):
    if ts_df.empty:
        logger.info("No time series data for ECDF.")
        return
    ecdf_dir = os.path.join(out_dir, 'ecdf')
    os.makedirs(ecdf_dir, exist_ok=True)

    metrics = sorted(ts_df['metric_name'].unique())
    # Dicionário para armazenar dados ECDF por métrica e fase para o plot consolidado
    all_ecdf_data = {}
    
    # Primeiro, gerar ECDFs individuais por métrica
    for metric in metrics:
        mdf = ts_df[ts_df['metric_name'] == metric]
        if mdf.empty:
            continue
        fig, ax = plt.subplots(figsize=(15, 8))
        csv_rows: list[pd.DataFrame] = []
        all_ecdf_data[metric] = {}
        
        for phase, phase_df in mdf.groupby('experimental_phase'):
            vals = phase_df['metric_value'].dropna().to_numpy()
            if vals.size == 0:
                continue
            x, y = compute_ecdf(vals)
            ax.step(x, y, where='post', label=format_phase_name(phase))
            csv_rows.append(pd.DataFrame({'metric': metric, 'phase': phase, 'value': x, 'ecdf': y}))
            # Armazenar para plot consolidado
            all_ecdf_data[metric][phase] = (x, y)
            
        ax.set_title(f"ECDF - {format_metric_name(metric)}")
        ax.set_xlabel('Value')
        ax.set_ylabel('Cumulative fraction')
        ax.legend(title='Phase', fontsize=8)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        fig.tight_layout()
        png_path = os.path.join(ecdf_dir, f'ecdf_{metric}.png')
        fig.savefig(png_path)
        plt.close(fig)
        if csv_rows:
            ecdf_csv = pd.concat(csv_rows, ignore_index=True)
            ecdf_csv.to_csv(os.path.join(ecdf_dir, f'ecdf_{metric}.csv'), index=False)
        logger.info(f"ECDF generated for {metric}: {png_path}")
    
    # Now, create a consolidated panel with all metrics
    if all_ecdf_data:
        # Determine number of subplots needed
        n_metrics = len(all_ecdf_data)
        if n_metrics > 0:
            # Calculate number of rows and columns for subplots
            n_cols = min(2, n_metrics)  # Maximum 2 columns
            n_rows = (n_metrics + n_cols - 1) // n_cols  # Number of rows rounded up
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 8), squeeze=False)
            # List to store handles and labels from all phases for the shared legend
            handles_labels = []
            
            # Plot each metric in a subplot
            for i, (metric, phase_data) in enumerate(all_ecdf_data.items()):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                
                for phase, (x, y) in phase_data.items():
                    line, = ax.step(x, y, where='post', label=format_phase_name(phase))
                    # Only collect handles/labels from the first subplot for shared legend
                    if i == 0:
                        handles_labels.append((line, format_phase_name(phase)))
                
                ax.set_title(f"{format_metric_name(metric)}")
                ax.set_xlabel('Value')
                ax.set_ylabel('Cumulative fraction')
                ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
            
            # Esconder subplots vazios extras
            for i in range(len(all_ecdf_data), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)
            
            # Shared legend - extract handles and labels
            if handles_labels:
                handles, labels = zip(*handles_labels)
                # Legenda em uma única linha
                fig.legend(handles, labels, title='Phase', loc='lower center',
                           bbox_to_anchor=(0.5, 0.0), ncol=len(labels),
                           fontsize=9, title_fontsize=10, frameon=False)
            
            fig.suptitle('Consolidated ECDFs by Metric', fontsize=14)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))  # Margem inferior reduzida
            
            # Save consolidated figure
            consolidated_path = os.path.join(ecdf_dir, 'ecdf_consolidated_all_metrics.png')
            fig.savefig(consolidated_path, dpi=300)
            plt.close(fig)
            logger.info(f"Consolidated ECDF panel saved: {consolidated_path}")


def generate_multi_round_metric_boxplots(ts_df: pd.DataFrame, out_dir: str):
    if ts_df.empty:
        logger.info("No data for multi-round boxplots.")
        return
    bdir = os.path.join(out_dir, 'boxplots_metric')
    csv_dir = os.path.join(out_dir, 'boxplots_metric', 'csv')
    os.makedirs(bdir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    metrics = sorted(ts_df['metric_name'].unique())
    metrics_data = {}  # Para armazenar dados para o painel consolidado
    all_stats_dfs = []  # Para armazenar estatísticas consolidadas
    
    for metric in metrics:
        mdf = ts_df[ts_df['metric_name'] == metric]
        if mdf.empty:
            continue
            
        # Armazenar os dados para o painel consolidado
        metrics_data[metric] = mdf.copy()
        
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Criar cópia do DataFrame e formatar os nomes das fases para visualização
        mdf_plot = mdf.copy()
        phase_mapping = {phase: format_phase_name(phase) for phase in mdf['experimental_phase'].unique()}
        mdf_plot['experimental_phase'] = mdf_plot['experimental_phase'].map(phase_mapping)
        
        # Calcular estatísticas do boxplot e salvar em CSV
        stats_df = calculate_boxplot_stats(mdf, metric)
        stats_csv_path = os.path.join(csv_dir, f'box_stats_{metric}.csv')
        stats_df.to_csv(stats_csv_path, index=False)
        logger.info(f"Boxplot statistics saved: {stats_csv_path}")
        
        # Adicionar ao dataframe consolidado
        all_stats_dfs.append(stats_df)
        
        sns.boxplot(data=mdf_plot, x='experimental_phase', y='metric_value', hue='tenant_id', showfliers=False, ax=ax, palette='viridis')
        ax.set_title(f'Multi-Round Distribution {format_metric_name(metric)} by Phase')
        ax.set_xlabel('Phase')
        ax.set_ylabel(format_metric_name(metric))
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Tenant', loc='upper left', fontsize=9, title_fontsize=9, frameon=True)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.28)
        out_path = os.path.join(bdir, f'boxplot_metric_{metric}.png')
        fig.savefig(out_path)
        plt.close(fig)
        logger.info(f"Multi-round boxplot saved: {out_path}")
    
    # Salvar estatísticas consolidadas
    if all_stats_dfs:
        all_stats = pd.concat(all_stats_dfs, ignore_index=True)
        consolidated_csv_path = os.path.join(csv_dir, 'box_stats_all_metrics.csv')
        all_stats.to_csv(consolidated_csv_path, index=False)
        logger.info(f"Consolidated boxplot statistics saved: {consolidated_csv_path}")
    
    # Gerar painel consolidado com todos os boxplots juntos
    if metrics_data:
        generate_consolidated_boxplot_panel(metrics_data, bdir)


def calculate_boxplot_stats(df, metric_name):
    """
    Calcula estatísticas para os boxplots (quartis, média, mediana, etc.) por fase e tenant.
    """
    stats_list = []
    
    for phase in sorted(df['experimental_phase'].unique()):
        phase_df = df[df['experimental_phase'] == phase]
        
        for tenant in sorted(phase_df['tenant_id'].unique()):
            tenant_df = phase_df[phase_df['tenant_id'] == tenant]
            
            # Calcular estatísticas
            values = tenant_df['metric_value']
            
            if not values.empty:
                stats = {
                    'metric_name': metric_name,
                    'formatted_metric_name': format_metric_name(metric_name),
                    'experimental_phase': phase,
                    'formatted_phase': format_phase_name(phase),
                    'tenant_id': tenant,
                    'count': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'q1': values.quantile(0.25),
                    'median': values.median(),
                    'q3': values.quantile(0.75),
                    'max': values.max(),
                    'iqr': values.quantile(0.75) - values.quantile(0.25)
                }
                
                stats_list.append(stats)
    
    return pd.DataFrame(stats_list)


def generate_consolidated_boxplot_panel(metrics_data, output_dir):
    """
    Gera um painel consolidado com boxplots de todas as métricas.
    """
    if not metrics_data:
        return
    
    n_metrics = len(metrics_data)
    if n_metrics == 0:
        return
    
    # Definir layout do grid
    n_cols = 2  # Usar 2 colunas para melhor visualização
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Arredondar para cima
    
    # Criar figura com tamanho adequado
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 7 * n_rows), squeeze=False)
    
    # Para legenda compartilhada
    legend_handles = []
    legend_labels = []
    
    # Plotar cada métrica em um subplot
    for i, (metric, data) in enumerate(metrics_data.items()):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Formatar nomes das fases
        data_plot = data.copy()
        phase_mapping = {phase: format_phase_name(phase) for phase in data['experimental_phase'].unique()}
        data_plot['experimental_phase'] = data_plot['experimental_phase'].map(phase_mapping)
        
        # Criar boxplot para esta métrica
        boxplot = sns.boxplot(data=data_plot, x='experimental_phase', y='metric_value', 
                   hue='tenant_id', showfliers=False, ax=ax)
        
        # Coletar handles e labels para a legenda compartilhada apenas do primeiro subplot
        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels)
        
        # Remover a legenda individual
        if ax.get_legend():
            ax.get_legend().remove()
        
        # Configurar o subplot
        ax.set_title(f'{format_metric_name(metric)}', fontsize=11)
        ax.set_xlabel('Phase', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', labelsize=9)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
    # Esconder subplots vazios extras
    for i in range(len(metrics_data), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    # Adicionar legenda compartilhada
    if legend_handles and legend_labels:
        fig.legend(legend_handles, legend_labels, title='Tenant', loc='lower center', 
                   bbox_to_anchor=(0.5, 0), ncol=len(legend_labels), 
                   fontsize=9, title_fontsize=10)
    
    # Ajustar layout
    fig.suptitle('Multi-Round Distribution of Metrics by Phase', fontsize=14)
    fig.tight_layout(rect=(0, 0.08, 1, 0.97))  # Ajuste para o título e legenda
    
    # Salvar figura consolidada
    consolidated_path = os.path.join(output_dir, 'boxplot_metrics_consolidated_panel.png')
    fig.savefig(consolidated_path, dpi=300)
    plt.close(fig)
    logger.info(f"Consolidated boxplot panel saved: {consolidated_path}")


def generate_simplified_boxplot_panel(metrics_data, output_dir):
    """
    Gera uma versão simplificada do painel de boxplots, focada apenas nas distribuições,
    sem anotações adicionais.
    """
    if not metrics_data:
        return
    
    n_metrics = len(metrics_data)
    if n_metrics == 0:
        return
    
    # Definir layout do grid
    n_cols = 2  # Usar 2 colunas para melhor visualização
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Arredondar para cima
    
    # Criar figura com tamanho adequado
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows), squeeze=False)
    
    # Plotar cada métrica em um subplot
    for i, (metric, data) in enumerate(metrics_data.items()):
        row, col = i // n_cols, i % n_cols
        ax = axes[row, col]
        
        # Formatar nomes das fases
        data_plot = data.copy()
        phase_mapping = {phase: format_phase_name(phase) for phase in data['experimental_phase'].unique()}
        data_plot['experimental_phase'] = data_plot['experimental_phase'].map(phase_mapping)
        
        # Criar boxplot para esta métrica - versão mais simples
        sns.boxplot(data=data_plot, x='experimental_phase', y='metric_value', 
                   hue='tenant_id', showfliers=False, ax=ax)
        
        # Configurar o subplot - mais minimalista
        ax.set_title(f'{format_metric_name(metric)}', fontsize=10)
        ax.set_xlabel('Phase', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        
        # Legenda mais compacta
        legend = ax.legend(title='Tenant', loc='upper left', fontsize=7, title_fontsize=7, frameon=True)
        legend.get_frame().set_alpha(0.7)
        
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Esconder subplots vazios extras
    for i in range(len(metrics_data), n_rows * n_cols):
        row, col = i // n_cols, i % n_cols
        axes[row, col].set_visible(False)
    
    # Ajustar layout
    fig.suptitle('Distribution of Metrics Across Experimental Phases', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))  # Ajuste para o título
    
    # Salvar figura simplificada
    simplified_path = os.path.join(output_dir, 'boxplot_metrics_simplified_panel.png')
    fig.savefig(simplified_path, dpi=300)
    plt.close(fig)
    logger.info(f"Simplified boxplot panel saved: {simplified_path}")


# --------------------------------------------------------------------------------------
# Aggregated impact (reuse CSV + mean/std barplot)
# --------------------------------------------------------------------------------------

def aggregate_impact(impact_df: pd.DataFrame, out_dir: str):
    if impact_df.empty:
        logger.info("No impact data to aggregate.")
        return
    idir = os.path.join(out_dir, 'impact')
    os.makedirs(idir, exist_ok=True)
    # mean and std per metric/tenant/phase
    grouping = ['tenant_id', 'metric_name', 'experimental_phase']
    agg = impact_df.groupby(grouping)['percentage_change'].agg(['mean', 'std', 'count']).reset_index()
    agg.rename(columns={'mean': 'mean_percentage_change', 'std': 'std_percentage_change'}, inplace=True)
    agg_path = os.path.join(idir, 'impact_aggregated_stats.csv')
    agg.to_csv(agg_path, index=False)
    logger.info(f"Aggregated impact saved: {agg_path}")

    # Barplot facet by metric
    for metric, g in agg.groupby('metric_name'):
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Criar cópia para visualização com nomes formatados
        g_plot = g.copy()
        phase_mapping = {phase: format_phase_name(phase) for phase in g['experimental_phase'].unique()}
        g_plot['experimental_phase'] = g_plot['experimental_phase'].map(phase_mapping)
        
        sns.barplot(data=g_plot, x='experimental_phase', y='mean_percentage_change', hue='tenant_id', ax=ax, capsize=0.1)
        ax.set_title(f'Mean Percentage Impact - {format_metric_name(str(metric))}')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Impact (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.legend(title='Tenant', loc='upper left', fontsize=9, title_fontsize=9, frameon=True)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.9, bottom=0.24)
        fig.savefig(os.path.join(idir, f'impact_bar_{metric}.png'))
        plt.close(fig)


# --------------------------------------------------------------------------------------
# Aggregated correlation (mean per metric/phase tenant1 tenant2) + heatmaps
# --------------------------------------------------------------------------------------

def aggregate_correlation(corr_long: pd.DataFrame, out_dir: str) -> pd.DataFrame | None:
    if corr_long.empty:
        logger.info("No correlation data.")
        return None
    cdir = os.path.join(out_dir, 'correlation')
    os.makedirs(cdir, exist_ok=True)
    agg = (corr_long
           .groupby(['metric', 'phase', 'tenant1', 'tenant2'])['correlation']
           .agg(['mean', 'std', 'count'])
           .reset_index())
    agg.rename(columns={'mean': 'mean_correlation', 'std': 'std_correlation'}, inplace=True)
    agg.to_csv(os.path.join(cdir, 'correlation_aggregated_long.csv'), index=False)

    for (metric, phase), g in agg.groupby(['metric', 'phase']):
        pivot = g.pivot(index='tenant1', columns='tenant2', values='mean_correlation')
        tenants = sorted(list(set(pivot.index) | set(pivot.columns)))
        pivot = pivot.reindex(index=tenants, columns=tenants)
        sym = pivot.copy()
        for i in tenants:
            for j in tenants:
                if pd.isna(sym.loc[i, j]) and not pd.isna(sym.loc[j, i]):
                    sym.loc[i, j] = sym.loc[j, i]
        np.fill_diagonal(sym.values, 1.0)
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(sym, vmin=-1, vmax=1, cmap='viridis', annot=True, fmt='.2f', linewidths=.5, ax=ax)
        ax.set_title(f'Mean Correlation - {format_metric_name(metric)} - {phase}')
        ax.set_xlabel('Tenant')
        ax.set_ylabel('Tenant')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout()
        safe_phase = phase.replace('/', '_')
        fig.savefig(os.path.join(cdir, f'correlation_heatmap_mean_{metric}_{safe_phase}.png'))
        plt.close(fig)
    return agg


def generate_tenant_coupling_index_per_metric_plots(corr_agg: pd.DataFrame | None,
                                                    out_dir: str,
                                                    exclude_baseline: bool = True,
                                                    exclude_recovery: bool = True,
                                                    bar_width: float = 0.8):
    """Generate per-metric coupling index grouped bar plots.

    For each metric we plot phases on the x-axis and, for every phase, grouped bars per tenant
    (hue=tenant) showing that tenant's coupling index for the metric/phase.

    Coupling index (per tenant, metric, phase) = sum(|correlations with others|) - 1.
    Saves one PNG per metric plus a long-form CSV with columns: phase, tenant, metric, coupling_index.
    """
    if corr_agg is None or corr_agg.empty:
        logger.info("No aggregated correlation data for per-metric coupling plots.")
        return None
    required = {'phase','metric', 'tenant1', 'tenant2', 'mean_correlation'}
    if not required.issubset(corr_agg.columns):
        logger.warning(f"Missing columns for per-metric coupling index: {required - set(corr_agg.columns)}")
        return None

    tenants = sorted(list(set(corr_agg['tenant1']).union(corr_agg['tenant2'])))
    metrics = sorted(corr_agg['metric'].unique())
    phases_all = sorted(corr_agg['phase'].unique())

    def _phase_keep(p: str) -> bool:
        p_lower = str(p).lower().replace(' ', '')
        if exclude_baseline and p_lower.startswith('1-baseline'):
            return False
        if exclude_recovery and 'recovery' in p_lower:
            return False
        return True

    def _phase_sort_key(p: str):
        try:
            return int(str(p).split('-', 1)[0])
        except Exception:
            return 9999

    phases = [p for p in phases_all if _phase_keep(p)]
    phases = sorted(phases, key=_phase_sort_key)
    if not phases:
        logger.warning("No phases left after filters for per-metric coupling index.")
        return None

    rows: list[dict] = []
    for metric in metrics:
        mdf_all = corr_agg[corr_agg['metric'] == metric]
        if mdf_all.empty:
            continue
        for phase in phases:
            phase_df = mdf_all[mdf_all['phase'] == phase]
            if phase_df.empty:
                continue
            # build square matrix (fill with identity)
            mat = pd.DataFrame(np.eye(len(tenants)), index=tenants, columns=tenants)
            for _, r in phase_df.iterrows():
                t1, t2, mc = r['tenant1'], r['tenant2'], r['mean_correlation']
                if t1 in mat.index and t2 in mat.columns:
                    mat.loc[t1, t2] = mc
                if t2 in mat.index and t1 in mat.columns:
                    mat.loc[t2, t1] = mc
            for t in tenants:
                rows.append({
                    'metric': metric,
                    'phase': phase,
                    'tenant': t,
                    'coupling_index': float(mat.loc[t].abs().sum() - 1.0)
                })

    if not rows:
        logger.warning("No rows computed for per-metric coupling index.")
        return None

    df_long = pd.DataFrame(rows)
    # Transform (metric, phase) so that metric incorporates submetric prefix from phase string.
    # Original: metric='disk', phase='io_total_5-Disk-Noise' -> metric='disk_io_total', phase='5-Disk-Noise'
    # Logic: find final underscore before the numeric phase token (\d+-...)
    pattern = re.compile(r'^(?P<submetric>.+)_(?P<phase>\d+-.*)$')
    new_metric_vals = []
    new_phase_vals = []
    for mval, phval in zip(df_long['metric'].astype(str), df_long['phase'].astype(str)):
        match = pattern.match(phval)
        if match:
            submetric = match.group('submetric')  # can contain internal underscores
            phase_clean = match.group('phase')
            new_metric_vals.append(f"{mval}_{submetric}")
            new_phase_vals.append(phase_clean)
        else:
            # If phase not matching pattern, leave as-is
            new_metric_vals.append(mval)
            new_phase_vals.append(phval)
    df_long['metric'] = new_metric_vals
    df_long['phase'] = new_phase_vals

    cdir = os.path.join(out_dir, 'correlation')
    os.makedirs(cdir, exist_ok=True)
    csv_path = os.path.join(cdir, 'tenant_coupling_index_per_metric_long.csv')
    df_long.to_csv(csv_path, index=False)

    # Plot per metric
    # Determine ordering of transformed phases (numeric prefix ascending)
    def _phase_sort_key2(p: str):
        try:
            return int(str(p).split('-', 1)[0])
        except Exception:
            return 9999
    phase_order_effective = sorted(df_long['phase'].unique(), key=_phase_sort_key2)
    # Canonical tenant order for publication (article order)
    tenant_order = ["tenant-cpu", "tenant-mem", "tenant-dsk", "tenant-ntk", "tenant-nsy"]
    
    # Dictionary to store data by metric for the consolidated plot
    metrics_data = {}
    
    # Generate individual plots by metric
    for metric, g in df_long.groupby('metric'):
        g = g.copy()
        g['phase'] = pd.Categorical(g['phase'], categories=phase_order_effective, ordered=True)
        # Apply tenant categorical ordering (drop tenants not present gracefully)
        present_tenants = [t for t in tenant_order if t in g['tenant'].unique().tolist()]
        g['tenant'] = pd.Categorical(g['tenant'], categories=present_tenants, ordered=True)
        # Wider figure: width scales on phases and tenants for better readability
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Aplicar formatação de fase para visualização
        g_plot = g.copy()
        phase_mapping = {phase: format_phase_name(phase) for phase in g['phase'].unique()}
        g_plot['display_phase'] = g_plot['phase'].map(phase_mapping)
        # Usar a coluna formatada para o eixo x
        
        sns.barplot(data=g_plot, x='display_phase', y='coupling_index', hue='tenant', ax=ax, palette='viridis', 
                   hue_order=present_tenants)
        display_metric = str(metric)
        ax.set_title(f'Coupling Index per Tenant Across Phases - {format_metric_name(display_metric)}')
        ax.set_xlabel('Phase')
        ax.set_ylabel('Coupling Index')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
        # Legend ordering already enforced by hue_order; place below or side depending on width
        # ax.legend(title='Tenant', loc='upper left', fontsize=9, title_fontsize=9, frameon=True)
        max_val = 0.0
        from matplotlib.patches import Rectangle
        for p in ax.patches:
            if not isinstance(p, Rectangle):
                continue
            height = getattr(p, 'get_height', lambda: None)()
            if height is None or np.isnan(height):
                continue
            max_val = max(max_val, height)
            x = p.get_x() + p.get_width() / 2
            ax.annotate(f"{height:.2f}", (x, height), ha='center', va='bottom', fontsize=8, xytext=(0, 2), textcoords='offset points')
        if max_val > 0:
            ax.set_ylim(0, max_val * 1.12)
            fig.tight_layout(rect=(0, 0, 0.88, 1))
            out_path = os.path.join(cdir, f'coupling_index_per_metric_bar_{metric}.png')
            fig.savefig(out_path, dpi=300)
            plt.close(fig)
            logger.info(f"Per-metric coupling bar plot saved: {out_path}")
        
        # Store data for the consolidated plot
        metrics_data[metric] = {
            'data': g,
            'present_tenants': present_tenants
        }
    
    # Generate consolidated plot with all metrics
    if metrics_data:
        # Determine number of subplots needed
        n_metrics = len(metrics_data)
        if n_metrics > 0:
            # Calculate number of rows and columns for subplots
            n_cols = min(2, n_metrics)  # Maximum 2 columns
            n_rows = (n_metrics + n_cols - 1) // n_cols  # Number of rows rounded up
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), squeeze=False)
            all_tenants = set()
            tenant_handles = {}
            from matplotlib.patches import Rectangle
            
            # Plot each metric in a subplot
            for i, (metric, metric_info) in enumerate(metrics_data.items()):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                data = metric_info['data']
                present_tenants = metric_info['present_tenants']
                all_tenants.update(present_tenants)
                
                # Aplicar formatação de fase para visualização
                plot_data = data.copy()
                phase_mapping = {phase: format_phase_name(phase) for phase in data['phase'].unique()}
                plot_data['display_phase'] = plot_data['phase'].map(phase_mapping)
                
                bars = sns.barplot(
                    data=plot_data, 
                    x='display_phase', 
                    y='coupling_index', 
                    hue='tenant', 
                    ax=ax, 
                    palette='viridis',
                    hue_order=present_tenants,
                    legend=False  # Desabilitar legendas individuais
                )
                
                # Capturar handles de legenda apenas para o primeiro plot
                if i == 0:
                    # Criar e capturar a legenda temporária para extrair handles
                    temp_legend = ax.legend(title='Tenant')
                    if temp_legend and hasattr(temp_legend, 'get_patches'):
                        handles = temp_legend.get_patches()
                        for tenant, handle in zip(present_tenants, handles):
                            tenant_handles[tenant] = handle
                    # Remover a legenda temporária
                    if ax.get_legend():
                        ax.get_legend().remove()
                # Garantir que não haja legenda em nenhum subplot
                if ax.get_legend():
                    ax.get_legend().remove()
                
                display_metric = str(metric)
                ax.set_title(f'{format_metric_name(display_metric)}')
                # Remover termo 'Phase' conforme solicitado
                ax.set_xlabel('')
                ax.set_ylabel('Coupling Index')
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
                ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.6)
                
                # Reduzir tamanho das anotações e colocá-las apenas para valores relevantes
                for p in ax.patches:
                    if not isinstance(p, Rectangle):
                        continue
                    height = getattr(p, 'get_height', lambda: None)()
                    if height is None or np.isnan(height) or height < 0.1:  # Ignorar valores muito pequenos
                        continue
                    x = p.get_x() + p.get_width() / 2
                    ax.annotate(f"{height:.1f}", (x, height), ha='center', va='bottom', fontsize=7, xytext=(0, 1), textcoords='offset points')
            
            # Esconder subplots vazios extras
            for i in range(len(metrics_data), n_rows * n_cols):
                row, col = i // n_cols, i % n_cols
                axes[row, col].set_visible(False)
            
            # Legenda compartilhada
            if tenant_handles:
                ordered_tenants = [t for t in tenant_order if t in all_tenants]
                legend_handles = []
                legend_labels = []
                for t in ordered_tenants:
                    if t in tenant_handles:
                        legend_handles.append(tenant_handles[t])
                        legend_labels.append(t)
                
                if legend_handles:
                    fig.legend(legend_handles, legend_labels,
                               title='Tenant', loc='lower center',
                               bbox_to_anchor=(0.5, 0.0), ncol=len(legend_labels),
                               fontsize=9, title_fontsize=10, frameon=False)
            
            fig.suptitle('Coupling Index per Metric', fontsize=14)
            fig.tight_layout(rect=(0, 0.07, 1, 0.95))  # Margem inferior ligeiramente reduzida
            
            # Save consolidated figure
            consolidated_path = os.path.join(cdir, 'coupling_index_consolidated_all_metrics.png')
            fig.savefig(consolidated_path, dpi=300)
            plt.close(fig)
            logger.info(f"Consolidated coupling index panel saved: {consolidated_path}")
    
    # Create and save the tenant_coupling_index files
    # 1. Create a copy of df_long for the metric-level file (already exists at csv_path)
    metric_level_df = df_long.copy()
    
    # 2. Create the aggregated ALL metric version for tenant_coupling_index_long.csv
    # Group by phase and tenant, taking mean of the coupling_index
    res_df = df_long.groupby(['phase', 'tenant']).agg(
        coupling_index=('coupling_index', 'mean')
    ).reset_index()
    # Add ALL metric column
    res_df['metric'] = 'ALL'
    
    # 3. Save both files
    # Save metric level CSV
    metric_level_df.to_csv(os.path.join(cdir, 'tenant_coupling_index_metric_level_long.csv'), index=False)
    # Save aggregated (metric=ALL) CSV
    res_df.to_csv(os.path.join(cdir, 'tenant_coupling_index_long.csv'), index=False)
    logger.info(f"Tenant coupling index files saved at {cdir}")

    return csv_path


# --------------------------------------------------------------------------------------
# Aggregated causality & consistency
# --------------------------------------------------------------------------------------

def aggregate_causality(causality_tidy: pd.DataFrame, out_dir: str, p_significance: float = 0.05, te_threshold: float | None = None, top_k: int = 15):
    if causality_tidy.empty:
        logger.info("No causality data.")
        return
    cdir = os.path.join(out_dir, 'causality')
    os.makedirs(cdir, exist_ok=True)

    # Normalize expected columns
    required_cols = {'target', 'source', 'metric', 'phase', 'round_id', 'method'}
    missing = required_cols - set(causality_tidy.columns)
    if missing:
        logger.warning(f"Colunas ausentes em causalidade tidy: {missing}")

    # Derived metrics: -log10(p-value) for granger; keep score for TE
    df = causality_tidy.copy()
    if 'p-value' in df.columns:
        df['neg_log_p'] = -np.log10(df['p-value'].replace(0, np.nextafter(0, 1)))
    if 'score' not in df.columns:
        df['score'] = np.nan

    # Mean aggregation per method/metric/phase/source/target
    agg = (df.groupby(['method', 'metric', 'phase', 'source', 'target'])
             .agg({
                 'neg_log_p': 'mean',
                 'p-value': 'mean',
                 'score': 'mean',
                 'round_id': 'count'
             })
             .rename(columns={'round_id': 'num_rounds'})
             .reset_index())
    agg.to_csv(os.path.join(cdir, 'causality_aggregated_long.csv'), index=False)

    # Mean heatmaps per method/metric/phase
    for (method, metric, phase), g in agg.groupby(['method', 'metric', 'phase']):
        pivot_val_col = 'score' if method == 'te' else 'neg_log_p'
        pivot = g.pivot(index='target', columns='source', values=pivot_val_col)
        targets = sorted(list(set(pivot.index) | set(pivot.columns)))
        pivot = pivot.reindex(index=targets, columns=targets)
        fig, ax = plt.subplots(figsize=(15, 8))
        sns.heatmap(pivot, cmap='viridis', annot=True, fmt='.2f', linewidths=.5, ax=ax)
        ax.set_title(f'Mean Causality ({method.upper()}) - {format_metric_name(metric)} - {phase}')
        ax.set_xlabel('Source')
        ax.set_ylabel('Target')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        fig.tight_layout()
        safe_phase = phase.replace('/', '_')
        hm_dir = os.path.join(cdir, method)
        os.makedirs(hm_dir, exist_ok=True)
        fig.savefig(os.path.join(hm_dir, f'causality_heatmap_{method}_{metric}_{safe_phase}.png'))
        plt.close(fig)

    # Consistency: frequency of significant links per method
    def is_significant(row):
        if row['method'] == 'granger' and 'p-value' in row and not pd.isna(row['p-value']):
            return row['p-value'] < p_significance
        if row['method'] == 'te':
            if te_threshold is None:
                # heurística: score > mediana global TE
                return row['score'] > df[df['method'] == 'te']['score'].median()
            return row['score'] > te_threshold
        return False

    df['is_significant'] = df.apply(is_significant, axis=1)
    sig = df[df['is_significant']]
    if sig.empty:
        logger.info("No significant link found for consistency.")
        return
    num_rounds_total = df['round_id'].nunique()
    freq = (sig.groupby(['method', 'metric', 'source', 'target'])['round_id']
              .nunique()
              .reset_index(name='frequency'))
    freq['consistency_rate'] = freq['frequency'] / num_rounds_total * 100.0
    # Global Top K (independent of method/metric) for synthetic view
    top15 = freq.sort_values('consistency_rate', ascending=False).head(top_k)
    top15_path = os.path.join(cdir, 'causality_consistency_top15.csv')
    top15.to_csv(top15_path, index=False)

    # Plot bar horizontal
    fig, ax = plt.subplots(figsize=(15, 8))
    top15_plot = top15.copy()
    top15_plot['label'] = (top15_plot['method'] + ':' + top15_plot['metric'] + '\n' + top15_plot['source'] + '→' + top15_plot['target'])
    sns.barplot(data=top15_plot, x='consistency_rate', y='label', hue='label', orient='h', ax=ax, palette='viridis', legend=False)
    ax.set_xlabel('Consistency (%)')
    ax.set_ylabel('Link (method:metric)')
    ax.set_title('Top-15 Causal Links by Multi-Round Consistency')
    for i, v in enumerate(top15_plot['consistency_rate']):
        ax.text(v + 0.5, i, f"{v:.1f}%", va='center')
    fig.tight_layout()
    fig.savefig(os.path.join(cdir, 'causality_consistency_top15.png'))
    plt.close(fig)
    logger.info(f"Top-15 causality saved: {top15_path}")


# --------------------------------------------------------------------------------------
# Additional visualizations: Impact Signature & Causality Reproducibility
# --------------------------------------------------------------------------------------

def generate_directional_causality_phase_heatmaps(causality_tidy: pd.DataFrame,
                                                  out_dir: str,
                                                  p_significance: float = 0.05,
                                                  te_percentile: float = 0.75,
                                                  exclude_recovery: bool = True,
                                                  n_cols: int = 3):
    """Generate directional heatmaps by phase with frequency of significant causal links.

    Significance criteria:
    - Granger: p-value < p_significance
    - TE: score > global quantile (te_percentile) of TE scores
    """
    if causality_tidy.empty:
        logger.info("No causality data for directional heatmaps.")
        return None

    df = causality_tidy.copy()
    # Limiares
    te_thr = None
    if 'score' in df.columns:
        te_scores = df[df.get('method') == 'te']['score'].dropna()
        if not te_scores.empty:
            te_thr = te_scores.quantile(te_percentile)

    def _is_sig(row) -> bool:
        m = row.get('method')
        if m == 'granger' and 'p-value' in row and not pd.isna(row['p-value']):
            return row['p-value'] < p_significance
        if m == 'te' and ('score' in row) and not pd.isna(row['score']) and te_thr is not None:
            return row['score'] > te_thr
        return False

    df['__sig__'] = df.apply(_is_sig, axis=1)
    sig = df[df['__sig__']].copy()
    if sig.empty:
        logger.info("No significant link for directional heatmaps.")
        return None

    # Fases e nós
    phases = sorted(sig['phase'].dropna().unique().tolist())
    if exclude_recovery:
        phases = [p for p in phases if 'Recovery' not in str(p)]
    if not phases:
        logger.info("No phases after filter for directional heatmaps.")
        return None
    nodes = sorted(list(set(sig['source'].dropna()).union(set(sig['target'].dropna()))))

    n_phases = len(phases)
    n_rows = math.ceil(n_phases / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows), sharex=True, sharey=True)
    axes = np.array(axes).flatten() if isinstance(axes, (list, np.ndarray)) else np.array([axes])

    dd = os.path.join(out_dir, 'causality', 'directional')
    os.makedirs(dd, exist_ok=True)

    # Pré-calcular matrizes para obter escala global e garantir uniformidade
    phase_matrices: dict[str, pd.DataFrame] = {}
    global_max = 0
    for phase in phases:
        phase_df = sig[sig['phase'] == phase]
        edge_counts = (phase_df.groupby(['source', 'target']).size().reset_index(name='frequency'))
        matrix = (edge_counts
                  .pivot_table(index='source', columns='target', values='frequency', fill_value=0)
                  .reindex(index=nodes, columns=nodes, fill_value=0))
        phase_matrices[phase] = matrix
        if not matrix.empty:
            vmax = matrix.values.max()
            if vmax > global_max:
                global_max = vmax

    for i, phase in enumerate(phases):
        ax = axes[i]
        matrix = phase_matrices[phase]
        sns.heatmap(matrix, ax=ax, annot=True, fmt='.0f', cmap='viridis', linewidths=.5,
                    cbar=False, vmin=0, vmax=global_max, square=True)
        ax.set_title(f'Phase: {format_phase_name(phase)}', fontsize=14)
        ax.set_ylabel('Source' if i % n_cols == 0 else '')
        ax.set_xlabel('Target')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        matrix.to_csv(os.path.join(dd, f'directional_counts_{phase}.csv'.replace('/', '_')))

    # Colorbar única compartilhada
    if phase_matrices:
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=0, vmax=global_max)
        sm = ScalarMappable(norm=norm, cmap='viridis')
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes[:n_phases], fraction=0.03, pad=0.02)
        cbar.set_label('Frequency')

    # Esconder eixos não usados
    # Esconder eixos além do número de fases plotadas
    for j in range(len(phases), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Directional Causality Analysis (Phase Heatmaps)', fontsize=20, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = os.path.join(dd, 'phase_causality_frequency_heatmaps.png')
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Directional heatmaps saved at {out_path}")
    return out_path

def generate_impact_signature_heatmap(out_dir: str, victim_tenant: str = 'tenant-cpu', exclude_baseline: bool = True):
    """Generate impact signature heatmap for a victim tenant.

    Looks for <out_dir>/impact/impact_aggregated_stats.csv with columns:
    tenant_id, metric_name, experimental_phase, mean_percentage_change.
    """
    impact_csv = os.path.join(out_dir, 'impact', 'impact_aggregated_stats.csv')
    if not os.path.exists(impact_csv):
        logger.warning("Aggregated impact file not found; skipping impact signature.")
        return None
    try:
        df = pd.read_csv(impact_csv)
    except Exception as e:
        logger.error(f"Failed to read aggregated impact: {e}")
        return None

    required = {'tenant_id', 'metric_name', 'experimental_phase', 'mean_percentage_change'}
    if not required.issubset(df.columns):
        logger.warning(f"Missing required columns for impact signature: {required - set(df.columns)}")
        return None

    victim_df = df[df['tenant_id'] == victim_tenant].copy()
    if victim_df.empty:
        logger.warning(f"No impact data for tenant {victim_tenant}")
        return None

    # Order phases numerically by prefix before first dash
    def _phase_order(ph: str):
        try:
            return int(ph.split('-', 1)[0])
        except Exception:
            return 999
    victim_df['__phase_order'] = victim_df['experimental_phase'].apply(_phase_order)
    victim_df = victim_df.sort_values('__phase_order')

    pivot = victim_df.pivot_table(
        index='experimental_phase',
        columns='metric_name',
        values='mean_percentage_change',
        aggfunc='mean'
    )

    if exclude_baseline:
        # Remove rows whose prefix starts with '1'
        baseline_rows = [idx for idx in pivot.index if idx.startswith('1')]
        pivot = pivot.drop(baseline_rows, errors='ignore')

    if pivot.empty:
        logger.warning("Impact signature matrix empty after filtering.")
        return None

    sig_dir = os.path.join(out_dir, 'impact')
    os.makedirs(sig_dir, exist_ok=True)
    # Save CSV of the signature
    pivot.to_csv(os.path.join(sig_dir, f'impact_signature_{victim_tenant}.csv'))

    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Create a copy of the pivot with formatted phase names and metric names
    pivot_formatted = pivot.copy()
    pivot_formatted = pivot_formatted.rename(index={idx: format_phase_name(idx) for idx in pivot.index})
    pivot_formatted = pivot_formatted.rename(columns={col: format_metric_name(col) for col in pivot.columns})
    
    center_val = 0 if (pivot_formatted.values.min() < 0 and pivot_formatted.values.max() > 0) else None
    sns.heatmap(pivot_formatted, annot=True, fmt='.1f', cmap='viridis', linewidths=.5, center=center_val, ax=ax)
    ax.set_title(f'Impact Signature - {victim_tenant}')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Experimental Phase' + (' (no baseline)' if exclude_baseline else ''))
    plt.xticks(rotation=45, ha='right')
    fig.tight_layout()
    out_path = os.path.join(sig_dir, f'heatmap_impact_signature_{victim_tenant}.png')
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info(f"Impact signature saved at {out_path}")
    return pivot, out_path


def generate_consolidated_impact_signatures(out_dir: str, exclude_baseline: bool = True):
    """Generate a consolidated panel with impact signatures for all victim tenants.
    
    This function creates both individual heatmaps (using generate_impact_signature_heatmap)
    and a consolidated panel with all victim tenants, and exports consolidated data to CSV.
    """
    impact_csv = os.path.join(out_dir, 'impact', 'impact_aggregated_stats.csv')
    if not os.path.exists(impact_csv):
        logger.warning("Aggregated impact file not found; skipping consolidated impact signatures.")
        return None
    
    try:
        df = pd.read_csv(impact_csv)
        # Apply formatting to phase names for visualization
        df_formatted = df.copy()
        if 'experimental_phase' in df.columns:
            df_formatted['phase_formatted'] = df['experimental_phase'].apply(format_phase_name)
        if 'metric_name' in df.columns:
            df_formatted['metric_formatted'] = df['metric_name'].apply(format_metric_name)
        
        all_tenants = sorted(df['tenant_id'].unique())
    except Exception as e:
        logger.error(f"Failed to read aggregated impact: {e}")
        return None
    
    tenant_pivots = {}
    
    # Generate individual heatmaps for each tenant
    for tenant in all_tenants:
        result = generate_impact_signature_heatmap(out_dir, tenant, exclude_baseline)
        if result:
            pivot, _ = result
            tenant_pivots[tenant] = pivot
    
    if not tenant_pivots:
        logger.warning("No impact signatures generated for any tenant.")
        return None
    
    # Create a consolidated CSV with all tenants
    all_data = []
    for tenant, pivot in tenant_pivots.items():
        tenant_data = pivot.reset_index()
        tenant_data['victim_tenant'] = tenant
        # Melt the dataframe to long format
        long_data = pd.melt(
            tenant_data, 
            id_vars=['experimental_phase', 'victim_tenant'],
            var_name='metric_name',
            value_name='mean_percentage_change'
        )
        all_data.append(long_data)
    
    consolidated_df = pd.concat(all_data)
    
    # Save consolidated CSV
    sig_dir = os.path.join(out_dir, 'impact')
    os.makedirs(sig_dir, exist_ok=True)
    consolidated_path = os.path.join(sig_dir, 'impact_signatures_consolidated.csv')
    consolidated_df.to_csv(consolidated_path, index=False)
    logger.info(f"Consolidated impact signatures CSV saved at {consolidated_path}")
    
    # Create consolidated visualization panel
    n_tenants = len(tenant_pivots)
    if n_tenants > 0:
        # Determine layout
        n_cols = min(2, n_tenants)  # Maximum 2 columns
        n_rows = (n_tenants + n_cols - 1) // n_cols  # Number of rows rounded up
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 10 * n_rows // 2), squeeze=False)
        
        # Create heatmap for each tenant
        for i, (tenant, pivot) in enumerate(tenant_pivots.items()):
            row, col = i // n_cols, i % n_cols
            ax = axes[row, col]
            
            # Create a copy of the pivot with formatted phase names
            pivot_formatted = pivot.copy()
            pivot_formatted = pivot_formatted.rename(index={idx: format_phase_name(idx) for idx in pivot.index})
            pivot_formatted = pivot_formatted.rename(columns={col: format_metric_name(col) for col in pivot.columns})
            
            center_val = 0 if (pivot_formatted.values.min() < 0 and pivot_formatted.values.max() > 0) else None
            sns.heatmap(pivot_formatted, annot=True, fmt='.1f', cmap='viridis', linewidths=.5, 
                       center=center_val, ax=ax)
            ax.set_title(f'Impact Signature - {tenant}')
            ax.set_xlabel('Metric')
            ax.set_ylabel('Experimental Phase' + (' (no baseline)' if exclude_baseline else ''))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Hide any unused subplots
        for i in range(len(tenant_pivots), n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle('Impact Signatures by Victim Tenant', fontsize=16, y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        
        # Save consolidated figure
        out_path = os.path.join(sig_dir, 'impact_signatures_consolidated_panel.png')
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info(f"Consolidated impact signatures panel saved at {out_path}")
        
    return consolidated_path


def generate_causality_reproducibility_barplot(out_dir: str, adv_csv_path: str | None = None, top_k: int = 15):
    """Generate barplots of most reproducible causal links per metric and export CSV.

    Requirements in analise_avancada.csv: phase, metric, source, target, significant_frequency.
    Outputs:
      - <out_dir>/causality/barplot_causality_reproducibility_<metric>.png (one per metric)
      - <out_dir>/causality/causality_reproducibility_top_links_long.csv (long form consolidated)
      - <out_dir>/causality/barplot_causality_reproducibility_consolidated.png (all metrics in a panel)
    """
    # Se não for especificado, procura o arquivo no diretório de saída primeiro, depois no diretório raiz
    if adv_csv_path is None:
        adv_csv_path = os.path.join(out_dir, 'analise_avancada.csv')
        if not os.path.exists(adv_csv_path):
            adv_csv_path = 'analise_avancada.csv'
    
    if not os.path.exists(adv_csv_path):
        logger.warning("analise_avancada.csv not found; skipping causal reproducibility plot.")
        return None
    try:
        df = pd.read_csv(adv_csv_path)
    except Exception as e:
        logger.error(f"Failed to read {adv_csv_path}: {e}")
        return None
    needed = {'metric', 'source', 'target', 'significant_frequency'}
    if not needed.issubset(df.columns):
        logger.warning(f"Insufficient columns for causal reproducibility: {needed - set(df.columns)}")
        return None
    cdir = os.path.join(out_dir, 'causality')
    os.makedirs(cdir, exist_ok=True)

    all_top_rows: list[pd.DataFrame] = []
    metrics_data = {}  # Armazenar os dados para o painel consolidado
    metrics = sorted(df['metric'].unique())
    
    # Gerar plots individuais por métrica
    for metric in metrics:
        mdf = df[df['metric'] == metric]
        grouped = (mdf.groupby(['metric', 'source', 'target'], as_index=False)['significant_frequency']
                     .sum())
        grouped = grouped.sort_values('significant_frequency', ascending=False).head(top_k)
        if grouped.empty:
            continue
        grouped['label'] = grouped['source'] + ' → ' + grouped['target']
        all_top_rows.append(grouped.copy())
        
        # Salvar dados para o painel consolidado
        metrics_data[metric] = grouped.copy()
        
        # Gerar plot individual
        fig, ax = plt.subplots(figsize=(15, 8))
        # Limitar a 5 barras
        grouped_plot = grouped.head(min(top_k, 5))
        sns.barplot(data=grouped_plot, x='significant_frequency', y='label', hue='label', palette='viridis', ax=ax, legend=False)
        ax.set_title(f'Top {min(top_k, 5)} Reproducible Causal Links - {format_metric_name(metric)}')
        ax.set_xlabel('Total Significance Frequency (across phases)')
        ax.set_ylabel('Link (Source → Target)')
        # Garantir que os valores no eixo x sejam inteiros
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        max_val = grouped_plot['significant_frequency'].max()
        from matplotlib.patches import Rectangle  # local import
        for p in ax.patches:
            if isinstance(p, Rectangle):
                w = p.get_width()  # type: ignore[attr-defined]
                y = p.get_y()      # type: ignore[attr-defined]
                h = p.get_height() # type: ignore[attr-defined]
                ax.text(w + 0.01 * max_val, y + h / 2, f"{int(w)}", va='center', fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(cdir, f'barplot_causality_reproducibility_{metric}.png')
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        logger.info(f"Causal reproducibility barplot saved at {out_path}")
    
    # Gerar gráfico consolidado com os 10 links mais frequentes de todas as métricas
    if metrics_data:
        if len(metrics_data) > 0:
            # Log para debug das métricas disponíveis
            for metric in metrics_data.keys():
                logger.info(f"Metric in consolidated panel: '{metric}' -> '{format_metric_name(metric)}'")
            
            # Combinar todos os dados em um único DataFrame
            all_data = []
            for metric, data in metrics_data.items():
                data_copy = data.copy()
                formatted_metric = format_metric_name(metric)
                # Adicionar a métrica formatada no label
                data_copy['formatted_label'] = data_copy['label'] + f" ({formatted_metric})"
                all_data.append(data_copy)
            
            # Criar um DataFrame combinado
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                # Ordenar por frequência e pegar os 15 primeiros (aumentando de 10 para 15)
                combined_df = combined_df.sort_values('significant_frequency', ascending=False).head(15)
                
                # Criar o gráfico consolidado
                fig, ax = plt.subplots(figsize=(15, 10))  # Aumentar um pouco a altura
                
                # Importar Rectangle para verificação de tipos
                from matplotlib.patches import Rectangle
                
                # Criar o barplot com os 15 links mais frequentes
                sns.barplot(
                    data=combined_df, 
                    x='significant_frequency', 
                    y='formatted_label', 
                    palette='viridis',
                    ax=ax,
                    hue='formatted_label',
                    legend=False,
                    width=0.6  # Reduzir a espessura das barras
                )
                
                # Garantir que os valores no eixo x sejam inteiros
                ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
                
                # Configurar o título e labels
                ax.set_title('Top 15 Most Reproducible Causal Links Across All Metrics', fontsize=12)
                ax.set_xlabel('Frequency (Number of Significant Occurrences)', fontsize=10)
                ax.set_ylabel('Causal Link (Source → Target)', fontsize=10)
                
                # Ajustar tamanho das labels dos eixos
                ax.tick_params(axis='both', which='major', labelsize=9)
                
                # Adicionar anotações aos valores das barras
                max_val = combined_df['significant_frequency'].max() if not combined_df.empty else 0
                for p in ax.patches:
                    if isinstance(p, Rectangle):
                        w = p.get_width()
                        y = p.get_y()
                        h = p.get_height()
                        ax.text(w + 0.01 * max_val, y + h / 2, f"{int(w)}", va='center', fontsize=9)
                
                # Verificar se temos métricas de rede separadas (receive/transmit)
                has_network_metrics = any(m in ['network_receive', 'network_transmit'] for m in metrics_data.keys())
                if has_network_metrics:
                    ax.text(0.5, -0.1, '(Network metrics shown as Receive/Transmit instead of Throughput)', 
                            ha='center', transform=ax.transAxes, fontsize=9, style='italic')
                
                # Ajustar layout
                fig.tight_layout(rect=(0, 0, 1, 0.98))  # Ajuste para o título
                
                # Ajustar layout
                fig.tight_layout(rect=(0, 0, 1, 0.98))  # Ajuste para o título
                
                # Salvar figura consolidada
                consolidated_path = os.path.join(cdir, 'barplot_causality_reproducibility_consolidated.png')
                fig.savefig(consolidated_path, dpi=300)
                plt.close(fig)
                logger.info(f"Consolidated causality reproducibility panel saved: {consolidated_path}")

    if all_top_rows:
        consolidated = pd.concat(all_top_rows, ignore_index=True)
        consolidated_csv = os.path.join(cdir, 'causality_reproducibility_top_links_long.csv')
        consolidated.to_csv(consolidated_csv, index=False)
        logger.info(f"Consolidated reproducibility CSV saved at {consolidated_csv}")
        return consolidated_csv
    else:
        logger.warning("No reproducibility data after processing all metrics.")
        return None


def generate_quantification_barplot(out_dir: str,
                                    phase: str = '2-CPU-Noise',
                                    metric: str = 'cpu_usage',
                                    victim_only: bool = False,
                                    sort_by_abs: bool = False,
                                    palette: str = 'viridis'):
    """Generate impact quantification barplot for a specific phase and metric.

    Reads impact/impact_aggregated_stats.csv and plots mean_percentage_change with error bars (std).
    """
    impact_csv = os.path.join(out_dir, 'impact', 'impact_aggregated_stats.csv')
    if not os.path.exists(impact_csv):
        logger.warning("impact_aggregated_stats.csv not found; skipping quantification barplot.")
        return None
    try:
        df = pd.read_csv(impact_csv)
    except Exception as e:
        logger.error(f"Failed to read aggregated impact: {e}")
        return None
    required = {'tenant_id', 'metric_name', 'experimental_phase', 'mean_percentage_change', 'std_percentage_change'}
    if not required.issubset(df.columns):
        logger.warning(f"Missing columns for quantification barplot: {required - set(df.columns)}")
        return None
    subset = df[(df['experimental_phase'] == phase) & (df['metric_name'] == metric)].copy()
    if subset.empty:
        logger.warning(f"No data for phase {phase} and metric {metric}.")
        return None
    if victim_only:
        subset = subset[~subset['tenant_id'].str.contains('nsy')]  # heurística
    if subset.empty:
        logger.warning("After victim filter no data left.")
        return None
    if sort_by_abs:
        subset = subset.reindex(subset['mean_percentage_change'].abs().sort_values().index)
    else:
        subset = subset.sort_values('mean_percentage_change')
    colors = sns.color_palette(palette, len(subset))
    fig, ax = plt.subplots(figsize=(15, 8))
    bars = ax.bar(subset['tenant_id'], subset['mean_percentage_change'],
                  yerr=subset['std_percentage_change'], capsize=5,
                  color=colors, edgecolor='black', linewidth=0.6)
    ax.set_title(f'Percentage Impact - {format_metric_name(metric)} - {phase}')
    ax.set_ylabel('Δ (%) vs Baseline (mean ± SD)')
    ax.set_xlabel('Tenant')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    for bar, mean, std in zip(bars, subset['mean_percentage_change'], subset['std_percentage_change']):
        y = bar.get_height()
        offset = 0.5 if y >= 0 else -0.5
        ax.text(bar.get_x() + bar.get_width()/2, y + offset, f"{mean:.1f}%\n±{std:.1f}",
                ha='center', va='bottom' if y >= 0 else 'top', fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'impact', f'quantification_barplot_{metric}_{phase}.png'.replace('/', '_'))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    # também salvar subset filtrado
    subset.to_csv(os.path.join(out_dir, 'impact', f'quantification_data_{metric}_{phase}.csv'.replace('/', '_')), index=False)
    logger.info(f"Quantification barplot saved at {out_path}")
    return out_path


# --------------------------------------------------------------------------------------
# Main execution
# --------------------------------------------------------------------------------------

def run(experiment_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    round_ids = list_round_ids(experiment_dir)
    if not round_ids:
        logger.error("No round-* directories found.")
        return
    logger.info(f"Rounds found: {round_ids}")

    # Data loading
    ts_df = load_all_processed_timeseries(experiment_dir, round_ids)
    impact_df = load_impact_summaries(experiment_dir, round_ids)
    corr_long = load_correlation_matrices(experiment_dir, round_ids)
    causality_tidy = load_causality_tidy(experiment_dir, round_ids)

    # ECDF & Boxplots
    generate_multi_round_ecdf(ts_df, out_dir)
    generate_multi_round_metric_boxplots(ts_df, out_dir)

    # Impact
    aggregate_impact(impact_df, out_dir)

    # Correlation
    corr_agg = aggregate_correlation(corr_long, out_dir)

    # Coupling index by tenant/phase
    generate_tenant_coupling_index_per_metric_plots(corr_agg, out_dir)

    # Causality
    aggregate_causality(causality_tidy, out_dir)
    
    # Gerar arquivo analise_avancada.csv com dados consolidados
    try:
        from generate_advanced_analysis import generate_advanced_analysis
        
        # Caminhos para os arquivos consolidados
        causality_csv = os.path.join(out_dir, 'causality', 'causality_aggregated_long.csv')
        correlation_csv = os.path.join(out_dir, 'correlation', 'correlation_aggregated_long.csv')
        impact_csv = os.path.join(out_dir, 'impact', 'impact_aggregated_stats.csv')
        
        # Verificar se os arquivos existem
        if os.path.exists(causality_csv) and os.path.exists(correlation_csv) and os.path.exists(impact_csv):
            # Carregar dados
            causality_df = pd.read_csv(causality_csv)
            correlation_df = pd.read_csv(correlation_csv)
            impact_df = pd.read_csv(impact_csv)
            
            # Gerar análise avançada
            adv_csv_path = os.path.join(out_dir, 'analise_avancada.csv')
            result_df = generate_advanced_analysis(
                causality_df, correlation_df, impact_df, adv_csv_path
            )
            
            if result_df is not None:
                logger.info(f"Arquivo analise_avancada.csv gerado em: {adv_csv_path}")
                # Também salvamos uma cópia na raiz do projeto para compatibilidade
                root_adv_path = 'analise_avancada.csv'
                result_df.to_csv(root_adv_path, index=False)
                logger.info(f"Cópia salva em: {root_adv_path}")
        else:
            logger.warning("Não foi possível gerar analise_avancada.csv. Arquivos necessários não encontrados.")
    except Exception as e:
        logger.error(f"Erro ao gerar analise_avancada.csv: {e}")

    # New: per-metric coupling index line plots across phases
    per_metric_coupling_csv = generate_tenant_coupling_index_per_metric_plots(corr_agg, out_dir)
    if per_metric_coupling_csv:
        logger.info(f"Per-metric coupling index long CSV saved to {per_metric_coupling_csv}")
    # Directional heatmaps by phase (frequency of significant links)
    generate_directional_causality_phase_heatmaps(causality_tidy, out_dir)

    # Impact Signature (victim tenant)
    generate_impact_signature_heatmap(out_dir, victim_tenant='tenant-cpu', exclude_baseline=True)
    
    # Generate consolidated impact signatures for all tenants
    generate_consolidated_impact_signatures(out_dir, exclude_baseline=True)
    
    # Generate impact signatures by metric
    try:
        from impact_signature_by_metric import generate_impact_signatures_by_metric
        generate_impact_signatures_by_metric(out_dir)
        logger.info("Generated impact signatures by metric panels")
    except Exception as e:
        logger.error(f"Could not generate impact signatures by metric: {e}")

    # Caminho para o arquivo analise_avancada.csv
    adv_csv_path = os.path.join(out_dir, 'analise_avancada.csv')
    
    # Reproducibility (Top links)
    generate_causality_reproducibility_barplot(out_dir, adv_csv_path)

    # Quantification barplot (global parameters via args stored in internal environment)
    if hasattr(run, '_quant_args'):
        qargs = run._quant_args  # type: ignore[attr-defined]
        if qargs['enable']:
            generate_quantification_barplot(
                out_dir=out_dir,
                phase=qargs['phase'],
                metric=qargs['metric'],
                victim_only=qargs['victim_only'],
                sort_by_abs=qargs['sort_by_abs'],
                palette=qargs['palette']
            )

    logger.info("Multi-round analysis completed successfully.")


def parse_args():
    ap = argparse.ArgumentParser(description="Consolidated Multi-Round Analysis")
    ap.add_argument('--experiment-dir', required=True, help='Experiment directory containing round-* folders')
    ap.add_argument('--out-dir', required=False, help='Output directory (default: <experiment-dir>/multi_round_full_analysis)')
    ap.add_argument('--log-level', default='INFO')
    # Quantification barplot options
    ap.add_argument('--quantify', action='store_true', help='Generate impact quantification barplot for a phase/metric')
    ap.add_argument('--quant-phase', default='2-CPU-Noise', help='Phase to quantify (e.g., 2-CPU-Noise)')
    ap.add_argument('--quant-metric', default='cpu_usage', help='Metric name (e.g., cpu_usage)')
    ap.add_argument('--quant-victim-only', action='store_true', help='Exclude noisy tenant')
    ap.add_argument('--quant-sort-abs', action='store_true', help='Sort by absolute impact value')
    ap.add_argument('--quant-palette', default='viridis', help='Seaborn palette')
    return ap.parse_args()


def configure_logging(level: str):
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format='[%(levelname)s] %(message)s')


if __name__ == '__main__':
    args = parse_args()
    configure_logging(args.log_level)
    experiment_dir = args.experiment_dir
    out_dir = args.out_dir or os.path.join(experiment_dir, 'multi_round_full_analysis')
    run(experiment_dir, out_dir)
    # Armazenar argumentos de quantificação para chamada dentro de run antes (workaround simples)
    # (Para ser chamado antes, teríamos que passar args; aqui reexecutar só se habilitado.)
    if args.quantify:
        # Chamada direta pós-run se preferir não reexecutar tudo
        generate_quantification_barplot(
            out_dir=out_dir,
            phase=args.quant_phase,
            metric=args.quant_metric,
            victim_only=args.quant_victim_only,
            sort_by_abs=args.quant_sort_abs,
            palette=args.quant_palette
        )
