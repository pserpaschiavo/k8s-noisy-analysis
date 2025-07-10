"""
Module: correlation_plots.py
Description: Module for intra-phase correlation visualizations.

This module implements functions to visualize correlations between tenants
within each experimental phase, including correlation networks, heatmaps,
and stability analyses across rounds.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

from src.visualization_config import PUBLICATION_CONFIG
from src.visualization.plots import save_plot

logger = logging.getLogger(__name__)

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
    subset_mask = (df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)
    subset = df.loc[subset_mask]
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
