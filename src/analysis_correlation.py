"""
Module: analysis_correlation.py
Description: Correlation analysis utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('tableau-colorblind10')

def compute_correlation_matrix(df: pd.DataFrame, metric: str, phase: str, round_id: str, method: str = 'pearson') -> pd.DataFrame:
    """
    Computes the correlation matrix (Pearson or Spearman) between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        return pd.DataFrame()
    # Pivot to wide format: index=timestamp, columns=tenant_id, values=metric_value
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    corr = wide.corr(method=method)
    return corr

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, method: str = 'pearson'):
    """
    Plots a heatmap of the correlation matrix between tenants for a given metric, phase, and round.
    """
    if corr_matrix.empty:
        return None
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='vlag', vmin=-1, vmax=1, center=0, square=True, linewidths=0.5, cbar_kws={"label": f"{method.title()} correlation"})
    plt.title(f'{method.title()} correlation between tenants\n{metric} - {phase} - {round_id}')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"correlation_heatmap_{metric}_{phase}_{round_id}_{method}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def compute_covariance_matrix(df: pd.DataFrame, metric: str, phase: str, round_id: str) -> pd.DataFrame:
    """
    Computes the covariance matrix between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    cov = wide.cov()
    return cov

def plot_covariance_heatmap(cov_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """
    Plots a heatmap of the covariance matrix between tenants for a given metric, phase, and round.
    """
    if cov_matrix.empty:
        return None
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, cmap='crest', center=0, square=True, linewidths=0.5, cbar_kws={"label": "Covariance"})
    plt.title(f'Covariance between tenants\n{metric} - {phase} - {round_id}')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"covariance_heatmap_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def compute_cross_correlation(df: pd.DataFrame, metric: str, phase: str, round_id: str, tenant_x: str, tenant_y: str, max_lag: int = 20) -> pd.Series:
    """
    Computes the cross-correlation (at different lags) between two tenants for a given metric, phase, and round.
    Returns a pandas Series indexed by lag.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        return pd.Series(dtype=float)
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    x = wide[tenant_x].dropna()
    y = wide[tenant_y].dropna()
    # Garantir que ambos são Index
    x_idx = pd.Index(x.index)
    y_idx = pd.Index(y.index)
    common_idx = x_idx.intersection(y_idx)
    x = x.loc[common_idx]
    y = y.loc[common_idx]
    result = {}
    for lag in range(-max_lag, max_lag+1):
        if lag < 0:
            corr = x[:lag].corr(y[-lag:])
        elif lag > 0:
            corr = x[lag:].corr(y[:-lag])
        else:
            corr = x.corr(y)
        result[lag] = corr
    return pd.Series(result)

def plot_cross_correlation(series: pd.Series, metric: str, phase: str, round_id: str, tenant_x: str, tenant_y: str, out_dir: str):
    """
    Plots the cross-correlation series between two tenants for a given metric, phase, and round.
    """
    if series.empty:
        return None
    plt.figure(figsize=(10, 5))
    plt.stem(series.index, series.values)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title(f'Cross-correlation: {tenant_x} vs {tenant_y}\n{metric} - {phase} - {round_id}')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"crosscorr_{tenant_x}_vs_{tenant_y}_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path
