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
import logging
from statsmodels.tsa.stattools import acf

# Logging setup
logger = logging.getLogger(__name__)

plt.style.use('tableau-colorblind10')

def compute_correlation_matrix(df_identifier, metric: str, phase: str, round_id: str, method: str = 'pearson') -> pd.DataFrame:
    """
    Computes the correlation matrix (Pearson or Spearman) between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    
    Args:
        df_identifier: DataFrame identifier or the DataFrame itself
        metric: Name of the metric for analysis
        phase: Experimental phase to filter
        round_id: Round ID to filter
        method: Correlation method ('pearson', 'kendall', or 'spearman')
        
    Returns:
        DataFrame with the correlation matrix
    """
    # Handle DataFrame identification for caching purposes
    if isinstance(df_identifier, pd.DataFrame):
        df = df_identifier
    else:
        # If it's a string identifier, try to load from parquet or other source
        df = pd.read_parquet(df_identifier) if isinstance(df_identifier, str) else df_identifier
    
    # Validate correlation method
    valid_methods = ['pearson', 'kendall', 'spearman']
    if method not in valid_methods:
        logger.warning(f"Invalid correlation method: {method}. Using 'pearson'.")
        method = 'pearson'
    
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for correlation calculation: {metric}, {phase}, {round_id}")
        return pd.DataFrame()
    
    logger.info(f"Pre-pivot subset for {metric}, {phase}, {round_id}:")
    logger.info(f"  Shape: {subset.shape}")
    logger.info(f"  Tenants: {subset['tenant_id'].unique()}")
    logger.info(f"  Timestamps: {subset['timestamp'].nunique()}")

    # Pivot to wide format: index=timestamp, columns=tenant_id, values=metric_value
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    
    logger.info(f"Post-pivot wide DataFrame for {metric}, {phase}, {round_id}:")
    logger.info(f"  Shape: {wide.shape}")
    logger.info(f"  Columns: {wide.columns.tolist()}")

    # Check for missing data and handle it
    if wide.isna().any().any():
        logger.info(f"Missing data detected in the correlation matrix. Applying linear interpolation.")
        wide = wide.interpolate(method='linear')
    
    # Workaround for type issue by running the correlation function with the correct type
    if method == 'pearson':
        corr = wide.corr(method='pearson')
    elif method == 'kendall':
        corr = wide.corr(method='kendall')
    elif method == 'spearman':
        corr = wide.corr(method='spearman')
    else:
        # Default to pearson
        logger.warning(f"Unknown method {method}, using pearson")
        corr = wide.corr(method='pearson')
        
    logger.info(f"Correlation matrix ({method}) calculated for {metric}, {phase}, {round_id}: {corr.shape}")
    return corr

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
        
    plt.figure(figsize=(10, 8))
    
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
        cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.8}
    )
    
    plt.title(f'{method.title()} correlation between tenants\n{metric} - {phase} - {round_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    # Corrected formatting to match the covariance plots standard
    out_path = os.path.join(out_dir, f"correlation_heatmap_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info(f"Correlation heatmap saved at {out_path}")
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

def compute_cross_correlation(df: pd.DataFrame, metric: str, phase: str, round_id: str, tenants: list[str] | None = None, max_lag: int = 20) -> dict:
    """
    Calculates the cross-correlation (CCF) between pairs of tenants with lag.
    
    Args:
        df: DataFrame in long format
        metric: Name of the metric
        phase: Experimental phase
        round_id: Round ID
        tenants: List of tenants to consider (if None, uses all)
        max_lag: Maximum number of lags to consider
        
    Returns:
        Dictionary with tenant pairs as keys and CCF arrays as values
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"No data for CCF calculation: {metric}, {phase}, {round_id}")
        return {}
    
    # Convert to wide format
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    if tenants is None:
        tenants = wide.columns.tolist()
    
    # Check if we have at least 2 tenants
    if len(tenants) < 2:
        logger.warning(f"Less than 2 tenants available for CCF: {tenants}")
        return {}
    
    # Handle missing data
    if wide.isna().any().any():
        wide = wide.interpolate(method='linear')
    
    ccf_results = {}
    for i, tenant1 in enumerate(tenants):
        for tenant2 in tenants[i+1:]:  # Avoid duplication
            if tenant1 not in wide.columns or tenant2 not in wide.columns:
                continue
                
            # Get both time series
            ts1 = wide[tenant1].fillna(wide[tenant1].mean())
            ts2 = wide[tenant2].fillna(wide[tenant2].mean())
            
            # Normalize data (important for CCF)
            ts1_norm = (ts1 - ts1.mean()) / ts1.std()
            ts2_norm = (ts2 - ts2.mean()) / ts2.std()
            
            # Calculate CCF for positive lags (tenant1 -> tenant2)
            ccf_vals = []
            for lag in range(max_lag+1):
                if lag == 0:
                    # For lag=0, correlation is symmetric
                    corr = np.corrcoef(ts1_norm, ts2_norm)[0, 1]
                    ccf_vals.append(corr)
                else:
                    # For lag > 0, calculate correlation with offset
                    corr = np.corrcoef(ts1_norm[:-lag], ts2_norm[lag:])[0, 1]
                    ccf_vals.append(corr)
            
            # Calculate CCF for negative lags (tenant2 -> tenant1) and reverse the order
            neg_ccf_vals = []
            for lag in range(1, max_lag+1):
                corr = np.corrcoef(ts1_norm[lag:], ts2_norm[:-lag])[0, 1]
                neg_ccf_vals.append(corr)
            
            # Combine negative (inverted) + zero + positive
            full_ccf = neg_ccf_vals[::-1] + ccf_vals
            ccf_results[(tenant1, tenant2)] = full_ccf
    
    return ccf_results


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
        plt.figure(figsize=(12, 6))
        
        # Adjust lags to center around zero
        lags = np.arange(-max_lag, max_lag + 1)
        plt.stem(lags, ccf_vals, linefmt='b-', markerfmt='bo', basefmt='r-')
        
        # Add reference lines
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.6)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.6)
        
        # Find the lag with the highest correlation (in absolute value)
        max_idx = np.argmax(np.abs(ccf_vals))
        max_lag_val = lags[max_idx]
        max_corr = ccf_vals[max_idx]
        
        # Mark the correlation peak
        plt.plot(max_lag_val, max_corr, 'ro', markersize=10)
        plt.annotate(f'Max: {max_corr:.3f} @ lag {max_lag_val}', 
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
        
        plt.title(f'Cross-correlation between {tenant1} and {tenant2}\n{metric} - {phase} - {round_id}\nRelationship: {direction}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add confidence bands (approximation with 1.96/sqrt(N))
        n_samples = 2 * max_lag + 1  # Approximation
        conf_interval = 1.96 / np.sqrt(n_samples)
        plt.axhspan(-conf_interval, conf_interval, alpha=0.2, color='gray')
        
        plt.tight_layout()
        
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ccf_{tenant1}_{tenant2}_{metric}_{phase}_{round_id}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        logger.info(f"CCF plot saved at {out_path}")
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
    
    plt.figure(figsize=(10, 8))
    
    # Calculate correlation on the paired data
    correlation = paired_data.corr().iloc[0, 1]
    
    # Scatter plot with trend line
    sns.regplot(x=tenant1, y=tenant2, data=paired_data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title(f'{title}\n{metric} - {phase} - {round_id}\nCorrelation: {correlation:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel(f'{tenant1} (t{"-"+str(lag) if lag > 0 else ""})')
    plt.ylabel(f'{tenant2} (t{"+"+str(abs(lag)) if lag < 0 else ""})')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lag_scatter_{tenant1}_{tenant2}_{lag}_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info(f"Lag scatter plot saved at {out_path}")
    return out_path
