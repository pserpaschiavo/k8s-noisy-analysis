"""
Module: analysis_correlation.py
Description: Correlation analysis utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
# Removendo configuração local do matplotlib
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.stattools import acf

from src.utils import configure_matplotlib
from src.gpu_acceleration import check_gpu_availability, calculate_correlation_matrix_gpu

# Configuração centralizada do matplotlib
configure_matplotlib()

from src.visualization.correlation_plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_ccf,
    plot_lag_scatter
)

# Logging setup
logger = logging.getLogger(__name__)

plt.style.use('tableau-colorblind10')

def compute_correlation_matrix(df_identifier, metric: str, phase: str, round_id: str, method: str = 'pearson', 
                              use_gpu: bool = False, large_dataset_threshold: int = 10000) -> pd.DataFrame:
    """
    Computes the correlation matrix (Pearson or Spearman) between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    
    Args:
        df_identifier: DataFrame identifier or the DataFrame itself
        metric: Name of the metric for analysis
        phase: Experimental phase to filter
        round_id: Round ID to filter
        method: Correlation method ('pearson', 'kendall', or 'spearman')
        use_gpu: Se True, tenta usar aceleração GPU para correlação de Pearson
        large_dataset_threshold: Número de linhas para considerar um dataset grande e usar GPU
        
    Returns:
        DataFrame with the correlation matrix or an empty DataFrame if data is insufficient.
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
    
    # Enhanced validation: Check if subset is empty
    if subset.empty:
        logger.warning(f"Skipping correlation for {metric}, {phase}, {round_id}: No data found after initial filtering.")
        return pd.DataFrame()
    
    # Enhanced validation: Check for sufficient number of tenants
    if subset['tenant_id'].nunique() < 2:
        logger.warning(f"Skipping correlation for {metric}, {phase}, {round_id}: Requires at least 2 tenants, but found {subset['tenant_id'].nunique()}.")
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

    # Final check on columns after pivot
    if wide.shape[1] < 2:
        logger.warning(f"Skipping correlation for {metric}, {phase}, {round_id}: Less than 2 tenants available after pivoting data.")
        return pd.DataFrame()

    # Check for missing data and handle it
    if wide.isna().any().any():
        logger.info(f"Missing data detected in the correlation matrix. Applying linear interpolation.")
        wide = wide.interpolate(method='linear')
    
    # Verificar se devemos usar GPU (apenas para correlação de Pearson)
    use_gpu_for_calc = use_gpu and method == 'pearson' and (len(wide) > large_dataset_threshold)
    if use_gpu_for_calc and check_gpu_availability():
        # Usar aceleração GPU para correlação
        logger.info(f"Usando aceleração GPU para cálculo de correlação com {len(wide)} registros")
        corr = calculate_correlation_matrix_gpu(wide, method='pearson')
    else:
        # Fallback para CPU - comportamento original
        if use_gpu and method == 'pearson' and (len(wide) > large_dataset_threshold):
            logger.info("GPU solicitada mas não disponível. Usando CPU para correlação.")
            
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


def compute_covariance_matrix(df: pd.DataFrame, metric: str, phase: str, round_id: str) -> pd.DataFrame:
    """
    Computes the covariance matrix between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    
    if subset.empty:
        logger.warning(f"Skipping covariance for {metric}, {phase}, {round_id}: No data found after filtering.")
        return pd.DataFrame()

    if subset['tenant_id'].nunique() < 2:
        logger.warning(f"Skipping covariance for {metric}, {phase}, {round_id}: Requires at least 2 tenants, but found {subset['tenant_id'].nunique()}.")
        return pd.DataFrame()

    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')

    if wide.shape[1] < 2:
        logger.warning(f"Skipping covariance for {metric}, {phase}, {round_id}: Less than 2 tenants available after pivoting.")
        return pd.DataFrame()

    cov = wide.cov()
    return cov

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


def compute_aggregated_correlation(df: pd.DataFrame, metrics: list[str], rounds: list[str], phases: list[str], method: str = 'pearson') -> dict[str, pd.DataFrame]:
    """
    Computes the aggregated correlation matrix across multiple rounds and phases for each metric.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        metrics (list[str]): A list of metrics to analyze.
        rounds (list[str]): A list of round IDs to include.
        phases (list[str]): A list of experimental phases to include.
        method (str): The correlation method to use ('pearson', 'spearman', etc.).

    Returns:
        dict[str, pd.DataFrame]: A dictionary mapping each metric to its aggregated correlation matrix.
    """
    aggregated_correlations = {}
    for metric in metrics:
        all_correlations = []
        for round_id in rounds:
            for phase in phases:
                # Assuming compute_correlation_matrix is available in this module
                corr_matrix = compute_correlation_matrix(df, metric, phase, round_id, method)
                if not corr_matrix.empty:
                    all_correlations.append(corr_matrix)
        
        if all_correlations:
            # Use pd.concat and groupby to calculate the mean of all correlation matrices
            aggregated_matrix = pd.concat(all_correlations).groupby(level=0).mean()
            
            # Clear the index name to prevent conflicts when resetting the index later
            aggregated_matrix.index.name = None
            
            aggregated_correlations[metric] = aggregated_matrix
            logger.info(f"Aggregated correlation matrix for metric '{metric}' computed with shape {aggregated_matrix.shape}.")
        else:
            logger.warning(f"No correlation matrices could be computed for metric '{metric}' across the specified rounds and phases.")

    return aggregated_correlations
