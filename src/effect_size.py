"""
Module: effect_size.py
Description: Functions for calculating effect size between different experimental phases.
"""
import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import multiprocessing
from functools import lru_cache
from pathlib import Path
import time

# Import GPU acceleration module
from src.gpu_acceleration import check_gpu_availability, to_gpu, to_cpu

logger = logging.getLogger(__name__)

def cohens_d(group1, group2):
    """
    Calculates Cohen's d for independent samples.
    
    Args:
        group1: Array-like with values from group 1
        group2: Array-like with values from group 2
        
    Returns:
        float: Effect size (Cohen's d)
    """
    # Check if groups have enough data
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    
    # Calculate means and standard deviations for both groups
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # If standard deviation is zero in both groups, return NaN
    if std1 == 0 and std2 == 0:
        return np.nan
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Prevention of division by zero
    if pooled_std == 0:
        return np.nan
    
    # Cohen's d
    d = (mean2 - mean1) / pooled_std
    return d

def cohens_d_gpu(group1, group2):
    """
    Calculates Cohen's d for independent samples using GPU.
    
    Args:
        group1: Array-like with values from group 1
        group2: Array-like with values from group 2
        
    Returns:
        float: Effect size (Cohen's d)
    """
    # Check if groups have enough data
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    
    # Transfer to GPU
    gpu_group1 = to_gpu(np.array(group1))
    gpu_group2 = to_gpu(np.array(group2))
    
    # Get backend
    backend = check_gpu_availability()
    
    # Calculate statistics using the appropriate backend
    if backend == "cupy":
        import cupy as cp
        mean1, mean2 = cp.mean(gpu_group1), cp.mean(gpu_group2)
        std1, std2 = cp.std(gpu_group1, ddof=1), cp.std(gpu_group2, ddof=1)
        
        # If standard deviation is zero in both groups, return NaN
        if std1 == 0 and std2 == 0:
            return np.nan
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = cp.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Prevention of division by zero
        if pooled_std == 0:
            return np.nan
        
        # Cohen's d
        d = (mean2 - mean1) / pooled_std
        return float(to_cpu(d))
    
    elif backend == "torch":
        import torch
        mean1, mean2 = torch.mean(gpu_group1), torch.mean(gpu_group2)
        std1, std2 = torch.std(gpu_group1, unbiased=True), torch.std(gpu_group2, unbiased=True)
        
        # If standard deviation is zero in both groups, return NaN
        if std1 == 0 and std2 == 0:
            return np.nan
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = torch.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Prevention of division by zero
        if pooled_std == 0:
            return np.nan
        
        # Cohen's d
        d = (mean2 - mean1) / pooled_std
        return float(to_cpu(d))
    
    elif backend == "tensorflow":
        import tensorflow as tf
        mean1, mean2 = tf.reduce_mean(gpu_group1), tf.reduce_mean(gpu_group2)
        std1 = tf.sqrt(tf.reduce_sum(tf.square(gpu_group1 - mean1)) / (len(group1) - 1))
        std2 = tf.sqrt(tf.reduce_sum(tf.square(gpu_group2 - mean2)) / (len(group2) - 1))
        
        # If standard deviation is zero in both groups, return NaN
        if std1 == 0 and std2 == 0:
            return np.nan
        
        # Pooled standard deviation
        n1, n2 = len(group1), len(group2)
        pooled_std = tf.sqrt(((n1 - 1) * tf.square(std1) + (n2 - 1) * tf.square(std2)) / (n1 + n2 - 2))
        
        # Prevention of division by zero
        if pooled_std == 0:
            return np.nan
        
        # Cohen's d
        d = (mean2 - mean1) / pooled_std
        return float(to_cpu(d))
    
    else:
        # Fallback to CPU version
        return cohens_d(group1, group2)

def eta_squared(group1, group2):
    """
    Calculates Eta-squared effect size between two groups.
    
    Args:
        group1: Array-like with values from group 1
        group2: Array-like with values from group 2
        
    Returns:
        float: Effect size (Eta-squared)
    """
    # Check if groups have enough data
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    
    # Combine groups into a single array
    all_data = np.concatenate([group1, group2])
    
    # If all values are the same, return NaN (no variation)
    if np.all(all_data == all_data[0]):
        return np.nan
    
    # Calculate total sum of squares
    ss_total = np.sum((all_data - np.mean(all_data))**2)
    
    # Prevention of division by zero
    if ss_total == 0:
        return np.nan
    
    # Calculate between-group sum of squares
    mean1, mean2 = np.mean(group1), np.mean(group2)
    ss_between = len(group1) * (mean1 - np.mean(all_data))**2 + len(group2) * (mean2 - np.mean(all_data))**2
    
    # Eta-squared
    eta_sq = ss_between / ss_total
    return eta_sq

def t_test(group1, group2):
    """
    Performs independent t-test between two groups.
    
    Args:
        group1: Array-like with values from group 1
        group2: Array-like with values from group 2
        
    Returns:
        Tuple[float, float]: t-statistic and p-value
    """
    # Check if groups have enough data
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan
    
    # Perform t-test
    try:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
        return t_stat, p_val
    except:
        return np.nan, np.nan

def _process_effect_size(args):
    """
    Helper function to process a single effect size calculation.
    
    Args:
        args: Tuple with (df, round_id, metric, phase, tenant, baseline_phase, use_gpu)
        
    Returns:
        Dict with effect size results or None if error occurs
    """
    # Unpack arguments
    if len(args) >= 7:
        df, round_id, metric, phase, tenant, baseline_phase, use_gpu = args
    else:
        df, round_id, metric, phase, tenant, baseline_phase = args
        use_gpu = False  # Default value if not provided
        
    try:
        # Filter data for baseline
        base_data = df[(df['round_id'] == round_id) & 
                    (df['metric_name'] == metric) & 
                    (df['experimental_phase'] == baseline_phase) &
                    (df['tenant_id'] == tenant)]
        
        # Filter data for experimental phase
        exp_data = df[(df['round_id'] == round_id) & 
                    (df['metric_name'] == metric) & 
                    (df['experimental_phase'] == phase) &
                    (df['tenant_id'] == tenant)]
        
        # Check for sufficient data
        if base_data.empty or exp_data.empty:
            return None
            
        base_values = base_data['metric_value'].values
        exp_values = exp_data['metric_value'].values
        
        if len(base_values) < 2 or len(exp_values) < 2:
            return None
            
        # Calculate effect size and statistical tests
        if use_gpu and check_gpu_availability() and (len(base_values) + len(exp_values) > 1000):
            # Use GPU versions of functions for large datasets
            d = cohens_d_gpu(base_values, exp_values)
            _, p_value = t_test(base_values, exp_values)  # Using t-test on CPU for simplicity
            eta_sq = eta_squared(base_values, exp_values)  # Using eta_squared on CPU for simplicity
        else:
            # Use CPU versions for smaller datasets
            d = cohens_d(base_values, exp_values)
            _, p_value = t_test(base_values, exp_values)
            eta_sq = eta_squared(base_values, exp_values)
        
        # Return results
        return {
            'round_id': round_id,
            'metric_name': metric,
            'experimental_phase': phase,
            'tenant_id': tenant,
            'baseline_phase': baseline_phase,
            'effect_size': d,
            'p_value': p_value,
            'eta_squared': eta_sq,
            'sample_size_baseline': len(base_values),
            'sample_size_experimental': len(exp_values)
        }
    except Exception as e:
        logger.warning(f"Error processing effect for {round_id}/{metric}/{phase}/{tenant}: {e}")
        return None

def extract_effect_sizes(
    df_long: pd.DataFrame,
    rounds: List[str],
    metrics: List[str],
    phases: List[str],
    tenants: List[str],
    baseline_phase: str = "1 - Baseline",
    use_cache: bool = True,
    parallel: bool = False,
    cache_dir: Optional[str] = None,
    use_gpu: bool = False,
    large_dataset_threshold: int = 10000
) -> pd.DataFrame:
    """
    Extracts effect size statistics (Cohen's d) and p-values for
    phase vs. baseline comparisons for each metric, tenant, and round.
    
    Args:
        df_long: Long-format DataFrame with all data
        rounds: List of rounds for analysis
        metrics: List of metrics for analysis
        phases: List of phases for analysis
        tenants: List of tenants for analysis
        baseline_phase: Name of the baseline phase (default: "1 - Baseline")
        use_cache: If True, uses cache to avoid recalculations
        parallel: If True, parallelizes processing
        cache_dir: Directory to store cache (optional)
        use_gpu: If True, attempts to use GPU acceleration for intensive calculations
        large_dataset_threshold: Number of rows to consider a large dataset
        
    Returns:
        DataFrame with columns: round_id, metric_name, experimental_phase, 
        tenant_id, baseline_phase, effect_size, p_value, eta_squared, etc.
    """
    start_time = time.time()
    logger.info("Starting effect size extraction...")
    
    # Check GPU availability if requested
    if use_gpu:
        gpu_available = check_gpu_availability()
        if gpu_available:
            logger.info(f"GPU acceleration enabled for effect size extraction")
        else:
            logger.info(f"Requested GPU not available. Using CPU for calculations.")
            use_gpu = False
    
    # Configure cache, if requested
    cache_file = None
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "effect_sizes_cache.parquet")
        
        # If cache exists, load it
        if os.path.exists(cache_file):
            try:
                logger.info(f"Loading results from cache: {cache_file}")
                cached_df = pd.read_parquet(cache_file)
                logger.info(f"Cache loaded successfully: {cached_df.shape[0]} records")
                return cached_df
            except Exception as e:
                logger.warning(f"Error loading cache: {e}. Recalculating...")
    
    # Prepare arguments for processing
    args_list = []
    for round_id in rounds:
        for metric_name in metrics:
            for phase in phases:
                if phase == baseline_phase:
                    continue  # Skip baseline comparison
                for tenant_id in tenants:
                    # Add GPU usage flag to arguments
                    args_list.append((df_long, round_id, metric_name, phase, tenant_id, baseline_phase, use_gpu))
    
    # Process data (in parallel or sequentially)
    results = []
    if parallel and len(args_list) > 10:  # Only use parallelism if many items
        logger.info(f"Processing {len(args_list)} combinations in parallel...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(_process_effect_size, args_list)
    else:
        logger.info(f"Processing {len(args_list)} combinations sequentially...")
        results = [_process_effect_size(args) for args in args_list]
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Convert results to DataFrame
    if not results:
        logger.warning("No results calculated. Check input data.")
        return pd.DataFrame()
    
    effect_sizes_df = pd.DataFrame(results)
    
    # Add quality assessment of results
    effect_sizes_df['result_quality'] = effect_sizes_df.apply(
        lambda row: 'high' if (not pd.isna(row['effect_size']) and 
                              not pd.isna(row['p_value']) and 
                              row['sample_size_baseline'] >= 30 and 
                              row['sample_size_experimental'] >= 30) else
                   ('medium' if (not pd.isna(row['effect_size']) and 
                                not pd.isna(row['p_value']) and 
                                row['sample_size_baseline'] >= 10 and 
                                row['sample_size_experimental'] >= 10) else 'low'),
        axis=1
    )
    
    # Save cache, if requested
    if use_cache and cache_file:
        try:
            effect_sizes_df.to_parquet(cache_file)
            logger.info(f"Results saved to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Error saving cache: {e}")
    
    elapsed = time.time() - start_time
    logger.info(f"Effect size extraction completed in {elapsed:.2f}s")
    
    return effect_sizes_df
