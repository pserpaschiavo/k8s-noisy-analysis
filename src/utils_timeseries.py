#!/usr/bin/env python3
"""
Module: utils_timeseries.py
Description: Time series analysis utility functions.

This module contains helper functions for manipulating and processing time series data,
including stationarity checks, differencing, and other common transformations.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union
import warnings
from statsmodels.tsa.stattools import adfuller, grangercausalitytests

# Setup logging
logger = logging.getLogger(__name__)

def check_and_transform_timeseries(
    data: pd.DataFrame, 
    max_lag: int = 5,
    min_points: int = 10,
    difference_if_nonstationary: bool = True
) -> Tuple[pd.DataFrame, bool]:
    """
    Checks and transforms a time series for causal analysis.
    
    Args:
        data: DataFrame with two columns (target and possible cause)
        max_lag: Maximum lag for analysis (requires sufficient data)
        min_points: Minimum number of points required after transformations
        difference_if_nonstationary: If True, applies differencing to non-stationary series
    
    Returns:
        Tuple[pd.DataFrame, bool]: Transformed DataFrame and flag indicating if valid for analysis
    """
    # Check if there is enough data
    if len(data) < max_lag + 5:  # Requires at least 5 points beyond maxlag
        logger.warning(f"Insufficient data: {len(data)} points, minimum {max_lag+5}")
        return data, False
    
    # Copy to avoid modifying the original DataFrame
    transformed_data = data.copy()
    
    # Check stationarity if requested
    if difference_if_nonstationary:
        try:
            # First-order differencing if not stationary
            for col in range(transformed_data.shape[1]):
                adf_result = adfuller(transformed_data.iloc[:, col], autolag='AIC')
                if adf_result[1] > 0.05:  # p-value > 0.05 indicates non-stationarity
                    # Apply differencing
                    transformed_data.iloc[:, col] = transformed_data.iloc[:, col].diff().dropna()
            
            # Remove NaN values after differencing
            transformed_data = transformed_data.dropna()
        except Exception as e:
            logger.warning(f"Error checking stationarity: {str(e)}")
    
    # Check size again after transformations
    if len(transformed_data) <= max_lag + 3:
        logger.warning(f"Insufficient data after transformations: {len(transformed_data)} points")
        return transformed_data, False
    
    return transformed_data, True

def resample_and_align_timeseries(
    data: pd.DataFrame,
    freq: str = '1min',
    fill_method: str = 'interpolate',
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Resamples and aligns time series to a uniform frequency.
    
    Args:
        data: DataFrame with timestamp index
        freq: Resampling frequency (e.g., '1min', '5min')
        fill_method: Method to fill missing values ('interpolate', 'ffill', etc)
        min_periods: Minimum number of observations for valid interpolation
    
    Returns:
        DataFrame with aligned and resampled time series
    """
    # Check if the index is datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        logger.warning("Index is not of datetime type, attempting to convert")
        try:
            data.index = pd.to_datetime(data.index)
        except:
            logger.error("Could not convert index to datetime")
            return data
    
    # Resample to regular frequency
    resampled = data.resample(freq).mean()
    
    # Fill missing values with the chosen method
    if fill_method == 'interpolate':
        filled = resampled.interpolate(method='time', limit_area='inside', min_periods=min_periods)
    elif fill_method == 'ffill':
        filled = resampled.ffill()
    elif fill_method == 'bfill':
        filled = resampled.bfill()
    else:
        filled = resampled.interpolate(method='time').ffill().bfill()
    
    return filled

def run_granger_causality_test(
    data: pd.DataFrame, 
    maxlag: int = 3,
    verbose: bool = False
) -> dict:
    """
    Runs the Granger causality test with robust error handling.
    
    Args:
        data: DataFrame with two columns (target and possible cause)
        maxlag: Maximum number of lags for the test
        verbose: If True, prints detailed results
    
    Returns:
        Dictionary with test results or None if it fails
    """
    if data.shape[1] != 2:
        logger.error(f"DataFrame must have exactly 2 columns, found {data.shape[1]}")
        return None
        
    # Check and transform the data
    transformed_data, valid = check_and_transform_timeseries(
        data, 
        max_lag=maxlag,
        min_points=maxlag + 5,
        difference_if_nonstationary=True
    )
    
    if not valid:
        logger.warning("Insufficient or invalid data for Granger test")
        return None
        
    # Run the test with exception handling
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            test_results = grangercausalitytests(
                transformed_data, 
                maxlag=maxlag, 
                verbose=verbose
            )
            
            # Extract the smallest p-value among all lags
            p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
            min_p_value = min(p_values) if p_values else np.nan
            
            return {
                'p_values': p_values,
                'min_p_value': min_p_value,
                'results': test_results
            }
            
    except Exception as e:
        logger.warning(f"Error running Granger test: {str(e)}")
        return None

def check_stationarity(series: pd.Series, p_threshold: float = 0.05) -> Tuple[bool, float]:
    """
    Checks if a time series is stationary using the Augmented Dickey-Fuller test.
    
    Args:
        series (pd.Series): The time series to check.
        p_threshold (float): The significance level to reject the null hypothesis.
        
    Returns:
        Tuple[bool, float]: A tuple containing a boolean indicating if the series is stationary
                            and the p-value of the test.
    """
    if series.empty or series.isnull().all():
        logger.warning("Input series is empty or all NaN. Cannot check for stationarity.")
        return False, 1.0 # Not stationary, p-value is 1.0

    try:
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < p_threshold
        logger.debug(f"ADF test p-value: {p_value}. Stationary: {is_stationary}")
        return is_stationary, p_value
    except Exception as e:
        logger.error(f"Error during ADF test: {e}")
        return False, 1.0

def make_series_stationary(series: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
    """
    Makes a time series stationary by applying differencing.
    
    Args:
        series (pd.Series): The time series to make stationary.
        max_diff (int): The maximum number of differencing operations to apply.
        
    Returns:
        Tuple[pd.Series, int]: A tuple containing the stationary series and the number of
                               differencing operations performed.
    """
    stationary_series = series.copy()
    diff_order = 0
    for i in range(max_diff):
        is_stationary, _ = check_stationarity(stationary_series)
        if is_stationary:
            logger.info(f"Series is stationary after {i} differencing operations.")
            return stationary_series, diff_order
            
        stationary_series = stationary_series.diff().dropna()
        diff_order += 1
        
    is_stationary, _ = check_stationarity(stationary_series)
    if not is_stationary:
        logger.warning(f"Series could not be made stationary after {max_diff} differencing operations.")
        
    return stationary_series, diff_order

def run_granger_causality(data: pd.DataFrame, cause_col: str, effect_col: str, max_lag: int = 5, p_threshold: float = 0.05) -> Optional[Dict]:
    """
    Performs the Granger causality test between two time series.
    
    Args:
        data (pd.DataFrame): DataFrame containing the two time series.
        cause_col (str): The name of the column representing the cause.
        effect_col (str): The name of the column representing the effect.
        max_lag (int): The maximum number of lags to test.
        p_threshold (float): The significance level.
        
    Returns:
        Optional[Dict]: A dictionary with the test results if successful, otherwise None.
    """
    if cause_col not in data.columns or effect_col not in data.columns:
        logger.error(f"Columns '{cause_col}' or '{effect_col}' not found in DataFrame.")
        return None

    test_data = data[[effect_col, cause_col]].dropna()
    
    if len(test_data) < 3 * max_lag:
        logger.warning(f"Not enough data to perform Granger causality test for {cause_col} -> {effect_col}. Need at least {3 * max_lag} observations, but have {len(test_data)}.")
        return None

    try:
        # The grangercausalitytests function prints results to stdout, which can be noisy.
        # We can capture this if needed, but for now, we let it print.
        gc_result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
        
        # Find the minimum p-value among all lags
        min_p_value = 1.0
        for lag in gc_result:
            p_value = gc_result[lag][0]['ssr_ftest'][1]
            if p_value < min_p_value:
                min_p_value = p_value
                
        is_causal = min_p_value < p_threshold
        
        return {
            "cause": cause_col,
            "effect": effect_col,
            "p_value": min_p_value,
            "is_causal": is_causal,
            "max_lag": max_lag
        }
    except Exception as e:
        logger.error(f"Error running Granger causality for {cause_col} -> {effect_col}: {e}")
        return None
