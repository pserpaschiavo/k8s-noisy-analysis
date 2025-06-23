"""
Module: analysis_causality.py
Description: Causality analysis utilities for multi-tenant time series analysis, including graph visualization.

This module implements methods for causality analysis between time series of different tenants,
including Granger causality and Transfer Entropy (TE), as well as graph visualizations.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import MissingDataError
from functools import lru_cache
import warnings

from src.visualization.plots import plot_causality_graph, plot_causality_heatmap

# Import utilities for time series processing
try:
    from src.utils_timeseries import check_and_transform_timeseries, resample_and_align_timeseries
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    logging.warning("utils_timeseries module not available. Some optimizations will not be applied.")

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific statsmodels warnings
warnings.filterwarnings("ignore", "Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", "Unable to solve the eigenvalue problem")

# Import the pyinform library for more robust Transfer Entropy
try:
    from pyinform.transferentropy import transfer_entropy
    PYINFORM_AVAILABLE = True
    logger.info("pyinform available. Using optimized implementation for Transfer Entropy.")
except ImportError:
    PYINFORM_AVAILABLE = False
    logger.warning("pyinform is not installed. Transfer Entropy will use a basic implementation. Install with: pip install pyinform")

plt.style.use('tableau-colorblind10')

def _transfer_entropy(x, y, bins=5, k=1):
    """
    Calculates the Transfer Entropy (TE) from y to x (y→x) using an optimized implementation.
    
    Args:
        x: 1D array representing the target time series
        y: 1D array representing the source time series
        bins: Number of bins for discretization (default=5)
        k: History of the target series to consider (default=1)
        
    Returns:
        Scalar TE value (y→x): how much y helps predict x beyond the history of x
    """
    # Check input data
    min_points = 8  # Reduced from 10 to allow more calculations
    if len(x) != len(y):
        logger.warning(f"Unequal lengths for TE: x={len(x)}, y={len(y)}. Aligning series.")
        # Truncate to the shorter length
        length = min(len(x), len(y))
        x = x[:length]
        y = y[:length]
    
    if len(x) < min_points:
        logger.warning(f"Insufficient data for TE: {len(x)} points available. Minimum {min_points} points.")
        return 0.0
    
    # Handle missing values (NaN)
    x = np.nan_to_num(x, nan=np.nanmean(x))
    y = np.nan_to_num(y, nan=np.nanmean(y))
    
    # If pyinform is available, use the optimized implementation
    if 'transfer_entropy' in globals() and PYINFORM_AVAILABLE:
        try:
            # Use the more robust implementation from the pyinform library
            # Automatic normalization and binning of series
            # Convert to integers as required by pyinform
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
            
            x_bin = np.floor(x_norm * (bins-1)).astype(int)
            y_bin = np.floor(y_norm * (bins-1)).astype(int)
            
            # Calculate TE using pyinform
            from pyinform.transferentropy import transfer_entropy
            te_value = transfer_entropy(y_bin, x_bin, k=k, local=False)
            return float(te_value)  # Ensure scalar type
        except Exception as e:
            logger.warning(f"Error using pyinform for TE: {e}. Using basic implementation.")
    
    # Basic fallback implementation using numpy histograms
    # Discretize the series
    x_binned = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_binned = np.digitize(y, np.histogram_bin_edges(y, bins=bins))
    
    # Calculate joint and conditional probabilities
    px = np.histogram(x_binned[1:], bins=bins, density=True)[0]
    pxy = np.histogram2d(x_binned[:-1], x_binned[1:], bins=bins, density=True)[0]
    pxyy = np.histogramdd(np.stack([x_binned[:-1], y_binned[:-1], x_binned[1:]], axis=1), 
                         bins=(bins, bins, bins), density=True)[0]
    
    # Add regularization to avoid log(0)
    pxyy = pxyy + 1e-12
    pxy = pxy + 1e-12
    px = px + 1e-12
    
    # TE(y→x) = sum p(x_{t+1}, x_t, y_t) * log [p(x_{t+1}|x_t, y_t) / p(x_{t+1}|x_t)]
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k_ in range(bins):
                pxyz = pxyy[j, k_, i]
                pxz = pxy[j, i]
                px_i = px[i]
                if pxyz > 0 and pxz > 0 and px_i > 0:
                    te += pxyz * np.log((pxyz / (np.sum(pxyy[j, k_, :]) + 1e-12)) / 
                                       (pxz / (np.sum(pxy[j, :]) + 1e-12)))
    return te

class CausalityAnalyzer:
    """
    Class responsible for causality calculations (e.g., Granger, Transfer Entropy) between tenants.
    """
    def __init__(self, df: pd.DataFrame, output_dir: str):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def run_and_plot_causality_analysis(self, metric: str, phase: str, round_id: str, p_value_threshold: float = 0.05) -> dict:
        """
        Orchestrates the full causality analysis pipeline for a given metric, phase, and round.
        It computes causality matrices, generates visualizations (graphs and heatmaps),
        and saves them to the output directory.

        Args:
            metric: The metric to analyze.
            phase: The experimental phase.
            round_id: The round identifier.
            p_value_threshold: Significance level for Granger causality.
            
        Returns:
            A dictionary containing the computed matrices and paths to the generated plots.
        """
        logger.info(f"Starting causality analysis for metric='{metric}', phase='{phase}', round='{round_id}'")
        plot_paths = []
        results = {
            "granger_matrix": pd.DataFrame(),
            "te_matrix": pd.DataFrame(),
            "plot_paths": []
        }

        # --- Granger Causality ---
        granger_p_matrix = self.compute_granger_matrix(metric, phase, round_id)
        results["granger_matrix"] = granger_p_matrix

        if not granger_p_matrix.empty:
            # Define output path for the graph
            graph_filename = os.path.join(self.output_dir, f"granger_graph_{metric}_{phase}_{round_id}.png")
            
            # Plot and save Granger causality graph
            path = plot_causality_graph(
                causality_matrix=granger_p_matrix,
                out_path=graph_filename,
                metric=metric,
                threshold=p_value_threshold,
                threshold_mode='less'
            )
            if path: plot_paths.append(path)

            # Plot and save Granger p-value heatmap
            path = plot_causality_heatmap(
                causality_matrix=granger_p_matrix,
                metric=metric,
                phase=phase,
                round_id=round_id,
                out_dir=self.output_dir,
                method='Granger',
                value_type='p-value'
            )
            if path: plot_paths.append(path)

        # --- Transfer Entropy ---
        te_matrix = self.compute_transfer_entropy_matrix(metric, phase, round_id)
        results["te_matrix"] = te_matrix

        if not te_matrix.empty:
            # Define a threshold for TE score to be considered significant
            te_threshold = 0.1  # This threshold may require tuning

            # Plot and save Transfer Entropy causality graph
            te_graph_filename = os.path.join(self.output_dir, f"te_graph_{metric}_{phase}_{round_id}.png")
            path = plot_causality_graph(
                causality_matrix=te_matrix,
                out_path=te_graph_filename,
                metric=metric,
                threshold=te_threshold,
                threshold_mode='greater',
                directed=True
            )
            if path: plot_paths.append(path)

            # Plot and save Transfer Entropy score heatmap
            path = plot_causality_heatmap(
                causality_matrix=te_matrix,
                metric=metric,
                phase=phase,
                round_id=round_id,
                out_dir=self.output_dir,
                method='TE',
                value_type='score'
            )
            if path: plot_paths.append(path)

        logger.info(f"Finished causality analysis for metric='{metric}', phase='{phase}', round='{round_id}'")
        results["plot_paths"] = plot_paths
        return results


    def _granger_causality_test(self, source_series: np.ndarray, target_series: np.ndarray, max_lag: int = 3) -> dict:
        """
        Executes a Granger causality test between two time series.
        
        Args:
            source_series: Source time series (potential cause)
            target_series: Target time series (potential effect)
            max_lag: Maximum number of lags to test
            
        Returns:
            Dictionary with test results or an empty dictionary if it fails
        """
        try:
            # Check if the series are constant (very low standard deviation)
            if np.std(source_series) < 1e-6 or np.std(target_series) < 1e-6:
                logger.warning("Constant series detected - cannot run Granger test")
                return {}
                
            # Check for missing values
            if np.isnan(source_series).any() or np.isnan(target_series).any():
                # Try to interpolate missing values
                source_series_pd = pd.Series(source_series).interpolate().bfill().ffill()
                target_series_pd = pd.Series(target_series).interpolate().bfill().ffill()
                
                # Convert back to numpy arrays
                source_series = source_series_pd.to_numpy()
                target_series = target_series_pd.to_numpy()
                
                # Check if NaNs still exist after interpolation
                if np.isnan(source_series).any() or np.isnan(target_series).any():
                    logger.warning("Missing values could not be corrected by interpolation")
                    return {}
            
            # Adjust max_lag if the series is too short
            if len(source_series) <= max_lag + 2:
                new_max_lag = max(1, len(source_series) // 3 - 1)
                logger.warning(f"Series too short for max_lag={max_lag}. Adjusting to {new_max_lag}")
                max_lag = new_max_lag
            
            # Create a DataFrame with the two series
            data = pd.DataFrame({
                'target': target_series,
                'source': source_series
            })
            
            # Execute the Granger test
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
            return results
        except Exception as e:
            logger.warning(f"Error in Granger causality test: {str(e)}")
            return {}

    @lru_cache(maxsize=32)
    def compute_granger_matrix(self, metric: str, phase: str, round_id: str, maxlag: int = 5) -> pd.DataFrame:
        """
        Calculates the matrix of p-values from the Granger causality test between all tenants for a specific metric.
        Results are cached to improve performance on repeated runs.
        
        Args:
            metric: Name of the metric for analysis
            phase: Experimental phase (e.g., "1-Baseline", "2-CPU-Noise")
            round_id: ID of the round (e.g., "round-1")
            maxlag: Maximum number of lags for the Granger test
            
        Returns:
            DataFrame where mat[i,j] is the lowest p-value of j causing i (considering lags from 1 to maxlag)
        """
        logger.info(f"Calculating Granger causality matrix for {metric}, {phase}, {round_id}, maxlag={maxlag}")
        subset = self.df[(self.df['metric_name'] == metric) & 
                        (self.df['experimental_phase'] == phase) & 
                        (self.df['round_id'] == round_id)]
        
        # Enhanced validation
        if subset.empty:
            logger.warning(f"Skipping Granger for {metric}, {phase}, {round_id}: No data found after filtering.")
            return pd.DataFrame()

        tenants = subset['tenant_id'].unique()
        if len(tenants) < 2:
            logger.warning(f"Skipping Granger for {metric}, {phase}, {round_id}: Requires at least 2 tenants, but found {len(tenants)}.")
            return pd.DataFrame(index=tenants, columns=tenants)

        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        
        # Transform to wide format for time series analysis
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Ensure all tenants are present as columns, filling with NaN if necessary
        wide = wide.reindex(columns=tenants, fill_value=np.nan)

        # Interpolation and filling to handle missing values
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        # Test Granger causality for each pair of tenants
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                
                try:
                    # Select only data with overlapping timestamps
                    data = pd.concat([wide[target], wide[source]], axis=1)
                    data = data.dropna()
                    
                    # Additional check to ensure sufficient data
                    if len(data) < maxlag + 5:  # Requires at least 5 points beyond maxlag
                        logger.warning(f"Insufficient data for Granger test {source}->{target}: {len(data)} points, minimum {maxlag+5}")
                        continue
                        
                    # Check for stationarity before the test
                    from statsmodels.tsa.stattools import adfuller
                    # First-order differentiation if not stationary
                    for col_idx, col_name in enumerate(data.columns):
                        adf_result = adfuller(data[col_name], autolag='AIC')
                        if adf_result[1] > 0.05:  # p-value > 0.05 indicates non-stationarity
                            original_len = len(data)
                            data[col_name] = data[col_name].diff().dropna()
                            logger.warning(f"Series {col_name} was not stationary. Differentiated and reduced from {original_len} to {len(data)} points.")
                            
                    data = data.dropna()  # Remove NaN values after differentiation
                    
                    if len(data) <= maxlag + 3:  # Check again after differentiation
                        logger.warning(f"Insufficient data after differentiation: {len(data)} points")
                        continue
                        
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        # Perform Granger test with improved error handling
                        test_results = grangercausalitytests(
                            data, 
                            maxlag=maxlag, 
                            verbose=False
                        )
                        
                        # Extract the lowest p-value among all lags
                        p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
                        min_p_value = min(p_values) if p_values else np.nan
                    
                    # Store the result in the matrix
                    mat.loc[target, source] = min_p_value
                except KeyError:
                    continue
                except MissingDataError:
                    continue
                except Exception as e:
                    logging.warning(f"Error calculating Granger for {source}->{target}: {str(e)}")
                    continue
                    
        return mat

    @lru_cache(maxsize=32)
    def compute_transfer_entropy_matrix(self, metric: str, phase: str, round_id: str, bins: int = 8, k: int = 1) -> pd.DataFrame:
        """
        Calculates the Transfer Entropy (TE) matrix between all tenants for a specific metric.
        Results are cached to improve performance on repeated runs.
        
        Args:
            metric: Name of the metric for analysis
            phase: Experimental phase (e.g., "1-Baseline", "2-CPU-Noise")
            round_id: ID of the round (e.g., "round-1") 
            bins: Number of bins for discretizing continuous series
            k: History of the target series to consider
            
        Returns:
            DataFrame where mat[i,j] is the Transfer Entropy value from j to i (j→i)
            Higher values indicate greater information transfer
        """
        # Log the start of the calculation
        logger.info(f"Calculating Transfer Entropy matrix for {metric} in {phase} ({round_id}), bins={bins}, k={k}")
        
        # Filter relevant data
        subset = self.df[(self.df['metric_name'] == metric) & 
                        (self.df['experimental_phase'] == phase) & 
                        (self.df['round_id'] == round_id)]
        
        # Enhanced validation
        if subset.empty:
            logger.warning(f"Skipping TE for {metric}, {phase}, {round_id}: No data found after filtering.")
            return pd.DataFrame()

        tenants = subset['tenant_id'].unique()
        if len(tenants) < 2:
            logger.warning(f"Skipping TE for {metric}, {phase}, {round_id}: Requires at least 2 tenants, but found {len(tenants)}.")
            return pd.DataFrame(index=tenants, columns=tenants)

        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        
        # Transform to wide format for analysis
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')

        # Ensure all tenants are present as columns, filling with NaN if necessary
        wide = wide.reindex(columns=tenants, fill_value=np.nan)
        
        # Interpolate and fill NaNs to align all tenants
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        # Calculate TE for each pair of tenants
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                
                try:
                    # Get time series for the tenants
                    target_series = wide[target]
                    source_series = wide[source]
                    
                    # Align indices to ensure temporal correspondence
                    common_idx = target_series.index.intersection(source_series.index)
                    target_values = target_series.loc[common_idx].values
                    source_values = source_series.loc[common_idx].values
                    
                    # Check if there are enough points for a meaningful calculation
                    if len(target_values) >= 8:  # Reduced from 10 to 8 minimum points
                        # Calculate TE and store it in the matrix
                        try:
                            te_value = _transfer_entropy(target_values, source_values, bins=bins, k=k)
                            mat.loc[target, source] = te_value
                        except Exception as calc_error:
                            logging.warning(f"Error in TE calculation for {source}->{target}: {calc_error}")
                    else:
                        logging.warning(f"Insufficient time series for pair {source}->{target}: {len(target_values)} points")
                        
                except Exception as e:
                    logging.error(f"Error calculating Transfer Entropy for {source}->{target}: {str(e)}")
        
        return mat
