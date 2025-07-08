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
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import MissingDataError, InfeasibleTestError
from functools import lru_cache
import warnings

from src.visualization.causality_plots import plot_causality_graph, plot_causality_heatmap
from src.utils import configure_matplotlib
from src.pipeline_stage import PipelineStage
from src.config import PipelineConfig

# Configure matplotlib using the centralized function
configure_matplotlib()

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

class CausalityAnalysisStage(PipelineStage):
    """
    Pipeline stage for performing causality analysis (Granger, Transfer Entropy).
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("causality_analysis", "Performs causality analysis between tenants.")
        self.config = config

    def _execute_implementation(self, context: dict) -> dict:
        """
        Executes the causality analysis for all specified rounds.
        """
        self.logger.info("Starting causality analysis stage.")
        rounds_data = context.get('rounds_data', {})
        if not rounds_data:
            self.logger.warning("No rounds data found in context. Skipping causality analysis.")
            return context

        for round_id, round_data in rounds_data.items():
            self.logger.info(f"Processing round: {round_id}")
            df = round_data.get('data')
            if df is None or df.empty:
                self.logger.warning(f"No data found for round {round_id}. Skipping.")
                continue

            output_dir = self.config.get_output_dir_for_round("causality_analysis", round_id)
            
            metrics = self.config.get('analysis_settings', {}).get('metrics', [])
            phases = self.config.get('analysis_settings', {}).get('phases', [])
            p_value_threshold = self.config.get('analysis_settings', {}).get('causality', {}).get('p_value_threshold', 0.05)
            te_threshold = self.config.get('analysis_settings', {}).get('causality', {}).get('te_threshold', 0.1)

            for metric in metrics:
                for phase in phases:
                    self.logger.info(f"Analyzing metric: {metric}, phase: {phase}")
                    
                    # --- Granger Causality ---
                    granger_p_matrix = self.compute_granger_matrix(df, metric, phase, round_id)
                    if not granger_p_matrix.empty:
                        granger_csv_path = os.path.join(output_dir, f"granger_matrix_{metric}_{phase}.csv")
                        granger_p_matrix.to_csv(granger_csv_path)
                        
                        plot_causality_graph(
                            causality_matrix=granger_p_matrix,
                            out_path=os.path.join(output_dir, f"granger_graph_{metric}_{phase}.png"),
                            metric=metric,
                            threshold=p_value_threshold,
                            threshold_mode='less'
                        )
                        plot_causality_heatmap(
                            causality_matrix=granger_p_matrix,
                            metric=metric,
                            phase=phase,
                            round_id=round_id,
                            out_dir=output_dir,
                            method='Granger',
                            value_type='p-value'
                        )

                    # --- Transfer Entropy ---
                    te_matrix = self.compute_transfer_entropy_matrix(df, metric, phase, round_id)
                    if not te_matrix.empty:
                        te_csv_path = os.path.join(output_dir, f"te_matrix_{metric}_{phase}.csv")
                        te_matrix.to_csv(te_csv_path)
                        
                        plot_causality_graph(
                            causality_matrix=te_matrix,
                            out_path=os.path.join(output_dir, f"te_graph_{metric}_{phase}.png"),
                            metric=metric,
                            threshold=te_threshold,
                            threshold_mode='greater',
                            directed=True
                        )
                        plot_causality_heatmap(
                            causality_matrix=te_matrix,
                            metric=metric,
                            phase=phase,
                            round_id=round_id,
                            out_dir=output_dir,
                            method='TE',
                            value_type='score'
                        )

        self.logger.info("Causality analysis stage finished.")
        return context

    @lru_cache(maxsize=32)
    def compute_granger_matrix(self, df: pd.DataFrame, metric: str, phase: str, round_id: str, maxlag: int = 5) -> pd.DataFrame:
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
        subset = df[(df['metric_name'] == metric) & 
                        (df['experimental_phase'] == phase)]
        
        if subset.empty:
            logger.warning(f"Skipping Granger for {metric}, {phase}, {round_id}: No data found after filtering.")
            return pd.DataFrame()

        tenants = subset['tenant_id'].unique()
        if len(tenants) < 2:
            logger.warning(f"Skipping Granger for {metric}, {phase}, {round_id}: Requires at least 2 tenants, but found {len(tenants)}.")
            return pd.DataFrame(index=tenants, columns=tenants)

        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        wide = wide.reindex(columns=tenants, fill_value=np.nan)
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                try:
                    data = pd.concat([wide[target], wide[source]], axis=1).dropna()
                    if data[source].std() < 1e-9 or data[target].std() < 1e-9:
                        continue
                    if len(data) < maxlag + 5:
                        continue
                    from statsmodels.tsa.stattools import adfuller
                    for col_name in data.columns:
                        try:
                            adf_result = adfuller(data[col_name], autolag='AIC')
                            if adf_result[1] > 0.05:
                                data[col_name] = data[col_name].diff()
                        except Exception:
                            continue
                    data = data.dropna()
                    if data[source].std() < 1e-9 or data[target].std() < 1e-9:
                        continue
                    if len(data) <= maxlag + 3:
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        test_results = grangercausalitytests(data[[target, source]], maxlag=maxlag, verbose=False)
                        p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
                        min_p_value = min(p_values) if p_values else np.nan
                    mat.loc[target, source] = min_p_value
                except (KeyError, MissingDataError, ValueError, np.linalg.LinAlgError, InfeasibleTestError):
                    continue
                except Exception as e:
                    logger.error(f"An unexpected error occurred calculating Granger for {source}->{target}: {str(e)}", exc_info=True)
                    continue
        return mat

    @lru_cache(maxsize=32)
    def compute_transfer_entropy_matrix(self, df: pd.DataFrame, metric: str, phase: str, round_id: str, bins: int = 8, k: int = 1) -> pd.DataFrame:
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
        subset = df[(df['metric_name'] == metric) & 
                        (df['experimental_phase'] == phase)]
        
        if subset.empty:
            logger.warning(f"Skipping TE for {metric}, {phase}, {round_id}: No data found after filtering.")
            return pd.DataFrame()

        tenants = subset['tenant_id'].unique()
        if len(tenants) < 2:
            logger.warning(f"Skipping TE for {metric}, {phase}, {round_id}: Requires at least 2 tenants, but found {len(tenants)}.")
            return pd.DataFrame(index=tenants, columns=tenants)

        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        wide = wide.reindex(columns=tenants, fill_value=np.nan)
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                try:
                    target_series = wide[target]
                    source_series = wide[source]
                    common_idx = target_series.index.intersection(source_series.index)
                    target_values = target_series.loc[common_idx].values
                    source_values = source_series.loc[common_idx].values
                    if len(target_values) >= 8:
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
