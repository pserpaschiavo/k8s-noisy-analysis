"""
Module: analysis_causality.py
Description: Causality analysis utilities for multi-tenant time series analysis,
             including Granger causality and Transfer Entropy (TE), as well as
             graph visualizations.
"""
import os
import logging
import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Optional

from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import MissingDataError, InfeasibleTestError

from src.pipeline_stage import PipelineStage
from src.config import PipelineConfig
from src.visualization.causality_plots import plot_causality_graph, plot_causality_heatmap

# Configure logging
logger = logging.getLogger(__name__)

# Suppress specific statsmodels warnings that can be noisy
warnings.filterwarnings("ignore", "Maximum Likelihood optimization failed to converge")
warnings.filterwarnings("ignore", "Unable to solve the eigenvalue problem")

# Attempt to import the pyinform library for a more robust Transfer Entropy implementation
try:
    from pyinform.transferentropy import transfer_entropy
    PYINFORM_AVAILABLE = True
    logger.info("pyinform library found. Using optimized implementation for Transfer Entropy.")
except ImportError:
    PYINFORM_AVAILABLE = False
    logger.warning("pyinform is not installed. Transfer Entropy will use a basic implementation. "
                   "For better results, install with: pip install pyinform")

def _transfer_entropy_fallback(x: np.ndarray, y: np.ndarray, bins: int = 5, k: int = 1) -> float:
    """
    Basic fallback implementation of Transfer Entropy (TE) from y to x (y→x).
    This is used if the 'pyinform' library is not available.
    """
    # Discretize the series
    x_binned = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_binned = np.digitize(y, np.histogram_bin_edges(y, bins=bins))
    
    # Calculate joint and conditional probabilities
    px = np.histogram(x_binned[1:], bins=bins, density=True)[0]
    pxy = np.histogram2d(x_binned[:-1], x_binned[1:], bins=bins, density=True)[0]
    pxyy = np.histogramdd(np.stack([x_binned[:-1], y_binned[:-1], x_binned[1:]], axis=1), 
                         bins=(bins, bins, bins), density=True)[0]
    
    # Add regularization to avoid log(0)
    epsilon = 1e-12
    pxyy += epsilon
    pxy += epsilon
    px += epsilon
    
    # TE(y→x) = sum p(x_{t+1}, x_t, y_t) * log [p(x_{t+1}|x_t, y_t) / p(x_{t+1}|x_t)]
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k_ in range(bins):
                pxyz = pxyy[j, k_, i]
                pxz = pxy[j, i]
                if pxyz > 0 and pxz > 0:
                    # Conditional probabilities
                    p_cond_xyz = pxyz / (np.sum(pxyy[j, k_, :]) + epsilon)
                    p_cond_xz = pxz / (np.sum(pxy[j, :]) + epsilon)
                    if p_cond_xyz > 0 and p_cond_xz > 0:
                        te += pxyz * np.log(p_cond_xyz / p_cond_xz)
    return te

def _calculate_transfer_entropy(x: np.ndarray, y: np.ndarray, bins: int = 5, k: int = 1) -> float:
    """
    Calculates the Transfer Entropy (TE) from y to x (y→x).
    Uses the optimized 'pyinform' library if available, otherwise falls back to a basic implementation.
    """
    min_points = 10
    if len(x) != len(y):
        length = min(len(x), len(y))
        x, y = x[:length], y[:length]
    
    if len(x) < min_points:
        logger.debug(f"Insufficient data for TE: {len(x)} points. Minimum required: {min_points}.")
        return 0.0
    
    # Handle potential NaN values from resampling or gaps
    if np.isnan(x).any(): x = np.nan_to_num(x, nan=np.nanmean(x))
    if np.isnan(y).any(): y = np.nan_to_num(y, nan=np.nanmean(y))
    
    if PYINFORM_AVAILABLE:
        try:
            # pyinform requires integer-discretized data
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
            x_bin = np.floor(x_norm * (bins - 1)).astype(int)
            y_bin = np.floor(y_norm * (bins - 1)).astype(int)
            return float(transfer_entropy(y_bin, x_bin, k=k, local=False))
        except Exception as e:
            logger.warning(f"Error using pyinform for TE: {e}. Falling back to basic implementation.")
    
    return _transfer_entropy_fallback(x, y, bins, k)

class CausalityAnalysisStage(PipelineStage):
    """
    Pipeline stage for performing causality analysis (Granger, Transfer Entropy).
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("causality_analysis", "Performs causality analysis between tenants.")
        self.config = config
        causality_settings = self.config.get('analysis_settings', {}).get('causality', {})
        self.p_value_threshold = causality_settings.get('p_value_threshold', 0.05)
        self.te_threshold = causality_settings.get('te_threshold', 0.1)
        self.max_lag = causality_settings.get('granger_max_lag', 5)
        self.te_bins = causality_settings.get('te_bins', 8)
        self.te_history = causality_settings.get('te_history', 1)

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the causality analysis for the current round.
        """
        self.logger.info(f"Starting causality analysis stage for round: {round_id}.")

        if data is None or data.empty:
            self.logger.warning("No data found in context. Skipping causality analysis.")
            return {}
            
        if not round_id:
            self.logger.error("Round ID is not specified. Skipping stage.")
            return {}

        output_dir = self.config.get_output_dir_for_round(self.stage_name, round_id)
        
        metrics = self.config.get_selected_metrics() or data['metric_name'].unique()
        phases = self.config.get_selected_phases() or data['experimental_phase'].unique()
        
        all_causality_results = []
        generated_artifacts = {'plots': {}, 'data': {}}

        for metric in metrics:
            for phase in phases:
                self.logger.info(f"Analyzing metric: {metric}, phase: {phase}")
                
                # --- Granger Causality ---
                granger_matrix = self.compute_granger_matrix(data, metric, phase, maxlag=self.max_lag)
                if not granger_matrix.empty:
                    self._process_causality_results(
                        matrix=granger_matrix,
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        method='granger',
                        output_dir=output_dir,
                        threshold=self.p_value_threshold,
                        threshold_mode='less',
                        value_type='p-value',
                        results_list=all_causality_results,
                        artifacts_dict=generated_artifacts
                    )

                # --- Transfer Entropy ---
                te_matrix = self.compute_transfer_entropy_matrix(data, metric, phase, bins=self.te_bins, k=self.te_history)
                if not te_matrix.empty:
                    self._process_causality_results(
                        matrix=te_matrix,
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        method='te',
                        output_dir=output_dir,
                        threshold=self.te_threshold,
                        threshold_mode='greater',
                        value_type='score',
                        results_list=all_causality_results,
                        artifacts_dict=generated_artifacts
                    )

        if not all_causality_results:
            self.logger.warning("No causality results were generated for this round.")
            return {"artifacts": generated_artifacts}
            
        # Consolidate all results for the round
        final_results_df = pd.concat(all_causality_results, ignore_index=True)
        
        # This is the main output for the multi-round analysis
        return {
            "causality_results": final_results_df,
            "artifacts": generated_artifacts
        }

    def _process_causality_results(self, matrix, metric, phase, round_id, method, output_dir, threshold, threshold_mode, value_type, results_list, artifacts_dict):
        """Helper to process and save results for a given causality method."""
        # Save matrix to CSV
        csv_path = os.path.join(output_dir, f"{method}_matrix_{metric}_{phase}.csv")
        matrix.to_csv(csv_path)
        artifacts_dict['data'][f"{method}_matrix_{metric}_{phase}"] = csv_path

        # Generate and save graph plot
        graph_path = os.path.join(output_dir, f"{method}_graph_{metric}_{phase}.png")
        plot_causality_graph(
            causality_matrix=matrix,
            out_path=graph_path,
            metric=metric,
            threshold=threshold,
            threshold_mode=threshold_mode,
            directed=True
        )
        artifacts_dict['plots'][f"{method}_graph_{metric}_{phase}"] = graph_path

        # Generate and save heatmap
        heatmap_path = plot_causality_heatmap(
            causality_matrix=matrix,
            metric=metric,
            phase=phase,
            round_id=round_id,
            out_dir=output_dir,
            method=method.title(),
            value_type=value_type
        )
        artifacts_dict['plots'][f"{method}_heatmap_{metric}_{phase}"] = heatmap_path

        # Convert matrix to tidy format for consolidation
        matrix.index.name = 'target'
        matrix.columns.name = 'source'
        tidy_df = matrix.stack().reset_index(name=value_type)
        tidy_df['metric'] = metric
        tidy_df['phase'] = phase
        tidy_df['round_id'] = round_id
        results_list.append(tidy_df)

    def compute_granger_matrix(self, df: pd.DataFrame, metric: str, phase: str, maxlag: int) -> pd.DataFrame:
        """
        Calculates the matrix of p-values from the Granger causality test.
        """
        self.logger.debug(f"Calculating Granger causality for {metric}, {phase}, maxlag={maxlag}")
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase)]
        
        if subset.empty:
            return pd.DataFrame()

        tenants = sorted(subset['tenant_id'].unique())
        if len(tenants) < 2:
            return pd.DataFrame(index=tenants, columns=tenants)

        p_value_matrix = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        wide_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        wide_df = wide_df.reindex(columns=tenants).interpolate(method='time').bfill().ffill()
        
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                
                data = wide_df[[target, source]].dropna()
                if data.shape[0] < maxlag + 5 or data[source].std() < 1e-9 or data[target].std() < 1e-9:
                    continue
                
                try:
                    # Check for stationarity and difference if needed
                    from statsmodels.tsa.stattools import adfuller
                    adf_p_target = adfuller(data[target], autolag='AIC')[1]
                    adf_p_source = adfuller(data[source], autolag='AIC')[1]
                    if adf_p_target > 0.05 or adf_p_source > 0.05:
                        data = data.diff().dropna()
                    
                    if data.shape[0] <= maxlag + 3 or data[source].std() < 1e-9 or data[target].std() < 1e-9:
                        continue

                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        test_results = grangercausalitytests(data, maxlag=[maxlag], verbose=False)
                        p_value = test_results[maxlag][0]['ssr_chi2test'][1]
                    p_value_matrix.loc[target, source] = p_value
                except (MissingDataError, ValueError, np.linalg.LinAlgError, InfeasibleTestError) as e:
                    self.logger.debug(f"Granger test failed for {source}->{target}: {e}")
                    continue
                except Exception as e:
                    self.logger.error(f"Unexpected error in Granger test for {source}->{target}: {e}", exc_info=True)
                    continue
        return p_value_matrix

    def compute_transfer_entropy_matrix(self, df: pd.DataFrame, metric: str, phase: str, bins: int, k: int) -> pd.DataFrame:
        """
        Calculates the Transfer Entropy (TE) matrix between all tenants.
        """
        self.logger.debug(f"Calculating Transfer Entropy for {metric}, {phase}, bins={bins}, k={k}")
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase)]
        
        if subset.empty:
            return pd.DataFrame()

        tenants = sorted(subset['tenant_id'].unique())
        if len(tenants) < 2:
            return pd.DataFrame(index=tenants, columns=tenants)

        te_matrix = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        wide_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        wide_df = wide_df.reindex(columns=tenants).interpolate(method='time').bfill().ffill()
        
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                try:
                    data = wide_df[[target, source]].dropna()
                    if data.shape[0] >= 10:
                        te_value = _calculate_transfer_entropy(data[target].values, data[source].values, bins=bins, k=k)
                        te_matrix.loc[target, source] = te_value
                except Exception as e:
                    self.logger.error(f"Error calculating TE for {source}->{target}: {e}", exc_info=True)
        return te_matrix
