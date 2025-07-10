"""
Module: analysis_descriptive.py
Description: Descriptive statistics and plotting utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional

from src.utils import configure_matplotlib
from src.pipeline_stage import PipelineStage
from src.config import PipelineConfig
from src.visualization.descriptive_plots import (
    plot_metric_timeseries_multi_tenant_all_phases,
    plot_metric_barplot_by_phase,
    plot_metric_boxplot
)

# Centralized matplotlib configuration
configure_matplotlib()

# Setup logging
logger = logging.getLogger(__name__)

def compute_descriptive_stats(df, groupby_cols=None) -> pd.DataFrame:
    """
    Compute descriptive statistics (count, mean, std, min, max, skewness, kurtosis) for metric_value,
    grouped by the specified columns.
    
    Args:
        df (pd.DataFrame): DataFrame with the data to be analyzed.
        groupby_cols (List[str], optional): List of columns to group by. 
                                            Defaults to ['tenant_id', 'metric_name', 'experimental_phase', 'round_id'].
        
    Returns:
        pd.DataFrame: DataFrame with descriptive statistics.
    """
    
    if groupby_cols is None:
        groupby_cols=['tenant_id', 'metric_name', 'experimental_phase', 'round_id']
    
    # More comprehensive statistics including skewness and kurtosis
    stats = df.groupby(groupby_cols)['metric_value'].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('skewness', lambda x: x.skew()),
        ('kurtosis', lambda x: x.kurtosis())
    ]).reset_index()
    
    logger.info(f"Computed descriptive stats for {len(stats)} rows, grouped by {len(groupby_cols)} columns.")
    return stats


class DescriptiveAnalysisStage(PipelineStage):
    """
    Pipeline stage for descriptive analysis.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("descriptive_analysis", "Descriptive statistics and anomaly detection")
        self.config = config

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Executes the descriptive analysis stage.
        
        Args:
            data (pd.DataFrame): The input data for this stage.
            all_results (Dict[str, Any]): Dictionary containing results from previous stages.
            round_id (str): The ID of the current processing round.

        Returns:
            Dict[str, Any]: A dictionary containing the results of this stage.
        """
        self.logger.info(f"Starting descriptive analysis for round '{round_id}'...")
        if data is None or data.empty:
            self.logger.error("Input DataFrame 'data' is not available for descriptive analysis.")
            return {}

        # Filter data for the current round
        round_df = data[data['round_id'] == round_id].copy()
        if round_df.empty:
            self.logger.warning(f"No data found for round '{round_id}' in DescriptiveAnalysisStage. Skipping.")
            return {}

        # Compute descriptive stats
        descriptive_stats = compute_descriptive_stats(round_df)
        
        # Save to CSV
        output_dir = self.config.get_output_dir_for_round("descriptive_analysis", round_id)
        csv_path = os.path.join(output_dir, f"descriptive_stats_{round_id}.csv")
        descriptive_stats.to_csv(csv_path, index=False)
        self.logger.info(f"Descriptive stats for round '{round_id}' saved to {csv_path}")

        # --- Generate plots ---
        self.logger.info(f"Generating descriptive plots for round '{round_id}'...")
        plot_paths = []
        
        selected_metrics = self.config.get_selected_metrics()
        if not selected_metrics:
            selected_metrics = round_df['metric_name'].unique()

        for metric in selected_metrics:
            metric_df = round_df[round_df['metric_name'] == metric]
            if metric_df.empty:
                self.logger.warning(f"No data for metric '{metric}' in round '{round_id}'. Skipping plot generation.")
                continue

            try:
                # Generate plots that require the full round data for a metric
                self.logger.debug(f"Generating barplot for {metric} in round {round_id}")
                path = plot_metric_barplot_by_phase(metric_df, metric, round_id, output_dir, self.config)
                if path: plot_paths.append(path)

                self.logger.debug(f"Generating boxplot for {metric} in round {round_id}")
                path = plot_metric_boxplot(metric_df, metric, round_id, output_dir, self.config)
                if path: plot_paths.append(path)

                self.logger.debug(f"Generating multi-tenant timeseries for {metric} in round {round_id}")
                path = plot_metric_timeseries_multi_tenant_all_phases(metric_df, metric, round_id, output_dir, self.config)
                if path: plot_paths.append(path)

            except Exception as e:
                self.logger.error(f"Error generating plots for metric '{metric}' in round '{round_id}': {e}", exc_info=True)

        self.logger.info(f"Generated {len(plot_paths)} descriptive plots for round '{round_id}'.")

        return {
            "descriptive_stats_path": csv_path,
            "plot_paths": plot_paths
        }


def detect_anomalies(df: pd.DataFrame, metric: str, phase: str, round_id: str, window_size: int = 10, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detects anomalies in time series using rolling window and Z-score.
    
    Args:
        df: DataFrame in long format
        metric: Metric name to analyze
        phase: Experimental phase to filter
        round_id: Round ID to filter
        window_size: Window size for rolling mean and std
        threshold: Z-score threshold to consider a point as an anomaly
        
    Returns:
        DataFrame with detected anomalies
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)].copy()
    if subset.empty:
        logger.warning(f"No data for anomaly detection: {metric}, {phase}, {round_id}")
        return pd.DataFrame()
    
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Sort by timestamp for rolling window
    subset = subset.sort_values(['tenant_id', 'timestamp'])
    
    # Initialize results DataFrame
    anomalies = pd.DataFrame()
    
    # Process each tenant separately
    for tenant, group in subset.groupby('tenant_id'):
        # Calculate rolling statistics (mean and std)
        rolling_mean = group['metric_value'].rolling(window=window_size, center=True).mean()
        rolling_std = group['metric_value'].rolling(window=window_size, center=True).std()
        
        # Replace NaN at the beginning/end of rolling with global means
        rolling_mean = rolling_mean.fillna(group['metric_value'].mean())
        rolling_std = rolling_std.fillna(group['metric_value'].std())
        rolling_std = rolling_std.replace(0, group['metric_value'].std())  # Avoid division by zero
        
        # Calculate Z-score
        z_scores = np.abs((group['metric_value'] - rolling_mean) / rolling_std)
        
        # Identify anomalies
        is_anomaly = z_scores > threshold
        
        # Filter anomalies and add to results DataFrame
        tenant_anomalies = group[is_anomaly].copy()
        if not tenant_anomalies.empty:
            tenant_anomalies['z_score'] = z_scores[is_anomaly]
            anomalies = pd.concat([anomalies, tenant_anomalies])
    
    # Sort by severity (Z-score)
    if not anomalies.empty:
        anomalies = anomalies.sort_values('z_score', ascending=False)
        logger.info(f"Detected {len(anomalies)} anomalies for {metric} in {phase}, {round_id}")
    else:
        logger.info(f"No anomalies detected for {metric} in {phase}, {round_id}")
    
    return anomalies
