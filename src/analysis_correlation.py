"""
Module: analysis_correlation.py
Description: Correlation analysis utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Literal

from src.pipeline_stage import PipelineStage
from src.config import PipelineConfig
from src.gpu_acceleration import check_gpu_availability, calculate_correlation_matrix_gpu
from src.visualization.correlation_plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
)

logger = logging.getLogger(__name__)

class CorrelationAnalysisStage(PipelineStage):
    """
    Pipeline stage for performing correlation analysis on time series data.
    """

    def __init__(self, config: PipelineConfig):
        super().__init__("correlation_analysis", "Performs correlation analysis between tenants.")
        self.config = config
        analysis_settings = self.config.get('analysis_settings', {})
        self.use_gpu = analysis_settings.get('gpu_acceleration', False)
        self.large_dataset_threshold = analysis_settings.get('large_dataset_threshold', 10000)

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Executes the correlation analysis for the current round.
        """
        self.logger.info(f"Starting correlation analysis stage for round: {round_id}.")
        
        if data is None or data.empty:
            self.logger.warning("No data found in context. Skipping correlation analysis.")
            return {}
        
        if not round_id:
            self.logger.error("Round ID is not specified. Skipping stage.")
            return {}

        output_dir = self.config.get_output_dir_for_round(self.stage_name, round_id)
        
        metrics = self.config.get_selected_metrics() or data['metric_name'].unique()
        phases = self.config.get_selected_phases() or data['experimental_phase'].unique()

        all_results_dfs = []
        generated_artifacts = {'plots': {}, 'data': {}}

        for metric in metrics:
            for phase in phases:
                self.logger.info(f"Analyzing metric: {metric}, phase: {phase}")

                # Compute and save correlation matrix
                corr_matrix = self._compute_correlation_matrix(data, metric, phase)
                if not corr_matrix.empty:
                    corr_path = os.path.join(output_dir, f"correlation_matrix_{metric}_{phase}.csv")
                    corr_matrix.to_csv(corr_path)
                    self.logger.info(f"Correlation matrix saved to {corr_path}")
                    generated_artifacts['data'][f"corr_matrix_{metric}_{phase}"] = corr_path

                    plot_path = plot_correlation_heatmap(
                        corr_matrix=corr_matrix,
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        out_dir=output_dir
                    )
                    generated_artifacts['plots'][f"corr_heatmap_{metric}_{phase}"] = plot_path
                    
                    # Prepare data for consolidated results
                    corr_matrix.index.name = 'tenant1'
                    corr_matrix.columns.name = 'tenant2'
                    corr_tidy = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))\
                                           .stack().reset_index()
                    corr_tidy.columns = ['tenant1', 'tenant2', 'correlation']
                    corr_tidy['metric'] = metric
                    corr_tidy['phase'] = phase
                    all_results_dfs.append(corr_tidy)

                # Compute and save covariance matrix
                cov_matrix = self._compute_covariance_matrix(data, metric, phase)
                if not cov_matrix.empty:
                    cov_path = os.path.join(output_dir, f"covariance_matrix_{metric}_{phase}.csv")
                    cov_matrix.to_csv(cov_path)
                    self.logger.info(f"Covariance matrix saved to {cov_path}")
                    generated_artifacts['data'][f"cov_matrix_{metric}_{phase}"] = cov_path

                    plot_path = plot_covariance_heatmap(
                        cov_matrix=cov_matrix,
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        out_dir=output_dir
                    )
                    generated_artifacts['plots'][f"cov_heatmap_{metric}_{phase}"] = plot_path

        # Consolidate all correlation results for this round
        if all_results_dfs:
            consolidated_df = pd.concat(all_results_dfs, ignore_index=True)
            summary_path = os.path.join(output_dir, "correlation_summary.csv")
            consolidated_df.to_csv(summary_path, index=False)
            self.logger.info(f"Consolidated correlation results for round {round_id} saved to {summary_path}")
            
            # This is the main output for the multi-round analysis
            return {
                "correlation_results": consolidated_df,
                "artifacts": generated_artifacts
            }
        else:
            self.logger.warning("No correlation results were generated for this round.")
            return {"artifacts": generated_artifacts}

    def _compute_correlation_matrix(self, df: pd.DataFrame, metric: str, phase: str) -> pd.DataFrame:
        """
        Computes the correlation matrix for a given metric and phase.
        """
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase)]
        if subset.empty:
            self.logger.warning(f"No data for metric '{metric}' and phase '{phase}'.")
            return pd.DataFrame()

        pivot_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Handle missing values, if any
        pivot_df.interpolate(method='time', inplace=True)
        pivot_df.bfill(inplace=True)
        pivot_df.ffill(inplace=True)
        
        if pivot_df.isnull().values.any():
            self.logger.warning(f"Could not fill all NaNs for metric '{metric}' and phase '{phase}'. Correlation might be inaccurate.")
            pivot_df.fillna(0, inplace=True)

        if self.use_gpu and check_gpu_availability():
            if pivot_df.shape[0] > self.large_dataset_threshold:
                self.logger.info(f"Using GPU for correlation matrix calculation (dataset size: {pivot_df.shape[0]}).")
                return calculate_correlation_matrix_gpu(pivot_df)

        return pivot_df.corr()

    def _compute_covariance_matrix(self, df: pd.DataFrame, metric: str, phase: str) -> pd.DataFrame:
        """
        Computes the covariance matrix for a given metric and phase.
        """
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase)]
        if subset.empty:
            return pd.DataFrame()

        pivot_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Handle missing values
        pivot_df.interpolate(method='time', inplace=True)
        pivot_df.bfill(inplace=True)
        pivot_df.ffill(inplace=True)
        pivot_df.fillna(0, inplace=True)

        return pivot_df.cov()
