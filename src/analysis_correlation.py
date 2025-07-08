"""
Module: analysis_correlation.py
Description: Correlation analysis utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.stattools import acf

from src.pipeline_stage import PipelineStage
from src.utils import configure_matplotlib
from src.gpu_acceleration import check_gpu_availability, calculate_correlation_matrix_gpu
from src.visualization.correlation_plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_ccf,
    plot_lag_scatter
)

# Configure logging
logger = logging.getLogger(__name__)
configure_matplotlib()


class CorrelationAnalysisStage(PipelineStage):
    """
    Pipeline stage for performing correlation analysis on time series data.
    """

    def __init__(self, config):
        super().__init__("correlation_analysis", "Performs correlation analysis between tenants.")
        self.config = config
        self.use_gpu = self.config.get('analysis_settings', {}).get('gpu_acceleration', False)
        self.large_dataset_threshold = self.config.get('analysis_settings', {}).get('large_dataset_threshold', 10000)

    def _execute_implementation(self, context: dict) -> dict:
        """
        Executes the correlation analysis for the current round.
        """
        self.logger.info("Starting correlation analysis stage.")
        
        df = context.get('data')
        if df is None or df.empty:
            self.logger.warning("No data found in context. Skipping correlation analysis for this round.")
            return context

        # A configuração do estágio é específica da rodada, então obtemos o ID da rodada a partir dela
        try:
            round_id = self.config.get_selected_rounds()[0]
        except (IndexError, TypeError):
            self.logger.error("Could not determine round_id from configuration. Skipping.")
            return context

        self.logger.info(f"Processing correlation analysis for round: {round_id}")
        
        output_dir = self.config.get_output_dir(self.name) # O diretório de saída agora inclui o nome do estágio
        os.makedirs(output_dir, exist_ok=True)

        metrics = self.config.get_selected_metrics()
        phases = self.config.get_selected_phases()

        all_results_dfs = []

        for metric in metrics:
            for phase in phases:
                self.logger.info(f"Analyzing metric: {metric}, phase: {phase}")

                # Compute and save correlation matrix
                corr_matrix = self._compute_correlation_matrix(df, metric, phase)
                if not corr_matrix.empty:
                    corr_path = os.path.join(output_dir, f"correlation_matrix_{metric}_{phase}.csv")
                    corr_matrix.to_csv(corr_path)
                    self.logger.info(f"Correlation matrix saved to {corr_path}")

                    plot_correlation_heatmap(
                        corr_matrix=corr_matrix,
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        out_dir=output_dir
                    )
                    
                    # Renomear os eixos para evitar conflito de nomes após o 'stack'
                    corr_matrix.index.name = 'tenant1'
                    corr_matrix.columns.name = 'tenant2'

                    # Converter a matriz para o formato longo (tidy), pegando apenas o triângulo superior
                    corr_tidy = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))\
                                           .stack()\
                                           .reset_index()
                    
                    corr_tidy.columns = ['tenant1', 'tenant2', 'correlation']
                    
                    # Adicionar informações de contexto
                    corr_tidy['metric'] = metric
                    corr_tidy['phase'] = phase
                    corr_tidy['round_id'] = round_id
                    
                    all_results_dfs.append(corr_tidy)

                # Compute and save covariance matrix
                cov_matrix = self._compute_covariance_matrix(df, metric, phase)
                if not cov_matrix.empty:
                    cov_path = os.path.join(output_dir, f"covariance_matrix_{metric}_{phase}.csv")
                    cov_matrix.to_csv(cov_path)
                    self.logger.info(f"Covariance matrix saved to {cov_path}")

                    plot_covariance_heatmap(
                        cov_matrix=cov_matrix,
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        out_dir=output_dir
                    )
        
        if all_results_dfs:
            results_df = pd.concat(all_results_dfs, ignore_index=True)
            # Reorder columns for clarity
            results_df = results_df[['round_id', 'metric', 'phase', 'tenant1', 'tenant2', 'correlation']]
            results_path = os.path.join(output_dir, "correlation_results.csv")
            results_df.to_csv(results_path, index=False)
            self.logger.info(f"Consolidated correlation results for round {round_id} saved to {results_path}")


        self.logger.info("Correlation analysis stage finished.")
        return context

    def _compute_correlation_matrix(self, df, metric, phase, method='pearson'):
        """
        Computes the correlation matrix for a given metric and phase.
        """
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase)]
        if subset.empty:
            self.logger.warning(f"No data for metric '{metric}' and phase '{phase}'.")
            return pd.DataFrame()

        pivot_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Lidar com valores ausentes, se houver
        pivot_df.interpolate(method='time', inplace=True)
        pivot_df.fillna(method='bfill', inplace=True)
        pivot_df.fillna(method='ffill', inplace=True)
        
        if pivot_df.isnull().values.any():
            self.logger.warning(f"Could not fill all NaNs for metric '{metric}' and phase '{phase}'. Correlation might be inaccurate.")
            pivot_df.fillna(0, inplace=True)

        if self.use_gpu and check_gpu_availability():
            if pivot_df.shape[0] > self.large_dataset_threshold:
                self.logger.info(f"Using GPU for correlation matrix calculation (dataset size: {pivot_df.shape[0]}).")
                return calculate_correlation_matrix_gpu(pivot_df)

        return pivot_df.corr(method=method)

    def _compute_covariance_matrix(self, df, metric, phase):
        """
        Computes the covariance matrix for a given metric and phase.
        """
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase)]
        if subset.empty:
            return pd.DataFrame()

        pivot_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Lidar com valores ausentes
        pivot_df.interpolate(method='time', inplace=True)
        pivot_df.fillna(method='bfill', inplace=True)
        pivot_df.fillna(method='ffill', inplace=True)
        pivot_df.fillna(0, inplace=True)

        return pivot_df.cov()

    def _compute_cross_correlation(self, df, metric, phase, tenant1, tenant2, max_lag=40):
        """
        Computes the cross-correlation between two tenants for a given metric and phase.
        """
        subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['tenant_id'].isin([tenant1, tenant2]))]
        
        if subset.empty:
            logger.warning(f"Skipping cross-correlation for {metric}, {phase}, {tenant1}, {tenant2}: Not enough data.")
            return pd.DataFrame()

        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='value')

        if wide.shape[1] < 2:
            logger.warning(f"Skipping cross-correlation for {metric}, {phase}, {tenant1}, {tenant2}: Less than 2 tenants after pivoting.")
            return pd.DataFrame()
        
        if wide.isna().any().any():
            wide = wide.interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')

        ccf_values = []
        lags = range(-max_lag, max_lag + 1)

        for lag in lags:
            if lag < 0:
                shifted = wide[tenant1].shift(-lag)
                ccf = np.corrcoef(wide[tenant2].iloc[:-lag], shifted.iloc[:-lag])[0, 1]
            elif lag > 0:
                shifted = wide[tenant2].shift(lag)
                ccf = np.corrcoef(wide[tenant1].iloc[lag:], shifted.iloc[lag:])[0, 1]
            else:
                ccf = np.corrcoef(wide[tenant1], wide[tenant2])[0, 1]

            ccf_values.append(ccf)

        ccf_df = pd.DataFrame(ccf_values, index=lags, columns=['ccf'])
        ccf_df['lag'] = ccf_df.index
        ccf_df['metric'] = metric
        ccf_df['phase'] = phase
        ccf_df['tenant1'] = tenant1
        ccf_df['tenant2'] = tenant2

        return ccf_df
