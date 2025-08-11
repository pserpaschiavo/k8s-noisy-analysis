"""
Module: src.analysis_impact
Description: Provides analysis to quantify the impact of noisy neighbors on tenants.
"""
import pandas as pd
import numpy as np
import logging
import os
from typing import Dict, Any, Optional
from scipy.stats import ttest_ind

from src.pipeline_stage import PipelineStage
from src.visualization.impact_plots import plot_impact_summary
from src.config import PipelineConfig

logger = logging.getLogger(__name__)

class ImpactAnalysisStage(PipelineStage):
    """
    Pipeline stage for analyzing the impact of different experimental phases on tenant metrics.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("impact_analysis", "Impact analysis of noisy neighbors")
        self.config = config

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculates the impact by comparing metrics from a baseline phase to other phases.

        Args:
            data: The DataFrame with time series data for the current round.
            all_results: A dictionary containing results from previous stages.
            round_id: The identifier for the current execution round.

        Returns:
            A dictionary containing the impact analysis results and artifact paths.
        """
        if data is None or data.empty:
            self.logger.error("Input data is not available for impact analysis. Skipping stage.")
            return {}

        if not round_id:
            self.logger.error("Round ID is not specified. Skipping stage.")
            return {}

        self.logger.info(f"Starting impact analysis for round: {round_id}...")

        impact_summary = self.calculate_impact(data)

        if impact_summary.empty:
            self.logger.warning("Impact analysis resulted in an empty DataFrame. Skipping artifact generation.")
            return {}

        # --- Artifact Generation ---
        output_dir = self.config.get_output_dir_for_round(self.stage_name, round_id)
        results: Dict[str, Any] = {"impact_metrics": impact_summary}

        # Ensure round_id is present for precision
        try:
            if 'round_id' not in impact_summary.columns:
                impact_summary['round_id'] = round_id
        except Exception as e:
            self.logger.warning(f"Could not annotate impact summary with round_id: {e}")

    # 1. Export results to CSV
    csv_dir = os.path.join(output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f'impact_analysis_summary_{round_id}.csv')
        try:
            impact_summary.to_csv(csv_path, index=False)
            self.logger.info(f"Impact analysis results saved to {csv_path}")
            results['csv_path'] = csv_path
        except Exception as e:
            self.logger.error(f"Failed to save impact analysis CSV: {e}")

        # 2. Generate impact plots
        try:
            plot_paths = plot_impact_summary(impact_summary, output_dir)
            self.logger.info(f"Generated {len(plot_paths)} impact plots.")
            results['plot_paths'] = plot_paths
        except Exception as e:
            self.logger.error(f"Failed to generate impact plots: {e}")

        self.logger.info("Impact analysis completed successfully.")
        return results

    def calculate_impact(self, df: pd.DataFrame, baseline_phase_name: str = 'Baseline') -> pd.DataFrame:
        """
        Calculates the performance impact on tenants by comparing metrics from a baseline
        phase to other experimental phases.

        Args:
            df: The long-format DataFrame with time series data.
            baseline_phase_name: The name of the baseline phase to use for comparison.

        Returns:
            A DataFrame containing the impact analysis results, including percentage changes.
        """
        # Find the full name of the baseline phase (it might have a numeric prefix)
        baseline_phases = [p for p in df['experimental_phase'].unique() if baseline_phase_name in p]
        if not baseline_phases:
            self.logger.warning(f"No baseline phase found containing the name '{baseline_phase_name}'. Skipping impact analysis.")
            return pd.DataFrame()
        
        # Assume the first one found is the baseline
        actual_baseline_phase = baseline_phases[0]
        self.logger.info(f"Using '{actual_baseline_phase}' as the baseline for impact calculation.")

        # Calculate the mean of metrics per tenant in the baseline phase
        baseline_df = df[df['experimental_phase'] == actual_baseline_phase]
        baseline_stats = baseline_df.groupby(['tenant_id', 'metric_name'])['metric_value'].agg(['mean', 'std', 'count']).reset_index()
        baseline_stats.rename(columns={'mean': 'baseline_mean', 'std': 'baseline_std', 'count': 'baseline_count'}, inplace=True)

        # Calculate the mean and std of metrics in all phases
        phase_stats = df.groupby(['tenant_id', 'metric_name', 'experimental_phase'])['metric_value'].agg(['mean', 'std', 'count']).reset_index()

        # Join baseline data with phase averages
        impact_df = pd.merge(phase_stats, baseline_stats, on=['tenant_id', 'metric_name'])

        # Calculate percentage change and volatility
        impact_df['percentage_change'] = ((impact_df['mean'] - impact_df['baseline_mean']) / impact_df['baseline_mean']) * 100
        impact_df.rename(columns={'std': 'volatility'}, inplace=True)

        # Calculate Cohen's d for effect size
        n1 = impact_df['baseline_count']
        n2 = impact_df['count']
        s1 = impact_df['baseline_std']
        s2 = impact_df['volatility'] # Renamed from 'std'

        # Pooled standard deviation, handle potential division by zero
        s_pooled_numerator = (n1 - 1) * s1**2 + (n2 - 1) * s2**2
        s_pooled_denominator = n1 + n2 - 2
        
        # Avoid division by zero or invalid values
        with np.errstate(divide='ignore', invalid='ignore'):
            s_pooled = np.sqrt(s_pooled_numerator / s_pooled_denominator)
            impact_df['cohen_d'] = (impact_df['mean'] - impact_df['baseline_mean']) / s_pooled

        # Replace inf/-inf with NaN and then fill NaN with 0
        impact_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        impact_df['cohen_d'].fillna(0, inplace=True)

        # Perform statistical significance test (t-test)
        p_values = []
        for _, row in impact_df.iterrows():
            baseline_data = df[
                (df['tenant_id'] == row['tenant_id']) &
                (df['metric_name'] == row['metric_name']) &
                (df['experimental_phase'] == actual_baseline_phase)
            ]['metric_value'].dropna()

            phase_data = df[
                (df['tenant_id'] == row['tenant_id']) &
                (df['metric_name'] == row['metric_name']) &
                (df['experimental_phase'] == row['experimental_phase'])
            ]['metric_value'].dropna()

            if len(baseline_data) > 1 and len(phase_data) > 1:
                _, p_value = ttest_ind(baseline_data, phase_data, equal_var=False) # Welch's t-test
                p_values.append(p_value)
            else:
                p_values.append(None)
        
        impact_df['p_value'] = p_values
        impact_df['is_significant'] = impact_df['p_value'] < 0.05 # 5% significance level

        # Remove the baseline phase itself from the results to focus on impact
        impact_df = impact_df[impact_df['experimental_phase'] != actual_baseline_phase]

        self.logger.info(f"Calculated impact for {len(impact_df)} combinations of tenant/metric/phase.")

        # Ensure the tenant_id column is present
        if 'tenant_id' not in impact_df.columns:
            self.logger.error("Column 'tenant_id' is missing from the impact analysis DataFrame.")
            # Try to recover tenant_id if there is only one in the context
            if len(df['tenant_id'].unique()) == 1:
                impact_df['tenant_id'] = df['tenant_id'].unique()[0]
                self.logger.info("Recovered single tenant_id for impact analysis.")
            else:
                 return pd.DataFrame() # Return empty DF if it cannot be resolved

        return impact_df

