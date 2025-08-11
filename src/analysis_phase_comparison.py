"""
Module: analysis_phase_comparison.py
Description: Implements comparative analyses between different experimental phases.

This module provides functionality to compare metrics and results between
experimental phases (baseline, attack, recovery), including:

1. Comparison of descriptive statistics
2. Comparison of correlation/covariance
3. Comparison of causality
"""
import os
import logging
import pandas as pd
from typing import Dict, List, Optional, Any

from src.config import PipelineConfig
from src.pipeline_stage import PipelineStage
from src.utils import normalize_phase_name
from src.visualization.phase_comparison_plots import (
    plot_phase_comparison
)


def compute_phase_stats_comparison(df: pd.DataFrame, metric: str, round_id: str, tenants: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculates comparative statistics for a metric across different phases for each tenant.
    This function is designed to work with the new 7-phase experimental structure.
    
    Args:
        df: A long-format DataFrame containing the data.
        metric: The name of the metric to analyze.
        round_id: The ID of the round to analyze.
        tenants: A list of tenants to analyze. If None, all available tenants are used.
        
    Returns:
        A DataFrame with comparative statistics per tenant and phase.
    """
    # Filter data for the specified metric and round
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)].copy()
    
    # If tenants are not specified, use all available ones from the subset
    if tenants is None:
        tenants = subset['tenant_id'].unique().tolist()
    
    assert tenants is not None, "Tenant list cannot be None after initialization."

    # Normalization is now handled during data ingestion. The code can rely on canonical phase names.
    available_phases = sorted(subset['experimental_phase'].unique())
    baseline_phase_name = normalize_phase_name('Baseline') # Get the canonical baseline name

    if baseline_phase_name not in available_phases:
        logging.warning(
            f"Canonical baseline phase '{baseline_phase_name}' not found for metric {metric} "
            f"in round {round_id}. Phase variation analysis will be skipped."
        )
        return pd.DataFrame()

    # Structure to store results
    results = []
    
    # For each combination of tenant and phase
    for tenant in tenants:
        tenant_data = {'tenant_id': tenant}
        
        # Base metrics for the tenant (during the baseline phase)
        baseline_data = subset[(subset['tenant_id'] == tenant) & 
                              (subset['experimental_phase'] == baseline_phase_name)]
        
        if baseline_data.empty:
            logging.warning(f"No data available for tenant {tenant} in the baseline phase. Skipping.")
            continue

        base_mean = baseline_data['metric_value'].mean()
        base_std = baseline_data['metric_value'].std()
        
        tenant_data[f'{baseline_phase_name}_mean'] = base_mean
        tenant_data[f'{baseline_phase_name}_std'] = base_std

        # For each stress phase, compare its metrics against the baseline
        for phase in available_phases:
            phase_key_mean = f'{phase}_mean'
            phase_key_std = f'{phase}_std'
            phase_key_vs_baseline_pct = f'{phase}_vs_baseline_pct'

            if phase == baseline_phase_name:
                continue

            phase_data = subset[(subset['tenant_id'] == tenant) & (subset['experimental_phase'] == phase)]
            
            if phase_data.empty:
                tenant_data[phase_key_mean] = None
                tenant_data[phase_key_std] = None
                tenant_data[phase_key_vs_baseline_pct] = None
            else:
                phase_mean = phase_data['metric_value'].mean()
                phase_std = phase_data['metric_value'].std()
                
                tenant_data[phase_key_mean] = phase_mean
                tenant_data[phase_key_std] = phase_std
                
                # Calculate percentage change relative to baseline, ensuring values are numeric
                if pd.notna(phase_mean) and pd.notna(base_mean):
                    if abs(base_mean) > 1e-9:  # Avoid division by zero
                        change_pct = ((phase_mean - base_mean) / abs(base_mean)) * 100
                        tenant_data[phase_key_vs_baseline_pct] = change_pct
                    else:
                        tenant_data[phase_key_vs_baseline_pct] = 0.0 if phase_mean == base_mean else float('inf')
                else:
                    tenant_data[phase_key_vs_baseline_pct] = None
        
        results.append(tenant_data)

    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)
    

class PhaseComparisonStage(PipelineStage):
    """
    Pipeline stage for comparative analysis between experimental phases.
    """
    
    def __init__(self, config: PipelineConfig):
        """Initializes the phase comparison stage."""
        super().__init__(
            stage_name="phase_comparison_analysis",
            description="Performs comparative analysis between experimental phases."
        )
        self.config = config

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Executes the phase comparison analysis for a specific round.
        
        Args:
            data: The DataFrame for the current round.
            all_results: A dictionary containing all results from previous stages.
            round_id: The identifier for the current processing round.
            
        Returns:
            A dictionary containing the results of this stage.
        """
        self.logger.info(f"Starting phase comparison analysis for round: {round_id}")
        
        if data is None or data.empty:
            self.logger.warning(f"No data for round {round_id}, skipping phase comparison.")
            return {}

        output_dir = self.config.get_output_dir_for_round(self.stage_name, round_id)
        metrics_to_analyze = data['metric_name'].unique()
        
        round_stats = []

        for metric in metrics_to_analyze:
            self.logger.info(f"Analyzing metric '{metric}' for round '{round_id}'")

            stats_df = compute_phase_stats_comparison(
                data, 
                metric=metric,
                round_id=round_id
            )
            
            if stats_df.empty:
                self.logger.warning(f"No stats computed for metric {metric} in round {round_id}.")
                continue

            # The plotting function may need a more explicit output path
            plot_phase_comparison(stats_df, metric, output_dir, round_id)
            
            stats_df['metric'] = metric
            round_stats.append(stats_df)

        if not round_stats:
            self.logger.warning(f"No phase comparison statistics were generated for round {round_id}.")
            return {}

        # Consolidate all stats for the current round into a single DataFrame
        round_stats_df = pd.concat(round_stats, ignore_index=True)

        # Save the consolidated stats for the round
        csv_dir = os.path.join(output_dir, 'csv')
        os.makedirs(csv_dir, exist_ok=True)
        round_csv_path = os.path.join(csv_dir, f'phase_comparison_stats_{round_id}.csv')
        if 'round_id' not in round_stats_df.columns:
            round_stats_df['round_id'] = round_id
        round_stats_df.to_csv(round_csv_path, index=False)
        self.logger.info(f"Round-specific phase comparison stats saved to {round_csv_path}")

        self.logger.info(f"Phase comparison analysis for round {round_id} completed successfully.")
        
        # The result is a dictionary containing the DataFrame of stats for this round
        return {'phase_comparison_stats': round_stats_df}
