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
    Recognizes the new 7-phase experimental structure.
    
    Args:
        df: Long DataFrame with the data.
        metric: Name of the metric for analysis.
        round_id: ID of the round for analysis.
        tenants: List of tenants to analyze. If None, uses all available.
        
    Returns:
        DataFrame with comparative statistics per tenant and phase.
    """
    # Filter data for the specified metric and round
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)].copy()
    
    # If tenants are not specified, use all available ones
    if tenants is None:
        tenants = subset['tenant_id'].unique().tolist()
    
    assert tenants is not None

    # A normalização agora é feita na ingestão. O código pode confiar nos nomes canônicos.
    available_phases = sorted(subset['experimental_phase'].unique())
    baseline_phase_name = normalize_phase_name('Baseline') # Obtém o nome canônico da baseline

    if baseline_phase_name not in available_phases:
        logging.warning(f"Fase de baseline canônica '\'{baseline_phase_name}\'' não encontrada para a métrica {metric} no round {round_id}. A análise de variação de fase será pulada.")
        return pd.DataFrame()

    # Estrutura para armazenar resultados
    results = []
    
    # For each combination of tenant and phase
    for tenant in tenants:
        tenant_data = {'tenant_id': tenant}
        
        # Métricas base para o tenant (na baseline)
        baseline_data = subset[(subset['tenant_id'] == tenant) & 
                              (subset['experimental_phase'] == baseline_phase_name)]
        
        if baseline_data.empty:
            logging.warning(f"Sem dados para o tenant {tenant} na fase baseline.")
            continue

        base_mean = baseline_data['metric_value'].mean()
        base_std = baseline_data['metric_value'].std()
        
        tenant_data[f'{baseline_phase_name}_mean'] = base_mean
        tenant_data[f'{baseline_phase_name}_std'] = base_std

        # Para cada fase de stress, compara com a baseline
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
                        tenant_data[phase_key_vs_baseline_pct] = 0.0
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
            name="phase_comparison_analysis",
            description="Performs comparative analysis between experimental phases."
        )
        self.config = config

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the phase comparison analysis.
        """
        self.logger.info("Starting phase comparison analysis.")
        rounds_data = context.get('rounds_data', {})
        if not rounds_data:
            self.logger.warning("No rounds data found in context. Skipping phase comparison.")
            return context

        all_stats = []

        for round_id, round_data in rounds_data.items():
            self.logger.info(f"Processing round: {round_id}")
            df = round_data.get('data')
            if df is None or df.empty:
                self.logger.warning(f"No data for round {round_id}, skipping.")
                continue

            output_dir = self.config.get_output_dir_for_round("phase_comparison_analysis", round_id)
            metrics_to_analyze = df['metric_name'].unique()

            for metric in metrics_to_analyze:
                self.logger.info(f"Analyzing metric '{metric}' for round '{round_id}'")

                stats_df = compute_phase_stats_comparison(
                    df, 
                    metric=metric,
                    round_id=round_id
                )
                
                if stats_df.empty:
                    self.logger.warning(f"No stats computed for metric {metric} in round {round_id}.")
                    continue

                # A função de plot agora pode precisar de um caminho de saída mais explícito
                plot_phase_comparison(stats_df, metric, output_dir, round_id)
                
                stats_df['round_id'] = round_id
                all_stats.append(stats_df)

        if all_stats:
            final_stats_df = pd.concat(all_stats, ignore_index=True)
            # Salvar o consolidado no diretório principal do experimento
            exp_output_dir = self.config.get_output_dir()
            final_csv_path = os.path.join(exp_output_dir, 'phase_comparison_all_rounds.csv')
            final_stats_df.to_csv(final_csv_path, index=False)
            self.logger.info(f"Aggregated phase comparison stats saved to {final_csv_path}")
            context['phase_comparison_stats'] = final_stats_df
            self.logger.info("Phase comparison analysis completed successfully.")
        else:
            self.logger.warning("No phase comparison statistics were generated.")

        return context
