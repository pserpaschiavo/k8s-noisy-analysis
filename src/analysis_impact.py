"""
Module: src.analysis_impact
Description: Provides analysis to quantify the impact of noisy neighbors on tenants.
"""
import pandas as pd
import logging
import os
from typing import Dict, Any, List, Optional
from scipy.stats import ttest_ind

from src.pipeline_stage import PipelineStage
from src.visualization.impact_plots import plot_impact_summary # Importa a função de plotagem

logger = logging.getLogger(__name__)

class ImpactAnalysisStage(PipelineStage):
    """
    Pipeline stage for analyzing the impact of different experimental phases on tenant metrics.
    """
    def __init__(self):
        super().__init__("impact_analysis", "Impact analysis of noisy neighbors")

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates the impact by comparing metrics from a baseline phase to other phases.

        Args:
            context: The pipeline context, containing the long DataFrame.

        Returns:
            The updated context with impact analysis results.
        """
        df_long = context.get('data')  # Padronizado para 'data'
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame 'data' not available for impact analysis.")
            return context

        self.logger.info("Starting impact analysis...")
        
        impact_results = self.calculate_impact(df_long)
        
        if impact_results.empty:
            self.logger.warning("Impact analysis resulted in an empty DataFrame. Skipping artifact generation.")
            return context
            
        context['impact_analysis_results'] = impact_results
        
        # --- Geração de Artefatos ---
        output_dir = context.get('output_dir', 'outputs')
        impact_output_dir = os.path.join(output_dir, 'impact_analysis')
        os.makedirs(impact_output_dir, exist_ok=True)

        # 1. Exportar resultados para CSV
        csv_path = os.path.join(impact_output_dir, 'impact_analysis_results.csv')
        try:
            impact_results.to_csv(csv_path, index=False)
            self.logger.info(f"Impact analysis results saved to {csv_path}")
            context['impact_analysis_csv_path'] = csv_path
        except Exception as e:
            self.logger.error(f"Failed to save impact analysis CSV: {e}")

        # 2. Gerar plots de impacto
        try:
            plot_paths = plot_impact_summary(impact_results, impact_output_dir)
            self.logger.info(f"Generated {len(plot_paths)} impact plots.")
            context['impact_plot_paths'] = plot_paths
        except Exception as e:
            self.logger.error(f"Failed to generate impact plots: {e}")
        
        self.logger.info("Impact analysis completed successfully.")
        return context

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
        # Encontrar o nome completo da fase de baseline (pode ter um prefixo numérico)
        baseline_phases = [p for p in df['experimental_phase'].unique() if baseline_phase_name in p]
        if not baseline_phases:
            self.logger.warning(f"No baseline phase found containing the name '{baseline_phase_name}'. Skipping impact analysis.")
            return pd.DataFrame()
        
        # Assumir a primeira encontrada como a baseline
        actual_baseline_phase = baseline_phases[0]
        self.logger.info(f"Using '{actual_baseline_phase}' as the baseline for impact calculation.")

        # Calcular a média das métricas por tenant na fase de baseline
        baseline_df = df[df['experimental_phase'] == actual_baseline_phase]
        baseline_means = baseline_df.groupby(['tenant_id', 'metric_name'])['metric_value'].mean().reset_index()
        baseline_means.rename(columns={'metric_value': 'baseline_mean'}, inplace=True)

        # Calcular a média das métricas em todas as fases
        phase_stats = df.groupby(['tenant_id', 'metric_name', 'experimental_phase'])['metric_value'].agg(['mean', 'std']).reset_index()

        # Juntar os dados de baseline com as médias das fases
        impact_df = pd.merge(phase_stats, baseline_means, on=['tenant_id', 'metric_name'])

        # Calcular a variação percentual e a volatilidade
        impact_df['percentage_change'] = ((impact_df['mean'] - impact_df['baseline_mean']) / impact_df['baseline_mean']) * 100
        impact_df.rename(columns={'std': 'volatility'}, inplace=True)

        # Realizar teste de significância estatística (t-test)
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
        impact_df['is_significant'] = impact_df['p_value'] < 0.05 # Nível de significância de 5%

        # Remover a própria fase de baseline dos resultados para focar no impacto
        impact_df = impact_df[impact_df['experimental_phase'] != actual_baseline_phase]

        self.logger.info(f"Calculated impact for {len(impact_df)} combinations of tenant/metric/phase.")

        # Garantir que a coluna tenant_id está presente
        if 'tenant_id' not in impact_df.columns:
            self.logger.error("Column 'tenant_id' is missing from the impact analysis DataFrame.")
            # Tentar recuperar o tenant_id se houver apenas um no contexto
            if len(df['tenant_id'].unique()) == 1:
                impact_df['tenant_id'] = df['tenant_id'].unique()[0]
                self.logger.info("Recovered single tenant_id for impact analysis.")
            else:
                 return pd.DataFrame() # Retorna DF vazio se não for possível resolver

        return impact_df

