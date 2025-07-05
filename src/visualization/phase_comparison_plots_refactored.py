"""
Module: src.visualization.phase_comparison_plots
Description: Plotting functions for phase comparison analysis.
"""
import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any

from src.visualization.plots import save_plot
from src.visualization_config import PUBLICATION_CONFIG

logger = logging.getLogger(__name__)


class PhaseComparisonPlots(PipelineStage):
    """
    Pipeline stage for generating and saving phase comparison plots.
    """
    def __init__(self, config: Dict[str, Any], output_dir: str):
        super().__init__(config, output_dir)
        self.logger = logging.getLogger("PhaseComparisonPlots")

    def run(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates and saves all configured phase comparison plots.

        Args:
            data: Dictionary containing data required for plotting.
                  - 'stats_df': DataFrame with comparative statistics.
                  - 'metric': The metric being plotted.

        Returns:
            The input data dictionary, unmodified.
        """
        stats_df = data.get('stats_df')
        metric = data.get('metric')

        if stats_df is None or metric is None:
            self.logger.warning("Stats DataFrame or metric not found in data. Skipping plot generation.")
            return data

        self._plot_phase_comparison(stats_df, metric)
        
        # Adicionar chamadas para outras visualizações se necessário
        # self._compare_correlation_matrices(data.get('corr_matrices'), metric)
        # self._compare_causality_graphs(data.get('causality_results'), metric)

        return data

    def _plot_phase_comparison(self, stats_df: pd.DataFrame, metric: str):
        """
        Generates a bar plot to compare metric statistics across different experimental phases.
        """
        if stats_df.empty:
            self.logger.warning(f"No stats data for metric '{metric}'. Phase comparison plot will not be generated.")
            return

        plot_data = stats_df.melt(id_vars=['tenant_id'], var_name='phase_metric', value_name='value')
        plot_data = plot_data[plot_data['phase_metric'].str.contains("_vs_baseline_pct")]
        plot_data['phase'] = plot_data['phase_metric'].str.replace('_vs_baseline_pct', '')

        if plot_data.empty:
            self.logger.warning(f"No baseline comparison data to plot for metric '{metric}'.")
            return

        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(data=plot_data, x='phase', y='value', hue='tenant_id', ax=ax, palette=PUBLICATION_CONFIG['tenant_colors'])

        ax.set_title(f"Percentage Change in {metric} vs. Baseline")
        ax.set_xlabel("Experimental Phase")
        ax.set_ylabel("Percentage Change (%)")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        ax.legend(title="Tenant")

        output_path = os.path.join(self.output_dir, f"phase_comparison_{metric}.png")
        save_plot(fig, output_path)

    def _compare_correlation_matrices(self, corr_matrices: Dict[str, pd.DataFrame], metric_name: str):
        """
        Compares correlation matrices from different phases using heatmaps.
        """
        if not corr_matrices:
            self.logger.warning("No correlation matrices to compare.")
            return

        num_matrices = len(corr_matrices)
        fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 5), sharey=True)
        if num_matrices == 1:
            axes = [axes]

        for ax, (phase, matrix) in zip(axes, corr_matrices.items()):
            sns.heatmap(matrix, ax=ax, annot=True, cmap="coolwarm", fmt=".2f")
            ax.set_title(f"{metric_name} - {phase}")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"correlation_comparison_{metric_name}.png")
        save_plot(fig, output_path)

    def _compare_causality_graphs(self, causality_results: Dict[str, Any], metric_name: str):
        """
        Placeholder for comparing causality graphs.
        """
        if not causality_results:
            return
        self.logger.info(f"Placeholder for causality graph comparison for {metric_name}. Data received for phases: {list(causality_results.keys())}")
        # In a real implementation, this would render and save plots comparing networkx graphs.
