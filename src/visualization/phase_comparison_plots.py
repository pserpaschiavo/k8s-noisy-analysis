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

def plot_phase_comparison(stats_df: pd.DataFrame, metric: str, output_dir: str, round_id: str):
    """
    Generates a bar plot to compare metric statistics across different experimental phases for a specific round.

    Args:
        stats_df: DataFrame with comparative statistics per tenant and phase.
        metric: The metric being plotted.
        output_dir: Directory to save the plot.
        round_id: The round identifier to be included in the plot title and filename.
    """
    if stats_df.empty:
        logger.warning(f"No stats data for metric '{metric}'. Phase comparison plot will not be generated.")
        return

    # Prepare data for plotting
    plot_data = stats_df.melt(id_vars=['tenant_id'], var_name='phase_metric', value_name='value')
    plot_data = plot_data[plot_data['phase_metric'].str.contains("_vs_baseline_pct")]
    plot_data['phase'] = plot_data['phase_metric'].str.replace('_vs_baseline_pct', '')

    if plot_data.empty:
        logger.warning(f"No baseline comparison data to plot for metric '{metric}' in round '{round_id}'.")
        return

    fig, ax = plt.subplots(figsize=(14, 8))
    sns.barplot(data=plot_data, x='phase', y='value', hue='tenant_id', ax=ax, palette=PUBLICATION_CONFIG['tenant_colors'])

    ax.set_title(f"Percentage Change in {metric} vs. Baseline (Round: {round_id})")
    ax.set_xlabel("Experimental Phase")
    ax.set_ylabel("Percentage Change (%)")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    ax.legend(title="Tenant")

    output_path = os.path.join(output_dir, f"phase_comparison_{metric}_{round_id}.png")
    save_plot(fig, output_path)

def compare_correlation_matrices(corr_matrices: Dict[str, pd.DataFrame], output_dir: str, metric_name: str):
    """
    Compares correlation matrices from different phases using heatmaps.

    Args:
        corr_matrices: Dictionary where keys are phase names and values are correlation matrices.
        output_dir: Directory to save the plots.
        metric_name: The name of the metric context for the title.
    """
    num_matrices = len(corr_matrices)
    if num_matrices == 0:
        logger.warning("No correlation matrices to compare.")
        return

    fig, axes = plt.subplots(1, num_matrices, figsize=(6 * num_matrices, 5), sharey=True)
    if num_matrices == 1:
        axes = [axes]

    for ax, (phase, matrix) in zip(axes, corr_matrices.items()):
        sns.heatmap(matrix, ax=ax, annot=True, cmap="coolwarm", fmt=".2f")
        ax.set_title(f"{metric_name} - {phase}")

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"correlation_comparison_{metric_name}.png")
    save_plot(fig, output_path)

def compare_causality_graphs(causality_results: Dict[str, Any], output_dir: str, metric_name: str):
    """
    Placeholder for comparing causality graphs.

    Args:
        causality_results: Dictionary with causality results from different phases.
        output_dir: Directory to save the plots.
        metric_name: The name of the metric context.
    """
    logger.info(f"Placeholder for causality graph comparison for {metric_name}. Data received for phases: {list(causality_results.keys())}")
    # In a real implementation, this would render and save plots comparing networkx graphs.
    # For now, we just log that the function was called.
    pass
