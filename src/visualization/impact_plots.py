"""
Module: src.visualization.impact_plots
Description: Provides functions to generate plots for impact analysis.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import logging
from typing import List

logger = logging.getLogger(__name__)

def plot_impact_summary(impact_df: pd.DataFrame, output_dir: str) -> List[str]:
    """
    Generates and saves bar plots summarizing the impact analysis results.

    For each tenant, it creates a plot showing the percentage change in metrics
    across different experimental phases.

    Args:
        impact_df: DataFrame with impact analysis results.
        output_dir: Directory to save the plots.

    Returns:
        A list of paths to the generated plot files.
    """
    if impact_df.empty:
        logger.warning("Impact DataFrame is empty. Skipping plot generation.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    plot_paths = []

    sns.set_theme(style="whitegrid")

    for tenant_id in impact_df['tenant_id'].unique():
        tenant_df = impact_df[impact_df['tenant_id'] == tenant_id]
        
        if tenant_df.empty:
            continue

        plt.figure(figsize=(14, 8))
        
        plot = sns.barplot(
            x='metric_name', 
            y='percentage_change', 
            hue='experimental_phase', 
            data=tenant_df,
            palette='viridis'
        )

        plt.title(f'Impact Analysis for Tenant: {tenant_id}', fontsize=16, fontweight='bold')
        plt.ylabel('Percentage Change (%)', fontsize=12)
        plt.xlabel('Metric', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Experimental Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=(0, 0, 0.85, 1))
        plt.axhline(0, color='grey', lw=1, linestyle='--')

        for p in plot.patches:
            plot.annotate(format(p.get_height(), '.1f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 9), 
                           textcoords = 'offset points')

        filename = f"impact_summary_{tenant_id}.png"
        plot_path = os.path.join(output_dir, filename)
        
        try:
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plot_paths.append(plot_path)
            logger.info(f"Saved impact plot for tenant {tenant_id} to {plot_path}")
        except Exception as e:
            logger.error(f"Failed to save plot {plot_path}: {e}")
        finally:
            plt.close()
            
    return plot_paths
