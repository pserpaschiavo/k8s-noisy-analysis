"""
Module: analysis_phase_comparison.py
Description: Implements comparative analyses between different experimental phases.

This module provides functionality to compare metrics and results between
experimental phases (baseline, attack, recovery), including:

1. Comparison of descriptive statistics
2. Comparison of correlation/covariance
3. Comparison of causality
4. Comparative visualizations
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union

from src.visualization_config import PUBLICATION_CONFIG
from src.utils import normalize_phase_name

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
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    
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
            if phase == baseline_phase_name:
                continue

            phase_data = subset[(subset['tenant_id'] == tenant) & (subset['experimental_phase'] == phase)]
            
            if phase_data.empty:
                tenant_data[f'{phase}_mean'] = 'N/A'
                tenant_data[f'{phase}_std'] = 'N/A'
                tenant_data[f'{phase}_vs_baseline_pct'] = 'N/A'
            else:
                phase_mean = phase_data['metric_value'].mean()
                phase_std = phase_data['metric_value'].std()
                
                tenant_data[f'{phase}_mean'] = phase_mean
                tenant_data[f'{phase}_std'] = phase_std
                
                # Calculate percentage change relative to baseline, ensuring values are numeric
                if isinstance(phase_mean, (int, float)) and isinstance(base_mean, (int, float)):
                    if abs(base_mean) > 1e-9:  # Avoid division by zero
                        change_pct = ((phase_mean - base_mean) / abs(base_mean)) * 100
                        tenant_data[f'{phase}_vs_baseline_pct'] = change_pct
                    else:
                        tenant_data[f'{phase}_vs_baseline_pct'] = 0.0
                else:
                    tenant_data[f'{phase}_vs_baseline_pct'] = 'N/A'
        
        results.append(tenant_data)

    if not results:
        return pd.DataFrame()
    
    return pd.DataFrame(results)
    


def plot_phase_comparison(stats_df: pd.DataFrame, metric: str, out_path: str) -> None:
    """
    Generates a comparative visualization of metrics by phase and tenant,
    using the centralized academic publication configuration.
    
    Args:
        stats_df: DataFrame with statistics per tenant and phase.
        metric: Name of the metric being visualized.
        out_path: Path to save the visualization.
    """
    if stats_df.empty:
        logging.warning(f"Empty DataFrame for visualization of {metric}")
        return

    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])

    # Get standardized names and units
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    
    # Extract relevant columns and map to display names
    phase_config = PUBLICATION_CONFIG['phase_display_names']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    
    tenants = stats_df['tenant_id'].tolist()
    
    # Prepare data for plotting
    plot_data = []
    for _, row in stats_df.iterrows():
        tenant_id = row['tenant_id']
        tenant_display = tenant_display_names.get(tenant_id, tenant_id)
        for phase_key, phase_display in phase_config.items():
            if f'{phase_key}_present' in row and row[f'{phase_key}_present']:
                plot_data.append({
                    'tenant_id': tenant_display,
                    'phase': phase_display,
                    'mean': row[f'{phase_key}_mean'],
                    'std': row.get(f'{phase_key}_std', 0),
                    'vs_baseline': row.get(f'{phase_key}_vs_baseline_pct', 0)
                })
    
    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        logging.warning(f"Insufficient data for visualization of {metric}")
        return
        
    # Define order for plotting
    phase_order = list(phase_config.values())
    tenant_order = [tenant_display_names.get(t, t) for t in tenants]

    # Create a color palette mapping display names to colors
    phase_color_map = {
        display_name: PUBLICATION_CONFIG['phase_colors'][phase_key]
        for phase_key, display_name in phase_config.items()
        if phase_key in PUBLICATION_CONFIG['phase_colors']
    }

    # Create subplots with adjusted size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8), gridspec_kw={'width_ratios': [3, 3]})
    
    # Plot 1: Mean values per phase and tenant
    sns.barplot(x='tenant_id', y='mean', hue='phase', data=plot_df, 
                ax=ax1, palette=phase_color_map, 
                hue_order=phase_order, order=tenant_order)
    ax1.set_title(f'Mean {metric_name} per Tenant and Phase')
    ax1.set_xlabel('Tenant')
    ax1.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax1.tick_params(axis='x', labelrotation=45)
    ax1.legend(title='Experimental Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Percentage change relative to baseline
    baseline_display_name = phase_config.get('baseline')
    if baseline_display_name:
        baseline_plot_df = plot_df[plot_df['phase'] != baseline_display_name]
        
        if not baseline_plot_df.empty:
            sns.barplot(x='tenant_id', y='vs_baseline', hue='phase', data=baseline_plot_df, 
                        ax=ax2, palette=phase_color_map, 
                        hue_order=[p for p in phase_order if p != baseline_display_name],
                        order=tenant_order)
            ax2.set_title(f'% Change in {metric_name} vs. Baseline')
            ax2.set_xlabel('Tenant')
            ax2.set_ylabel('% Change vs. Baseline')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
            ax2.tick_params(axis='x', labelrotation=45)
            if ax2.get_legend():
                ax2.get_legend().remove() # Remove redundant legend
    
    fig.suptitle(f'Comparative Analysis of {metric_name} Across Experimental Phases', fontsize=16, y=1.02)
    plt.tight_layout(rect=(0, 0, 0.9, 1)) # Adjust layout to make space for legend
    
    # Ensure output directory exists
    os.makedirs(out_path, exist_ok=True)
    
    # Save figure
    fig_path = os.path.join(out_path, f'phase_comparison_{metric}.png')
    try:
        fig.savefig(fig_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        logging.info(f"Saved phase comparison plot for {metric} to {fig_path}")
    except Exception as e:
        logging.error(f"Failed to save plot {fig_path}: {e}")
        plt.close(fig)


def compare_correlation_matrices(corr_matrices: Dict[str, pd.DataFrame], out_path: str, metric_name: str) -> None:
    """
    Compares correlation matrices between different experimental phases using heatmaps.
    
    Args:
        corr_matrices: Dictionary with correlation matrices per phase.
        out_path: Path to save the visualizations.
        metric_name: Name of the metric for context in the title.
    """
    if not corr_matrices:
        logging.warning("No correlation matrices to compare.")
        return

    # Determine the number of phases and set up subplots
    num_phases = len(corr_matrices)
    if num_phases == 0:
        return
        
    # Use display names from config
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Create a combined heatmap of differences
    if 'baseline' in corr_matrices:
        baseline_corr = corr_matrices['baseline']
        other_phases = {p: m for p, m in corr_matrices.items() if p != 'baseline'}
        
        if other_phases:
            num_other_phases = len(other_phases)
            fig_diff, axes_diff = plt.subplots(1, num_other_phases, 
                                               figsize=(6 * num_other_phases, 5), 
                                               sharey=True)
            if num_other_phases == 1:
                axes_diff = [axes_diff]

            for ax, (phase, corr_matrix) in zip(axes_diff, other_phases.items()):
                diff_matrix = corr_matrix - baseline_corr
                display_name = phase_display_names.get(phase, phase)
                sns.heatmap(diff_matrix, ax=ax, cmap='coolwarm', vmin=-1, vmax=1, annot=True, fmt=".2f")
                ax.set_title(f'Difference: {display_name} vs. Baseline')
            
            fig_diff.suptitle(f'Correlation Difference from Baseline for {metric_name}', fontsize=16)
            diff_path = os.path.join(out_path, f'correlation_difference_{metric_name}.png')
            fig_diff.savefig(diff_path, bbox_inches='tight')
            plt.close(fig_diff)
            logging.info(f"Saved correlation difference plot to {diff_path}")

def compare_causality_graphs(causality_results: Dict[str, Any], out_path: str, metric_name: str) -> None:
    """
    Compares causality analysis results between phases.
    For now, this is a placeholder for future, more complex comparisons.
    
    Args:
        causality_results: Dictionary with causality results per phase.
        out_path: Path to save visualizations.
        metric_name: Name of the metric for context.
    """
    logging.info(f"Causality comparison for {metric_name} is a placeholder.")
    # Future implementation could compare graph structures, link strengths, etc.
    pass

class PhaseComparisonAnalyzer:
    """
    Class for comparative analysis between experimental phases.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the analyzer.
        
        Args:
            df: Long DataFrame with data from all phases.
        """
        self.df = df
        self.logger = logging.getLogger("phase_comparison")
    
    def analyze_metric_across_phases(self, metric: str, round_id: str, output_dir: str) -> pd.DataFrame:
        """
        Performs a complete analysis of a metric across different phases.
        
        Args:
            metric: Name of the metric for analysis.
            round_id: ID of the round for analysis.
            output_dir: Directory to save visualizations.
            
        Returns:
            DataFrame with comparative data.
        """
        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Calculate comparative statistics
        stats_df = compute_phase_stats_comparison(
            self.df, 
            metric=metric,
            round_id=round_id
        )
        
        # 2. Generate comparative visualization
        out_path = os.path.join(output_dir, f'phase_comparison_{metric}_{round_id}.png')
        plot_phase_comparison(stats_df, metric, out_path)
        
        return stats_df
