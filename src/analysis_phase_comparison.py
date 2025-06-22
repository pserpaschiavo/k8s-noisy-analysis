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

    # List of phases in the expected format
    phases = [
        '1 - Baseline',
        '2 - CPU Noise',
        '3 - Memory Noise',
        '4 - Network Noise',
        '5 - Disk Noise',
        '6 - Combined Noise',
        '7 - Recovery'
    ]
    available_phases = [p for p in phases if p in subset['experimental_phase'].unique()]
    
    # Structure to store results
    results = []
    
    # For each combination of tenant and phase
    for tenant in tenants:
        tenant_data = {}
        tenant_data['tenant_id'] = tenant
        
        # Base metrics for the tenant (in baseline)
        baseline_data = subset[(subset['tenant_id'] == tenant) & 
                              (subset['experimental_phase'] == '1 - Baseline')]
        
        # If no baseline data, use general averages
        if len(baseline_data) == 0:
            baseline_stats = {
                'mean': subset['metric_value'].mean(),
                'std': subset['metric_value'].std(),
                'median': subset['metric_value'].median(),
                'min': subset['metric_value'].min(),
                'max': subset['metric_value'].max()
            }
        else:
            baseline_stats = {
                'mean': baseline_data['metric_value'].mean(),
                'std': baseline_data['metric_value'].std(),
                'median': baseline_data['metric_value'].median(),
                'min': baseline_data['metric_value'].min(),
                'max': baseline_data['metric_value'].max()
            }
        
        # For each phase, calculate statistics and relative variation to baseline
        for phase in available_phases:
            phase_data = subset[(subset['tenant_id'] == tenant) & 
                               (subset['experimental_phase'] == phase)]
            
            if len(phase_data) == 0:
                # Tenant not present in this phase
                tenant_data[f'{phase}_present'] = False
                continue
                
            tenant_data[f'{phase}_present'] = True
            tenant_data[f'{phase}_mean'] = phase_data['metric_value'].mean()
            tenant_data[f'{phase}_std'] = phase_data['metric_value'].std()
            tenant_data[f'{phase}_median'] = phase_data['metric_value'].median()
            tenant_data[f'{phase}_min'] = phase_data['metric_value'].min()
            tenant_data[f'{phase}_max'] = phase_data['metric_value'].max()
            
            # Calculate percentage change relative to baseline (if baseline exists)
            if phase != '1 - Baseline' and '1 - Baseline' in available_phases:
                baseline_mean = baseline_stats['mean']
                if baseline_mean != 0:  # Avoid division by zero
                    tenant_data[f'{phase}_vs_baseline_pct'] = ((tenant_data[f'{phase}_mean'] - baseline_mean) / 
                                                            abs(baseline_mean)) * 100
                else:
                    tenant_data[f'{phase}_vs_baseline_pct'] = np.nan
        
        results.append(tenant_data)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

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
    phases = list(PUBLICATION_CONFIG['phase_display_names'].keys())
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    
    tenants = stats_df['tenant_id'].tolist()
    
    # Prepare data for plotting
    plot_data = []
    for _, row in stats_df.iterrows():
        tenant_id = row['tenant_id']
        tenant_display = tenant_display_names.get(tenant_id, tenant_id)
        for phase_key in phases:
            if f'{phase_key}_present' in row and row[f'{phase_key}_present']:
                plot_data.append({
                    'tenant_id': tenant_display,
                    'phase': phase_display_names[phase_key],
                    'mean': row[f'{phase_key}_mean'],
                    'std': row.get(f'{phase_key}_std', 0),
                    'vs_baseline': row.get(f'{phase_key}_vs_baseline_pct', 0)
                })
    
    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        logging.warning(f"Insufficient data for visualization of {metric}")
        return
        
    # Define order for plotting
    phase_order = [phase_display_names[p] for p in phases]
    tenant_order = [tenant_display_names.get(t, t) for t in tenants]

    # Create a color palette mapping display names to colors
    phase_color_map = {
        display_name: PUBLICATION_CONFIG['phase_colors'][phase_key]
        for phase_key, display_name in phase_display_names.items()
        if phase_key in PUBLICATION_CONFIG['phase_colors']
    }

    # Create subplots with adjusted size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    
    # Plot 1: Mean values per phase and tenant
    sns.barplot(x='tenant_id', y='mean', hue='phase', data=plot_df, 
                ax=ax1, palette=phase_color_map, 
                hue_order=phase_order, order=tenant_order)
    ax1.set_title(f'Mean {metric_name} per Tenant and Phase')
    ax1.set_xlabel('Tenant')
    ax1.set_ylabel(f'{metric_name} ({metric_unit})' if metric_unit else metric_name)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(title='Experimental Phase', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Percentage change relative to baseline
    baseline_display_name = phase_display_names['1 - Baseline']
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
        ax2.tick_params(axis='x', rotation=45)
    
    # Adjust layout
    plt.tight_layout(rect=(0, 0, 0.9, 1)) # Adjust for legend
    
    # Save figure
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Phase comparison visualization saved to {out_path}")

def compare_correlation_matrices(corr_matrices: Dict[str, pd.DataFrame], metric: str, out_path: str) -> pd.DataFrame:
    """
    Compares correlation matrices between different experimental phases using
    the centralized academic publication configuration.
    
    Args:
        corr_matrices: Dictionary of correlation matrices by phase.
        metric: Name of the metric being analyzed.
        out_path: Path to save the visualization.
        
    Returns:
        DataFrame showing the most significant differences between phases.
    """
    if not corr_matrices or len(corr_matrices) < 2:
        logging.warning("Insufficient number of matrices for comparison")
        return pd.DataFrame()

    # Apply centralized figure style
    plt.rcParams.update(PUBLICATION_CONFIG['figure_style'])
    
    # Get standardized names
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric})
    metric_name = metric_info['name']
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']

    # Get all unique tenant IDs from all matrices
    all_tenant_ids = set()
    for mat in corr_matrices.values():
        all_tenant_ids.update(mat.index)
    
    # Get the corresponding display names, maintaining order
    sorted_tenant_ids = sorted(list(all_tenant_ids))
    display_tenants = [tenant_display_names.get(t, t) for t in sorted_tenant_ids]

    # Standardize all matrices (rename and re-index)
    standardized_matrices = {}
    for phase_key, mat in corr_matrices.items():
        # Rename index and columns to display names
        renamed_mat = mat.rename(index=tenant_display_names, columns=tenant_display_names)
        # Reindex to ensure all matrices have the same tenants in the same order
        standardized_mat = renamed_mat.reindex(index=display_tenants, columns=display_tenants)
        standardized_matrices[phase_key] = standardized_mat

    # Calculate differences between phases
    phases = sorted(list(standardized_matrices.keys()))
    diff_matrices = {}
    
    for i, phase1_key in enumerate(phases):
        for j, phase2_key in enumerate(phases):
            if i >= j:
                continue
                
            mat1 = standardized_matrices[phase1_key]
            mat2 = standardized_matrices[phase2_key]
            
            # Ensure matrices are aligned and fill missing values with 0
            mat1_aligned, mat2_aligned = mat1.align(mat2, join='outer', axis=None, fill_value=0)
            
            diff_mat = mat2_aligned - mat1_aligned
            
            phase1_display = phase_display_names.get(phase1_key, phase1_key)
            phase2_display = phase_display_names.get(phase2_key, phase2_key)
            diff_name = f"{phase1_display} to {phase2_display}"
            diff_matrices[diff_name] = diff_mat
    
    # Prepare visualization of differences
    if diff_matrices:
        n_diffs = len(diff_matrices)
        fig, axes = plt.subplots(1, n_diffs, figsize=(7 * n_diffs + 1, 6), squeeze=False)
        axes = axes.flatten()
            
        for ax, (diff_name, diff_mat) in zip(axes, diff_matrices.items()):
            sns.heatmap(diff_mat, vmin=-1, vmax=1, center=0, cmap='RdBu_r',
                      annot=True, fmt=".2f", ax=ax, annot_kws={"size": 8})
            ax.set_title(f'Correlation Change: {diff_name}\n({metric_name})')
            ax.tick_params(axis='x', rotation=45, labelrotation=45)
            ax.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        
        # Find the most significant differences
        results = []
        for diff_name, diff_mat in diff_matrices.items():
            # Transform matrix into a "long" format for analysis
            diff_long = diff_mat.stack().reset_index()
            diff_long.columns = ['tenant1', 'tenant2', 'correlation_change']
            
            # Remove self-comparisons (diagonal)
            diff_long = diff_long[diff_long['tenant1'] != diff_long['tenant2']].copy()
            
            # Remove duplicate pairs by sorting tenant names and dropping duplicates
            diff_long['pair'] = diff_long.apply(lambda row: tuple(sorted((row['tenant1'], row['tenant2']))), axis=1)
            diff_long = diff_long.drop_duplicates(subset='pair').drop(columns='pair')

            # Select top 5 differences based on absolute magnitude
            diff_long['abs_change'] = diff_long['correlation_change'].abs()
            top_changes = diff_long.nlargest(5, 'abs_change')
            
            for _, row in top_changes.iterrows():
                results.append({
                    'phase_transition': diff_name,
                    'tenant1': row['tenant1'],
                    'tenant2': row['tenant2'],
                    'correlation_change': row['correlation_change'],
                    'metric': metric
                })
        
        return pd.DataFrame(results)
    
    return pd.DataFrame()

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
