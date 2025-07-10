# -*- coding: utf-8 -*-
"""
Module: src.visualization.multi_round_plots
Description: Functions to generate aggregated visualizations from multi-round analysis results.
"""

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from typing import List
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)

def plot_aggregated_impact_boxplots(
    consolidated_df: pd.DataFrame,
    output_dir: str
) -> List[str]:
    """
    Generates box plots of percentage change in impact per tenant and metric.

    Args:
        consolidated_df: DataFrame with consolidated results from all rounds.
        output_dir: Directory to save the plots.

    Returns:
        A list of paths to the generated plot files.
    """
    logger.info("Generating aggregated impact box plots...")
    plot_paths = []
    
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        for tenant_id, group in consolidated_df.groupby('tenant_id'):
            plt.figure(figsize=(15, 8))
            sns.boxplot(data=group, x='metric_name', y='percentage_change', hue='experimental_phase')
            
            plt.title(f'Aggregated Impact Distribution for Tenant: {tenant_id}')
            plt.ylabel('Percentage Change (%)')
            plt.xlabel('Metric Name')
            plt.xticks(rotation=45, ha='right')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.tight_layout()
            
            plot_filename = f"boxplot_impact_{tenant_id}.png"
            plot_path = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(plot_path)
            logger.info(f"Saved box plot to {plot_path}")

    except Exception as e:
        logger.error(f"Error generating aggregated impact box plots: {e}", exc_info=True)

    return plot_paths


def plot_aggregated_impact_bar_charts(
    consolidated_df: pd.DataFrame,
    output_dir: str,
    ci: float = 95
) -> List[str]:
    """
    Generates bar charts of the average percentage change in impact with confidence intervals.

    Args:
        consolidated_df: DataFrame with consolidated results from all rounds.
        output_dir: Directory to save the plots.
        ci: Confidence interval level (e.g., 95 for 95% CI).

    Returns:
        A list of paths to the generated plot files.
    """
    logger.info(f"Generating aggregated impact bar charts with {ci}% confidence intervals...")
    plot_paths = []
    
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Use 'catplot' for better control over the figure and axes
        g = sns.catplot(
            data=consolidated_df,
            x='metric_name',
            y='percentage_change',
            hue='experimental_phase',
            col='tenant_id',
            kind='bar',
            ci=ci,
            height=6,
            aspect=1.5,
            col_wrap=2,  # Wrap plots into 2 columns
            sharex=False, # Metric names can be different
            legend_out=True
        )
        
        g.fig.suptitle('Average Impact per Tenant with 95% CI', y=1.03, fontsize=16)
        g.set_axis_labels('Metric Name', f'Average Percentage Change (CI {ci}%)')
        g.set_titles("Tenant: {col_name}")
        
        # Rotate x-axis labels for each subplot
        for ax in g.axes.flat:
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_horizontalalignment('right')

        plt.tight_layout(rect=(0, 0, 1, 0.97)) # Adjust for the supertitle
        
        plot_filename = "barchart_impact_ci_all_tenants.png"
        plot_path = os.path.join(output_dir, plot_filename)
        g.savefig(plot_path)
        plt.close('all')
        plot_paths.append(plot_path)
        logger.info(f"Saved aggregated bar chart to {plot_path}")

    except Exception as e:
        logger.error(f"Error generating aggregated impact bar charts: {e}", exc_info=True)

    return plot_paths

def plot_correlation_consistency_heatmap(
    correlation_consistency_df: pd.DataFrame, 
    output_dir: str
) -> List[str]:
    """
    Generates heatmaps showing the mean correlation between tenants for each 
    metric and phase, based on multi-round consistency data.

    Args:
        correlation_consistency_df: DataFrame with correlation consistency data.
        output_dir: Directory to save the plots.

    Returns:
        A list of paths to the generated plot files.
    """
    if correlation_consistency_df.empty:
        logging.warning("Correlation consistency DataFrame is empty. Skipping heatmap generation.")
        return []

    plot_paths = []
    
    # Group by metric and phase to create a plot for each combination
    for (metric, phase), group in correlation_consistency_df.groupby(['metric', 'phase']):
        try:
            # Create a pivot table for the heatmap
            pivot_table = group.pivot(
                index='tenant1', 
                columns='tenant2', 
                values='mean_correlation'
            )
            
            # Ensure the matrix is symmetric and complete
            all_tenants = sorted(list(set(group['tenant1']) | set(group['tenant2'])))
            pivot_table = pivot_table.reindex(index=all_tenants, columns=all_tenants)
            
            # Fill the matrix symmetrically and the diagonal
            # We use .add with fill_value=0 to sum the pivot with its transpose
            symmetric_matrix = pivot_table.add(pivot_table.T, fill_value=0)
            
            # For values that are not NaN in both (i.e., the original pairs), the sum doubled the value.
            # We divide by 2 where the original pivot was not null.
            symmetric_matrix[pivot_table.notna()] = symmetric_matrix[pivot_table.notna()] / 2

            np.fill_diagonal(symmetric_matrix.values, 1.0) # Correlation of a tenant with itself is 1

            fig, ax = plt.subplots(figsize=(14, 12))
            sns.heatmap(
                symmetric_matrix, 
                annot=True, 
                cmap='viridis', 
                fmt='.2f', 
                linewidths=.5, 
                ax=ax,
                vmin=-1, vmax=1 # Fix the color scale from -1 to 1
            )
            
            # Clean the phase name for the filename
            safe_phase_name = phase.replace(' ', '_').replace('/', '_')
            title = f'Mean Correlation Consistency - {metric} / {phase}'
            ax.set_title(title, fontsize=16)
            ax.set_xlabel('Tenant', fontsize=12)
            ax.set_ylabel('Tenant', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            
            plot_filename = f"heatmap_correlation_consistency_{metric}_{safe_phase_name}.png"
            output_path = os.path.join(output_dir, plot_filename)
            plt.tight_layout()
            fig.savefig(output_path)
            logging.info(f"Correlation consistency heatmap saved to: {output_path}")
            plt.close(fig)
            plot_paths.append(output_path)

        except Exception as e:
            logger.error(f"Failed to generate heatmap for metric '{metric}' and phase '{phase}': {e}", exc_info=True)

    return plot_paths

def plot_causality_consistency_matrix(causality_frequency: pd.DataFrame, output_dir: str) -> str:
    """
    Generates a heatmap of the causality consistency matrix.
    """
    if causality_frequency.empty:
        logging.warning("Causality frequency DataFrame is empty. Skipping plot.")
        return ""

    causality_matrix = causality_frequency.pivot(index='source', columns='target', values='consistency_rate').fillna(0)
    
    all_vars = sorted(list(set(causality_frequency['source']) | set(causality_frequency['target'])))
    causality_matrix = causality_matrix.reindex(index=all_vars, columns=all_vars).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(causality_matrix, annot=True, cmap='viridis', fmt='.1f', linewidths=.5, ax=ax)
    
    ax.set_title('Multi-Round Causality Consistency Matrix (%)', fontsize=16)
    ax.set_xlabel('Target Variable', fontsize=12)
    ax.set_ylabel('Source Variable', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    output_path = os.path.join(output_dir, 'causality_consistency_matrix.png')
    plt.tight_layout()
    fig.savefig(output_path)
    logging.info(f"Causality consistency matrix saved to: {output_path}")
    plt.close(fig)
    return output_path

def plot_aggregated_causality_graph(causality_frequency: pd.DataFrame, output_dir: str) -> str:
    """
    Generates an aggregated causality graph from the frequency of links.
    """
    if causality_frequency.empty:
        logging.warning("Causality frequency DataFrame is empty. Skipping plot.")
        return ""

    G = nx.from_pandas_edgelist(
        causality_frequency[causality_frequency['consistency_rate'] > 0],
        source='source',
        target='target',
        edge_attr='consistency_rate',
        create_using=nx.DiGraph()
    )

    if G.number_of_nodes() == 0:
        logging.warning("Causality graph is empty. Skipping plot.")
        return ""

    fig, ax = plt.subplots(figsize=(15, 15))
    pos = nx.circular_layout(G)

    edge_widths = [d['consistency_rate'] / 10 for _, _, d in G.edges(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_size=2500, node_color='skyblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        width=edge_widths,  # The linter might complain, but networkx accepts a list here # type: ignore
        edge_color='gray',
        arrows=True,
        arrowstyle='->',
        arrowsize=20,
        ax=ax
    )

    edge_labels = {(u, v): f"{d['consistency_rate']:.1f}%" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)

    ax.set_title('Aggregated Multi-Round Causality Graph (Consistency %)', fontsize=20)
    plt.axis('off')
    
    output_path = os.path.join(output_dir, 'aggregated_causality_graph.png')
    plt.tight_layout()
    fig.savefig(output_path)
    logging.info(f"Aggregated causality graph saved to: {output_path}")
    plt.close(fig)
    return output_path

