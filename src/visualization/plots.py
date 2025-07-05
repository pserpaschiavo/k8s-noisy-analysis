"""
Module: src.visualization.plots
Description: Centralized plotting functions for the analysis pipeline, using a unified configuration.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
import networkx as nx
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from typing import Dict, List, Optional

from src.visualization_config import PUBLICATION_CONFIG

logger = logging.getLogger(__name__)

# Apply global style settings from the configuration
plt.rcParams.update(PUBLICATION_CONFIG.get('figure_style', {}))


def save_plot(fig, out_path: str):
    """Saves a matplotlib figure to a file, creating directories if needed."""
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        dpi = PUBLICATION_CONFIG.get('figure_style', {}).get('figure.dpi', 300)
        fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Plot saved to {out_path}")
        plt.close(fig)
    except Exception as e:
        logger.error(f"Failed to save plot to {out_path}: {e}")


def generate_enhanced_consolidated_boxplot(
    df_long: pd.DataFrame,
    metric: str,
    output_dir: str,
    normalize: bool = False,
    baseline_phase_name: Optional[str] = "Baseline"
) -> Optional[str]:
    """
    Generates an enhanced consolidated boxplot (violin plot) for a metric,
    comparing experimental phases across all rounds, with optional normalization.

    Args:
        df_long: DataFrame in long format with data from all rounds.
        metric: The metric to be plotted.
        output_dir: Directory to save the plot.
        normalize: If True, normalizes the data based on the baseline phase mean.
        baseline_phase_name: The name of the phase to use for normalization.

    Returns:
        Path to the generated plot file or None if an error occurs.
    """
    logger.info(f"Generating enhanced consolidated boxplot for metric: {metric} (Normalized: {normalize})")

    metric_df = df_long[df_long['metric_name'] == metric].copy()
    if metric_df.empty:
        logger.warning(f"No data for metric '{metric}'. Boxplot will not be generated.")
        return None

    # Get display names and colors from config
    metric_info = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric, 'unit': ''})
    metric_name = metric_info['name']
    metric_unit = metric_info['unit']
    
    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    phase_display_names = PUBLICATION_CONFIG['phase_display_names']
    color_palette = {tenant_display_names.get(k, k): v for k, v in PUBLICATION_CONFIG['tenant_colors'].items()}

    # Map internal names to display names for plotting
    metric_df['tenant_id'] = metric_df['tenant_id'].map(tenant_display_names).fillna(metric_df['tenant_id'])
    
    # Ensure experimental phases are ordered correctly
    phase_order = sorted(metric_df['experimental_phase'].unique())
    metric_df['experimental_phase'] = pd.Categorical(metric_df['experimental_phase'], categories=phase_order, ordered=True)

    y_axis_label = f"{metric_name} ({metric_unit})"
    plot_title = f"Distribution of {metric_name} by Phase"

    if normalize:
        # Find the actual baseline phase name (e.g., "1 - Baseline")
        actual_baseline_phase = None
        for phase in metric_df['experimental_phase'].unique():
            if baseline_phase_name in phase:
                actual_baseline_phase = phase
                break
        
        if not actual_baseline_phase:
            logger.error(f"Baseline phase containing '{baseline_phase_name}' not found for metric '{metric}'. Cannot normalize.")
            return None

        # Calculate the mean of the baseline phase for each tenant
        baseline_means = metric_df[metric_df['experimental_phase'] == actual_baseline_phase]
        if baseline_means.empty:
            logger.warning(f"No data for baseline phase '{actual_baseline_phase}' in metric '{metric}'. Skipping normalization.")
        else:
            tenant_baselines = baseline_means.groupby('tenant_id')['metric_value'].mean().to_dict()
            
            # Avoid division by zero
            for tenant, mean_val in tenant_baselines.items():
                if mean_val == 0:
                    logger.warning(f"Baseline mean for tenant '{tenant}' is zero. Using 1.0 to avoid division by zero.")
                    tenant_baselines[tenant] = 1.0

            metric_df['metric_value'] = metric_df.apply(
                lambda row: row['metric_value'] / tenant_baselines.get(row['tenant_id'], 1.0),
                axis=1
            )
            y_axis_label = f"Normalized {metric_name} (Ratio to Baseline)"
            plot_title += "\n(Normalized by Tenant's Baseline Mean)"


    fig, ax = plt.subplots(figsize=(16, 9))
    
    # Use violinplot for a richer distribution view
    sns.violinplot(
        data=metric_df,
        x='experimental_phase',
        y='metric_value',
        hue='tenant_id',
        palette=color_palette,
        inner='quartile',  # Shows quartiles inside the violin
        linewidth=1.5,
        ax=ax
    )

    ax.set_title(plot_title, fontweight='bold')
    ax.set_xlabel("Experimental Phase")
    ax.set_ylabel(y_axis_label)
    ax.tick_params(axis='x', rotation=30, labelsize=12)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.5)
    
    ax.legend(title='Tenant', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout(rect=(0, 0, 0.9, 1))

    norm_suffix = "_normalized" if normalize else ""
    output_path = os.path.join(output_dir, f"enhanced_boxplot_{metric}{norm_suffix}.png")
    save_plot(fig, output_path)
    return output_path


def generate_all_enhanced_consolidated_boxplots(df_long: pd.DataFrame, output_dir: str) -> Dict[str, str]:
    """
    Generates and saves all enhanced consolidated boxplots for each metric in the dataframe.

    Args:
        df_long: The dataframe containing all data.
        output_dir: The directory to save the plots.

    Returns:
        A dictionary mapping metric names to the paths of their generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    plot_paths = {}
    
    metrics = df_long['metric_name'].unique()
    for metric in metrics:
        # Generate both regular and normalized plots
        path = generate_enhanced_consolidated_boxplot(df_long, metric, output_dir, normalize=False)
        if path:
            plot_paths[f"{metric}_raw"] = path
            
        path_norm = generate_enhanced_consolidated_boxplot(df_long, metric, output_dir, normalize=True)
        if path_norm:
            plot_paths[f"{metric}_normalized"] = path_norm
            
    return plot_paths


def generate_consolidated_boxplot(
    df_long: pd.DataFrame,
    metric: str,
    output_dir: str
) -> Optional[str]:
    """
    Gera um boxplot consolidado para uma métrica, comparando fases experimentais
    entre todos os rounds.

    Args:
        df_long: DataFrame em formato long com dados de todos os rounds.
        metric: A métrica a ser plotada.
        output_dir: Diretório para salvar o gráfico.

    Returns:
        Caminho do arquivo de imagem do gráfico gerado ou None em caso de erro.
    """
    logger.info(f"Gerando boxplot consolidado para a métrica: {metric}")
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metric_df = df_long[df_long['metric_name'] == metric]
    
    if metric_df.empty:
        logger.warning(f"Não há dados para a métrica '{metric}'. O boxplot não será gerado.")
        plt.close(fig)
        return None

    sns.boxplot(data=metric_df, x='experimental_phase', y='metric_value', hue='tenant_id', ax=ax)
    
    ax.set_title(f"Boxplot Consolidado para {metric} (Todos os Rounds)")
    ax.set_xlabel("Fase Experimental")
    ax.set_ylabel("Valor da Métrica")
    plt.xticks(rotation=45)
    
    output_path = os.path.join(output_dir, f"consolidated_boxplot_{metric}.png")
    save_plot(fig, output_path)
    return output_path

def generate_consolidated_heatmap(
    aggregated_matrix: pd.DataFrame,
    output_dir: str,
    title: str,
    filename: str
) -> Optional[str]:
    """
    Gera um heatmap consolidado a partir de uma matriz agregada.

    Args:
        aggregated_matrix: Matriz de dados agregados (e.g., Jaccard, Spearman).
        output_dir: Diretório para salvar o gráfico.
        title: Título do gráfico.
        filename: Nome do arquivo de saída.

    Returns:
        Caminho do arquivo de imagem do gráfico gerado ou None em caso de erro.
    """
    logger.info(f"Gerando heatmap consolidado para: {title}")
    
    if aggregated_matrix.empty:
        logger.warning(f"Matriz agregada vazia para '{title}'. O heatmap não será gerado.")
        return None

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(aggregated_matrix, annot=True, cmap="viridis", fmt=".2f", ax=ax)
    
    ax.set_title(title)
    ax.set_xlabel("Par de Rounds")
    ax.set_ylabel("Métrica")
    
    output_path = os.path.join(output_dir, filename)
    save_plot(fig, output_path)
    return output_path
