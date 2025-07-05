"""
Module: src.visualization.causality_plots
Description: Plotting functions for causality analysis, including graphs and heatmaps.
"""
import os
import logging
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional

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


def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', threshold_mode: str = 'less'):
    """Plots a causality graph using the centralized configuration."""
    if causality_matrix.empty:
        logger.warning(f"Causality matrix is empty for {metric}. Skipping plot.")
        return

    G = nx.DiGraph() if directed else nx.Graph()
    tenants = causality_matrix.index.tolist()
    G.add_nodes_from(tenants)

    tenant_display_names = PUBLICATION_CONFIG['tenant_display_names']
    node_colors: list[str] = [PUBLICATION_CONFIG['tenant_colors'].get(t, '#8c564b') for t in tenants]
    node_labels = {t: tenant_display_names.get(t, t) for t in tenants}

    edges, edge_weights, edge_labels = [], [], {}
    edge_weights_list: list[float] = []
    for src in tenants:
        for tgt in tenants:
            if src == tgt: continue
            val = causality_matrix.at[src, tgt]
            is_significant, weight = False, 0
            if threshold_mode == 'less' and not pd.isna(val) and float(val) < threshold:
                is_significant, weight = True, 1 - float(val)
            elif threshold_mode == 'greater' and not pd.isna(val) and float(val) > threshold:
                is_significant, weight = True, float(val)
            
            if is_significant:
                G.add_edge(src, tgt, weight=weight)
                edges.append((src, tgt))
                edge_weights_list.append(weight * 5 + 1)
                edge_labels[(src, tgt)] = f"{weight:.2f}"

    if not G.nodes:
        logger.warning(f"No nodes in graph for {metric}. Skipping plot.")
        return

    pos = nx.spring_layout(G, seed=42, k=0.9, iterations=50)
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_facecolor('#f8f8f8')

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=3000, node_shape='o', edgecolors='darkblue', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=12, font_weight='bold', font_color='white', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=edges, arrowstyle='->' if directed else '-', width=edge_weights_list, edge_color='gray', alpha=0.8, connectionstyle='arc3,rad=0.1', ax=ax)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='darkred', font_size=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7), ax=ax)

    metric_name = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric})['name']
    title = f'Causality: {metric_name} (Edges: {threshold_mode.replace("less", "p <").replace("greater", "TE >")} {threshold:.2g})'
    ax.set_title(title, fontweight='bold')
    ax.axis('off')

    legend_patches = [mpatches.Patch(color=c, label=l) for t, l, c in zip(tenants, node_labels.values(), node_colors)]
    ax.legend(handles=legend_patches, title="Tenants", bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.tight_layout(rect=(0, 0, 0.85, 1))
    save_plot(fig, out_path)
    return out_path


def plot_causality_heatmap(causality_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, method: str, value_type: str = 'p-value') -> str | None:
    """Plots a causality heatmap using the centralized configuration."""
    if causality_matrix.empty:
        logger.warning(f"Empty causality matrix for {metric}, {phase}, {round_id}")
        return None

    metric_display = PUBLICATION_CONFIG['metric_display_names'].get(metric, {'name': metric})['name']
    phase_display = PUBLICATION_CONFIG['phase_display_names'].get(phase, phase)
    cmap = PUBLICATION_CONFIG['heatmap_colormaps'].get(value_type, 'viridis')
    cbar_label = f"{method} {value_type.replace('_', ' ').title()}"

    fig, ax = plt.subplots()
    sns.heatmap(causality_matrix, annot=True, cmap=cmap, fmt=".3f", linewidths=0.5, cbar_kws={"label": cbar_label}, ax=ax)

    title = f'{method} Causality: {metric_display}\n{phase_display} - Round {round_id}'
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Caused by (Source Tenant)")
    ax.set_ylabel("Affected (Target Tenant)")
    
    out_path = os.path.join(out_dir, f"{method.lower()}_causality_heatmap_{metric}_{phase}_{round_id}.png")
    save_plot(fig, out_path)
    return out_path
