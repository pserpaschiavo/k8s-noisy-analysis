"""
Module: src.visualization.advanced_plots
Description: Advanced visualization functions for scientific-grade multi-round analysis
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings
import networkx as nx

from src.visualization_config import PUBLICATION_CONFIG

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# Apply global style settings
plt.rcParams.update(PUBLICATION_CONFIG.get('figure_style', {}))


def _save_figure(fig, output_dir, filename, dpi=300):
    """Helper to save a matplotlib figure."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    try:
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Figura salva em: {output_path}")
        plt.close(fig)
        return output_path
    except Exception as e:
        logger.error(f"Erro ao salvar figura {filename}: {e}")
        plt.close(fig)
        return None

def _plot_timeseries_by_round(metric_df, metric, time_col, xlabel, add_confidence_bands):
    """Plota a evolução temporal por round (média entre tenants)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_rounds = sorted(metric_df['round_id'].unique())
    round_colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(unique_rounds)))

    for i, round_id in enumerate(unique_rounds):
        round_data = metric_df[metric_df['round_id'] == round_id]
        aggregated = round_data.groupby(time_col)['metric_value'].agg(['mean', 'std']).reset_index()
        
        ax.plot(aggregated[time_col], aggregated['mean'], color=round_colors[i], label=f'Round {round_id}', linewidth=2, alpha=0.8)
        
        if add_confidence_bands and not aggregated['std'].isna().all():
            ax.fill_between(aggregated[time_col], aggregated['mean'] - aggregated['std'], aggregated['mean'] + aggregated['std'], color=round_colors[i], alpha=0.2)

    ax.set_title(f'Evolução por Round (Média de Tenants) - {metric.replace("_", " ").title()}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def _plot_timeseries_by_tenant(metric_df, metric, time_col, xlabel):
    """Plota a evolução temporal por tenant (todos os rounds sobrepostos)."""
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_rounds = sorted(metric_df['round_id'].unique())
    unique_tenants = sorted(metric_df['tenant_id'].unique())
    tenant_colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_tenants)))

    for i, tenant in enumerate(unique_tenants):
        tenant_data = metric_df[metric_df['tenant_id'] == tenant]
        for j, round_id in enumerate(unique_rounds):
            round_tenant_data = tenant_data[tenant_data['round_id'] == round_id]
            if not round_tenant_data.empty:
                linestyle = ['-', '--', '-.', ':'][j % 4]
                ax.plot(round_tenant_data[time_col], round_tenant_data['metric_value'], color=tenant_colors[i], linestyle=linestyle, alpha=0.6, label=f'{tenant} (R{round_id})' if j == 0 else None, linewidth=1.5)

    ax.set_title(f'Evolução por Tenant (Todos os Rounds) - {metric.replace("_", " ").title()}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " ").title())
    
    from matplotlib.lines import Line2D
    tenant_legend = [Line2D([0], [0], color=tenant_colors[i], lw=2, label=t) for i, t in enumerate(unique_tenants)]
    round_legend = [Line2D([0], [0], color='gray', linestyle=['-', '--', '-.', ':'][j % 4], lw=2, label=f'R{r}') for j, r in enumerate(unique_rounds)]
    leg1 = ax.legend(handles=tenant_legend, title='Tenants', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.add_artist(leg1)
    ax.legend(handles=round_legend, title='Rounds', bbox_to_anchor=(1.05, 0), loc='lower left')

    ax.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0, 0.85, 1))
    return fig

def _plot_smoothed_trends(metric_df, metric, time_col, xlabel):
    """Plota tendências suavizadas por round."""
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_rounds = sorted(metric_df['round_id'].unique())
    round_colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(unique_rounds)))
    
    num_points = len(metric_df)
    num_rounds = len(unique_rounds) if unique_rounds else 1
    window_size = max(5, num_points // (100 * num_rounds))

    for i, round_id in enumerate(unique_rounds):
        round_data = metric_df[metric_df['round_id'] == round_id]
        aggregated = round_data.groupby(time_col)['metric_value'].mean().reset_index().sort_values(time_col)
        aggregated['rolling_mean'] = aggregated['metric_value'].rolling(window=window_size, center=True, min_periods=1).mean()
        ax.plot(aggregated[time_col], aggregated['rolling_mean'], color=round_colors[i], label=f'Round {round_id}', linewidth=3, alpha=0.9)

    ax.set_title(f'Tendências Suavizadas (Média Móvel) - {metric.replace("_", " ").title()}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def _plot_distribution_by_phase(metric_df, metric):
    """Plota a distribuição de valores por fase experimental."""
    fig, ax = plt.subplots(figsize=(12, 7))
    unique_phases = sorted(metric_df['experimental_phase'].unique())
    
    sns.boxplot(data=metric_df, x='experimental_phase', y='metric_value', hue='round_id', ax=ax, order=unique_phases)
    ax.set_title(f'Distribuição por Fase e Round - {metric.replace("_", " ").title()}')
    ax.set_xlabel('Fase Experimental')
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title='Round')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def generate_consolidated_timeseries(
    df_long: pd.DataFrame,
    metric: str,
    output_dir: str,
    rounds: Optional[List[str]] = None,
    tenants: Optional[List[str]] = None,
    normalize_time: bool = True,
    add_confidence_bands: bool = True,
) -> Optional[Dict[str, str]]:
    """
    Gera um conjunto de plots de time series consolidados para uma métrica,
    salvando cada plot em um arquivo separado.
    
    Args:
        df_long: DataFrame em formato long com dados de todos os rounds.
        metric: Nome da métrica a ser plotada.
        output_dir: Diretório de saída para os plots.
        rounds: Lista de rounds a incluir (se None, usa todos).
        tenants: Lista de tenants a incluir (se None, usa todos).
        normalize_time: Se True, alinha o tempo pelo início de cada round.
        add_confidence_bands: Se True, adiciona bandas de confiança ao plot por round.
        
    Returns:
        Dicionário mapeando tipo de plot para o caminho do arquivo gerado, ou None.
    """
    logger.info(f"Gerando time series consolidados para métrica: {metric}")
    
    metric_df = df_long[df_long['metric_name'] == metric].copy()
    if metric_df.empty:
        logger.warning(f"Nenhum dado encontrado para métrica '{metric}'")
        return None
    
    if rounds:
        metric_df = metric_df[metric_df['round_id'].isin(rounds)]
    if tenants:
        metric_df = metric_df[metric_df['tenant_id'].isin(tenants)]
    
    if metric_df.empty:
        logger.warning(f"Nenhum dado após filtros para métrica '{metric}'")
        return None

    # Garantir que o tempo relativo é calculado por round
    if normalize_time:
        metric_df['timestamp'] = pd.to_datetime(metric_df['timestamp'])
        metric_df['relative_time'] = metric_df.groupby('round_id')['timestamp'].transform(lambda x: (x - x.min()).dt.total_seconds())
        time_col = 'relative_time'
        xlabel = 'Tempo Relativo (segundos)'
    else:
        time_col = 'timestamp'
        xlabel = 'Timestamp'

    dpi = PUBLICATION_CONFIG.get('figure_style', {}).get('figure.dpi', 300)
    plot_paths = {}

    try:
        # Plot 1: Por Round
        fig1 = _plot_timeseries_by_round(metric_df, metric, time_col, xlabel, add_confidence_bands)
        path1 = _save_figure(fig1, output_dir, f'consolidated_ts_by_round_{metric}.png', dpi)
        if path1: plot_paths['by_round'] = path1

        # Plot 2: Por Tenant
        fig2 = _plot_timeseries_by_tenant(metric_df, metric, time_col, xlabel)
        path2 = _save_figure(fig2, output_dir, f'consolidated_ts_by_tenant_{metric}.png', dpi)
        if path2: plot_paths['by_tenant'] = path2

        # Plot 3: Suavizado
        fig3 = _plot_smoothed_trends(metric_df, metric, time_col, xlabel)
        path3 = _save_figure(fig3, output_dir, f'consolidated_ts_smoothed_{metric}.png', dpi)
        if path3: plot_paths['smoothed'] = path3

        # Plot 4: Distribuição por Fase
        fig4 = _plot_distribution_by_phase(metric_df, metric)
        path4 = _save_figure(fig4, output_dir, f'consolidated_ts_by_phase_{metric}.png', dpi)
        if path4: plot_paths['by_phase'] = path4

    except Exception as e:
        logger.error(f"Erro fatal ao gerar plots para a métrica {metric}: {e}", exc_info=True)
        plt.close('all')
        return None

    return plot_paths


def generate_all_consolidated_timeseries(
    df_long: pd.DataFrame,
    output_dir: str,
    **kwargs
) -> Dict[str, Dict[str, str]]:
    """
    Gera time series consolidados para todas as métricas disponíveis.
    
    Args:
        df_long: DataFrame em formato long.
        output_dir: Diretório de saída.
        **kwargs: Argumentos adicionais para generate_consolidated_timeseries.
        
    Returns:
        Dicionário mapeando métrica -> dicionário de caminhos de plots.
    """
    logger.info("Gerando time series consolidados para todas as métricas...")
    
    timeseries_dir = os.path.join(output_dir, 'timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)
    
    results = {}
    metrics = sorted(df_long['metric_name'].unique())
    
    for metric in metrics:
        try:
            output_paths = generate_consolidated_timeseries(
                df_long=df_long,
                metric=metric,
                output_dir=timeseries_dir,
                **kwargs
            )
            if output_paths:
                results[metric] = output_paths
                logger.info(f"✅ Time series gerados para {metric}")
            else:
                logger.warning(f"❌ Falha ao gerar time series para {metric}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao processar métrica {metric}: {e}", exc_info=True)
    
    logger.info(f"Time series consolidados concluídos: {len(results)} métricas processadas")
    return results

def plot_aggregated_correlation_graph(correlation_matrix: pd.DataFrame, title: str, output_dir: str, filename: str, threshold: float = 0.5):
    """
    Plota um grafo de correlações agregadas entre tenants.

    Args:
        correlation_matrix: DataFrame contendo a matriz de correlação.
        title: Título do gráfico.
        output_dir: Diretório para salvar o gráfico.
        filename: Nome do arquivo de saída.
        threshold: Limiar para exibir apenas correlações acima deste valor (em módulo).
    """
    if correlation_matrix.empty:
        logger.warning(f"Skipping correlation graph for {filename}: matrix is empty.")
        return None

    # Filtrar a matriz para remover auto-correlações e aplicar o threshold
    corr_graph_data = correlation_matrix.stack().reset_index()
    corr_graph_data.columns = ['tenant_a', 'tenant_b', 'correlation']
    corr_graph_data = corr_graph_data[corr_graph_data['tenant_a'] != corr_graph_data['tenant_b']]
    corr_graph_data = corr_graph_data[abs(corr_graph_data['correlation']) >= threshold]

    if corr_graph_data.empty:
        logger.info(f"No correlations above threshold {threshold} for {filename}. Skipping graph.")
        return None

    # Criar o grafo
    G = nx.from_pandas_edgelist(corr_graph_data, 'tenant_a', 'tenant_b', edge_attr='correlation')

    # Configurações de visualização
    fig, ax = plt.subplots(figsize=PUBLICATION_CONFIG.get('graph_figsize', (14, 14)))
    pos = nx.spring_layout(G, k=0.7, iterations=50, seed=42)

    # Pesos e cores das arestas
    edges = G.edges()
    weights = [abs(G[u][v]['correlation']) * 5 for u, v in edges]  # Multiplicador para visibilidade
    edge_colors = [G[u][v]['correlation'] for u, v in edges]

    # Desenhar o grafo
    nodes = nx.draw_networkx_nodes(G, pos, node_size=3000, node_color=PUBLICATION_CONFIG.get('node_color', '#B4C8E4'), ax=ax)
    edges = nx.draw_networkx_edges(G, pos, width=weights, edge_color=edge_colors, edge_cmap=plt.cm.coolwarm,
                                   edge_vmin=-1, edge_vmax=1, alpha=0.7, ax=ax)
    labels = nx.draw_networkx_labels(G, pos, font_size=12, font_family='sans-serif', ax=ax)

    # Estilo dos nós
    nodes.set_edgecolor(PUBLICATION_CONFIG.get('node_edge_color', '#3B5998'))
    nodes.set_linewidth(1.5)

    # Adicionar uma colorbar para a legenda das arestas
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
    cbar.set_label('Força da Correlação', weight='bold')

    ax.set_title(title, fontsize=20, weight='bold')
    plt.axis('off')
    
    return _save_figure(fig, output_dir, filename)
