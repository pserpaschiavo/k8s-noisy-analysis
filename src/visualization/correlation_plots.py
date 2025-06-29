"""
Module: correlation_plots.py
Description: Módulo para visualizações de correlações intra-fase.

Este módulo implementa funções para visualizar correlações entre tenants
dentro de cada fase experimental, incluindo redes de correlação, heatmaps
e análises de estabilidade entre rounds.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

def plot_correlation_heatmap(
    correlation_df: pd.DataFrame,
    output_dir: str,
    metric: Optional[str] = None,
    phase: Optional[str] = None,
    round_id: Optional[str] = None,
    min_correlation: float = 0.0,
    cmap: str = 'coolwarm',
    filename_prefix: str = ''
) -> str:
    """
    Gera heatmaps de correlação entre tenants para uma fase específica.
    
    Args:
        correlation_df: DataFrame com dados de correlação intra-fase
                       (colunas: round_id, metric_name, experimental_phase, tenant_pair, correlation, method, sample_size)
        output_dir: Diretório para salvar os gráficos
        metric: Métrica específica a visualizar (se None, gera para todas)
        phase: Fase específica a visualizar (se None, gera para todas)
        round_id: Round específico a visualizar (se None, gera para todos)
        min_correlation: Valor mínimo de correlação para considerar (filtro)
        cmap: Colormap para visualização
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o último arquivo gerado ou string vazia se nenhum
    """
    if correlation_df.empty:
        logger.warning("DataFrame de correlações vazio. Não é possível gerar heatmap.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['round_id', 'metric_name', 'experimental_phase', 'tenant_pair', 'correlation']
    missing_cols = [col for col in required_cols if col not in correlation_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica, fase e round se especificados
    filtered_df = correlation_df.copy()
    
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{metric}'.")
            return ""
    
    if phase:
        filtered_df = filtered_df[filtered_df['experimental_phase'] == phase]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a fase '{phase}'.")
            return ""
    
    if round_id:
        filtered_df = filtered_df[filtered_df['round_id'] == round_id]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para o round '{round_id}'.")
            return ""
    
    # Filtrar por valor mínimo de correlação
    filtered_df = filtered_df[abs(filtered_df['correlation']) >= min_correlation]
    if filtered_df.empty:
        logger.warning(f"Nenhuma correlação com magnitude acima de {min_correlation}.")
        return ""
    
    # Determinar combinações únicas de métrica, fase e round
    combinations = filtered_df.groupby(['metric_name', 'experimental_phase', 'round_id'])
    output_paths = []
    
    for (current_metric, current_phase, current_round), group_df in combinations:
        # Extrair pares de tenants e valores de correlação
        tenant_pairs = []
        for _, row in group_df.iterrows():
            tenant_pair = row['tenant_pair'].split('-')
            if len(tenant_pair) == 2:
                tenant1, tenant2 = tenant_pair
                tenant_pairs.append((tenant1, tenant2))
        
        # Obter lista única de tenants
        unique_tenants = sorted(set([tenant for pair in tenant_pairs for tenant in pair]))
        
        if len(unique_tenants) < 2:
            logger.warning(f"Insuficientes tenants para métrica='{current_metric}', fase='{current_phase}', round='{current_round}'")
            continue
        
        # Criar matriz de correlação
        corr_matrix = pd.DataFrame(index=unique_tenants, columns=unique_tenants, dtype=float)
        corr_matrix.fillna(0, inplace=True)
        
        # Preencher matriz com valores de correlação
        for _, row in group_df.iterrows():
            tenant_pair = row['tenant_pair'].split('-')
            if len(tenant_pair) == 2:
                tenant1, tenant2 = tenant_pair
                corr_matrix.loc[tenant1, tenant2] = row['correlation']
                corr_matrix.loc[tenant2, tenant1] = row['correlation']  # Matriz simétrica
        
        # Definir valores diagonais como 1 (autocorrelação)
        for tenant in unique_tenants:
            corr_matrix.loc[tenant, tenant] = 1.0
        
        # Criar a visualização
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Máscara para manter metade
        
        ax = sns.heatmap(
            corr_matrix,
            annot=True,
            cmap=cmap,
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            linewidths=0.5,
            mask=mask,
            cbar_kws={"shrink": 0.8, "label": "Correlação"}
        )
        
        # Título e rótulos
        title = f"Correlações Intra-Fase - {current_metric.replace('_', ' ').title()}\n"
        title += f"Fase: {current_phase}, Round: {current_round}"
        plt.title(title)
        
        # Ajustes finais
        plt.tight_layout()
        
        # Salvar figura
        safe_metric = current_metric.replace(' ', '_')
        safe_phase = current_phase.replace(' ', '_')
        filename = f"{filename_prefix}correlation_heatmap_{safe_metric}_{safe_phase}_{current_round}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Heatmap de correlação salvo em: {output_path}")
        output_paths.append(output_path)
    
    # Se não foi gerado nenhum gráfico, retorna string vazia
    if not output_paths:
        logger.warning("Nenhum heatmap de correlação foi gerado.")
        return ""
    
    # Retorna o caminho do último arquivo salvo se vários foram gerados
    return output_paths[-1]

def plot_correlation_network(
    correlation_df: pd.DataFrame,
    output_dir: str,
    metric: str,
    phase: str,
    round_id: Optional[str] = None,
    min_correlation: float = 0.5,
    layout: str = 'spring',
    filename_prefix: str = ''
) -> str:
    """
    Gera um gráfico de rede para visualizar correlações entre tenants.
    
    Args:
        correlation_df: DataFrame com dados de correlação intra-fase
        output_dir: Diretório para salvar os gráficos
        metric: Métrica a visualizar
        phase: Fase a visualizar
        round_id: Round específico a visualizar (se None, usa média entre rounds)
        min_correlation: Valor mínimo de correlação para considerar
        layout: Layout da rede ('spring', 'circular', 'kamada_kawai', etc.)
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo gerado
    """
    if correlation_df.empty:
        logger.warning("DataFrame de correlações vazio. Não é possível gerar rede.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['metric_name', 'experimental_phase', 'tenant_pair', 'correlation']
    missing_cols = [col for col in required_cols if col not in correlation_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica e fase
    filtered_df = correlation_df[
        (correlation_df['metric_name'] == metric) & 
        (correlation_df['experimental_phase'] == phase)
    ]
    
    if filtered_df.empty:
        logger.warning(f"Nenhum dado para métrica='{metric}', fase='{phase}'")
        return ""
    
    # Filtrar por round se especificado ou calcular média entre rounds
    if round_id:
        filtered_df = filtered_df[filtered_df['round_id'] == round_id]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para o round '{round_id}'")
            return ""
    else:
        # Agrupar por par de tenants e calcular média entre rounds
        filtered_df = filtered_df.groupby(['metric_name', 'experimental_phase', 'tenant_pair'])['correlation'].mean().reset_index()
    
    # Criar grafo
    G = nx.Graph()
    
    # Adicionar nós e arestas
    for _, row in filtered_df.iterrows():
        tenant_pair = row['tenant_pair'].split('-')
        if len(tenant_pair) == 2:
            tenant1, tenant2 = tenant_pair
            correlation = row['correlation']
            
            # Adicionar nós se ainda não existirem
            if tenant1 not in G:
                G.add_node(tenant1)
            if tenant2 not in G:
                G.add_node(tenant2)
            
            # Adicionar aresta se correlação acima do limiar
            if abs(correlation) >= min_correlation:
                G.add_edge(
                    tenant1, 
                    tenant2, 
                    weight=abs(correlation),
                    correlation=correlation,
                    sign=1 if correlation >= 0 else -1
                )
    
    if len(G.edges()) == 0:
        logger.warning(f"Nenhuma correlação acima do limiar de {min_correlation}")
        return ""
    
    # Criar figura
    plt.figure(figsize=(12, 10))
    
    # Posicionamento dos nós
    if layout == 'spring':
        pos = nx.spring_layout(G, k=1.5/np.sqrt(len(G.nodes())))
    elif layout == 'circular':
        pos = nx.circular_layout(G)
    elif layout == 'kamada_kawai':
        pos = nx.kamada_kawai_layout(G)
    else:
        pos = nx.spring_layout(G)
    
    # Extrair pesos para espessura das arestas
    weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    signs = [G[u][v]['sign'] for u, v in G.edges()]
    
    # Desenhar o grafo
    # Arestas positivas em azul
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] > 0]
    positive_weights = [G[u][v]['weight'] * 5 for u, v in positive_edges]
    
    # Arestas negativas em vermelho
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] < 0]
    negative_weights = [G[u][v]['weight'] * 5 for u, v in negative_edges]
    
    # Desenhar arestas
    # Desenhar arestas positivas
    for i, (u, v) in enumerate(positive_edges):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=positive_weights[i],
            edge_color='blue',
            alpha=0.6
        )
    
    # Desenhar arestas negativas
    for i, (u, v) in enumerate(negative_edges):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            width=negative_weights[i],
            edge_color='red',
            alpha=0.6,
            style='dashed'
        )
    
    # Desenhar nós
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=800, 
        node_color='lightgray',
        edgecolors='black',
        alpha=0.9
    )
    
    # Desenhar rótulos
    nx.draw_networkx_labels(
        G, pos, 
        font_size=10,
        font_family='sans-serif'
    )
    
    # Título
    title_round = f"Round: {round_id}" if round_id else "Média entre Rounds"
    plt.title(f"Rede de Correlações - {metric.replace('_', ' ').title()}\nFase: {phase}, {title_round}")
    
    # Legenda
    plt.figtext(0.01, 0.01, f"Correlação mínima: {min_correlation}", ha="left", fontsize=8)
    plt.figtext(0.01, 0.03, "Azul: correlação positiva, Vermelho: correlação negativa", ha="left", fontsize=8)
    plt.figtext(0.01, 0.05, "Espessura proporcional à magnitude da correlação", ha="left", fontsize=8)
    
    # Remover eixos
    plt.axis('off')
    
    # Salvar figura
    safe_metric = metric.replace(' ', '_')
    safe_phase = phase.replace(' ', '_')
    round_suffix = f"_{round_id}" if round_id else "_avg"
    filename = f"{filename_prefix}correlation_network_{safe_metric}_{safe_phase}{round_suffix}.png"
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Gráfico de rede de correlação salvo em: {output_path}")
    
    return output_path

def plot_correlation_stability(
    correlation_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    output_dir: str,
    metric: Optional[str] = None,
    phase: Optional[str] = None,
    filename_prefix: str = ''
) -> str:
    """
    Gera visualizações de estabilidade das correlações entre rounds.
    
    Args:
        correlation_df: DataFrame com dados de correlação intra-fase
        stability_df: DataFrame com métricas de estabilidade das correlações
                     (output de analyze_correlation_stability)
        output_dir: Diretório para salvar os gráficos
        metric: Métrica específica a visualizar (se None, gera para todas)
        phase: Fase específica a visualizar (se None, gera para todas)
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o último arquivo gerado
    """
    if correlation_df.empty or stability_df.empty:
        logger.warning("DataFrames vazios. Não é possível gerar visualizações de estabilidade.")
        return ""
    
    # Verificar colunas necessárias no DataFrame de estabilidade
    required_cols = ['metric_name', 'experimental_phase', 'tenant_pair', 'cv', 'consistency_score']
    missing_cols = [col for col in required_cols if col not in stability_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes no DataFrame de estabilidade: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica e fase se especificados
    filtered_stability = stability_df.copy()
    
    if metric:
        filtered_stability = filtered_stability[filtered_stability['metric_name'] == metric]
        if filtered_stability.empty:
            logger.warning(f"Nenhum dado de estabilidade para a métrica '{metric}'.")
            return ""
    
    if phase:
        filtered_stability = filtered_stability[filtered_stability['experimental_phase'] == phase]
        if filtered_stability.empty:
            logger.warning(f"Nenhum dado de estabilidade para a fase '{phase}'.")
            return ""
    
    # Determinar métricas e fases exclusivas
    metrics_to_plot = [metric] if metric else filtered_stability['metric_name'].unique()
    phases_to_plot = [phase] if phase else filtered_stability['experimental_phase'].unique()
    
    output_paths = []
    
    for current_metric in metrics_to_plot:
        for current_phase in phases_to_plot:
            # Filtrar dados para métrica e fase atuais
            curr_stability = filtered_stability[
                (filtered_stability['metric_name'] == current_metric) & 
                (filtered_stability['experimental_phase'] == current_phase)
            ]
            
            if curr_stability.empty:
                continue
            
            # Filtrar dados de correlação correspondentes
            curr_correlation = correlation_df[
                (correlation_df['metric_name'] == current_metric) & 
                (correlation_df['experimental_phase'] == current_phase)
            ]
            
            if curr_correlation.empty:
                continue
            
            # Criar gráfico de estabilidade
            plt.figure(figsize=(12, 8))
            
            # Ordenar por CV para melhor visualização
            curr_stability = curr_stability.sort_values('cv')
            
            # Criar subplot para CV
            plt.subplot(2, 1, 1)
            plt.bar(
                range(len(curr_stability)), 
                curr_stability['cv'],
                alpha=0.7,
                color=['red' if x > 0.5 else 'orange' if x > 0.3 else 'green' for x in curr_stability['cv']]
            )
            plt.xticks(
                range(len(curr_stability)),
                [str(pair) for pair in curr_stability['tenant_pair'].tolist()],
                rotation=90
            )
            plt.ylabel('Coeficiente de Variação')
            plt.title(f'Estabilidade das Correlações - {current_metric.replace("_", " ").title()}\nFase: {current_phase}')
            plt.grid(axis='y', alpha=0.3)
            
            # Criar subplot para boxplot de correlações por par de tenant
            plt.subplot(2, 1, 2)
            
            # Preparar dados para boxplot
            boxplot_data = []
            labels = []
            
            for tenant_pair in curr_stability['tenant_pair']:
                pair_data = curr_correlation[curr_correlation['tenant_pair'] == tenant_pair]['correlation'].values
                if len(pair_data) > 0:
                    boxplot_data.append(pair_data)
                    labels.append(tenant_pair)
            
            # Criar boxplot
            if boxplot_data:
                bp = plt.boxplot(
                    boxplot_data,
                    patch_artist=True,
                    vert=True,
                    showfliers=True
                )
                
                # Colorir boxplots de acordo com média
                for i, box in enumerate(bp['boxes']):
                    # Cor baseada na média da correlação
                    median_val = np.median(boxplot_data[i])
                    if median_val >= 0:
                        box.set_facecolor('skyblue')
                    else:
                        box.set_facecolor('lightcoral')
            
            plt.xticks(rotation=90)
            plt.ylabel('Correlação')
            plt.grid(axis='y', alpha=0.3)
            plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            
            # Ajustar layout
            plt.tight_layout()
            
            # Salvar figura
            safe_metric = current_metric.replace(' ', '_')
            safe_phase = current_phase.replace(' ', '_')
            filename = f"{filename_prefix}correlation_stability_{safe_metric}_{safe_phase}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Gráfico de estabilidade de correlação salvo em: {output_path}")
            output_paths.append(output_path)
    
    # Se não foi gerado nenhum gráfico, retorna string vazia
    if not output_paths:
        logger.warning("Nenhum gráfico de estabilidade de correlação foi gerado.")
        return ""
    
    # Retorna o caminho do último arquivo salvo
    return output_paths[-1]
