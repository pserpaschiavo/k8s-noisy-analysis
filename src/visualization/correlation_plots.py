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
    metric: Optional[str] = None,
    phase: Optional[str] = None,
    round_id: Optional[str] = None,
    min_abs_correlation: float = 0.5,
    layout: str = 'spring',
    node_size_factor: float = 1000.0,
    edge_width_factor: float = 2.0,
    filename_prefix: str = '',
    max_nodes: int = 50,          # Novo: limitar número de nós para otimização
    color_scheme: str = 'viridis', # Novo: esquema de cores personalizado
    edge_alpha: float = 0.7,      # Novo: controle de transparência
    filter_by_degree: int = 0,    # Novo: filtrar nós por grau mínimo
    highlight_top_correlations: bool = False,  # Novo: destacar correlações mais fortes
    community_detection: bool = False,  # Novo: detectar e colorir comunidades
    include_labels: bool = True   # Novo: opção para incluir ou não labels
) -> str:
    """
    Gera grafos de rede de correlação entre tenants para uma fase específica,
    com opções avançadas de estética e filtragem.
    
    Args:
        correlation_df: DataFrame com dados de correlação intra-fase
                       (colunas: round_id, metric_name, experimental_phase, tenant1, tenant2, correlation, method, sample_size)
        output_dir: Diretório para salvar os gráficos
        metric: Métrica específica a visualizar (se None, gera para todas)
        phase: Fase específica a visualizar (se None, gera para todas)
        round_id: Round específico a visualizar (se None, gera para todos)
        min_abs_correlation: Valor absoluto mínimo de correlação para incluir uma aresta
        layout: Algoritmo de layout do grafo ('spring', 'circular', 'kamada_kawai', etc.)
        node_size_factor: Fator para ajustar o tamanho dos nós
        edge_width_factor: Fator para ajustar a largura das arestas
        filename_prefix: Prefixo para o nome do arquivo
        max_nodes: Limite máximo de nós para incluir no grafo (otimização)
        color_scheme: Esquema de cores para nós ('viridis', 'plasma', 'inferno', 'magma', 'cividis')
        edge_alpha: Transparência das arestas (0.0 a 1.0)
        filter_by_degree: Filtrar nós com menos conexões que este valor
        highlight_top_correlations: Se True, destaca as correlações mais fortes
        community_detection: Se True, aplica detecção de comunidades e colorização
        include_labels: Se True, inclui rótulos nos nós
        
    Returns:
        str: Caminho para o último arquivo gerado
    """
    if correlation_df.empty:
        logger.warning("DataFrame de correlações vazio. Não é possível gerar grafo de rede.")
        return ""
    
    # Verificar se as colunas necessárias existem
    required_cols = ['round_id', 'metric_name', 'experimental_phase', 'tenant1', 'tenant2', 'correlation']
    missing_cols = [col for col in required_cols if col not in correlation_df.columns]
    if missing_cols:
        # Verificar se estamos usando 'tenant_pair' em vez de 'tenant1'/'tenant2'
        if 'tenant_pair' in correlation_df.columns and 'tenant1' in missing_cols and 'tenant2' in missing_cols:
            # Extrair tenant1 e tenant2 do campo tenant_pair
            correlation_df[['tenant1', 'tenant2']] = correlation_df['tenant_pair'].str.split(':', expand=True)
        else:
            logger.error(f"Colunas necessárias ausentes: {missing_cols}")
            return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar dados
    filtered_df = correlation_df.copy()
    
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
    
    if phase:
        filtered_df = filtered_df[filtered_df['experimental_phase'] == phase]
    
    if round_id:
        filtered_df = filtered_df[filtered_df['round_id'] == round_id]
    
    if filtered_df.empty:
        logger.warning("Sem dados após filtragem. Não é possível gerar grafo de rede.")
        return ""
    
    # Gerar gráficos para cada combinação de round, métrica e fase
    output_paths = []
    
    # Determinar as combinações únicas para gerar os gráficos
    unique_combinations = []
    
    if round_id and metric and phase:
        # Caso específico: uma única combinação
        unique_combinations = [(round_id, metric, phase)]
    else:
        # Múltiplas combinações
        for r in filtered_df['round_id'].unique() if not round_id else [round_id]:
            for m in filtered_df['metric_name'].unique() if not metric else [metric]:
                for p in filtered_df['experimental_phase'].unique() if not phase else [phase]:
                    combo_df = filtered_df[
                        (filtered_df['round_id'] == r) &
                        (filtered_df['metric_name'] == m) &
                        (filtered_df['experimental_phase'] == p)
                    ]
                    if not combo_df.empty:
                        unique_combinations.append((r, m, p))
    
    for r, m, p in unique_combinations:
        # Filtrar para esta combinação
        combo_df = filtered_df[
            (filtered_df['round_id'] == r) &
            (filtered_df['metric_name'] == m) &
            (filtered_df['experimental_phase'] == p)
        ]
        
        # Filtrar por correlação mínima absoluta
        combo_df = combo_df[combo_df['correlation'].abs() >= min_abs_correlation]
        
        if combo_df.empty:
            logger.warning(f"Sem correlações significativas para round={r}, metric={m}, phase={p}")
            continue
        
        # Criar o grafo
        G = nx.Graph()
        
        # Adicionar nós (todos os tenants únicos)
        all_tenants = set(combo_df['tenant1'].unique()) | set(combo_df['tenant2'].unique())
        for tenant in all_tenants:
            G.add_node(tenant)
        
        # Adicionar arestas com peso baseado na correlação
        for _, row in combo_df.iterrows():
            # A largura da aresta é proporcional ao valor absoluto da correlação
            width = abs(row['correlation']) * edge_width_factor
            
            # A cor da aresta é baseada no sinal da correlação (positiva=azul, negativa=vermelha)
            color = 'blue' if row['correlation'] > 0 else 'red'
            
            # O estilo da linha é baseado na magnitude da correlação
            style = 'solid' if abs(row['correlation']) >= 0.7 else ('dashed' if abs(row['correlation']) >= 0.5 else 'dotted')
            
            G.add_edge(
                row['tenant1'],
                row['tenant2'],
                weight=row['correlation'],
                width=width,
                color=color,
                style=style
            )
        
        # Novo: Filtrar nós por grau (número de conexões)
        if filter_by_degree > 0:
            nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree < filter_by_degree]
            G.remove_nodes_from(nodes_to_remove)
            if len(G.nodes()) == 0:
                logger.warning(f"Sem nós restantes após filtragem por grau >= {filter_by_degree}")
                continue
                
        # Novo: Limitar o número de nós para otimização
        if max_nodes > 0 and len(G.nodes()) > max_nodes:
            # Manter apenas os nós com as conexões mais fortes (maior grau ponderado)
            weighted_degrees = {}
            for node in G.nodes():
                weighted_degree = sum(abs(G[node][neighbor]['weight']) for neighbor in G.neighbors(node))
                weighted_degrees[node] = weighted_degree
            
            # Ordenar nós por grau ponderado e manter apenas os top max_nodes
            top_nodes = sorted(weighted_degrees.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
            top_node_ids = [node for node, _ in top_nodes]
            
            # Criar subgrafo com apenas os nós top
            G = G.subgraph(top_node_ids).copy()
            logger.info(f"Limitando o grafo a {max_nodes} nós com maior peso para melhor visualização")
        
        # Determinar o layout do grafo
        if layout == 'spring':
            pos = nx.spring_layout(G, seed=42, k=0.3, iterations=50)  # Mais iterações para melhor layout
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'shell':
            pos = nx.shell_layout(G)
        elif layout == 'spectral':  # Novo: layout espectral
            pos = nx.spectral_layout(G)
        else:
            # Fallback para spring layout
            pos = nx.spring_layout(G, seed=42)
        
        # Novo: Detectar comunidades se solicitado
        node_colors = ['lightgray'] * len(G.nodes())  # Cor padrão
        if community_detection and len(G.nodes()) >= 3:
            try:
                import community as community_louvain
                from matplotlib.cm import get_cmap
                
                # Verificar se há arestas no grafo
                if len(G.edges()) == 0:
                    logger.warning("Sem arestas no grafo. Detecção de comunidades não é possível.")
                    # Usar cores padrão baseadas no grau
                    if color_scheme != 'viridis':
                        degrees = dict(G.degree())
                        max_degree = max(degrees.values()) if degrees else 1
                        min_degree = min(degrees.values()) if degrees else 0
                        cmap = get_cmap(color_scheme)
                        
                        # Normalizar graus entre 0 e 1 para usar com colormap
                        if max_degree > min_degree:
                            node_colors = [cmap((degrees[node] - min_degree) / (max_degree - min_degree)) 
                                          for node in G.nodes()]
                        else:
                            node_colors = [cmap(0.5) for _ in G.nodes()]
                else:
                    # Detectar comunidades usando o algoritmo de Louvain
                    # Ajuste para garantir que o grafo tenha pesos positivos para o algoritmo de Louvain
                    G_community = G.copy()
                    for u, v, data in G_community.edges(data=True):
                        # Usar valor absoluto dos pesos para o algoritmo de comunidade
                        G_community[u][v]['weight'] = abs(data.get('weight', 1.0))
                    
                    partition = community_louvain.best_partition(G_community)
                    communities = set(partition.values())
                    
                    # Mapear comunidades para cores
                    cmap = get_cmap(color_scheme, max(len(communities), 2))  # Garantir pelo menos 2 cores
                    node_colors = [cmap(partition[node]) for node in G.nodes()]
                    
                    logger.info(f"Detecção de comunidades: {len(communities)} comunidades identificadas")
            except ImportError:
                logger.warning("Módulo 'community' não encontrado. Instalação: pip install python-louvain")
                # Continua sem detecção de comunidades
            except Exception as e:
                logger.warning(f"Erro na detecção de comunidades: {e}")
                # Em caso de erro, usar cores padrão
                if color_scheme != 'viridis':
                    try:
                        from matplotlib.cm import get_cmap
                        degrees = dict(G.degree())
                        max_degree = max(degrees.values()) if degrees and degrees.values() else 1
                        min_degree = min(degrees.values()) if degrees and degrees.values() else 0
                        cmap = get_cmap(color_scheme)
                        
                        if max_degree > min_degree:
                            node_colors = [cmap((degrees[node] - min_degree) / (max_degree - min_degree)) 
                                          for node in G.nodes()]
                        else:
                            node_colors = [cmap(0.5) for _ in G.nodes()]
                    except Exception:
                        # Fallback para cores padrão
                        node_colors = ['lightgray'] * len(G.nodes())
        elif color_scheme != 'viridis':
            # Se não estiver usando detecção de comunidade mas tiver um esquema de cores personalizado
            try:
                from matplotlib.cm import get_cmap
                # Usar grau do nó para colorir
                degrees = dict(G.degree())
                max_degree = max(degrees.values()) if degrees else 1
                min_degree = min(degrees.values()) if degrees else 0
                cmap = get_cmap(color_scheme)
                
                # Normalizar graus entre 0 e 1 para usar com colormap
                if max_degree > min_degree:
                    node_colors = [cmap((degrees[node] - min_degree) / (max_degree - min_degree)) 
                                  for node in G.nodes()]
                else:
                    node_colors = [cmap(0.5) for _ in G.nodes()]
            except Exception as e:
                logger.warning(f"Erro ao aplicar esquema de cores: {e}")
        
        # Novo: Calcular tamanho dos nós baseado no grau ponderado (não apenas contagem)
        weighted_degrees = {}
        for node in G.nodes():
            # Soma dos valores absolutos dos pesos (correlações)
            weighted_degree = sum(abs(G[node][neighbor]['weight']) for neighbor in G.neighbors(node))
            weighted_degrees[node] = weighted_degree + 1  # +1 para evitar tamanho zero
        
        # Normalizar para o intervalo desejado
        max_weighted_degree = max(weighted_degrees.values()) if weighted_degrees else 1
        min_weighted_degree = min(weighted_degrees.values()) if weighted_degrees else 0
        
        # Escalar entre 100 e node_size_factor
        node_sizes = []
        if max_weighted_degree > min_weighted_degree:
            for node in G.nodes():
                normalized = (weighted_degrees[node] - min_weighted_degree) / (max_weighted_degree - min_weighted_degree)
                size = 100 + normalized * node_size_factor
                node_sizes.append(size)
        else:
            node_sizes = [300] * len(G.nodes())
        
        # Criar figura
        plt.figure(figsize=(14, 12))
        
        # Desenhar arestas por estilo de linha
        for style in ['solid', 'dashed', 'dotted']:
            # Filtrar arestas por estilo
            style_edges = [(u, v) for u, v in G.edges() if G[u][v]['style'] == style]
            if style_edges:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=style_edges,
                    width=[G[u][v]['width'] for u, v in style_edges],
                    edge_color=[G[u][v]['color'] for u, v in style_edges],
                    style=style,
                    alpha=edge_alpha
                )
        
        # Novo: Destacar correlações mais fortes se solicitado
        if highlight_top_correlations:
            # Encontrar o top 10% ou pelo menos 5 arestas mais fortes
            edges_with_weights = [(u, v, abs(G[u][v]['weight'])) for u, v in G.edges()]
            sorted_edges = sorted(edges_with_weights, key=lambda x: x[2], reverse=True)
            
            top_count = max(5, int(len(sorted_edges) * 0.1))
            top_edges = sorted_edges[:top_count]
            
            # Desenhar as arestas mais importantes com destaque
            top_edge_list = [(u, v) for u, v, _ in top_edges]
            if top_edge_list:
                nx.draw_networkx_edges(
                    G, pos,
                    edgelist=top_edge_list,
                    width=[G[u][v]['width'] * 1.5 for u, v in top_edge_list],  # 50% mais largas
                    edge_color=[G[u][v]['color'] for u, v in top_edge_list],
                    style='solid',
                    alpha=1.0,  # Sem transparência
                    arrowsize=15  # Setas maiores se for um grafo direcionado
                )
        
        # Desenhar nós
        nx.draw_networkx_nodes(
            G, pos,
            node_size=node_sizes,
            node_color=node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5
        )
        
        # Adicionar rótulos se solicitado
        if include_labels:
            # Ajustar tamanho da fonte com base no número de nós
            font_size = max(6, min(10, 14 - 0.05 * len(G.nodes())))
            
            nx.draw_networkx_labels(G, pos, font_size=font_size, font_weight='bold')
        
        # Adicionar legenda para correlações positivas e negativas
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', linewidth=2, label='Correlação Positiva'),
            Line2D([0], [0], color='red', linewidth=2, label='Correlação Negativa'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='solid', label='Forte (|r| ≥ 0.7)'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='dashed', label='Média (0.5 ≤ |r| < 0.7)'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='dotted', label='Fraca (|r| < 0.5)')
        ]
        
        # Adicionar informações sobre filtragem na legenda
        if filter_by_degree > 0:
            legend_elements.append(
                Line2D([0], [0], color='white', label=f'Filtro: min. {filter_by_degree} conexões')
            )
            
        if max_nodes > 0 and len(all_tenants) > max_nodes:
            legend_elements.append(
                Line2D([0], [0], color='white', label=f'Mostrando top {max_nodes} de {len(all_tenants)} tenants')
            )
            
        plt.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Título com informações de filtragem
        title = f"Rede de Correlação entre Tenants\n{m.replace('_', ' ').title()} - Fase: {p} - Round: {r}"
        title += f"\nCorrelação mínima: |r| ≥ {min_abs_correlation}"
        
        plt.title(title, fontsize=12)
        plt.axis('off')  # Desativar eixos
        
        # Adicionar estatísticas do grafo
        plt.figtext(0.02, 0.02, 
                  f"Estatísticas: {len(G.nodes())} nós, {len(G.edges())} arestas\n"
                  f"Densidade: {nx.density(G):.3f}", 
                  ha="left", fontsize=8)
        
        plt.tight_layout()
        
        # Salvar figura
        safe_metric = m.replace(' ', '_')
        safe_phase = p.replace(' ', '_')
        safe_round = r.replace(' ', '_')
        filename = f"{filename_prefix}correlation_network_{safe_metric}_{safe_phase}_{safe_round}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Grafo de rede de correlação avançado salvo em: {output_path}")
        output_paths.append(output_path)
    
    # Verificar se geramos algum gráfico
    if not output_paths:
        logger.warning("Nenhum grafo de rede de correlação foi gerado.")
        return ""
    
    # Retorna o caminho para o último arquivo gerado
    return output_paths[-1]

def plot_correlation_stability(
    stability_df: pd.DataFrame,
    output_dir: str,
    metric: Optional[str] = None,
    phase: Optional[str] = None,
    min_correlation: float = 0.0,
    filename_prefix: str = ''
) -> str:
    """
    Gera gráficos para visualizar a estabilidade das correlações entre rounds.
    
    Args:
        stability_df: DataFrame com dados de estabilidade de correlação
                     (colunas: metric_name, experimental_phase, tenant_pair, mean_correlation, 
                      std_correlation, cv_correlation, correlation_stability, etc.)
        output_dir: Diretório para salvar os gráficos
        metric: Métrica específica a visualizar (se None, gera para todas)
        phase: Fase específica a visualizar (se None, gera para todas)
        min_correlation: Valor absoluto mínimo de correlação média para incluir no gráfico
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o último arquivo gerado
    """
    if stability_df.empty:
        logger.warning("DataFrame de estabilidade vazio. Não é possível gerar gráficos de estabilidade.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['metric_name', 'experimental_phase', 'tenant_pair', 
                    'mean_correlation', 'std_correlation', 'cv_correlation']
    missing_cols = [col for col in required_cols if col not in stability_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar dados
    filtered_df = stability_df.copy()
    
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
    
    if phase:
        filtered_df = filtered_df[filtered_df['experimental_phase'] == phase]
    
    # Filtrar por correlação mínima (em valor absoluto)
    filtered_df = filtered_df[filtered_df['mean_correlation'].abs() >= min_correlation]
    
    if filtered_df.empty:
        logger.warning("Sem dados após filtragem. Não é possível gerar gráficos de estabilidade.")
        return ""
    
    # Gerar gráficos para cada combinação de métrica e fase
    output_paths = []
    
    # Determinar as combinações únicas para gerar os gráficos
    unique_combinations = []
    
    if metric and phase:
        # Caso específico: uma única combinação
        unique_combinations = [(metric, phase)]
    else:
        # Múltiplas combinações
        for m in filtered_df['metric_name'].unique() if not metric else [metric]:
            for p in filtered_df['experimental_phase'].unique() if not phase else [phase]:
                combo_df = filtered_df[
                    (filtered_df['metric_name'] == m) &
                    (filtered_df['experimental_phase'] == p)
                ]
                if not combo_df.empty:
                    unique_combinations.append((m, p))
    
    for m, p in unique_combinations:
        # Filtrar para esta combinação
        combo_df = filtered_df[
            (filtered_df['metric_name'] == m) &
            (filtered_df['experimental_phase'] == p)
        ]
        
        if combo_df.empty:
            continue
        
        # Ordenar por magnitude da correlação média
        combo_df['abs_mean_corr'] = combo_df['mean_correlation'].abs()
        combo_df = combo_df.sort_values('abs_mean_corr', ascending=False).drop(columns=['abs_mean_corr'])
        
        # Limitar a 20 pares para legibilidade
        if len(combo_df) > 20:
            combo_df = combo_df.head(20)
            logger.info(f"Limitando a 20 pares para métrica {m}, fase {p}")
        
        # Criar figura
        plt.figure(figsize=(14, 10))
        
        # Criar barplot com barras de erro
        ax = plt.gca()
        x = np.arange(len(combo_df))
        
        # Barras coloridas por sinal da correlação
        bar_colors = ['blue' if corr >= 0 else 'red' for corr in combo_df['mean_correlation']]
        
        # Plotar barras com erros
        bars = ax.bar(x, combo_df['mean_correlation'], 
               yerr=combo_df['std_correlation'],
               color=bar_colors, alpha=0.7, edgecolor='black',
               capsize=5)
        
        # Adicionar linha horizontal em y=0
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        # Rótulos e título
        ax.set_xlabel('Par de Tenants')
        ax.set_ylabel('Correlação Média ± DP')
        ax.set_title(f"Estabilidade de Correlação entre Rounds\n{m.replace('_', ' ').title()} - Fase: {p}")
        
        # Configurar eixo x
        ax.set_xticks(x)
        ax.set_xticklabels(combo_df['tenant_pair'], rotation=45, ha='right')
        
        # Adicionar valores de CV como texto
        for i, (bar, cv) in enumerate(zip(bars, combo_df['cv_correlation'])):
            height = bar.get_height()
            if np.isnan(cv):
                cv_text = "CV: N/A"
            else:
                cv_text = f"CV: {cv:.2f}"
                
            # Posicionar acima ou abaixo da barra dependendo do sinal
            y_pos = height + 0.05 if height >= 0 else height - 0.1
            va = 'bottom' if height >= 0 else 'top'
            
            ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                    cv_text, ha='center', va=va, fontsize=8)
        
        # Adicionar legenda
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='blue', lw=4, label='Correlação Positiva'),
            Line2D([0], [0], color='red', lw=4, label='Correlação Negativa'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Ajustar limites do eixo y para garantir espaço para os rótulos
        y_min, y_max = ax.get_ylim()
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        
        # Layout
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        
        # Salvar figura
        safe_metric = m.replace(' ', '_')
        safe_phase = p.replace(' ', '_')
        filename = f"{filename_prefix}correlation_stability_{safe_metric}_{safe_phase}.png"
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de estabilidade de correlação salvo em: {output_path}")
        output_paths.append(output_path)
    
    # Verificar se geramos algum gráfico
    if not output_paths:
        logger.warning("Nenhum gráfico de estabilidade de correlação foi gerado.")
        return ""
    
    # Retorna o caminho para o último arquivo gerado
    return output_paths[-1]
