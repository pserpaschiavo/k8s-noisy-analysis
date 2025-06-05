"""
Implementação de melhorias na visualização de grafos de causalidade:
1. Correção do problema onde nós ficam escondidos atrás de arestas
2. Implementação de visualização consolidada multi-métrica
3. Melhorias na legibilidade e interpretabilidade dos grafos
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx
import logging

logger = logging.getLogger(__name__)

def plot_improved_causality_graph(
    causality_matrix, 
    out_path, 
    threshold=0.05, 
    directed=True, 
    metric='', 
    metric_color=''
):
    """
    Versão melhorada da função plot_causality_graph para melhor visibilidade dos nós.
    """
    if causality_matrix.empty:
        logger.warning("Matriz de causalidade vazia. Grafo não gerado.")
        return None
        
    # Cria grafo direcionado ou não
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
        
    tenants = causality_matrix.index.tolist()
    G.add_nodes_from(tenants)
    
    # Listas para armazenar informações sobre as arestas
    edge_labels = {}
    edges = []
    edge_weights = []
    edge_colors = []
    
    # Paleta de cores para métricas
    metric_palette = {
        'cpu_usage': 'tab:blue',
        'memory_usage': 'tab:orange',
        'disk_io': 'tab:green',
        'network_io': 'tab:red',
        'network_traffic': 'tab:purple'
    }
    
    # Determina a cor a ser usada
    color = metric_color if metric_color else metric_palette.get(metric, 'tab:blue')
    
    # Adiciona arestas baseadas no threshold
    for src in tenants:
        for tgt in tenants:
            if src != tgt:
                val = causality_matrix.at[src, tgt]
                if not pd.isna(val) and float(val) < threshold:
                    weight = 1 - float(val)
                    G.add_edge(src, tgt, weight=weight)
                    edges.append((src, tgt))
                    edge_weights.append(weight * 6 + 1)
                    edge_colors.append(color)
                    edge_labels[(src, tgt)] = f"{weight:.2f}"
                    
    # Posicionamento circular dos nós
    sorted_tenants = sorted(tenants)
    pos = {}
    
    if len(sorted_tenants) <= 1:
        pos[sorted_tenants[0]] = np.array([0.0, 0.0])
    else:
        angles = np.linspace(0, 2 * np.pi, len(sorted_tenants), endpoint=False)
        radius = 0.8
        for tenant, angle in zip(sorted_tenants, angles):
            pos[tenant] = np.array([radius * np.cos(angle), radius * np.sin(angle)])
    
    # Adiciona jitter para evitar sobreposições perfeitas
    for node in pos:
        pos[node] = pos[node] + np.random.normal(0, 0.02, size=2)
    
    # Cria figura
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.set_facecolor('#f8f8f8')
    
    # Desenha as arestas (primeiro, para ficarem atrás)
    drawn_edges = []
    for idx, (edge, w, c) in enumerate(zip(edges, edge_weights, edge_colors)):
        drawn_edges.append(
            nx.draw_networkx_edges(
                G, pos, edgelist=[edge],
                arrowstyle='->' if directed else '-',
                arrows=directed,
                width=w,
                edge_color=c,
                alpha=0.75,
                connectionstyle='arc3,rad=0.15',
                min_source_margin=20,
                min_target_margin=20
            )
        )
    
    # Desenha os nós (por cima das arestas)
    nx.draw_networkx_nodes(
        G, pos,
        node_color='skyblue',
        node_size=1600,
        edgecolors='darkblue',
        linewidths=2.0,
        alpha=1.0
    )
    
    # Adiciona os rótulos dos nós
    nx.draw_networkx_labels(
        G, pos,
        font_size=13,
        font_weight='bold',
        font_color='black'
    )
    
    # Adiciona rótulos das arestas
    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_color='darkred',
        font_size=11,
        font_weight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7)
    )
    
    # Adiciona título
    plt.title(
        f'Gráfico de Causalidade ({"Direcionado" if directed else "Não-direcionado"})\n'
        f'Métrica: {metric if metric else "?"} | Arestas: p < {threshold:.2g}',
        fontsize=14,
        fontweight='bold'
    )
    
    plt.axis('off')
    
    # Adiciona legenda
    legend_elements = [
        mlines.Line2D([0], [0], color=color, lw=3, label=f'{metric if metric else "Métrica"}')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    
    # Salva o gráfico
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120)
    plt.close()
    
    return out_path


def plot_consolidated_causality_graph(
    causality_matrices, 
    out_path, 
    threshold=0.05, 
    directed=True, 
    metric_palette=None,
    phase='',
    round_id='',
    title_prefix='Análise Multi-Métrica',
    figsize=(12, 12),
    node_color='skyblue',
    node_size=1600
):
    """
    Plota um grafo de causalidade consolidado que combina múltiplas métricas.
    Cada métrica é representada com uma cor diferente de aresta.
    """
    if not causality_matrices:
        logger.warning("Nenhuma matriz de causalidade fornecida para o grafo consolidado")
        return None
        
    # Paleta de cores padrão se não fornecida
    if not metric_palette:
        metric_palette = {
            'cpu_usage': 'tab:blue',
            'memory_usage': 'tab:orange',
            'disk_io': 'tab:green',
            'network_io': 'tab:red',
            'network_traffic': 'tab:purple',
            'memory_utilization': 'tab:orange',
            'cpu_utilization': 'tab:blue'
        }
    
    # Detectar o tipo de matriz (p-valor ou TE) para cada métrica
    metric_modes = {}
    for metric, mat in causality_matrices.items():
        if mat.isnull().all().all():
            metric_modes[metric] = 'unknown'  # Matriz vazia ou inválida
        elif (mat.max().max() <= 1.0) and (mat.min().min() >= 0.0):
            # Parece ser p-valor (Granger) - valores baixos indicam causalidade
            metric_modes[metric] = 'p'
        else:
            # Parece ser Transfer Entropy - valores altos indicam causalidade
            metric_modes[metric] = 'TE'
            
    # Determinar o modo predominante para interpretação do threshold
    threshold_mode = 'p' if 'p' in metric_modes.values() else 'TE'
    
    # Unir todos os nós de todas as matrizes
    all_tenants = set()
    for mat in causality_matrices.values():
        all_tenants.update(mat.index.tolist())
    tenants = sorted(all_tenants)
    
    # Criar grafo
    G = nx.DiGraph() if directed else nx.Graph()
    G.add_nodes_from(tenants)
    
    # Estruturas para armazenar atributos das arestas
    edge_labels = {}     # Rótulos para cada aresta
    edge_colors = []     # Cor de cada aresta
    edge_widths = []     # Largura de cada aresta
    edge_list = []       # Lista de arestas
    edge_metrics = []    # Métrica associada a cada aresta
    
    # Processar cada matriz de causalidade
    for metric, mat in causality_matrices.items():
        color = metric_palette.get(metric, 'tab:blue')
        mode = metric_modes.get(metric, threshold_mode)
        
        for src in tenants:
            for tgt in tenants:
                if src != tgt and src in mat.index and tgt in mat.columns:
                    val = mat.at[src, tgt]
                    
                    # Verificar significância conforme o tipo de matriz
                    if mode == 'p':
                        # Para p-valores (Granger), menor é mais significativo
                        significant = (not pd.isna(val)) and (float(val) < threshold)
                        weight = 1 - float(val) if significant else 0
                    else:
                        # Para TE, maior é mais significativo
                        significant = (not pd.isna(val)) and (float(val) > threshold)
                        weight = float(val) if significant else 0
                        
                    if significant:
                        G.add_edge(src, tgt)
                        edge_list.append((src, tgt))
                        edge_colors.append(color)
                        edge_widths.append(weight * 5 + 1)
                        edge_labels[(src, tgt)] = f"{weight:.2f}"
                        edge_metrics.append(metric)
    
    # Verificar se alguma aresta foi encontrada
    if not edge_list:
        logger.warning(f"Nenhuma aresta significativa encontrada com threshold={threshold}")
        return None
        
    # Layout circular para melhor visualização
    pos = nx.circular_layout(G, scale=1)
    
    # Adicionar pequeno jitter para evitar sobreposição perfeita
    pos_dict = dict(pos)
    for node in pos_dict:
        pos_dict[node] = pos_dict[node] + np.random.normal(0, 0.015, size=2)
    pos = pos_dict
    
    # Criar figura
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_facecolor('#f8f8f8')
    
    # Desenhar arestas primeiro (para ficarem atrás dos nós)
    for (edge, color, width, metric) in zip(edge_list, edge_colors, edge_widths, edge_metrics):
        nx.draw_networkx_edges(
            G, pos, 
            edgelist=[edge],
            arrowstyle='-|>' if directed else '-',
            arrows=directed,
            width=width,
            edge_color=color,
            alpha=0.85,
            connectionstyle='arc3,rad=0.18',
            min_source_margin=18,
            min_target_margin=18,
            arrowsize=20
            # NetworkX não suporta zorder
        )
    
    # Desenhar nós sobre as arestas
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_color, 
        node_size=node_size,
        edgecolors='darkblue', 
        linewidths=1.8,
        alpha=1.0
        # NetworkX não suporta zorder
    )
    
    # Adicionar rótulos dos nós
    nx.draw_networkx_labels(
        G, pos, 
        font_size=13, 
        font_weight='bold', 
        font_color='black'
        # NetworkX não suporta zorder
    )
    
    # Adicionar rótulos das arestas
    nx.draw_networkx_edge_labels(
        G, pos, 
        edge_labels=edge_labels, 
        font_color='darkred', 
        font_size=11,
        font_weight='bold',
        bbox=dict(
            boxstyle="round,pad=0.3", 
            fc="white", 
            ec="gray", 
            alpha=0.85
        )
        # NetworkX não suporta zorder
    )
    
    # Adicionar legenda para cada métrica
    legend_elements = [
        mlines.Line2D([0], [0], color=metric_palette.get(m, 'tab:blue'), lw=3, label=m) 
        for m in causality_matrices.keys()
    ]
    plt.legend(handles=legend_elements, loc='lower left', fontsize=11)
    
    # Texto de legenda explicativo baseado no tipo predominante
    if threshold_mode == 'p':
        legend_str = f'Arestas: p-valor < {threshold:.2g}'
    else:
        legend_str = f'Arestas: TE > {threshold:.2g}'
    
    # Título contextualizado
    context_str = ''
    if phase:
        context_str += phase
    if round_id:
        if context_str:
            context_str += f', {round_id}'
        else:
            context_str += round_id
            
    title = f'{title_prefix}'
    if context_str:
        title += f' ({context_str})'
    title += f'\n{legend_str}'
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Salvar figura
    plt.savefig(out_path, dpi=150)
    plt.close()
    
    return out_path
