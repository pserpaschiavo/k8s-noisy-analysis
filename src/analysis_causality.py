"""
Module: analysis_causality.py
Description: Causality analysis utilities for multi-tenant time series analysis, including graph visualization.

Este módulo implementa métodos para análise de causalidade entre séries temporais de diferentes tenants,
incluindo causalidade de Granger e Transfer Entropy (TE), além de visualizações em formato de grafo.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tools.sm_exceptions import MissingDataError

# Importação da biblioteca pyinform para Transfer Entropy mais robusta
try:
    from pyinform.transferentropy import transfer_entropy
    PYINFORM_AVAILABLE = True
except ImportError:
    PYINFORM_AVAILABLE = False
    logging.warning("pyinform não está instalado. Transfer Entropy usará implementação básica.")

plt.style.use('tableau-colorblind10')

def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', metric_color: str = ''):
    """
    Plots a causality graph from a causality matrix (e.g., Granger p-values or scores).
    Edges are drawn where causality_matrix.loc[src, tgt] < threshold.
    Edge width = intensity (1-p ou score), color = metric.
    """
    if causality_matrix.empty:
        return None
    if directed:
        G = nx.DiGraph()
    else:
        G = nx.Graph()
    tenants = causality_matrix.index.tolist()
    G.add_nodes_from(tenants)
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
    }
    color = metric_color if metric_color else metric_palette.get(metric, 'tab:blue')
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
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(7, 7))
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    # Desenha cada aresta individualmente para aplicar peso e cor
    for idx, (edge, w, c) in enumerate(zip(edges, edge_weights, edge_colors)):
        nx.draw_networkx_edges(G, pos, edgelist=[edge], arrowstyle='->' if directed else '-', arrows=directed, width=w, edge_color=c, alpha=0.8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=10)
    plt.title(f'Causality Graph ({"Directed" if directed else "Undirected"})\nMétrica: {metric if metric else "?"} | Edges: p < {threshold:.2g}')
    plt.axis('off')
    # Legenda customizada
    import matplotlib.lines as mlines
    legend_elements = [
        mlines.Line2D([0], [0], color=color, lw=3, label=f'{metric if metric else "Métrica"}')
    ]
    plt.legend(handles=legend_elements, loc='lower left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()
    return out_path

def _transfer_entropy(x, y, bins=8, k=1):
    """
    Calcula a Transfer Entropy (TE) de y para x (y→x) usando implementação otimizada.
    
    Args:
        x: Array 1D representando a série temporal destino (target)
        y: Array 1D representando a série temporal fonte (source)
        bins: Número de bins para discretização (default=8)
        k: Histórico da série destino a considerar (default=1)
        
    Returns:
        valor escalar de TE (y→x): quanto y ajuda a prever x além da história de x
    """
    if PYINFORM_AVAILABLE:
        try:
            # Usa a implementação mais robusta da biblioteca pyinform
            # Normalização e binning automático de séries
            # Converte para inteiros conforme requisitos do pyinform
            x_norm = (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)
            y_norm = (y - np.min(y)) / (np.max(y) - np.min(y) + 1e-8)
            
            x_bin = np.floor(x_norm * (bins-1)).astype(int)
            y_bin = np.floor(y_norm * (bins-1)).astype(int)
            
            # Calcula TE usando pyinform
            te_value = transfer_entropy(y_bin, x_bin, k=k, local=False)
            return te_value
        except Exception as e:
            logging.warning(f"Erro ao usar pyinform para TE: {e}. Usando implementação básica.")
    
    # Implementação básica fallback usando histogramas numpy
    # Discretiza as séries
    x_binned = np.digitize(x, np.histogram_bin_edges(x, bins=bins))
    y_binned = np.digitize(y, np.histogram_bin_edges(y, bins=bins))
    
    # Calcula as probabilidades conjuntas e condicionais
    px = np.histogram(x_binned[1:], bins=bins, density=True)[0]
    pxy = np.histogram2d(x_binned[:-1], x_binned[1:], bins=bins, density=True)[0]
    pxyy = np.histogramdd(np.stack([x_binned[:-1], y_binned[:-1], x_binned[1:]], axis=1), 
                         bins=(bins, bins, bins), density=True)[0]
    
    # Adiciona regularização para evitar log(0)
    pxyy = pxyy + 1e-12
    pxy = pxy + 1e-12
    px = px + 1e-12
    
    # TE(y→x) = sum p(x_{t+1}, x_t, y_t) * log [p(x_{t+1}|x_t, y_t) / p(x_{t+1}|x_t)]
    te = 0.0
    for i in range(bins):
        for j in range(bins):
            for k in range(bins):
                pxyz = pxyy[j, k, i]
                pxz = pxy[j, i]
                px_i = px[i]
                if pxyz > 0 and pxz > 0 and px_i > 0:
                    te += pxyz * np.log((pxyz / (np.sum(pxyy[j, k, :]) + 1e-12)) / 
                                       (pxz / (np.sum(pxy[j, :]) + 1e-12)))
    return te

class CausalityAnalyzer:
    """
    Classe responsável por cálculos de causalidade (ex: Granger, Transfer Entropy) entre tenants.
    """
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute_granger_matrix(self, metric: str, phase: str, round_id: str, maxlag: int = 5) -> pd.DataFrame:
        """
        Calcula a matriz de p-valores do teste de causalidade de Granger entre todos os tenants para uma métrica específica.
        
        Args:
            metric: Nome da métrica para análise
            phase: Fase experimental (ex: "1 - Baseline", "2 - Attack")
            round_id: ID do round (ex: "round-1")
            maxlag: Número máximo de lags para o teste de Granger
            
        Returns:
            DataFrame onde mat[i,j] é o menor p-valor de j causando i (considerando lags de 1 a maxlag)
        """
        subset = self.df[(self.df['metric_name'] == metric) & 
                        (self.df['experimental_phase'] == phase) & 
                        (self.df['round_id'] == round_id)]
        
        tenants = subset['tenant_id'].unique()
        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        
        # Transforma para formato amplo para análise de séries temporais
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Interpolação e preenchimento para lidar com valores ausentes
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        # Testa causalidade de Granger para cada par de tenants
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                    
                # Obtém as séries relevantes
                try:
                    # Seleciona apenas dados com sobreposição de timestamps
                    data = pd.concat([wide[target], wide[source]], axis=1)
                    data = data.dropna()
                    
                    if len(data) < maxlag + 2:
                        continue
                        
                    # Realiza teste de Granger
                    try:
                        test_results = grangercausalitytests(
                            data, 
                            maxlag=maxlag, 
                            verbose=False
                        )
                        
                        # Extrai o menor p-valor entre todos os lags
                        p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
                        min_p_value = min(p_values) if p_values else np.nan
                        
                        # Armazena o resultado na matriz
                        mat.loc[target, source] = min_p_value
                    except MissingDataError:
                        continue
                    except Exception as e:
                        logging.warning(f"Erro ao calcular Granger para {source}->{target}: {str(e)}")
                        continue
                except KeyError:
                    continue
                    
        return mat

    def compute_transfer_entropy_matrix(self, metric: str, phase: str, round_id: str, bins: int = 8, k: int = 1) -> pd.DataFrame:
        """
        Calcula a matriz de Transfer Entropy (TE) entre todos os tenants para uma métrica específica.
        
        Args:
            metric: Nome da métrica para análise
            phase: Fase experimental (ex: "1 - Baseline", "2 - Attack")
            round_id: ID do round (ex: "round-1") 
            bins: Número de bins para discretização das séries contínuas
            k: Histórico da série destino a considerar
            
        Returns:
            DataFrame onde mat[i,j] é o valor da Transfer Entropy de j para i (j→i)
            Valores mais altos indicam maior transferência de informação
        """
        # Log do início do cálculo
        logging.info(f"Calculando matriz de Transfer Entropy para {metric} em {phase} ({round_id})")
        
        # Filtra os dados relevantes
        subset = self.df[(self.df['metric_name'] == metric) & 
                        (self.df['experimental_phase'] == phase) & 
                        (self.df['round_id'] == round_id)]
        
        tenants = subset['tenant_id'].unique()
        mat = pd.DataFrame(np.nan, index=tenants, columns=tenants)
        
        # Transforma para formato amplo para análise
        wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Interpola e preenche NaNs para alinhar todos os tenants
        wide = wide.sort_index().interpolate(method='time').ffill().bfill()
        
        # Calcula TE para cada par de tenants
        for target in tenants:
            for source in tenants:
                if target == source:
                    continue
                
                try:
                    # Obtém séries temporais dos tenants
                    target_series = wide[target]
                    source_series = wide[source]
                    
                    # Alinha os índices para garantir correspondência temporal
                    common_idx = target_series.index.intersection(source_series.index)
                    target_values = target_series.loc[common_idx].values
                    source_values = source_series.loc[common_idx].values
                    
                    # Verifica se há pontos suficientes para cálculo significativo
                    if len(target_values) > 10:
                        # Calcula TE e armazena na matriz
                        te_value = _transfer_entropy(target_values, source_values, bins=bins, k=k)
                        mat.loc[target, source] = te_value
                    else:
                        logging.warning(f"Série temporal insuficiente para par {source}->{target}: {len(target_values)} pontos")
                        
                except Exception as e:
                    logging.error(f"Erro ao calcular Transfer Entropy para {source}->{target}: {str(e)}")
        
        return mat

class CausalityVisualizer:
    """
    Classe responsável por visualizações de causalidade (ex: grafos, heatmaps).
    """
    @staticmethod
    def plot_causality_graph(causality_matrix: pd.DataFrame, out_path: str, threshold: float = 0.05, directed: bool = True, metric: str = '', metric_color: str = ''):
        return plot_causality_graph(causality_matrix, out_path, threshold, directed, metric, metric_color)

    @staticmethod
    def plot_causality_graph_multi(
        causality_matrices: dict,  # {metric: matrix}
        out_path: str,
        threshold: float = 0.05,
        directed: bool = True,
        metric_palette: dict = {},
        threshold_mode: str = ''  # 'p' para Granger, 'TE' para Transfer Entropy
    ):
        """
        Plota um grafo de causalidade comparando múltiplas métricas.
        Cada métrica é uma cor de aresta diferente.
        threshold_mode: 'p' (arestas p < threshold), 'TE' (arestas TE > threshold), ou '' (auto).
        """
        if not causality_matrices:
            return None
        # Paleta padrão
        if not metric_palette:
            metric_palette = {
                'cpu_usage': 'tab:blue',
                'memory_usage': 'tab:orange',
                'disk_io': 'tab:green',
                'network_io': 'tab:red',
            }
        # --- Lógica aprimorada para threshold_mode e legenda ---
        # Detecta para cada métrica se é p-valor (Granger real) ou TE
        metric_modes = {}
        for metric, mat in causality_matrices.items():
            if mat.isnull().all().all():
                metric_modes[metric] = 'unknown'
            elif (mat.max().max() <= 1.0) and (mat.min().min() >= 0.0):
                # Pode ser p-valor (Granger real ou placeholder)
                # Se não for placeholder (tudo NaN), considera p-valor
                metric_modes[metric] = 'p'
            else:
                metric_modes[metric] = 'TE'
        # Prioriza p-valor se houver pelo menos uma métrica real de Granger
        if threshold_mode == '':
            if 'p' in metric_modes.values():
                threshold_mode = 'p'
            else:
                threshold_mode = 'TE'
        # Unir todos os nós
        all_tenants = set()
        for mat in causality_matrices.values():
            all_tenants.update(mat.index.tolist())
        tenants = sorted(all_tenants)
        G = nx.DiGraph() if directed else nx.Graph()
        G.add_nodes_from(tenants)
        edge_labels = {}
        edge_colors = []
        edge_widths = []
        edge_list = []
        edge_metrics = []
        for metric, mat in causality_matrices.items():
            color = metric_palette.get(metric, 'tab:blue')
            mode = metric_modes.get(metric, threshold_mode)
            for src in tenants:
                for tgt in tenants:
                    if src != tgt and src in mat.index and tgt in mat.columns:
                        val = mat.at[src, tgt]
                        if mode == 'p':
                            cond = (not pd.isna(val)) and (float(val) < threshold)
                        else:
                            cond = (not pd.isna(val)) and (float(val) > threshold)
                        if cond:
                            weight = 1 - float(val) if mode == 'p' else float(val)
                            G.add_edge(src, tgt)
                            edge_list.append((src, tgt))
                            edge_colors.append(color)
                            edge_widths.append(weight * 6 + 1)
                            edge_labels[(src, tgt)] = f"{weight:.2f}"
                            edge_metrics.append(metric)
        # Layout circular para garantir visibilidade
        pos = nx.circular_layout(G)
        plt.figure(figsize=(8, 8))
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200)
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
        # Desenha cada aresta individualmente, com offset e seta visível
        for (edge, color, width, metric) in zip(edge_list, edge_colors, edge_widths, edge_metrics):
            nx.draw_networkx_edges(
                G, pos, edgelist=[edge],
                arrowstyle='-|>' if directed else '-',
                arrows=directed,
                width=width,
                edge_color=color,
                alpha=0.8,
                connectionstyle='arc3,rad=0.18',
                min_source_margin=25, min_target_margin=25,
                arrowsize=28
            )
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=10)
        # Legenda para cada métrica
        import matplotlib.lines as mlines
        legend_elements = [mlines.Line2D([0], [0], color=metric_palette.get(m, 'tab:blue'), lw=3, label=m) for m in causality_matrices.keys()]
        plt.legend(handles=legend_elements, loc='lower left')
        # Legenda contextual automática
        if threshold_mode == 'p':
            legend_str = f'Arestas: p-valor < {threshold:.2g}'
        else:
            legend_str = f'Arestas: TE > {threshold:.2g}'
        plt.title(f'Causality Graph (Comparativo de Métricas) | {legend_str}')
        plt.axis('off')
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path)
        plt.close()
        return out_path

class CausalityModule:
    """
    Módulo de alto nível para análise de causalidade, integrando cálculo e visualização.
    """
    def __init__(self, df: pd.DataFrame):
        self.analyzer = CausalityAnalyzer(df)
        self.visualizer = CausalityVisualizer()

    def run_granger_and_plot(self, metric: str, phase: str, round_id: str, out_dir: str, maxlag: int = 5, threshold: float = 0.05):
        os.makedirs(out_dir, exist_ok=True)
        mat = self.analyzer.compute_granger_matrix(metric, phase, round_id, maxlag)
        out_path = os.path.join(out_dir, f"causality_graph_granger_{metric}_{phase}_{round_id}.png")
        # Paleta de cores igual à usada na função de plot
        metric_palette = {
            'cpu_usage': 'tab:blue',
            'memory_usage': 'tab:orange',
            'disk_io': 'tab:green',
            'network_io': 'tab:red',
        }
        color = metric_palette.get(metric, 'tab:blue')
        return self.visualizer.plot_causality_graph(mat, out_path, threshold, directed=True, metric=metric, metric_color=color)

    def run_transfer_entropy_and_plot(self, metric: str, phase: str, round_id: str, out_dir: str, bins: int = 8, threshold: float = 0.05):
        os.makedirs(out_dir, exist_ok=True)
        mat = self.analyzer.compute_transfer_entropy_matrix(metric, phase, round_id, bins)
        out_path = os.path.join(out_dir, f"causality_graph_te_{metric}_{phase}_{round_id}.png")
        # Paleta de cores igual à usada na função de plot
        metric_palette = {
            'cpu_usage': 'tab:blue',
            'memory_usage': 'tab:orange',
            'disk_io': 'tab:green',
            'network_io': 'tab:red',
        }
        color = metric_palette.get(metric, 'tab:blue')
        # Para TE, threshold destaca relações mais fortes (TE > threshold)
        def plot_te_graph(te_matrix, out_path, threshold, metric, color):
            if te_matrix.empty:
                return None
            G = nx.DiGraph()
            tenants = te_matrix.index.tolist()
            G.add_nodes_from(tenants)
            edge_labels = {}
            edges = []
            edge_weights = []
            edge_colors = []
            for src in tenants:
                for tgt in tenants:
                    if src != tgt:
                        val = te_matrix.at[src, tgt]
                        if not pd.isna(val) and float(val) > threshold:
                            weight = float(val)
                            G.add_edge(src, tgt, weight=weight)
                            edges.append((src, tgt))
                            edge_weights.append(weight * 6 + 1)
                            edge_colors.append(color)
                            edge_labels[(src, tgt)] = f"{weight:.2f}"
            pos = nx.spring_layout(G, seed=42)
            plt.figure(figsize=(7, 7))
            nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=1200)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            for idx, (edge, w, c) in enumerate(zip(edges, edge_weights, edge_colors)):
                nx.draw_networkx_edges(G, pos, edgelist=[edge], arrowstyle='->', arrows=True, width=w, edge_color=c, alpha=0.8)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=10)
            plt.title(f'Transfer Entropy Graph (TE > {threshold:.2g})\nMétrica: {metric}')
            plt.axis('off')
            import matplotlib.lines as mlines
            legend_elements = [
                mlines.Line2D([0], [0], color=color, lw=3, label=f'{metric}')
            ]
            plt.legend(handles=legend_elements, loc='lower left')
            plt.tight_layout()
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            plt.savefig(out_path)
            plt.close()
            return out_path
        return plot_te_graph(mat, out_path, threshold, metric, color)
