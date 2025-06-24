#!/usr/bin/env python3
"""
Module: analysis_multi_round.py
Description: Módulo para análise de experimentos com múltiplos rounds.

Este módulo implementa metodologias para análise de consistência entre rounds,
avaliação de robustez causal, análise de divergência comportamental e
agregação de consenso para experimentos com múltiplos rounds.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx  # Adicionando importação do NetworkX
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from datetime import datetime

from src.pipeline_stage import PipelineStage
from src.visualization.plots import generate_consolidated_boxplot, generate_consolidated_heatmap
from src.visualization.advanced_plots import generate_all_consolidated_timeseries

# Configuração de logging e estilo
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("analysis_multi_round")
plt.style.use('tableau-colorblind10')

class MultiRoundAnalysisStage(PipelineStage):
    """
    Estágio do pipeline para análise de experimentos com múltiplos rounds.
    Implementa as funcionalidades descritas na seção 3.6 do plano de trabalho.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            name="Multi-Round Analysis", 
            description="Análise de consistência e robustez entre múltiplos rounds de experimento"
        )
        self.output_dir = output_dir
        # Adicionando um logger específico para esta classe para melhor rastreabilidade
        self.logger = logging.getLogger(self.__class__.__name__)

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of the abstract method from PipelineStage.
        Bridges to the existing run method by extracting config from context.
        """
        # Extract config from context - pipeline stages should include their config
        config = context.get('config', {})
        return self.run(context, config)

    def run(self, context: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a análise multi-round, combinando consistência de métricas,
        robustez causal e análise de divergência.
        """
        self.logger.info("Iniciando a análise multi-round completa...")

        if 'error' in context:
            self.logger.error(f"Erro em estágio anterior: {context['error']}")
            return context

        df_long = context.get('df_long')
        if df_long is None:
            self.logger.warning("DataFrame principal (df_long) não encontrado no contexto.")
            context['error'] = "DataFrame principal não disponível para análise multi-round"
            return context

        rounds = config.get('selected_rounds', sorted(df_long['round_id'].unique()))
        if len(rounds) <= 1:
            self.logger.info("Apenas um round encontrado ou selecionado. Pulando análise multi-round.")
            context['multi_round_analysis'] = {
                'status': 'skipped',
                'reason': 'O dataset contém apenas um round de experimento ou apenas um foi selecionado.'
            }
            return context

        # Configuração do diretório de saída
        base_output_dir = self.output_dir or context.get('output_dir')
        if not base_output_dir:
            experiment_id = context.get('experiment_id', 'unknown_experiment')
            base_output_dir = os.path.join(os.getcwd(), 'outputs', experiment_id)
            self.logger.warning(f"Diretório de saída não especificado. Usando fallback: {base_output_dir}")

        output_dir = os.path.join(base_output_dir, 'multi_round_analysis')
        os.makedirs(output_dir, exist_ok=True)
        self.logger.info(f"Diretório de saída para análise multi-round: {output_dir}")
        
        # Atualiza o output_dir da instância para que outros métodos o utilizem
        self.output_dir = output_dir

        metrics = context.get('selected_metrics', sorted(df_long['metric_name'].unique()))
        tenants = context.get('selected_tenants', sorted(df_long['tenant_id'].unique()))
        
        df_filtered = df_long[df_long['round_id'].isin(rounds)]

        try:
            results = {}

            # 1. Análise de consistência de causalidade (Jaccard/Spearman)
            self.logger.info("Analisando a consistência da estrutura causal (Jaccard/Spearman)...")
            causality_data = self._load_causality_matrices(context, rounds)
            te_matrices_by_round = causality_data.get('te_matrices_by_round', {})
            granger_matrices_by_round = causality_data.get('granger_matrices_by_round', {})
            
            # Reformatar dados para o método de análise de consistência
            causality_matrices_reformatted = {}
            for r in rounds:
                if r in te_matrices_by_round or r in granger_matrices_by_round:
                     causality_matrices_reformatted[r] = {
                         'te_matrices': te_matrices_by_round.get(r, {}),
                         'granger_matrices': granger_matrices_by_round.get(r, {})
                     }

            if causality_matrices_reformatted:
                graph_consistency_results = self.analyze_causality_consistency(causality_matrices_reformatted)
                results['graph_consistency'] = graph_consistency_results
                self.generate_consistency_visualizations(graph_consistency_results)
            else:
                self.logger.warning("Nenhuma matriz de causalidade carregada. Pulando análise de consistência da estrutura causal.")
                results['graph_consistency'] = {}


            # 2. Análise de consistência de métricas (CV/Friedman)
            self.logger.info("Analisando a consistência dos valores das métricas (CV/Friedman)...")
            results['metric_consistency'] = analyze_round_consistency(
                df_long=df_filtered, metrics=metrics, tenants=tenants, output_dir=output_dir
            )

            # 3. Análise de robustez de causalidade (CV sobre TE)
            if te_matrices_by_round:
                self.logger.info("Analisando a robustez da força causal (CV sobre TE)...")
                causality_robustness = analyze_causality_robustness(
                    te_matrices_by_round=te_matrices_by_round,
                    granger_matrices_by_round=granger_matrices_by_round,
                    output_dir=output_dir
                )
                results['causality_robustness'] = causality_robustness

                if 'robust_causal_relationships' in causality_robustness:
                    robust_graphs = {}
                    for metric in causality_robustness.get('metrics_with_consistent_causality', []):
                        graph_path = generate_robust_causality_graph(
                            robust_relationships=causality_robustness['robust_causal_relationships'],
                            output_dir=output_dir, metric=metric
                        )
                        if graph_path:
                            robust_graphs[metric] = graph_path
                    if robust_graphs:
                        results['robust_causality_graphs'] = robust_graphs
            else:
                self.logger.warning("Matrizes de Transfer Entropy não disponíveis. Pulando análise de robustez causal.")

            # 4. Análise de divergência comportamental (KL-Divergence)
            self.logger.info("Analisando a divergência comportamental entre rounds (KL-Divergence)...")
            results['behavioral_divergence'] = analyze_behavioral_divergence(
                df_long=df_filtered, metrics=metrics, tenants=tenants, output_dir=output_dir
            )

            # 5. Agregação de consenso
            if te_matrices_by_round:
                self.logger.info("Agregando um consenso entre os rounds...")
                results['consensus'] = aggregate_round_consensus(
                    df_long=df_filtered,
                    te_matrices_by_round=te_matrices_by_round,
                    consistency_results=results.get('metric_consistency', {}),
                    output_dir=output_dir
                )
            else:
                 self.logger.warning("Matrizes de Transfer Entropy não disponíveis. Pulando agregação de consenso.")

            # 6. Visualizações consolidadas
            self.logger.info("Gerando visualizações consolidadas...")
            results['visualization_paths'] = generate_round_consistency_visualizations(
                df_long=df_filtered,
                consistency_results=results.get('metric_consistency', {}),
                causality_robustness=results.get('causality_robustness', {}),
                output_dir=output_dir
            )

            # 6.1 Time Series Consolidados (NOVA FUNCIONALIDADE v2.0)
            self.logger.info("Gerando time series consolidados para todas as métricas...")
            try:
                timeseries_paths = generate_all_consolidated_timeseries(
                    df_long=df_filtered,
                    output_dir=output_dir,
                    rounds=rounds,
                    tenants=tenants,
                    normalize_time=True,
                    add_confidence_bands=True,
                    add_phase_annotations=True
                )
                results['consolidated_timeseries'] = timeseries_paths
                self.logger.info(f"✅ Time series consolidados gerados: {len(timeseries_paths)} métricas")
            except Exception as e:
                self.logger.error(f"Erro ao gerar time series consolidados: {e}")
                results['consolidated_timeseries'] = {}

            # 7. Gerar relatório consolidado
            self.logger.info("Gerando relatório consolidado da análise multi-round...")
            self.generate_multi_round_report(results) # Passando todos os resultados

            context['multi_round_analysis'] = results
            context['multi_round_analysis_dir'] = output_dir
            self.logger.info(f"Análise multi-round concluída com sucesso. Resultados em {output_dir}")

        except Exception as e:
            self.logger.error(f"Erro fatal durante a análise multi-round: {str(e)}", exc_info=True)
            context['error'] = f"Erro na análise multi-round: {str(e)}"

        return context

    def analyze_causality_consistency(self, causality_matrices: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa a consistência das matrizes de causalidade entre os rounds.
        Calcula a similaridade de Jaccard para as relações causais (Granger)
        e a correlação de Spearman para a força da causalidade (TE).
        """
        self.logger.info("Analisando a consistência da causalidade entre os rounds...")
        
        consistency_results = {
            "granger_jaccard": pd.DataFrame(),
            "te_spearman": pd.DataFrame()
        }
        
        rounds = list(causality_matrices.keys())
        if len(rounds) < 2:
            self.logger.warning("São necessários pelo menos dois rounds para a análise de consistência.")
            return consistency_results

        # Extrai as matrizes de Granger e TE
        granger_matrices_by_round = {r: v.get('granger_matrices', {}) for r, v in causality_matrices.items()}
        te_matrices_by_round = {r: v.get('te_matrices', {}) for r, v in causality_matrices.items()}

        metrics = list(granger_matrices_by_round[rounds[0]].keys())
        
        jaccard_scores = []
        spearman_scores = []

        for metric in metrics:
            for i in range(len(rounds)):
                for j in range(i + 1, len(rounds)):
                    round1, round2 = rounds[i], rounds[j]
                    
                    # Consistência para Causalidade de Granger (Jaccard)
                    g1 = granger_matrices_by_round.get(round1, {}).get(metric)
                    g2 = granger_matrices_by_round.get(round2, {}).get(metric)
                    
                    if g1 is not None and g2 is not None:
                        # Binariza a matriz com base no p-valor < 0.05
                        g1_bin = (g1 < 0.05).values.flatten()
                        g2_bin = (g2 < 0.05).values.flatten()
                        
                        jaccard_sim = 1 - distance.jaccard(g1_bin, g2_bin)
                        jaccard_scores.append((metric, f"{round1}-{round2}", jaccard_sim))

                    # Consistência para Entropia de Transferência (Spearman)
                    te1 = te_matrices_by_round.get(round1, {}).get(metric)
                    te2 = te_matrices_by_round.get(round2, {}).get(metric)

                    if te1 is not None and te2 is not None:
                        rho, _ = stats.spearmanr(te1.values.flatten(), te2.values.flatten())
                        spearman_scores.append((metric, f"{round1}-{round2}", rho))

        if jaccard_scores:
            df_jaccard = pd.DataFrame(jaccard_scores, columns=['Metric', 'Round Pair', 'Jaccard Similarity'])
            consistency_results['granger_jaccard'] = df_jaccard.pivot(index='Metric', columns='Round Pair', values='Jaccard Similarity')

        if spearman_scores:
            df_spearman = pd.DataFrame(spearman_scores, columns=['Metric', 'Round Pair', 'Spearman Correlation'])
            consistency_results['te_spearman'] = df_spearman.pivot(index='Metric', columns='Round Pair', values='Spearman Correlation')
            
        self.logger.info("Análise de consistência de causalidade concluída.")
        self.logger.debug(f"Resultados de Jaccard (Granger):\n{consistency_results['granger_jaccard']}")
        self.logger.debug(f"Resultados de Spearman (TE):\n{consistency_results['te_spearman']}")

        return consistency_results

    def generate_consistency_visualizations(self, consistency_results: Dict[str, Any]):
        """
        Gera visualizações para os resultados da análise de consistência.
        Cria heatmaps para a similaridade de Jaccard (Granger) e correlação de Spearman (TE).
        """
        self.logger.info("Gerando visualizações de consistência...")
        
        if not self.output_dir:
            self.logger.warning("Diretório de saída não configurado. As visualizações não serão salvas.")
            return

        # Visualização para Jaccard (Granger)
        df_jaccard = consistency_results.get('granger_jaccard')
        if df_jaccard is not None and not df_jaccard.empty:
            output_path = generate_consolidated_heatmap(
                aggregated_matrix=df_jaccard,
                output_dir=self.output_dir,
                title="Consistência da Causalidade de Granger (Similaridade de Jaccard)",
                filename="granger_consistency_heatmap.png"
            )
            if output_path:
                self.logger.info(f"Heatmap de consistência de Granger salvo em: {output_path}")

        # Visualização para Spearman (TE)
        df_spearman = consistency_results.get('te_spearman')
        if df_spearman is not None and not df_spearman.empty:
            output_path = generate_consolidated_heatmap(
                aggregated_matrix=df_spearman,
                output_dir=self.output_dir,
                title="Consistência da Força Causal (Correlação de Spearman para TE)",
                filename="te_consistency_heatmap.png"
            )
            if output_path:
                self.logger.info(f"Heatmap de consistência de TE salvo em: {output_path}")

    def generate_multi_round_report(self, all_results: Dict[str, Any]):
        """
        Gera um relatório markdown consolidado com todos os resultados da análise multi-round.
        """
        self.logger.info("Gerando relatório multi-round consolidado...")
        if not self.output_dir:
            self.logger.warning("Diretório de saída não configurado. O relatório não será salvo.")
            return

        report_path = os.path.join(self.output_dir, "multi_round_analysis_report.md")
        
        with open(report_path, "w") as f:
            f.write("# Relatório Consolidado de Análise Multi-Round\n\n")
            f.write(f"Relatório gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("Este relatório apresenta uma análise compreensiva de múltiplos rounds de um experimento, avaliando a consistência, robustez e divergências comportamentais para fornecer um veredito consolidado sobre os resultados.\n\n")

            # --- Seção 1: Consistência da Estrutura Causal (Jaccard/Spearman) ---
            f.write("## 1. Consistência da Estrutura Causal\n\n")
            f.write("Avalia a consistência das relações causais identificadas entre os rounds.\n\n")
            
            graph_consistency = all_results.get('graph_consistency', {})
            df_jaccard = graph_consistency.get('granger_jaccard')
            if df_jaccard is not None and not df_jaccard.empty:
                f.write("### 1.1. Causalidade de Granger (Similaridade de Jaccard)\n")
                f.write("A tabela a seguir mostra a similaridade de Jaccard entre os conjuntos de relações causais (p < 0.05) para cada par de rounds. Valores mais próximos de 1 indicam maior consistência na ESTRUTURA do grafo causal.\n\n")
                f.write(df_jaccard.to_markdown())
                f.write("\n\n![Heatmap de Consistência de Granger](./granger_consistency_heatmap.png)\n\n")

            df_spearman = graph_consistency.get('te_spearman')
            if df_spearman is not None and not df_spearman.empty:
                f.write("### 1.2. Força Causal - Transfer Entropy (Correlação de Spearman)\n")
                f.write("A tabela a seguir mostra a correlação de Spearman entre as matrizes de Transferência de Entropia (TE). Esta métrica avalia a consistência na FORÇA da causalidade. Valores próximos de 1 indicam uma forte correlação positiva na força causal entre os rounds.\n\n")
                f.write(df_spearman.to_markdown())
                f.write("\n\n![Heatmap de Consistência de TE](./te_consistency_heatmap.png)\n\n")

            # --- Seção 2: Robustez Causal e Grafos Robustos ---
            f.write("## 2. Robustez das Relações Causais\n\n")
            causality_robustness = all_results.get('causality_robustness', {})
            if causality_robustness:
                 f.write("Análise da robustez das relações causais individuais com base na sua consistência (baixo Coeficiente de Variação) através dos rounds.\n\n")
                 f.write("### Relações Causais Robustas (Consenso)\n")
                 f.write("Relações que aparecem consistentemente com força similar em múltiplos rounds.\n\n")
                 # Link para o CSV
                 f.write("Para uma lista detalhada, veja o arquivo `robust_causal_relationships.csv`.\n\n")
                 
                 robust_graphs = all_results.get('robust_causality_graphs', {})
                 if robust_graphs:
                     f.write("### Grafos de Causalidade Robustos\n")
                     f.write("Grafos mostrando apenas as relações causais mais robustas para métricas selecionadas.\n\n")
                     for metric, path in robust_graphs.items():
                         f.write(f"**Métrica: {metric}**\n")
                         f.write(f"![Grafo Robusto para {metric}](./{os.path.basename(path)})\n\n")
            else:
                f.write("Análise de robustez causal não foi executada ou não produziu resultados.\n\n")


            # --- Seção 3: Consistência de Métricas (CV) ---
            f.write("## 3. Consistência dos Valores de Métricas\n\n")
            f.write("Análise da estabilidade dos valores das métricas através dos rounds, utilizando o Coeficiente de Variação (CV). Baixo CV indica alta consistência.\n\n")
            f.write("![Heatmap de CV por Tenant e Métrica](./cv_heatmap_by_tenant_metric.png)\n\n")
            f.write("Para dados detalhados, veja `round_consistency_cv.csv`.\n\n")

            # --- Seção 4: Divergência Comportamental ---
            f.write("## 4. Análise de Divergência Comportamental\n\n")
            f.write("Identifica rounds com comportamento anômalo e mede a estabilidade do comportamento dos tenants através dos rounds usando a Divergência de Kullback-Leibler.\n\n")
            f.write("Para dados detalhados, veja `tenant_stability_scores.csv`.\n\n")


            # --- Seção 5: Veredictos de Consenso ---
            f.write("## 5. Veredictos de Consenso\n\n")
            consensus = all_results.get('consensus', {})
            if consensus:
                f.write("Agregação dos resultados de todos os rounds para produzir um veredito final.\n\n")
                if consensus.get('noisy_tenants_consensus'):
                    f.write("### Tenants Barulhentos (Consenso)\n")
                    f.write("Tenants identificados como fontes de causalidade de forma consistente na maioria dos rounds. Veja `consensus_noisy_tenants.csv`.\n\n")
                if consensus.get('tenant_influence_ranking'):
                    f.write("### Ranking de Influência de Tenants (Consenso)\n")
                    f.write("Ranking de tenants com base na sua influência causal consolidada. Veja `tenant_influence_ranking.csv`.\n\n")
            else:
                f.write("Análise de consenso não foi executada ou não produziu resultados.\n\n")

            # --- Seção de Sumário ---
            f.write("## Sumário Final\n\n")
            f.write("A análise multi-round fornece insights sobre a estabilidade e reprodutibilidade dos resultados do experimento. Alta consistência sugere que as relações causais e comportamentos observados são robustos. Baixa consistência pode indicar que o sistema exibe comportamento variável ou que os resultados são sensíveis a condições iniciais, necessitando de investigação adicional.\n")

        self.logger.info(f"Relatório consolidado de análise multi-round salvo em: {report_path}")

    def _load_causality_matrices(self, context: Dict[str, Any], rounds: List[str]) -> Dict[str, Any]:
        """
        Carrega as matrizes de causalidade (TE e Granger) para cada round.
        Primeiro, tenta carregar do contexto. Se não encontrar, busca os arquivos CSV
        no diretório de saída do pipeline principal.
        """
        te_matrices_by_round = {}
        granger_matrices_by_round = {}
        
        # Prioriza o diretório de saída principal do contexto para maior robustez
        base_output_dir = context.get('output_dir')
        if not base_output_dir:
            # Fallback para o caso de o diretório do contexto não estar disponível
            base_output_dir = os.path.dirname(self.output_dir) if self.output_dir else None

        if not base_output_dir:
            self.logger.warning("Não foi possível determinar o diretório base de saídas. Não será possível carregar matrizes de causalidade.")
            return {}

        causality_output_dir = os.path.join(base_output_dir, 'causality')
        
        self.logger.info(f"Procurando matrizes de causalidade em: {causality_output_dir}")

        for round_id in rounds:
            # Tenta carregar do contexto primeiro
            te_key = f'te_matrices_round_{round_id}'
            granger_key = f'granger_matrices_round_{round_id}'
            
            if te_key in context and granger_key in context:
                te_matrices_by_round[round_id] = context[te_key]
                granger_matrices_by_round[round_id] = context[granger_key]
                self.logger.info(f"Matrizes de causalidade para o round '{round_id}' carregadas do contexto.")
                continue

            # Se não estiver no contexto, carregar dos arquivos
            self.logger.info(f"Matrizes para o round '{round_id}' não encontradas no contexto. Tentando carregar de arquivos...")
            round_causality_path = os.path.join(causality_output_dir, round_id)
            
            if not os.path.isdir(round_causality_path):
                self.logger.warning(f"Diretório de causalidade para o round '{round_id}' não encontrado em '{round_causality_path}'.")
                continue

            te_matrices = {}
            granger_matrices = {}
            
            try:
                for file_name in os.listdir(round_causality_path):
                    if file_name.startswith('te_matrix_') and file_name.endswith('.csv'):
                        metric = file_name.replace('te_matrix_', '').replace('.csv', '')
                        file_path = os.path.join(round_causality_path, file_name)
                        te_matrices[metric] = pd.read_csv(file_path, index_col=0)
                    elif file_name.startswith('granger_matrix_') and file_name.endswith('.csv'):
                        metric = file_name.replace('granger_matrix_', '').replace('.csv', '')
                        file_path = os.path.join(round_causality_path, file_name)
                        granger_matrices[metric] = pd.read_csv(file_path, index_col=0)
            except Exception as e:
                self.logger.error(f"Erro ao carregar arquivos de matriz do diretório {round_causality_path}: {e}")

            if te_matrices:
                te_matrices_by_round[round_id] = te_matrices
                self.logger.info(f"  - {len(te_matrices)} matrizes de Transfer Entropy carregadas para o round '{round_id}'.")
            if granger_matrices:
                granger_matrices_by_round[round_id] = granger_matrices
                self.logger.info(f"  - {len(granger_matrices)} matrizes de Granger carregadas para o round '{round_id}'.")

        return {
            'te_matrices_by_round': te_matrices_by_round,
            'granger_matrices_by_round': granger_matrices_by_round
        }

def generate_robust_causality_graph(
    robust_relationships: Dict[str, Dict[Tuple[str, str], Dict[str, float]]],
    output_dir: str,
    metric: str
) -> Optional[str]:
    """
    Gera um grafo de causalidade robusto para uma métrica específica.

    Args:
        robust_relationships: Dicionário com as relações causais robustas.
        output_dir: Diretório para salvar o grafo.
        metric: Métrica para a qual o grafo será gerado.

    Returns:
        Caminho do arquivo de imagem do grafo gerado ou None se não houver dados.
    """
    if not robust_relationships.get(metric):
        logger.warning(f"Não há relações causais robustas para a métrica '{metric}'. O grafo não será gerado.")
        return None

    G = nx.DiGraph()
    
    # Adicionar arestas com pesos, ignorando NaNs, para garantir que os dados são numéricos
    for (source, target), data in robust_relationships[metric].items():
        weight = data.get('mean_te')
        if weight is not None and pd.notna(weight):
            G.add_edge(source, target, weight=float(weight))

    if not G.nodes():
        logger.warning(f"O grafo para a métrica '{metric}' não possui nós. A visualização será pulada.")
        return None

    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.9, iterations=50)

    # Desenhar nós com tamanho proporcional ao out-degree (influência).
    node_sizes: List[int] = [int(500 + 1000 * G.out_degree(n)) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='skyblue', alpha=0.8)

    # Desenhar arestas com espessura proporcional à força da causalidade (peso).
    edge_weights: List[float] = [float(G[u][v]['weight']) for u, v in G.edges()]
    
    if edge_weights:
        # Normalizar pesos para uma faixa de espessura visualmente agradável (e.g., 1 a 10).
        min_w = min(edge_weights)
        max_w = max(edge_weights)
        
        edge_widths: List[float]
        if max_w > min_w:
            edge_widths = [float(1.0 + 9.0 * (w - min_w) / (max_w - min_w)) for w in edge_weights]
        else:
            # Todos os pesos são iguais, usar uma espessura média.
            edge_widths = [5.0] * len(edge_weights)
            
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', arrowsize=20)

    # Desenhar rótulos dos nós
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

    plt.title(f"Grafo de Causalidade Robusto para a Métrica: {metric}", fontsize=16)
    plt.axis('off')
    
    output_path = os.path.join(output_dir, f"robust_causality_graph_{metric}.png")
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Grafo de causalidade robusto para '{metric}' salvo em: {output_path}")
        plt.close()
        return output_path
    except Exception as e:
        logger.error(f"Erro ao salvar o grafo de causalidade para '{metric}': {e}")
        plt.close()
        return None


def analyze_round_consistency(
    df_long: pd.DataFrame,
    metrics: List[str],
    tenants: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Analisa a consistência dos dados entre diferentes rounds do experimento usando
    coeficiente de variação (CV) e teste de Friedman.
    
    Args:
        df_long: DataFrame consolidado em formato long.
        metrics: Lista de métricas para analisar.
        tenants: Lista de tenants para analisar.
        output_dir: Diretório para salvar resultados.
        
    Returns:
        Dicionário com resultados da análise de consistência.
    """
    # Inicializar resultados
    consistency_results = {
        'cv_by_metric': {},
        'cv_by_tenant': {},
        'friedman_tests': {},
        'consistent_tenants': {},
        'inconsistent_tenants': {}
    }
    
    # Para cada métrica e tenant, calcular o CV entre rounds
    for metric in metrics:
        cv_values = {}
        friedman_p_values = {}
        
        for tenant in tenants:
            # Filtrar dados para este tenant e métrica
            filtered_df = df_long[
                (df_long['metric_name'] == metric) & 
                (df_long['tenant_id'] == tenant)
            ]
            
            if filtered_df.empty:
                continue
                
            # Agrupar por round e calcular estatísticas
            stats_by_round = filtered_df.groupby('round_id')['metric_value'].agg(
                ['mean', 'std', 'count']
            )
            
            if len(stats_by_round) <= 1:
                continue
            
            # Calcular CV (Coeficiente de Variação) entre as médias dos rounds
            cv = stats_by_round['mean'].std() / stats_by_round['mean'].mean() * 100 if stats_by_round['mean'].mean() != 0 else np.nan
            cv_values[tenant] = cv
            
            # Preparar dados para teste de Friedman
            if len(stats_by_round) >= 3:  # Friedman requer pelo menos 3 grupos
                # Reorganizar dados para o formato necessário para o teste de Friedman
                round_data = {}
                for round_id in stats_by_round.index:
                    round_data[round_id] = filtered_df[filtered_df['round_id'] == round_id]['metric_value'].values
                
                # Equalizar tamanhos para o teste de Friedman (requer mesmo número de observações)
                min_size = min(len(data) for data in round_data.values())
                if min_size > 5:  # Garantir amostra mínima significativa
                    friedman_data = np.array([data[:min_size] for data in round_data.values()])
                    statistic, p_value = stats.friedmanchisquare(*friedman_data)
                    friedman_p_values[tenant] = p_value
        
        if cv_values:
            consistency_results['cv_by_metric'][metric] = cv_values
            
            # Classificar tenants como consistentes ou inconsistentes
            consistent = {tenant: cv for tenant, cv in cv_values.items() if cv < 15}  # CV < 15% indica boa consistência
            inconsistent = {tenant: cv for tenant, cv in cv_values.items() if cv >= 15}
            
            consistency_results['consistent_tenants'][metric] = consistent
            consistency_results['inconsistent_tenants'][metric] = inconsistent
        
        if friedman_p_values:
            consistency_results['friedman_tests'][metric] = friedman_p_values
    
    # Calcular CV médio por tenant (em todas as métricas)
    tenant_avg_cv = {}
    for metric, tenant_cvs in consistency_results['cv_by_metric'].items():
        for tenant, cv in tenant_cvs.items():
            if not np.isnan(cv):
                if tenant not in tenant_avg_cv:
                    tenant_avg_cv[tenant] = []
                tenant_avg_cv[tenant].append(cv)
    
    # Finalizar cálculo da média
    for tenant, cvs in tenant_avg_cv.items():
        consistency_results['cv_by_tenant'][tenant] = np.mean(cvs)
    
    # Salvar resultados
    cv_df = pd.DataFrame.from_dict({
        metric: pd.Series(tenant_cvs) for metric, tenant_cvs in consistency_results['cv_by_metric'].items()
    })
    cv_df.to_csv(os.path.join(output_dir, 'round_consistency_cv.csv'))
    
    return consistency_results


def analyze_causality_robustness(
    te_matrices_by_round: Dict[str, Dict[str, pd.DataFrame]],
    granger_matrices_by_round: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: str
) -> Dict[str, Any]:
    """
    Analisa a robustez das relações causais detectadas entre diferentes rounds.
    
    Args:
        te_matrices_by_round: Matrizes de transfer entropy por round.
        granger_matrices_by_round: Matrizes de causalidade de Granger por round.
        output_dir: Diretório para salvar resultados.
        
    Returns:
        Dicionário com resultados da análise de robustez causal.
    """
    results = {
        'robust_causal_relationships': {},
        'robust_te_scores': {},
        'inconsistent_causal_relationships': {},
        'metrics_with_consistent_causality': []
    }
    
    # Verificar quais métricas existem em todos os rounds (tanto TE quanto Granger)
    common_metrics_te = set()
    for round_id, te_matrices in te_matrices_by_round.items():
        if round_id == list(te_matrices_by_round.keys())[0]:
            common_metrics_te = set(te_matrices.keys())
        else:
            common_metrics_te = common_metrics_te.intersection(set(te_matrices.keys()))
    
    # Analisar robustez do Transfer Entropy
    for metric in common_metrics_te:
        # Coletar matrizes de TE para esta métrica em todos os rounds
        te_matrices = [
            matrices[metric] for round_id, matrices in te_matrices_by_round.items() 
            if metric in matrices
        ]
        
        if len(te_matrices) < 2:
            continue
            
        # Identificar pares de tenants presentes em todas as matrizes
        common_sources = set(te_matrices[0].columns)
        common_targets = set(te_matrices[0].index)
        
        for matrix in te_matrices[1:]:
            common_sources = common_sources.intersection(set(matrix.columns))
            common_targets = common_targets.intersection(set(matrix.index))
        
        # Calcular média e desvio padrão do TE para cada par de tenants
        robust_pairs = {}
        inconsistent_pairs = {}
        for source in common_sources:
            for target in common_targets:
                if source == target:  # Pular auto-relações
                    continue
                    
                # Extrair valores de TE de todos os rounds
                te_values = [matrix.loc[target, source] for matrix in te_matrices]
                
                # Calcular média, desvio e CV
                mean_te = np.mean(np.array(te_values))
                std_te = np.std(np.array(te_values))
                cv = (std_te / mean_te * 100) if mean_te > 0 else np.inf
                
                # Determinar se a relação causal é robusta (CV < 25% e média > 0.05)
                if cv < 25 and mean_te > 0.05:
                    robust_pairs[(source, target)] = {
                        'mean_te': mean_te,
                        'std_te': std_te,
                        'cv': cv,
                        'values': te_values
                    }
                elif mean_te > 0.05:  # Relação causal significativa, mas inconsistente
                    inconsistent_pairs[(source, target)] = {
                        'mean_te': mean_te,
                        'std_te': std_te,
                        'cv': cv,
                        'values': te_values
                    }
        
        if robust_pairs:
            results['robust_causal_relationships'][metric] = robust_pairs
            results['robust_te_scores'][metric] = {
                f"{source}->{target}": data['mean_te'] 
                for (source, target), data in robust_pairs.items()
            }
            
            if len(robust_pairs) >= 2:  # Pelo menos duas relações causais robustas
                results['metrics_with_consistent_causality'].append(metric)
        
        if inconsistent_pairs:
            results['inconsistent_causal_relationships'][metric] = inconsistent_pairs
    
    # Salvar resultados em CSV
    robust_data = []
    for metric, pairs in results['robust_causal_relationships'].items():
        for (source, target), stats in pairs.items():
            robust_data.append({
                'metric': metric,
                'source': source,
                'target': target,
                'mean_te': stats['mean_te'],
                'std_te': stats['std_te'],
                'cv': stats['cv']
            })
    
    if robust_data:
        pd.DataFrame(robust_data).to_csv(os.path.join(output_dir, 'robust_causal_relationships.csv'), index=False)
    
    return results


def analyze_behavioral_divergence(
    df_long: pd.DataFrame,
    metrics: List[str],
    tenants: List[str],
    output_dir: str
) -> Dict[str, Any]:
    """
    Realiza análise de divergência comportamental entre rounds usando distância de
    Kullback-Leibler e clustering hierárquico.
    
    Args:
        df_long: DataFrame consolidado em formato long.
        metrics: Lista de métricas para analisar.
        tenants: Lista de tenants para analisar.
        output_dir: Diretório para salvar resultados.
        
    Returns:
        Dicionário com resultados da análise de divergência.
    """
    results = {
        'kl_divergence': {},
        'clustered_rounds': {},
        'anomalous_rounds': {},
        'tenant_stability': {}
    }
    
    # Para cada métrica e tenant, calcular a divergência de KL entre distributions por round
    for metric in metrics:
        metric_results = {}
        
        for tenant in tenants:
            # Filtrar dados para este tenant e métrica
            filtered_df = df_long[
                (df_long['metric_name'] == metric) & 
                (df_long['tenant_id'] == tenant)
            ]
            
            if filtered_df.empty:
                continue
            
            rounds = filtered_df['round_id'].unique()
            if len(rounds) < 2:
                continue
                
            # Calcular KL Divergence entre cada par de rounds
            kl_distances = {}
            distributions = {}
            
            # Preparar distribuições dos valores para cada round
            for round_id in rounds:
                round_data = filtered_df[filtered_df['round_id'] == round_id]['metric_value']
                
                # Criar histograma para estimar densidade
                hist, bin_edges = np.histogram(round_data, bins=10, density=True)
                # Evitar zeros (que causam infinitos na KL divergence)
                hist = hist + 1e-10
                hist = hist / hist.sum()  # Normalizar novamente
                
                distributions[round_id] = hist
            
            # Calcular KL divergence para cada par de rounds
            for i, round1 in enumerate(rounds):
                for round2 in rounds[i+1:]:
                    # Cálculo simétrico de divergência: KL(P||Q) + KL(Q||P)
                    kl1 = distance.jensenshannon(distributions[round1], distributions[round2])
                    kl2 = distance.jensenshannon(distributions[round2], distributions[round1])
                    symmetric_kl = (kl1 + kl2) / 2
                    
                    kl_distances[(round1, round2)] = symmetric_kl
            
            # Se temos pelo menos 3 rounds, podemos fazer clustering
            if len(rounds) >= 3:
                # Criar matriz de distância para clustering
                n_rounds = len(rounds)
                distance_matrix = np.zeros((n_rounds, n_rounds))
                
                for i, round1 in enumerate(rounds):
                    for j, round2 in enumerate(rounds):
                        if i == j:
                            distance_matrix[i, j] = 0
                        elif (round1, round2) in kl_distances:
                            distance_matrix[i, j] = kl_distances[(round1, round2)]
                        else:
                            distance_matrix[i, j] = kl_distances[(round2, round1)]
                
                # Aplicar clustering hierárquico
                clustering = AgglomerativeClustering(
                    n_clusters=None,
                    distance_threshold=0.05,  # Ajustar conforme necessário
                    metric='precomputed',
                    linkage='average'
                ).fit(distance_matrix)
                
                # Verificar se algum round está isolado (potencialmente anômalo)
                labels = clustering.labels_
                unique_clusters = np.unique(labels)
                anomalous_rounds = []
                
                for cluster in unique_clusters:
                    cluster_size = sum(labels == cluster)
                    if cluster_size == 1:  # Round isolado
                        anomalous_round_idx = np.where(labels == cluster)[0][0]
                        anomalous_rounds.append(rounds[anomalous_round_idx])
                
                # Organizar rounds por cluster
                clusters = {}
                for i, label in enumerate(labels):
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(rounds[i])
                
                # Armazenar resultado do clustering
                if clusters:
                    results['clustered_rounds'][(tenant, metric)] = clusters
                    
                if anomalous_rounds:
                    results['anomalous_rounds'][(tenant, metric)] = anomalous_rounds
            
            # Armazenar KL divergences
            if kl_distances:
                metric_results[(tenant)] = {
                    'kl_distances': kl_distances,
                    # Calcular métrica de estabilidade (inverso da média das divergências)
                    'stability_score': 1 / (1 + np.mean(list(kl_distances.values())))
                }
        
        if metric_results:
            results['kl_divergence'][metric] = metric_results
            
            # Armazenar scores de estabilidade por tenant
            for tenant, data in metric_results.items():
                if tenant not in results['tenant_stability']:
                    results['tenant_stability'][tenant] = []
                results['tenant_stability'][tenant].append(data['stability_score'])
    
    # Finalizar scores de estabilidade por tenant
    for tenant in results['tenant_stability']:
        results['tenant_stability'][tenant] = np.mean(results['tenant_stability'][tenant])
    
    # Salvar resultados em CSV
    stability_df = pd.DataFrame({
        'tenant': list(results['tenant_stability'].keys()),
        'stability_score': list(results['tenant_stability'].values())
    })
    
    if not stability_df.empty:
        stability_df.to_csv(os.path.join(output_dir, 'tenant_stability_scores.csv'), index=False)
    
    return results


def aggregate_round_consensus(
    df_long: pd.DataFrame,
    te_matrices_by_round: Dict[str, Dict[str, pd.DataFrame]],
    consistency_results: Dict[str, Any],
    output_dir: str
) -> Dict[str, Any]:
    """
    Realiza agregação de consenso entre rounds, produzindo veredictos consolidados
    sobre o comportamento do sistema.
    
    Args:
        df_long: DataFrame consolidado em formato long.
        te_matrices_by_round: Matrizes de transfer entropy por round.
        consistency_results: Resultados da análise de consistência.
        output_dir: Diretório para salvar resultados.
        
    Returns:
        Dicionário com resultados da agregação de consenso.
    """
    results = {
        'noisy_tenants_consensus': {},
        'causal_relationships_consensus': {},
        'tenant_influence_ranking': {}
    }
    
    # Extrair rounds e métricas
    rounds = df_long['round_id'].unique()
    metrics = df_long['metric_name'].unique()
    
    # Para cada métrica, determinar os "noisy tenants" e relações causais por consenso
    for metric in metrics:
        # Verificar se temos matrizes de TE para esta métrica em todos os rounds
        te_available = all(
            round_id in te_matrices_by_round and 
            metric in te_matrices_by_round[round_id]
            for round_id in rounds
        )
        
        if not te_available:
            continue
        
        # Coletar "votos" para relações causais significativas em cada round
        causal_relationship_votes = {}
        noisy_tenant_votes = {}
        
        for round_id in rounds:
            te_matrix = te_matrices_by_round[round_id][metric]
            
            # Identificar tenants com maior "outgoing causality"
            tenant_outgoing_te = {}
            for source in te_matrix.columns:
                outgoing = te_matrix[source].drop(source) if source in te_matrix.index else te_matrix[source]
                significant_outgoing = outgoing[outgoing > 0.05]  # Threshold de significância
                
                # Calcular score de influência do tenant
                influence_score = significant_outgoing.sum() * len(significant_outgoing)
                tenant_outgoing_te[source] = influence_score
                
                # Registrar "voto" para tenant barulhento se tiver influência significativa
                if influence_score > 0.1:  # Threshold empírico
                    if source not in noisy_tenant_votes:
                        noisy_tenant_votes[source] = 0
                    noisy_tenant_votes[source] += 1
                
                # Registrar votos para cada relação causal significativa
                for target, te_value in significant_outgoing.items():
                    relationship = (source, target)
                    if relationship not in causal_relationship_votes:
                        causal_relationship_votes[relationship] = 0
                    causal_relationship_votes[relationship] += 1
        
        # Determinar consenso para "noisy tenants" (mais de 75% dos rounds)
        min_votes_noisy = len(rounds) * 0.75
        consensus_noisy_tenants = {
            tenant: votes for tenant, votes in noisy_tenant_votes.items() 
            if votes > min_votes_noisy
        }
        
        # Determinar consenso para relações causais (mais de 75% dos rounds)
        min_votes_causal = len(rounds) * 0.75
        consensus_causal_relationships = {
            relationship: votes for relationship, votes in causal_relationship_votes.items()
            if votes > min_votes_causal
        }
        
        # Armazenar resultados
        if consensus_noisy_tenants:
            results['noisy_tenants_consensus'][metric] = consensus_noisy_tenants
            
        if consensus_causal_relationships:
            results['causal_relationships_consensus'][metric] = consensus_causal_relationships
            
        # Criar ranking de influência dos tenants por consenso
        tenant_scores = {}
        for (source, target), votes in consensus_causal_relationships.items():
            if source not in tenant_scores:
                tenant_scores[source] = {'total_influence': 0, 'targets': 0, 'vote_strength': 0}
                
            # Ponderação pelo número de votos e alvos
            tenant_scores[source]['targets'] += 1
            tenant_scores[source]['vote_strength'] += votes
            
        # Calcular score final
        for tenant in tenant_scores:
            tenant_scores[tenant]['total_influence'] = (
                tenant_scores[tenant]['targets'] * 
                tenant_scores[tenant]['vote_strength'] / len(rounds)
            )
        
        # Ordenar por influência
        tenant_ranking = {
            tenant: score['total_influence'] 
            for tenant, score in sorted(
                tenant_scores.items(), 
                key=lambda x: x[1]['total_influence'], 
                reverse=True
            )
        }
        
        if tenant_ranking:
            results['tenant_influence_ranking'][metric] = tenant_ranking
    
    # Salvar resultados
    # Tenants barulhentos por consenso
    noisy_data = []
    for metric, tenants in results['noisy_tenants_consensus'].items():
        for tenant, votes in tenants.items():
            noisy_data.append({
                'metric': metric,
                'tenant': tenant,
                'votes': votes,
                'total_rounds': len(rounds),
                'vote_percentage': (votes / len(rounds)) * 100
            })
    
    if noisy_data:
        pd.DataFrame(noisy_data).to_csv(
            os.path.join(output_dir, 'consensus_noisy_tenants.csv'), 
            index=False
        )
    
    # Relações causais por consenso
    causal_data = []
    for metric, relationships in results['causal_relationships_consensus'].items():
        for (source, target), votes in relationships.items():
            causal_data.append({
                'metric': metric,
                'source': source,
                'target': target,
                'votes': votes,
                'total_rounds': len(rounds),
                'vote_percentage': (votes / len(rounds)) * 100
            })
    
    if causal_data:
        pd.DataFrame(causal_data).to_csv(
            os.path.join(output_dir, 'consensus_causal_relationships.csv'), 
            index=False
        )
    
    # Ranking de influência
    ranking_data = []
    for metric, ranking in results['tenant_influence_ranking'].items():
        for tenant, score in ranking.items():
            ranking_data.append({
                'metric': metric,
                'tenant': tenant,
                'influence_score': score
            })
    
    if ranking_data:
        pd.DataFrame(ranking_data).to_csv(
            os.path.join(output_dir, 'tenant_influence_ranking.csv'), 
            index=False
        )
    
    return results


def generate_round_consistency_visualizations(
    df_long: pd.DataFrame,
    consistency_results: Dict[str, Any],
    causality_robustness: Dict[str, Any],
    output_dir: str
) -> List[str]:
    """
    Gera visualizações para análise de consistência entre rounds, incluindo
    gráficos com intervalos de confiança e heatmaps.
    
    Args:
        df_long: DataFrame consolidado em formato long.
        consistency_results: Resultados da análise de consistência.
        causality_robustness: Resultados da análise de robustez causal.
        output_dir: Diretório para salvar resultados.
        
    Returns:
        Lista com caminhos das visualizações geradas.
    """
    visualization_paths = []
    
    # 1. Gráfico de CV por métrica e tenant
    if consistency_results.get('cv_by_metric'):
        cv_data = []
        for metric, tenant_cvs in consistency_results['cv_by_metric'].items():
            for tenant, cv in tenant_cvs.items():
                if not np.isnan(cv):
                    cv_data.append({
                        'metric': metric,
                        'tenant': tenant,
                        'cv': cv
                    })
        
        if cv_data:
            cv_df = pd.DataFrame(cv_data)
            pivot_table = cv_df.pivot_table(
                index='tenant', 
                columns='metric', 
                values='cv',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(
                pivot_table,
                annot=True,
                cmap='viridis',
                fmt=".1f",
                linewidths=0.5,
                cbar_kws={'label': 'Coeficiente de Variação (%)'}
            )
            plt.title('Coeficiente de Variação entre Rounds por Tenant e Métrica')
            plt.tight_layout()
            
            cv_heatmap_path = os.path.join(output_dir, 'cv_heatmap_by_tenant_metric.png')
            plt.savefig(cv_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            visualization_paths.append(cv_heatmap_path)
    
    # 2. Gerar boxplots consolidados
    logger.info("Gerando boxplots consolidados...")
    metrics = df_long['metric_name'].unique()
    for metric in metrics:
        try:
            path = generate_consolidated_boxplot(
                df_long=df_long,
                metric=metric,
                output_dir=output_dir
            )
            if path:
                visualization_paths.append(path)
        except Exception as e:
            logger.warning(f"Erro ao gerar boxplot consolidado para a métrica {metric}: {e}")
            
    return visualization_paths
