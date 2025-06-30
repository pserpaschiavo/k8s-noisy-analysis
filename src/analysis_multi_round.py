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
from src.visualization.plots import (
    generate_consolidated_heatmap,
    generate_all_enhanced_consolidated_boxplots
)
from src.visualization.advanced_plots import (
    generate_all_consolidated_timeseries,
    plot_aggregated_correlation_graph
)
from src.analysis_correlation import compute_aggregated_correlation
from src.effect_size import extract_effect_sizes  # Importar o novo módulo
from src.effect_aggregation import aggregate_effect_sizes  # Importar a função de agregação de efeitos
from src.phase_correlation import extract_phase_correlations, analyze_correlation_stability  # Importar funções de correlação intra-fase
from src.visualization.effect_plots import (
    generate_effect_size_heatmap,
    plot_effect_error_bars,
    plot_effect_scatter,
    generate_effect_forest_plot
)
from src.visualization.correlation_plots import (
    plot_correlation_heatmap,
    plot_correlation_network,
    plot_correlation_stability
)
from src.robustness_analysis import perform_robustness_analysis, generate_robustness_summary
from src.insights_generation import generate_automated_insights, generate_markdown_report

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

            # 0. Extração de tamanhos de efeito para todos os rounds
            self.logger.info("Realizando extração de tamanhos de efeito em todas as combinações...")
            effect_sizes_df = self._extract_effect_sizes(context, config, df_filtered, rounds)
            if not effect_sizes_df.empty:
                results['effect_sizes'] = {
                    'dataframe': effect_sizes_df,
                    'summary': {
                        'total_comparisons': effect_sizes_df.shape[0],
                        'significant_effects': effect_sizes_df[effect_sizes_df['p_value'] < 0.05].shape[0],
                        'large_effects': effect_sizes_df[effect_sizes_df['effect_size'].abs() > 0.8].shape[0],
                        'medium_effects': effect_sizes_df[(effect_sizes_df['effect_size'].abs() > 0.5) & 
                                                         (effect_sizes_df['effect_size'].abs() <= 0.8)].shape[0],
                        'small_effects': effect_sizes_df[(effect_sizes_df['effect_size'].abs() > 0.2) & 
                                                        (effect_sizes_df['effect_size'].abs() <= 0.5)].shape[0]
                    }
                }
                
                # 0.1 Agregação de tamanhos de efeito entre rounds
                self.logger.info("Agregando tamanhos de efeito entre rounds...")
                aggregated_effects_df = self._aggregate_effect_sizes(effect_sizes_df, config)
                if not aggregated_effects_df.empty:
                    results['aggregated_effects'] = {
                        'dataframe': aggregated_effects_df,
                        'summary': {
                            'total_combinations': aggregated_effects_df.shape[0],
                            'significant_combinations': aggregated_effects_df['is_significant'].sum(),
                            'high_reliability': (aggregated_effects_df['reliability_category'] == 'high').sum(),
                            'medium_reliability': (aggregated_effects_df['reliability_category'] == 'medium').sum(),
                            'low_reliability': (aggregated_effects_df['reliability_category'] == 'low').sum()
                        }
                    }
                    
                    # Realizar análise de robustez dos tamanhos de efeito
                    self.logger.info("Realizando análise de robustez dos tamanhos de efeito...")
                    robustness_output_dir = os.path.join(self.output_dir, 'robustness') if self.output_dir else None
                    
                    robustness_df, robustness_plots = perform_robustness_analysis(
                        effect_sizes_df=effect_sizes_df,
                        output_dir=robustness_output_dir
                    )
                    
                    if not robustness_df.empty:
                        results['effect_robustness'] = {
                            'dataframe': robustness_df,
                            'summary': {
                                'total_analyzed': robustness_df.shape[0],
                                'high_robustness': (robustness_df['overall_robustness'] == 'Alta').sum(),
                                'medium_robustness': (robustness_df['overall_robustness'] == 'Média').sum(),
                                'low_robustness': (robustness_df['overall_robustness'] == 'Baixa').sum()
                            },
                            'plots': robustness_plots
                        }
                        
                        # Adicionar resumo de robustez ao relatório
                        robustness_summary = generate_robustness_summary(robustness_df)
                        results['effect_robustness']['report'] = robustness_summary
                        
                    # Gerar visualizações para os tamanhos de efeito agregados
                    self.logger.info("Gerando visualizações para os tamanhos de efeito agregados...")
                    
                    if self.output_dir:
                        effect_viz_dir = os.path.join(self.output_dir, 'effect_visualizations')
                        os.makedirs(effect_viz_dir, exist_ok=True)
                        
                        # Heatmap de tamanhos de efeito
                        heatmap_paths = []
                        for metric in aggregated_effects_df['metric_name'].unique():
                            path = generate_effect_size_heatmap(
                                aggregated_effects_df=aggregated_effects_df,
                                output_dir=effect_viz_dir,
                                metric=metric
                            )
                            if path:
                                heatmap_paths.append(path)
                                
                        # Error bars com IC95%
                        error_bar_paths = []
                        for metric in aggregated_effects_df['metric_name'].unique():
                            path = plot_effect_error_bars(
                                aggregated_effects_df=aggregated_effects_df,
                                output_dir=effect_viz_dir,
                                metric=metric
                            )
                            if path:
                                error_bar_paths.append(path)
                                
                        # Gráficos de dispersão
                        scatter_paths = []
                        for metric in aggregated_effects_df['metric_name'].unique():
                            path = plot_effect_scatter(
                                aggregated_effects_df=aggregated_effects_df,
                                output_dir=effect_viz_dir,
                                metric=metric
                            )
                            if path:
                                scatter_paths.append(path)
                                
                        results['aggregated_effects']['visualizations'] = {
                            'heatmaps': heatmap_paths,
                            'error_bars': error_bar_paths,
                            'scatter_plots': scatter_paths
                        }
                    
            # 0.2 Extração de correlações intra-fase
            self.logger.info("Realizando extração de correlações intra-fase entre tenants...")
            phase_correlations_df = self._extract_phase_correlations(context, config, df_filtered, rounds)
            if not phase_correlations_df.empty:
                results['phase_correlations'] = {
                    'dataframe': phase_correlations_df,
                    'summary': {
                        'total_correlations': phase_correlations_df.shape[0],
                        'strong_correlations': (phase_correlations_df['correlation_strength'] == 'strong').sum(),
                        'moderate_correlations': (phase_correlations_df['correlation_strength'] == 'moderate').sum(),
                        'weak_correlations': (phase_correlations_df['correlation_strength'] == 'weak').sum(),
                        'positive_correlations': (phase_correlations_df['correlation'] > 0).sum(),
                        'negative_correlations': (phase_correlations_df['correlation'] < 0).sum()
                    }
                }
                
                # Analisar estabilidade das correlações intra-fase
                self.logger.info("Analisando estabilidade das correlações intra-fase...")
                correlation_stability_df = analyze_correlation_stability(phase_correlations_df)
                
                if isinstance(correlation_stability_df, pd.DataFrame) and not correlation_stability_df.empty:
                    results['phase_correlations']['stability'] = {
                        'dataframe': correlation_stability_df,
                        'summary': {
                            'total_analyzed': correlation_stability_df.shape[0],
                            'high_stability': len(correlation_stability_df[correlation_stability_df['stability_category'] == 'high']),
                            'medium_stability': len(correlation_stability_df[correlation_stability_df['stability_category'] == 'medium']),
                            'low_stability': len(correlation_stability_df[correlation_stability_df['stability_category'] == 'low'])
                        }
                    }
                    
                    # Gerar visualizações para as correlações intra-fase
                    self.logger.info("Gerando visualizações para correlações intra-fase...")
                    
                    if self.output_dir:
                        correlation_viz_dir = os.path.join(self.output_dir, 'correlation_visualizations')
                        os.makedirs(correlation_viz_dir, exist_ok=True)
                        
                        # 1. Heatmaps de correlação para cada combinação de métrica, fase e round
                        heatmap_paths = []
                        for metric in phase_correlations_df['metric_name'].unique():
                            for phase in phase_correlations_df['experimental_phase'].unique():
                                for round_id in phase_correlations_df['round_id'].unique():
                                    path = plot_correlation_heatmap(
                                        correlation_df=phase_correlations_df,
                                        output_dir=correlation_viz_dir,
                                        metric=metric,
                                        phase=phase,
                                        round_id=round_id
                                    )
                                    if path:
                                        heatmap_paths.append(path)
                        
                        # 2. Redes de correlação para cada combinação de métrica e fase
                        network_paths = []
                        for metric in phase_correlations_df['metric_name'].unique():
                            for phase in phase_correlations_df['experimental_phase'].unique():
                                path = plot_correlation_network(
                                    correlation_df=phase_correlations_df,
                                    output_dir=correlation_viz_dir,
                                    metric=metric,
                                    phase=phase
                                )
                                if path:
                                    network_paths.append(path)
                        
                        # 3. Visualizações de estabilidade para cada combinação de métrica e fase
                        stability_paths = []
                        for metric in correlation_stability_df['metric_name'].unique():
                            for phase in correlation_stability_df['experimental_phase'].unique():
                                path = plot_correlation_stability(
                                    correlation_df=phase_correlations_df,
                                    stability_df=results['phase_correlations']['stability']['dataframe'],
                                    output_dir=correlation_viz_dir,
                                    metric=metric,
                                    phase=phase
                                )
                                if path:
                                    stability_paths.append(path)
                        
                        results['phase_correlations']['visualizations'] = {
                            'heatmaps': heatmap_paths,
                            'networks': network_paths,
                            'stability_plots': stability_paths
                        }

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

            # 6. Geração de insights automáticos
            self.logger.info("Gerando insights automáticos...")
            try:
                # Determinar quais DataFrames estão disponíveis para insights
                effect_sizes_available = 'effect_sizes' in results and 'dataframe' in results['effect_sizes']
                aggregated_effects_available = 'aggregated_effects' in results and 'dataframe' in results['aggregated_effects']
                robustness_available = 'effect_robustness' in results and 'dataframe' in results['effect_robustness']
                correlations_available = 'phase_correlations' in results and 'dataframe' in results['phase_correlations']
                correlation_stability_available = correlations_available and 'stability' in results['phase_correlations']
                
                # Gerar insights apenas se houver dados agregados disponíveis
                if aggregated_effects_available:
                    aggregated_effects_df = results['aggregated_effects']['dataframe']
                    robustness_df = results['effect_robustness']['dataframe'] if robustness_available else None
                    phase_correlations_df = results['phase_correlations']['dataframe'] if correlations_available else None
                    correlation_stability_df = results['phase_correlations']['stability']['dataframe'] if correlation_stability_available else None
                    
                    # Gerar insights
                    # Extrair fases experimentais dos dados para o contexto
                    all_phases = sorted(aggregated_effects_df['experimental_phase'].unique().tolist())
                    baseline_phases = aggregated_effects_df['baseline_phase'].unique().tolist()
                    experimental_phases = [phase for phase in all_phases if phase not in baseline_phases]
                    
                    insights = generate_automated_insights(
                        aggregated_effects_df=aggregated_effects_df,
                        robustness_df=robustness_df,
                        phase_correlations_df=phase_correlations_df,
                        correlation_stability_df=correlation_stability_df,
                        context={
                            'experiment_name': context.get('experiment_name', 'Análise Multi-round'),
                            'rounds': rounds,
                            'metrics': metrics,
                            'phases': experimental_phases,  # Agora definido corretamente
                            'tenants': tenants
                        }
                    )
                    
                    # Adicionar insights aos resultados
                    results['automated_insights'] = insights
                    
                    # Gerar relatório markdown
                    if self.output_dir:
                        insights_dir = os.path.join(self.output_dir, 'insights')
                        os.makedirs(insights_dir, exist_ok=True)
                        report_path = os.path.join(insights_dir, 'automated_insights_report.md')
                        markdown_report = generate_markdown_report(insights, report_path)
                        
                        self.logger.info(f"✅ Insights automáticos gerados e salvos em: {report_path}")
                    else:
                        self.logger.info("✅ Insights automáticos gerados (sem diretório de saída definido)")
                else:
                    self.logger.warning("Não há tamanhos de efeito agregados disponíveis. Pulando geração de insights.")
            except Exception as e:
                self.logger.error(f"Erro ao gerar insights automáticos: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())
            
            # 7. Visualizações consolidadas
            self.logger.info("Gerando visualizações consolidadas...")
            # Removido o antigo `generate_round_consistency_visualizations`
            # As visualizações agora são geradas por tipo.

            # 6.1 Boxplots Consolidados (Aprimorado v2.1)
            self.logger.info("Gerando boxplots consolidados (violin) para todas as métricas...")
            try:
                boxplot_dir = os.path.join(output_dir, "boxplots")
                os.makedirs(boxplot_dir, exist_ok=True)
                boxplot_paths = generate_all_enhanced_consolidated_boxplots(
                    df_long=df_filtered,
                    output_dir=boxplot_dir
                )
                results['consolidated_boxplots'] = boxplot_paths
                self.logger.info(f"✅ Boxplots consolidados gerados: {len(boxplot_paths)} arquivos")
            except Exception as e:
                self.logger.error(f"Erro ao gerar boxplots consolidados: {e}")
                results['consolidated_boxplots'] = {}

            # 6.2 Time Series Consolidados (v2.0)
            self.logger.info("Gerando time series consolidados para todas as métricas...")
            try:
                timeseries_dir = os.path.join(output_dir, "timeseries")
                os.makedirs(timeseries_dir, exist_ok=True)
                timeseries_paths = generate_all_consolidated_timeseries(
                    df_long=df_filtered,
                    output_dir=timeseries_dir, # Diretório específico
                    rounds=rounds,
                    tenants=tenants,
                    normalize_time=True,
                    add_confidence_bands=True
                )
                results['consolidated_timeseries'] = timeseries_paths
                self.logger.info(f"✅ Time series consolidados gerados: {len(timeseries_paths)} métricas")
            except Exception as e:
                self.logger.error(f"Erro ao gerar time series consolidados: {e}")
                results['consolidated_timeseries'] = {}

            # 6.3 Gráfico de Correlação Agregado
            self.logger.info("Gerando gráfico de correlação agregado...")
            try:
                correlation_dir = os.path.join(output_dir, "correlation")
                os.makedirs(correlation_dir, exist_ok=True)
                
                phases = sorted(df_filtered['experimental_phase'].unique())

                aggregated_correlations = compute_aggregated_correlation(
                    df=df_filtered,
                    metrics=metrics,
                    rounds=rounds,
                    phases=phases,
                    method='pearson'
                )
                
                correlation_plot_paths = {}
                if aggregated_correlations:
                    for metric, corr_matrix in aggregated_correlations.items():
                        filename = f"aggregated_correlation_graph_{metric}.png"
                        title = f"Grafo de Correlação Agregada - {metric.replace('_', ' ').title()}"
                        
                        plot_path = plot_aggregated_correlation_graph(
                            correlation_matrix=corr_matrix,
                            title=title,
                            output_dir=correlation_dir,
                            filename=filename,
                            threshold=0.5
                        )
                        if plot_path:
                            correlation_plot_paths[metric] = plot_path
                
                results['aggregated_correlation_graphs'] = correlation_plot_paths
                self.logger.info(f"✅ Gráficos de correlação agregada gerados: {len(correlation_plot_paths)} arquivos")

            except Exception as e:
                self.logger.error(f"Erro ao gerar gráficos de correlação agregada: {e}", exc_info=True)
                results['aggregated_correlation_graphs'] = {}

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
            # Tratar NaNs que podem surgir se os dados de entrada forem constantes
            df_spearman['Spearman Correlation'] = df_spearman['Spearman Correlation'].fillna(0)
            consistency_results['te_spearman'] = df_spearman.pivot(index='Metric', columns='Round Pair', values='Spearman Correlation')
        else:
            # Garante que a chave exista mesmo que não haja dados
            consistency_results['te_spearman'] = pd.DataFrame()
            
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
                f.write("A tabela a seguir mostra a correlação de Spearman entre as matrizes de Transferência de Entropia (TE). Esta métrica avalia a consistência na FORÇA da causalidade. Valores próximos de 1 indicam uma forte correlação positiva na força causal entre os rounds.\n")
                f.write("*Nota: Valores de correlação nulos (NaN) foram convertidos para 0, indicando ausência de correlação consistente ou dados de entrada constantes (sem variabilidade na força causal).*\n\n")
                f.write(df_spearman.to_markdown(floatfmt=".4f"))
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
            f.write("Identifica rounds com comportamento anômalo e mede a estabilidade do comportamento dos tenants através dos rounds usando a Divergência de Kullback-Leibniz.\n\n")
            f.write("Para dados detalhados, veja `tenant_stability_scores.csv`.\n\n")

            # --- Seção 4.1: Boxplots Consolidados (Aprimorado v2.1) ---
            f.write("## 4.1. Boxplots Consolidados (Violin Plots)\n\n")
            f.write("**🆕 Visualizações aprimoradas** que mostram a distribuição de cada métrica por fase experimental, agregando dados de todos os rounds. Os violin plots oferecem uma visão mais rica da densidade dos dados em comparação com os boxplots tradicionais.\n\n")
            
            consolidated_boxplots = all_results.get('consolidated_boxplots', {})
            if consolidated_boxplots:
                f.write("### Boxplots por Métrica\n")
                f.write("Para cada métrica, são gerados dois gráficos:\n")
                f.write("- **Valores Brutos**: Mostra a distribuição real dos dados.\n")
                f.write("- **Valores Normalizados**: Normaliza os dados pela média da fase 'Baseline' de cada tenant, permitindo uma comparação justa do *impacto relativo* das fases de stress.\n\n")
                
                # Organizar por métrica para apresentar lado a lado
                metrics_boxplot = sorted(list(set([k.replace('_raw', '').replace('_normalized', '') for k in consolidated_boxplots.keys()])))
                
                for metric in metrics_boxplot:
                    metric_display = metric.replace("_", " ").title()
                    f.write(f"#### {metric_display}\n")
                    
                    raw_path = consolidated_boxplots.get(f"{metric}_raw")
                    norm_path = consolidated_boxplots.get(f"{metric}_normalized")
                    
                    if raw_path:
                        f.write(f"![Boxplot {metric_display}](./boxplots/{os.path.basename(raw_path)})\n")
                    if norm_path:
                        f.write(f"![Boxplot Normalizado {metric_display}](./boxplots/{os.path.basename(norm_path)})\n")
                    f.write("\n")

            else:
                f.write("Boxplots consolidados não foram gerados nesta execução.\n\n")

            # --- Seção 4.2: Time Series Consolidados (v2.0) ---
            f.write("## 4.2. Time Series Consolidados\n\n")
            f.write("**Visualizações avançadas** que agregam a evolução temporal de todas as métricas através dos rounds, facilitando a identificação de padrões, tendências e divergências comportamentais.\n\n")
            
            consolidated_timeseries = all_results.get('consolidated_timeseries', {})
            if consolidated_timeseries:
                f.write("### Time Series por Métrica\n")
                f.write("Cada visualização inclui:\n")
                f.write("- **Evolução por Round**: Tendências agregadas entre todos os tenants\n")
                f.write("- **Evolução por Tenant**: Comportamento individual de cada tenant em todos os rounds\n")
                f.write("- **Tendências Suavizadas**: Médias móveis para identificar padrões de longo prazo\n")
                f.write("- **Distribuições por Fase**: Boxplots comparando fases experimentais\n\n")
                
                for metric, paths in consolidated_timeseries.items():
                    metric_display = metric.replace("_", " ").title()
                    f.write(f"#### {metric_display}\n")
                    if isinstance(paths, dict):
                        for plot_type, path in paths.items():
                            if path: # Garante que o caminho não é nulo
                                f.write(f"![{plot_type.replace('_', ' ').title()}](./timeseries/{os.path.basename(path)})\n")
                        f.write("\n")
                    elif isinstance(paths, str): # Fallback para o formato antigo
                         f.write(f"![Time Series Consolidado - {metric_display}](./timeseries/{os.path.basename(paths)})\n\n")

                f.write("**Interpretação**: \n")
                f.write("- **Convergência entre rounds** indica comportamento reproduzível\n")
                f.write("- **Divergências significativas** podem indicar efeitos de noisy neighbors\n")
                f.write("- **Padrões temporais consistentes** sugerem relações causais estáveis\n\n")
            else:
                f.write("Time series consolidados não foram gerados nesta execução.\n\n")

            # --- Seção 4.3: Gráficos de Correlação Agregada ---
            f.write("## 4.3. Gráficos de Correlação Agregada\n\n")
            f.write("Estes grafos mostram as correlações médias entre os tenants, agregadas através de todos os rounds e fases. As arestas representam a força da correlação (positiva ou negativa) entre os pares de tenants.\n\n")
            
            aggregated_correlation_graphs = all_results.get('aggregated_correlation_graphs', {})
            if aggregated_correlation_graphs:
                for metric, path in aggregated_correlation_graphs.items():
                    metric_display = metric.replace("_", " ").title()
                    f.write(f"### {metric_display}\n")
                    f.write(f"![Grafo de Correlação Agregada - {metric_display}](./correlation/{os.path.basename(path)})\n\n")
            else:
                f.write("Gráficos de correlação agregada não foram gerados nesta execução.\n\n")


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

        causality_output_dir = os.path.join(base_output_dir, 'plots', 'causality')
        
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


    def _extract_effect_sizes(self, context: Dict[str, Any], config: Dict[str, Any], df_long: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
        """
        Extrai tamanhos de efeito e estatísticas relacionadas para cada 
        combinação de métrica × fase × tenant × round.
        
        Args:
            context: Contexto da análise
            config: Configuração da análise
            df_long: DataFrame em formato longo
            rounds: Lista de rounds a analisar
            
        Returns:
            DataFrame com tamanhos de efeito e estatísticas relacionadas
        """
        self.logger.info("Extraindo tamanhos de efeito para todas as combinações...")
        
        # Obtém parâmetros de configuração
        multi_round_config = config.get('multi_round_analysis', {})
        effect_size_config = multi_round_config.get('effect_size', {})
        
        # Define fase de baseline
        baseline_phase = effect_size_config.get('baseline_phase', "1 - Baseline")
        
        # Verifica se a baseline existe nos dados
        available_phases = df_long['experimental_phase'].unique()
        if baseline_phase not in available_phases:
            self.logger.warning(f"Fase de baseline '{baseline_phase}' não encontrada nos dados.")
            available_phases_str = ", ".join(sorted(available_phases))
            self.logger.info(f"Fases disponíveis: {available_phases_str}")
            self.logger.info(f"Usando a primeira fase como baseline: {sorted(available_phases)[0]}")
            baseline_phase = sorted(available_phases)[0]
        
        # Obtém métricas, fases e tenants
        metrics = context.get('selected_metrics', sorted(df_long['metric_name'].unique()))
        phases = sorted(df_long['experimental_phase'].unique())
        tenants = context.get('selected_tenants', sorted(df_long['tenant_id'].unique()))
        
        # Configurações de cache e paralelismo
        perf_config = multi_round_config.get('performance', {})
        use_cache = perf_config.get('use_cache', True)
        parallel = perf_config.get('parallel_processing', False)
        cache_dir = os.path.join(self.output_dir, 'cache') if self.output_dir else None
        
        # Extrai tamanhos de efeito
        effect_sizes_df = extract_effect_sizes(
            df_long=df_long,
            rounds=rounds,
            metrics=metrics,
            phases=phases,
            tenants=tenants,
            baseline_phase=baseline_phase,
            use_cache=use_cache,
            parallel=parallel,
            cache_dir=cache_dir
        )
        
        if effect_sizes_df.empty:
            self.logger.warning("Nenhum tamanho de efeito extraído. Verifique os dados e parâmetros.")
        else:
            self.logger.info(f"Extração concluída: {effect_sizes_df.shape[0]} tamanhos de efeito calculados.")
            
            # Salva os resultados em CSV se houver diretório de saída
            if self.output_dir:
                effect_sizes_path = os.path.join(self.output_dir, 'effect_sizes.csv')
                effect_sizes_df.to_csv(effect_sizes_path, index=False)
                self.logger.info(f"Tamanhos de efeito salvos em: {effect_sizes_path}")
        
        return effect_sizes_df

    def _aggregate_effect_sizes(self, effect_sizes_df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """
        Agrega os tamanhos de efeito por métrica × fase × tenant através dos rounds,
        calculando médias, desvios padrão, intervalos de confiança e p-valores combinados.
        
        Args:
            effect_sizes_df: DataFrame com tamanhos de efeito por round
            config: Configuração da análise
            
        Returns:
            DataFrame com estatísticas agregadas
        """
        self.logger.info("Agregando tamanhos de efeito por métrica, fase e tenant...")
        
        if effect_sizes_df.empty:
            self.logger.warning("DataFrame de tamanhos de efeito vazio. Pulando agregação.")
            return pd.DataFrame()
        
        # Obtém parâmetros de configuração
        multi_round_config = config.get('multi_round_analysis', {})
        meta_config = multi_round_config.get('meta_analysis', {})
        
        # Define parâmetros para agregação
        alpha = meta_config.get('alpha', 0.05)
        p_value_method = meta_config.get('p_value_combination', 'fisher')
        confidence_level = meta_config.get('confidence_level', 0.95)
        use_bootstrap = meta_config.get('use_bootstrap', True)
        n_bootstrap = meta_config.get('n_bootstrap', 1000)
        
        # Realiza agregação
        aggregated_df = aggregate_effect_sizes(
            effect_sizes_df=effect_sizes_df,
            alpha=alpha,
            p_value_method=p_value_method,
            confidence_level=confidence_level,
            use_bootstrap=use_bootstrap,
            n_bootstrap=n_bootstrap
        )
        
        if aggregated_df.empty:
            self.logger.warning("Nenhuma estatística agregada calculada.")
        else:
            self.logger.info(f"Agregação concluída: {aggregated_df.shape[0]} estatísticas agregadas.")
            
            # Salva os resultados em CSV se houver diretório de saída
            if self.output_dir:
                aggregated_path = os.path.join(self.output_dir, 'aggregated_effects.csv')
                aggregated_df.to_csv(aggregated_path, index=False)
                self.logger.info(f"Estatísticas agregadas salvas em: {aggregated_path}")
                
                # Gera um resumo das estatísticas
                summary_path = os.path.join(self.output_dir, 'effect_size_summary.md')
                with open(summary_path, 'w') as f:
                    f.write("# Resumo da Análise de Tamanhos de Efeito\n\n")
                    f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    
                    # Resumo de estatísticas gerais
                    f.write("## Estatísticas Gerais\n\n")
                    f.write(f"- Total de combinações analisadas: {aggregated_df.shape[0]}\n")
                    f.write(f"- Efeitos estatisticamente significativos: {aggregated_df['is_significant'].sum()}\n")
                    
                    # Resumo por magnitude de efeito
                    magnitude_counts = aggregated_df['effect_magnitude'].value_counts()
                    f.write("\n### Distribuição por Magnitude de Efeito\n\n")
                    for magnitude, count in magnitude_counts.items():
                        f.write(f"- {str(magnitude).title()}: {count}\n")
                    
                    # Resumo por confiabilidade
                    reliability_counts = aggregated_df['reliability_category'].value_counts()
                    f.write("\n### Distribuição por Confiabilidade\n\n")
                    for reliability, count in reliability_counts.items():
                        f.write(f"- {str(reliability).title()}: {count}\n")
                    
                    # Top 5 efeitos mais fortes e significativos
                    significant_df = aggregated_df[aggregated_df['is_significant']]
                    if not significant_df.empty:
                        top_effects = significant_df.sort_values('mean_effect_size', key=abs, ascending=False).head(5)
                        f.write("\n## Top 5 Efeitos Mais Fortes (Significativos)\n\n")
                        for _, row in top_effects.iterrows():
                            f.write(f"- **{row['experimental_phase']} em {row['tenant_id']} (métrica: {row['metric_name']})**\n")
                            f.write(f"  - Tamanho de efeito: {row['mean_effect_size']:.3f} (IC95%: {row['ci_lower']:.3f} a {row['ci_upper']:.3f})\n")
                            f.write(f"  - p-valor combinado: {row['combined_p_value']:.6f} ({row['rounds_count']} rounds)\n")
                            f.write(f"  - Confiabilidade: {row['reliability_category'].title()}\n")
        
        return aggregated_df

    def _extract_phase_correlations(self, context: Dict[str, Any], config: Dict[str, Any], df_long: pd.DataFrame, rounds: List[str]) -> pd.DataFrame:
        """
        Extrai as correlações intra-fase entre tenants para cada 
        combinação de métrica × fase × round.
        
        Args:
            context: Contexto da análise
            config: Configuração da análise
            df_long: DataFrame em formato longo
            rounds: Lista de rounds a analisar
            
        Returns:
            DataFrame com correlações intra-fase
        """
        self.logger.info("Extraindo correlações intra-fase para todas as combinações...")
        
        # Obtém parâmetros de configuração
        multi_round_config = config.get('multi_round_analysis', {})
        correlation_config = multi_round_config.get('correlation', {})
        
        # Define parâmetros para correlação
        method = correlation_config.get('method', 'pearson')
        min_periods = correlation_config.get('min_periods', 3)
        
        # Obtém métricas, fases e tenants
        metrics = context.get('selected_metrics', sorted(df_long['metric_name'].unique()))
        phases = sorted(df_long['experimental_phase'].unique())
        tenants = context.get('selected_tenants', sorted(df_long['tenant_id'].unique()))
        
        # Configurações de cache e paralelismo
        perf_config = multi_round_config.get('performance', {})
        use_cache = perf_config.get('use_cache', True)
        parallel = perf_config.get('parallel_processing', False)
        cache_dir = os.path.join(self.output_dir, 'cache') if self.output_dir else None
        
        # Extrai correlações intra-fase
        correlations_df = extract_phase_correlations(
            df_long=df_long,
            rounds=rounds,
            metrics=metrics,
            phases=phases,
            tenants=tenants,
            method=method,
            min_periods=min_periods,
            use_cache=use_cache,
            parallel=parallel,
            cache_dir=cache_dir
        )
        
        if correlations_df.empty:
            self.logger.warning("Nenhuma correlação intra-fase extraída. Verifique os dados e parâmetros.")
        else:
            self.logger.info(f"Extração concluída: {correlations_df.shape[0]} correlações intra-fase calculadas.")
            
            # Salva os resultados em CSV se houver diretório de saída
            if self.output_dir:
                correlations_path = os.path.join(self.output_dir, 'phase_correlations.csv')
                correlations_df.to_csv(correlations_path, index=False)
                self.logger.info(f"Correlações intra-fase salvas em: {correlations_path}")
                
                # Analisa a estabilidade das correlações entre rounds
                min_rounds = correlation_config.get('min_stable_rounds', 2)
                correlation_threshold = correlation_config.get('significance_threshold', 0.5)
                
                stability_results = analyze_correlation_stability(
                    phase_correlations_df=correlations_df,
                    min_rounds=min_rounds,
                    correlation_threshold=correlation_threshold
                )
                
                if stability_results:
                    # Gera um resumo da estabilidade das correlações
                    stability_path = os.path.join(self.output_dir, 'correlation_stability_summary.md')
                    with open(stability_path, 'w') as f:
                        f.write("# Resumo da Estabilidade das Correlações Intra-Fase\n\n")
                        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                        
                        # Resumo de estatísticas gerais
                        f.write("## Estatísticas Gerais\n\n")
                        summary = stability_results.get('summary', {})
                        f.write(f"- Total de pares analisados: {summary.get('total_pairs', 0)}\n")
                        f.write(f"- Pares com correlação estável: {summary.get('stable_pairs', 0)}\n")
                        f.write(f"- Pares com correlação instável: {summary.get('unstable_pairs', 0)}\n")
                        
                        # Correlações estáveis por métrica e fase
                        stable_correlations = stability_results.get('stable_correlations', {})
                        if stable_correlations:
                            f.write("\n## Correlações Estáveis por Métrica e Fase\n\n")
                            for (metric, phase), correlations in stable_correlations.items():
                                f.write(f"### {metric} - {phase}\n\n")
                                # Ordena por correlação média (valor absoluto) decrescente
                                correlations.sort(key=lambda x: abs(x['mean_correlation']), reverse=True)
                                for corr_info in correlations:
                                    tenant_pair = corr_info['tenant_pair']
                                    mean_corr = corr_info['mean_correlation']
                                    std_corr = corr_info['std_correlation']
                                    variability = corr_info['variability']
                                    rounds_count = corr_info['round_count']
                                    
                                    f.write(f"- **{tenant_pair}**: {mean_corr:.3f} ± {std_corr:.3f} ({variability} variability, {rounds_count} rounds)\n")
                                f.write("\n")
                        
                        # Variabilidade por métrica e fase
                        variability = stability_results.get('correlation_variability', [])
                        if variability:
                            f.write("\n## Variabilidade da Correlação por Métrica e Fase\n\n")
                            f.write("| Métrica | Fase | Desvio Padrão Médio | CV Médio | % Pares Estáveis |\n")
                            f.write("|---------|------|---------------------|----------|------------------|\n")
                            for var_info in variability:
                                metric = var_info['metric_name']
                                phase = var_info['experimental_phase']
                                mean_std = var_info['mean_std']
                                mean_cv = var_info['mean_cv']
                                stable_ratio = var_info['stable_ratio'] * 100
                                
                                f.write(f"| {metric} | {phase} | {mean_std:.3f} | {mean_cv:.3f} | {stable_ratio:.1f}% |\n")
                    
                    self.logger.info(f"Resumo de estabilidade das correlações salvo em: {stability_path}")
        
        return correlations_df

def analyze_round_consistency(df_long: pd.DataFrame, metrics: List[str], tenants: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Analisa a consistência das métricas entre os rounds usando o Coeficiente de Variação (CV)
    e testes estatísticos de Friedman para diferenças entre rounds.
    
    Args:
        df_long: DataFrame em formato longo
        metrics: Lista de métricas para análise
        tenants: Lista de tenants para análise
        output_dir: Diretório para salvar os resultados
        
    Returns:
        Dict[str, Any]: Dicionário com resultados da análise
    """
    logger.info("Analisando a consistência das métricas entre os rounds...")
    
    results = {
        'cv_by_metric_tenant': {},
        'friedman_tests': {},
        'round_outliers': {}
    }
    
    # Verificar se há mais de um round para análise
    rounds = sorted(df_long['round_id'].unique())
    if len(rounds) < 2:
        logger.warning("Pelo menos dois rounds são necessários para análise de consistência. Pulando.")
        return results
    
    # Calcular CV para cada combinação de métrica x tenant x fase
    cv_results = []
    
    for metric in metrics:
        for tenant in tenants:
            df_filtered = df_long[(df_long['metric_name'] == metric) & 
                                 (df_long['tenant_id'] == tenant)]
            
            if df_filtered.empty:
                continue
            
            for phase in df_filtered['experimental_phase'].unique():
                phase_data = df_filtered[df_filtered['experimental_phase'] == phase]
                
                # Agregar por round para obter um valor médio por round
                agg_by_round = phase_data.groupby('round_id')['metric_value'].mean()
                
                if len(agg_by_round) > 1:  # Só podemos calcular CV se tivermos mais de um valor
                    mean_val = agg_by_round.mean()
                    std_val = agg_by_round.std()
                    cv = (std_val / mean_val) if mean_val != 0 else float('inf')
                    
                    cv_results.append({
                        'metric_name': metric,
                        'tenant_id': tenant,
                        'experimental_phase': phase,
                        'mean': mean_val,
                        'std': std_val,
                        'cv': cv,
                        'rounds_count': len(agg_by_round),
                        'rounds': ','.join(agg_by_round.index.tolist())
                    })
    
    # Converter para DataFrame
    if cv_results:
        cv_df = pd.DataFrame(cv_results)
        results['cv_by_metric_tenant'] = cv_df
        
        # Salvar resultados
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            cv_path = os.path.join(output_dir, 'round_consistency_cv.csv')
            cv_df.to_csv(cv_path, index=False)
            logger.info(f"Resultados de CV salvos em: {cv_path}")
            
            # Gerar visualização de heatmap para CV
            try:
                # Preparar dados para heatmap
                pivot_df = cv_df.pivot_table(
                    index='tenant_id', 
                    columns='metric_name', 
                    values='cv', 
                    aggfunc='mean'
                )
                
                # Criar diretório para visualizações
                os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
                
                # Gerar heatmap
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    pivot_df, 
                    annot=True, 
                    fmt=".2f", 
                    cmap='viridis_r',  # Invertido para que valores baixos (mais consistentes) sejam verde escuro
                    linewidths=0.5
                )
                plt.title('Coeficiente de Variação (CV) por Tenant e Métrica\nValores mais baixos indicam maior consistência entre rounds')
                plt.tight_layout()
                
                heatmap_path = os.path.join(output_dir, 'cv_heatmap_by_tenant_metric.png')
                plt.savefig(heatmap_path, dpi=300)
                plt.close()
                
                results['heatmap_path'] = heatmap_path
                logger.info(f"Heatmap de CV salvo em: {heatmap_path}")
            
            except Exception as e:
                logger.error(f"Erro ao gerar heatmap de CV: {str(e)}")
    
    # Realizar testes de Friedman para cada métrica e tenant
    # Apenas se tivermos pelo menos 3 rounds (requisito do teste de Friedman)
    if len(rounds) >= 3:
        friedman_results = []
        
        for metric in metrics:
            for tenant in tenants:
                # Preparar dados para o teste de Friedman
                # Precisa de uma matriz com fases nas linhas e rounds nas colunas
                phases_data = {}
                
                for phase in df_long['experimental_phase'].unique():
                    phase_by_round = {}
                    
                    for round_id in rounds:
                        data = df_long[
                            (df_long['metric_name'] == metric) & 
                            (df_long['tenant_id'] == tenant) &
                            (df_long['experimental_phase'] == phase) &
                            (df_long['round_id'] == round_id)
                        ]
                        
                        if not data.empty:
                            phase_by_round[round_id] = data['metric_value'].mean()
                    
                    if len(phase_by_round) == len(rounds):  # Temos dados para todos os rounds
                        phases_data[phase] = [phase_by_round[r] for r in rounds]
                
                if phases_data:
                    try:
                        phases_df = pd.DataFrame(phases_data).T  # Transpor para ter fases nas linhas
                        
                        # Realizar teste de Friedman
                        friedman_stat, p_value = stats.friedmanchisquare(*[phases_df[col] for col in phases_df.columns])
                        
                        friedman_results.append({
                            'metric_name': metric,
                            'tenant_id': tenant,
                            'friedman_statistic': friedman_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05,
                            'phases_count': len(phases_data)
                        })
                    except Exception as e:
                        logger.warning(f"Erro no teste de Friedman para {metric}, {tenant}: {str(e)}")
        
        if friedman_results:
            friedman_df = pd.DataFrame(friedman_results)
            results['friedman_tests'] = friedman_df
            
            # Salvar resultados
            if output_dir:
                friedman_path = os.path.join(output_dir, 'friedman_tests.csv')
                friedman_df.to_csv(friedman_path, index=False)
                logger.info(f"Resultados dos testes de Friedman salvos em: {friedman_path}")
    
    # Detectar rounds outliers com base nos CVs
    if 'cv_by_metric_tenant' in results and not results['cv_by_metric_tenant'].empty:
        cv_df = results['cv_by_metric_tenant']
        
        # Para cada métrica e fase, identificar quais rounds são outliers
        outliers_by_metric = {}
        
        for metric in metrics:
            metric_cv = cv_df[cv_df['metric_name'] == metric]
            if metric_cv.empty:
                continue
                
            # Calcular média global do CV para esta métrica
            mean_cv = metric_cv['cv'].mean()
            std_cv = metric_cv['cv'].std()
            
            for phase in metric_cv['experimental_phase'].unique():
                phase_cv = metric_cv[metric_cv['experimental_phase'] == phase]
                
                # Identificar valores de CV muito altos (outliers)
                outlier_threshold = mean_cv + 2 * std_cv  # 2 desvios padrão acima da média
                
                outliers = phase_cv[phase_cv['cv'] > outlier_threshold]
                if not outliers.empty:
                    for _, row in outliers.iterrows():
                        metric_key = f"{metric}_{row['tenant_id']}"
                        if metric_key not in outliers_by_metric:
                            outliers_by_metric[metric_key] = []
                            
                        outliers_by_metric[metric_key].append({
                            'phase': row['experimental_phase'],
                            'cv': row['cv'],
                            'threshold': outlier_threshold,
                            'rounds': row['rounds']
                        })
        
        results['round_outliers'] = outliers_by_metric
    
    logger.info("Análise de consistência entre rounds concluída.")
    return results

def analyze_behavioral_divergence(df_long: pd.DataFrame, metrics: List[str], tenants: List[str], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Analisa a divergência comportamental entre rounds usando a Divergência de Kullback-Leibler (KL).
    Identifica rounds com comportamento anômalo e mede a estabilidade comportamental dos tenants.
    
    Args:
        df_long: DataFrame em formato longo
        metrics: Lista de métricas para análise
        tenants: Lista de tenants para análise
        output_dir: Diretório para salvar os resultados
        
    Returns:
        Dict[str, Any]: Dicionário com resultados da análise
    """
    logger.info("Analisando a divergência comportamental entre rounds...")
    
    results = {
        'kl_divergence': [],
        'round_distances': {},
        'tenant_stability_scores': []
    }
    
    # Verificar se há mais de um round para análise
    rounds = sorted(df_long['round_id'].unique())
    if len(rounds) < 2:
        logger.warning("Pelo menos dois rounds são necessários para análise de divergência. Pulando.")
        return results
    
    # Para cada métrica e tenant, calcular divergência KL entre distribuições por round
    for metric in metrics:
        metric_results = {'metric_name': metric, 'tenant_divergences': {}}
        
        for tenant in tenants:
            tenant_data = df_long[(df_long['metric_name'] == metric) & 
                                 (df_long['tenant_id'] == tenant)]
            
            if tenant_data.empty:
                continue
            
            round_distributions = {}
            
            # Construir distribuições empíricas para cada round
            for round_id in rounds:
                round_data = tenant_data[tenant_data['round_id'] == round_id]
                
                if round_data.empty:
                    continue
                
                # Usar histograma para estimar a distribuição
                hist, bin_edges = np.histogram(round_data['metric_value'], bins=20, density=True)
                
                # Suavizar zeros para evitar divergência infinita
                hist = np.where(hist == 0, 1e-10, hist)
                # Normalizar para garantir que soma a 1
                hist = hist / np.sum(hist)
                
                round_distributions[round_id] = hist
            
            # Calcular matriz de divergência KL entre todos os pares de rounds
            if len(round_distributions) >= 2:
                divergence_matrix = np.zeros((len(round_distributions), len(round_distributions)))
                round_ids = list(round_distributions.keys())
                
                for i, round1 in enumerate(round_ids):
                    for j, round2 in enumerate(round_ids):
                        if i == j:
                            divergence_matrix[i, j] = 0
                        else:
                            # Calcular divergência KL simétrica
                            p = round_distributions[round1]
                            q = round_distributions[round2]
                            
                            # Calcular KL em ambas direções e tomar a média
                            # KL(P||Q) = sum_i P_i * log(P_i/Q_i)
                            kl_pq = np.sum(p * np.log(p / q))
                            kl_qp = np.sum(q * np.log(q / p))
                            
                            # Divergência simétrica
                            sym_kl = (kl_pq + kl_qp) / 2
                            divergence_matrix[i, j] = sym_kl
                
                # Armazenar resultados
                metric_results['tenant_divergences'][tenant] = {
                    'round_ids': round_ids,
                    'divergence_matrix': divergence_matrix.tolist()
                }
                
                # Calcular estabilidade do tenant com base na divergência média
                avg_divergence = np.mean(divergence_matrix[np.triu_indices_from(divergence_matrix, k=1)])
                
                results['kl_divergence'].append({
                    'metric_name': metric,
                    'tenant_id': tenant,
                    'mean_divergence': avg_divergence,
                    'stability_score': 1 / (1 + avg_divergence)  # Converter divergência para score de estabilidade [0,1]
                })
        
        results['round_distances'][metric] = metric_results
    
    # Calcular scores de estabilidade por tenant
    if results['kl_divergence']:
        stability_df = pd.DataFrame(results['kl_divergence'])
        
        # Calcular estabilidade média por tenant entre todas as métricas
        tenant_stability = stability_df.groupby('tenant_id')['stability_score'].mean().reset_index()
        tenant_stability = tenant_stability.sort_values('stability_score', ascending=False)
        
        results['tenant_stability_scores'] = tenant_stability.to_dict('records')
        
        # Salvar resultados
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Salvar scores de estabilidade
            stability_path = os.path.join(output_dir, 'tenant_stability_scores.csv')
            tenant_stability.to_csv(stability_path, index=False)
            logger.info(f"Scores de estabilidade de tenant salvos em: {stability_path}")
            
            # Salvar divergências KL
            kl_path = os.path.join(output_dir, 'kl_divergence.csv')
            pd.DataFrame(results['kl_divergence']).to_csv(kl_path, index=False)
            logger.info(f"Divergências KL salvas em: {kl_path}")
            
            # Gerar visualização de barras para estabilidade de tenant
            try:
                plt.figure(figsize=(12, 6))
                bars = plt.bar(tenant_stability['tenant_id'], tenant_stability['stability_score'], alpha=0.7)
                
                # Adicionar valores nas barras
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.01,
                        f"{height:.2f}",
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )
                
                plt.title('Scores de Estabilidade por Tenant\nValores mais altos indicam comportamento mais estável entre rounds')
                plt.ylim(0, max(tenant_stability['stability_score']) * 1.1)  # Ajustar limites do eixo y
                plt.ylabel('Score de Estabilidade')
                plt.tight_layout()
                
                stability_plot_path = os.path.join(output_dir, 'tenant_stability_scores.png')
                plt.savefig(stability_plot_path, dpi=300)
                plt.close()
                
                results['stability_plot_path'] = stability_plot_path
                logger.info(f"Plot de estabilidade de tenant salvo em: {stability_plot_path}")
            
            except Exception as e:
                logger.error(f"Erro ao gerar plot de estabilidade de tenant: {str(e)}")
    
    logger.info("Análise de divergência comportamental concluída.")
    return results

def aggregate_round_consensus(df_long: pd.DataFrame, te_matrices_by_round: Dict[str, Dict[str, pd.DataFrame]], consistency_results: Dict[str, Any], output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Agrega os resultados de múltiplos rounds para gerar um consenso final.
    
    Args:
        df_long: DataFrame em formato longo
        te_matrices_by_round: Dicionário de matrizes de Transfer Entropy por round
        consistency_results: Resultados da análise de consistência
        output_dir: Diretório para salvar os resultados
        
    Returns:
        Dict[str, Any]: Dicionário com resultados do consenso
    """
    logger.info("Agregando consenso entre rounds...")
    
    results = {
        'noisy_tenants_consensus': [],
        'tenant_influence_ranking': []
    }
    
    # Verificar se temos matrizes de TE suficientes
    rounds = list(te_matrices_by_round.keys())
    if len(rounds) < 2:
        logger.warning("Pelo menos dois rounds com matrizes TE são necessários para agregação de consenso. Pulando.")
        return results
    
    # Métricas disponíveis (interseção de todas as matrizes)
    available_metrics = set(te_matrices_by_round[rounds[0]].keys())
    for round_id in rounds[1:]:
        available_metrics &= set(te_matrices_by_round[round_id].keys())
    
    if not available_metrics:
        logger.warning("Nenhuma métrica comum encontrada em todas as matrizes TE. Pulando agregação de consenso.")
        return results
    
    # Para cada métrica, agregar matrizes TE de todos os rounds
    aggregated_te = {}
    
    for metric in available_metrics:
        # Verificar se todas as matrizes têm as mesmas dimensões e tenants
        matrices = [te_matrices_by_round[r][metric] for r in rounds]
        
        # Verificar se todas as matrizes têm os mesmos tenants
        tenant_sets = [set(m.index) for m in matrices]
        common_tenants = tenant_sets[0]
        for tenant_set in tenant_sets[1:]:
            common_tenants &= tenant_set
        
        if not common_tenants:
            logger.warning(f"Nenhum tenant comum encontrado para a métrica {metric}. Pulando.")
            continue
        
        # Converter para lista ordenada para consistência
        common_tenants = sorted(common_tenants)
        
        # Inicializar matriz agregada
        aggregated_matrix = pd.DataFrame(0, index=common_tenants, columns=common_tenants)
        
        # Calcular média ponderada baseada na consistência (se disponível)
        if consistency_results and 'cv_by_metric_tenant' in consistency_results:
            # Tentar obter pesos baseados no CV (menor CV = maior peso)
            try:
                cv_df = consistency_results['cv_by_metric_tenant']
                if isinstance(cv_df, pd.DataFrame) and not cv_df.empty and 'cv' in cv_df.columns:
                    # Filtrar para a métrica atual
                    metric_cv = cv_df[cv_df['metric_name'] == metric]
                    
                    if not metric_cv.empty:
                        # Calcular pesos por round (inverso do CV médio)
                        round_weights = {}
                        for round_id in rounds:
                            round_data = df_long[df_long['round_id'] == round_id]
                            if round_data.empty:
                                round_weights[round_id] = 1.0  # Peso padrão
                                continue
                            
                            # Filtrar dados para a métrica atual
                            round_metric_data = round_data[round_data['metric_name'] == metric]
                            if round_metric_data.empty:
                                round_weights[round_id] = 1.0  # Peso padrão
                                continue
                            
                            # Calcular CV médio para este round e métrica
                            tenants = round_metric_data['tenant_id'].unique()
                            tenant_cvs = []
                            
                            for tenant in tenants:
                                tenant_rows = metric_cv[metric_cv['tenant_id'] == tenant]
                                if not tenant_rows.empty:
                                    tenant_cvs.append(tenant_rows['cv'].mean())
                            
                            if tenant_cvs:
                                mean_cv = np.mean(tenant_cvs)
                                # Peso = 1 / (1 + CV) para dar mais peso a rounds com menor variabilidade
                                round_weights[round_id] = 1.0 / (1.0 + mean_cv)
                            else:
                                round_weights[round_id] = 1.0  # Peso padrão
                        
                        # Normalizar pesos para somar 1
                        total_weight = sum(round_weights.values())
                        if total_weight > 0:
                            round_weights = {r: w / total_weight for r, w in round_weights.items()}
                        
                        # Aplicar pesos na agregação
                        for i, round_id in enumerate(rounds):
                            weight = round_weights.get(round_id, 1.0 / len(rounds))
                            matrix = matrices[i]
                            # Filtrar para os tenants comuns
                            filtered_matrix = matrix.loc[common_tenants, common_tenants]
                            aggregated_matrix += filtered_matrix * weight
                    else:
                        # Se não houver dados de CV, usar média simples
                        for matrix in matrices:
                            filtered_matrix = matrix.loc[common_tenants, common_tenants]
                            aggregated_matrix += filtered_matrix / len(matrices)
            except Exception as e:
                logger.error(f"Erro ao calcular pesos para agregação de TE: {str(e)}")
                # Fallback para média simples
                for matrix in matrices:
                    filtered_matrix = matrix.loc[common_tenants, common_tenants]
                    aggregated_matrix += filtered_matrix / len(matrices)
        else:
            # Se não houver dados de CV, usar média simples
            for matrix in matrices:
                filtered_matrix = matrix.loc[common_tenants, common_tenants]
                aggregated_matrix += filtered_matrix / len(matrices)
        
        aggregated_te[metric] = aggregated_matrix
    
        # Identificar tenants "barulhentos" (causam impacto em outros)
        noisy_tenants = []
        
        # Calcular "outflow causal" (soma da força causal que sai de cada tenant)
        for tenant in common_tenants:
            outflow = aggregated_matrix.loc[tenant].sum() - aggregated_matrix.loc[tenant, tenant]
            
            # Calcular estatísticas para o ranking
            mean_te = outflow / (len(common_tenants) - 1) if len(common_tenants) > 1 else 0
            
            # Considerar como "barulhento" se o valor médio de TE for maior que um limite
            te_threshold = 0.1  # Ajustar conforme necessário
            
            if mean_te > te_threshold:
                noisy_tenants.append({
                    'metric_name': metric,
                    'tenant_id': tenant,
                    'outflow_causality': float(outflow),
                    'mean_te': float(mean_te),
                    'affected_tenants': len([t for t in common_tenants if aggregated_matrix.loc[tenant, t] > te_threshold and t != tenant])
                })
            
            # Adicionar ao ranking geral
            results['tenant_influence_ranking'].append({
                'metric_name': metric,
                'tenant_id': tenant,
                'influence_score': float(outflow),
                'mean_outgoing_te': float(mean_te)
            })
        
        # Adicionar tenants barulhentos ao consenso
        noisy_tenants.sort(key=lambda x: x['outflow_causality'], reverse=True)
        results['noisy_tenants_consensus'].extend(noisy_tenants)
    
    # Salvar resultados do consenso
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Salvar matrizes TE agregadas
        te_dir = os.path.join(output_dir, 'aggregated_te')
        os.makedirs(te_dir, exist_ok=True)
        
        for metric, matrix in aggregated_te.items():
            matrix_path = os.path.join(te_dir, f'aggregated_te_matrix_{metric}.csv')
            matrix.to_csv(matrix_path)
            logger.info(f"Matriz TE agregada para {metric} salva em: {matrix_path}")
        
        # Salvar ranking de influência
        if results['tenant_influence_ranking']:
            ranking_df = pd.DataFrame(results['tenant_influence_ranking'])
            ranking_path = os.path.join(output_dir, 'tenant_influence_ranking.csv')
            ranking_df.to_csv(ranking_path, index=False)
            logger.info(f"Ranking de influência de tenants salvo em: {ranking_path}")
        
        # Salvar consenso de tenants barulhentos
        if results['noisy_tenants_consensus']:
            consensus_df = pd.DataFrame(results['noisy_tenants_consensus'])
            consensus_path = os.path.join(output_dir, 'consensus_noisy_tenants.csv')
            consensus_df.to_csv(consensus_path, index=False)
            logger.info(f"Consenso de tenants barulhentos salvo em: {consensus_path}")
    
    logger.info(f"Agregação de consenso concluída. {len(results['noisy_tenants_consensus'])} tenants barulhentos identificados.")
    return results
