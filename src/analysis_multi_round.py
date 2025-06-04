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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx  # Adicionando importação do NetworkX
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from datetime import datetime

from src.pipeline import PipelineStage

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
    
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementação da análise de múltiplos rounds.
        
        Args:
            context: Contexto atual do pipeline com resultados anteriores.
            
        Returns:
            Contexto atualizado com resultados da análise multi-round.
        """
        self.logger.info("Iniciando análise de experimentos com múltiplos rounds")
        
        if 'error' in context:
            self.logger.error(f"Erro em estágio anterior: {context['error']}")
            return context
        
        # Verificar se temos múltiplos rounds para analisar
        df_long = context.get('df_long')
        if df_long is None:
            self.logger.warning("DataFrame principal não encontrado no contexto")
            context['error'] = "DataFrame principal não disponível para análise multi-round"
            return context
        
        rounds = df_long['round_id'].unique()
        if len(rounds) <= 1:
            self.logger.info("Apenas um round encontrado. Pulando análise multi-round.")
            context['multi_round_analysis'] = {
                'status': 'skipped',
                'reason': 'O dataset contém apenas um round de experimento'
            }
            return context
        
        # Diretório de saída
        output_dir = self.output_dir or context.get('output_dir')
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), 'outputs', 'multi_round')
        else:
            output_dir = os.path.join(output_dir, 'multi_round')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extrair dados necessários do contexto
        experiment_id = context.get('experiment_id', df_long['experiment_id'].iloc[0])
        phases = sorted(df_long['experimental_phase'].unique())
        metrics = context.get('selected_metrics', df_long['metric_name'].unique())
        tenants = context.get('selected_tenants', df_long['tenant_id'].unique())
        
        try:
            results = {}
            
            # 1. Análise de consistência entre rounds
            self.logger.info("Realizando análise de consistência entre rounds")
            consistency_results = analyze_round_consistency(
                df_long=df_long,
                metrics=metrics,
                tenants=tenants,
                output_dir=output_dir
            )
            results['consistency'] = consistency_results
            
            # 2. Análise de robustez de causalidade
            self.logger.info("Realizando análise de robustez de causalidade")
            # Verificar se temos as matrizes de causalidade de todos os rounds
            te_matrices_by_round = {}
            granger_matrices_by_round = {}
            
            # Coletar matrizes de TE e Granger por round do contexto
            for round_id in rounds:
                te_key = f'te_matrices_round_{round_id}'
                granger_key = f'granger_matrices_round_{round_id}'
                
                if te_key in context:
                    te_matrices_by_round[round_id] = context[te_key]
                if granger_key in context:
                    granger_matrices_by_round[round_id] = context[granger_key]
            
            if te_matrices_by_round:
                causality_robustness = analyze_causality_robustness(
                    te_matrices_by_round=te_matrices_by_round,
                    granger_matrices_by_round=granger_matrices_by_round,
                    output_dir=output_dir
                )
                results['causality_robustness'] = causality_robustness
            else:
                self.logger.warning("Não foram encontradas matrizes de causalidade por round no contexto")
            
            # 3. Análise de divergência comportamental
            self.logger.info("Realizando análise de divergência comportamental entre rounds")
            divergence_results = analyze_behavioral_divergence(
                df_long=df_long,
                metrics=metrics,
                tenants=tenants,
                output_dir=output_dir
            )
            results['divergence'] = divergence_results
            
            # 4. Agregação de consenso
            self.logger.info("Realizando agregação de consenso entre rounds")
            consensus_results = aggregate_round_consensus(
                df_long=df_long,
                te_matrices_by_round=te_matrices_by_round,
                consistency_results=consistency_results,
                output_dir=output_dir
            )
            results['consensus'] = consensus_results
            
            # 5. Visualizações de consistência
            self.logger.info("Gerando visualizações de consistência entre rounds")
            visualization_paths = generate_round_consistency_visualizations(
                df_long=df_long,
                consistency_results=consistency_results,
                causality_robustness=results.get('causality_robustness', {}),
                output_dir=output_dir
            )
            results['visualization_paths'] = visualization_paths
            
            # Armazenar resultados no contexto
            context['multi_round_analysis'] = results
            context['multi_round_analysis_dir'] = output_dir
            
            self.logger.info(f"Análise multi-round concluída. Resultados salvos em {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Erro durante análise multi-round: {str(e)}", exc_info=True)
            context['error'] = f"Erro na análise multi-round: {str(e)}"
        
        return context


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
        
        # Determinar consenso para "noisy tenants" (mais de 50% dos rounds)
        min_votes = len(rounds) / 2
        consensus_noisy_tenants = {
            tenant: votes for tenant, votes in noisy_tenant_votes.items() 
            if votes > min_votes
        }
        
        # Determinar consenso para relações causais (mais de 50% dos rounds)
        consensus_causal_relationships = {
            relationship: votes for relationship, votes in causal_relationship_votes.items()
            if votes > min_votes
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
    gráficos com intervalos de confiança, heatmaps, dendrogramas.
    
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
        plt.figure(figsize=(12, 8))
        
        # Preparar dados para plot
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
            
            # Criar heatmap de CV
            pivot_table = cv_df.pivot_table(
                index='tenant', 
                columns='metric', 
                values='cv',
                aggfunc='mean'
            )
            
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
            
            # Salvar figura
            cv_heatmap_path = os.path.join(output_dir, 'cv_heatmap_by_tenant_metric.png')
            plt.savefig(cv_heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(cv_heatmap_path)
    
    # 2. Gráficos de intervalo de confiança para métricas importantes
    metrics = df_long['metric_name'].unique()
    tenants = df_long['tenant_id'].unique()
    rounds = sorted(df_long['round_id'].unique())
    
    for metric in metrics[:5]:  # Limitar a 5 métricas para não gerar muitos gráficos
        try:
            plt.figure(figsize=(12, 8))
            
            for tenant in tenants[:8]:  # Limitar a 8 tenants por gráfico
                tenant_data = []
                confidence_intervals = []
                
                for round_id in rounds:
                    # Filtrar dados
                    data = df_long[
                        (df_long['metric_name'] == metric) &
                        (df_long['tenant_id'] == tenant) &
                        (df_long['round_id'] == round_id)
                    ]['metric_value']
                    
                    if not data.empty:
                        mean_val = data.mean()
                        std_val = data.std()
                        tenant_data.append(mean_val)
                        
                        # Calcular intervalo de confiança de 95%
                        n = len(data)
                        ci = 1.96 * std_val / np.sqrt(n)
                        confidence_intervals.append([mean_val - ci, mean_val + ci])
                    else:
                        tenant_data.append(np.nan)
                        confidence_intervals.append([np.nan, np.nan])
                
                # Plotar linha com média
                plt.plot(rounds, tenant_data, marker='o', label=tenant)
                
                # Adicionar intervalos de confiança
                for i, (lower, upper) in enumerate(confidence_intervals):
                    if not np.isnan(lower) and not np.isnan(upper):
                        plt.fill_between([rounds[i]], [lower], [upper], alpha=0.2)
            
            plt.title(f'Comparação entre Rounds: {metric}')
            plt.xlabel('Round')
            plt.ylabel('Valor Médio')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            
            # Salvar figura
            ci_plot_path = os.path.join(output_dir, f'confidence_interval_{metric}.png')
            plt.savefig(ci_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            visualization_paths.append(ci_plot_path)
            
        except Exception as e:
            logger.warning(f"Erro ao gerar visualização de IC para {metric}: {str(e)}")
    
    # 3. Visualização de relações causais robustas
    if causality_robustness and 'robust_causal_relationships' in causality_robustness:
        for metric, relationships in causality_robustness['robust_causal_relationships'].items():
            if not relationships:
                continue
                
            try:
                plt.figure(figsize=(10, 8))
                
                # Criar grafo direcionado
                G = nx.DiGraph()
                
                # Adicionar nós e edges
                nodes = set()
                for (source, target), data in relationships.items():
                    nodes.add(source)
                    nodes.add(target)
                    # Usar média do TE como peso da edge
                    G.add_edge(source, target, weight=data['mean_te'])
                
                # Adicionar nós
                for node in nodes:
                    G.add_node(node)
                
                # Posicionamento dos nós
                pos = nx.spring_layout(G, seed=42)
                
                # Desenhar nós
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')
                
                # Desenhar arestas com largura proporcional ao peso (TE)
                edges = list(G.edges())
                edge_weights = [G[u][v]['weight'] * 10 for u, v in edges]
                nx.draw_networkx_edges(G, pos, edgelist=edges, width=1.0, arrowsize=20, alpha=0.7)
                # Ou alternativamente, desenhar cada aresta individualmente com sua largura
                # for i, (u, v) in enumerate(edges):
                #     nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=edge_weights[i], arrowsize=20, alpha=0.7)
                
                # Adicionar labels
                nx.draw_networkx_labels(G, pos)
                
                # Adicionar edge labels (valores de TE)
                edge_labels = {(u, v): f"{G[u][v]['weight']:.3f}" for u, v in G.edges()}
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
                
                plt.title(f'Relações Causais Robustas: {metric}')
                plt.axis('off')
                plt.tight_layout()
                
                # Salvar figura
                causal_graph_path = os.path.join(output_dir, f'robust_causal_graph_{metric}.png')
                plt.savefig(causal_graph_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                visualization_paths.append(causal_graph_path)
                
            except Exception as e:
                logger.warning(f"Erro ao gerar grafo de causalidade para {metric}: {str(e)}")
    
    return visualization_paths
