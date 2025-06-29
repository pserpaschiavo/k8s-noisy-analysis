"""
Module: phase_correlation.py
Description: Módulo para extração e análise de correlações intra-fase.

Este módulo implementa funções para calcular e analisar correlações entre
tenants dentro de cada fase experimental, permitindo avaliar a consistência
comportamental entre tenants e o seu padrão de correlação em diferentes
fases e rounds do experimento.
"""

import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import multiprocessing
import time
from pathlib import Path
from functools import lru_cache

# Configuração de logging
logger = logging.getLogger(__name__)

def _process_correlation_pair(args):
    """
    Função auxiliar para processamento paralelo que calcula a correlação
    entre um par de tenants para uma combinação específica de round, métrica e fase.
    
    Args:
        args: Tupla com argumentos (df_long, round_id, metric_name, phase, tenant1, tenant2, method, min_periods)
        
    Returns:
        Dict: Dicionário com os resultados calculados
    """
    df_long, round_id, metric_name, phase, tenant1, tenant2, method, min_periods = args
    
    # Filtra os dados para o primeiro tenant
    tenant1_data = df_long[(df_long['round_id'] == round_id) & 
                         (df_long['metric_name'] == metric_name) & 
                         (df_long['experimental_phase'] == phase) & 
                         (df_long['tenant_id'] == tenant1)]
    
    # Filtra os dados para o segundo tenant
    tenant2_data = df_long[(df_long['round_id'] == round_id) & 
                         (df_long['metric_name'] == metric_name) & 
                         (df_long['experimental_phase'] == phase) & 
                         (df_long['tenant_id'] == tenant2)]
    
    # Se não houver dados suficientes, retorna None
    if tenant1_data.empty or tenant2_data.empty:
        return None
    
    # Mescla os dados por timestamp para garantir alinhamento temporal
    tenant1_series = tenant1_data.set_index('timestamp')['metric_value']
    tenant2_series = tenant2_data.set_index('timestamp')['metric_value']
    
    # Recorta para timestamps comuns
    common_timestamps = tenant1_series.index.intersection(tenant2_series.index)
    
    # Se não houver timestamps comuns suficientes, retorna None
    if len(common_timestamps) < min_periods:
        return None
    
    tenant1_values = tenant1_series.loc[common_timestamps]
    tenant2_values = tenant2_series.loc[common_timestamps]
    
    # Calcula a correlação
    try:
        corr = tenant1_values.corr(tenant2_values, method=method)
    except Exception as e:
        logger.warning(f"Erro ao calcular correlação entre {tenant1} e {tenant2}: {e}")
        return None
    
    # Verifica se a correlação é válida
    if pd.isna(corr):
        return None
    
    # Retorna um dicionário com os resultados
    return {
        'round_id': round_id,
        'metric_name': metric_name,
        'experimental_phase': phase,
        'tenant_pair': f"{tenant1}:{tenant2}",
        'tenant1': tenant1,
        'tenant2': tenant2,
        'correlation': corr,
        'method': method,
        'sample_size': len(common_timestamps)
    }

def extract_phase_correlations(
    df_long: pd.DataFrame,
    rounds: List[str],
    metrics: List[str],
    phases: List[str],
    tenants: Optional[List[str]] = None,
    method: str = 'pearson',
    min_periods: int = 3,
    use_cache: bool = True,
    parallel: bool = False,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Extrai as correlações intra-fase entre tenants para cada métrica × fase × round.
    
    Args:
        df_long: DataFrame em formato longo com todos os dados
        rounds: Lista de rounds para análise
        metrics: Lista de métricas para análise
        phases: Lista de fases para análise
        tenants: Lista de tenants para análise (opcional)
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        min_periods: Número mínimo de períodos para calcular correlação
        use_cache: Se True, usa cache para evitar recálculos
        parallel: Se True, paraleliza o processamento
        cache_dir: Diretório para armazenar o cache (opcional)
        
    Returns:
        DataFrame com colunas: round_id, metric_name, experimental_phase, 
        tenant_pair, correlation
    """
    start_time = time.time()
    logger.info("Iniciando extração de correlações intra-fase...")
    
    # Verifica o método de correlação
    valid_methods = ['pearson', 'spearman', 'kendall']
    if method not in valid_methods:
        logger.warning(f"Método de correlação inválido: {method}. Usando 'pearson'.")
        method = 'pearson'
    
    # Se tenants não for fornecido, usa todos os tenants disponíveis
    if tenants is None:
        tenants = sorted(df_long['tenant_id'].unique())
    
    # Configurar cache, se solicitado
    cache_file = None
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"phase_correlations_{method}_cache.parquet")
        
        # Se o cache existir, carregá-lo
        if os.path.exists(cache_file):
            try:
                logger.info(f"Carregando resultados do cache: {cache_file}")
                cached_df = pd.read_parquet(cache_file)
                logger.info(f"Cache carregado com sucesso: {cached_df.shape[0]} registros")
                return cached_df
            except Exception as e:
                logger.warning(f"Erro ao carregar cache: {e}. Recalculando...")
    
    # Prepara argumentos para processamento
    args_list = []
    for round_id in rounds:
        for metric_name in metrics:
            for phase in phases:
                # Para cada par de tenants não redundante
                for i, tenant1 in enumerate(tenants):
                    for tenant2 in tenants[i+1:]:  # Evita pares redundantes
                        args_list.append((df_long, round_id, metric_name, phase, tenant1, tenant2, method, min_periods))
    
    # Processa os dados (em paralelo ou sequencial)
    results = []
    if parallel and len(args_list) > 10:  # Só usa paralelismo se tiver muitos itens
        logger.info(f"Processando {len(args_list)} combinações em paralelo...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(_process_correlation_pair, args_list)
    else:
        logger.info(f"Processando {len(args_list)} combinações sequencialmente...")
        results = [_process_correlation_pair(args) for args in args_list]
    
    # Filtra resultados None
    results = [r for r in results if r is not None]
    
    # Converte resultados para DataFrame
    if not results:
        logger.warning("Nenhuma correlação calculada. Verifique os dados de entrada ou ajuste min_periods.")
        return pd.DataFrame()
    
    correlations_df = pd.DataFrame(results)
    
    # Adiciona avaliação de qualidade dos resultados
    correlations_df['result_quality'] = correlations_df.apply(
        lambda row: 'high' if row['sample_size'] >= 30 else
                   ('medium' if row['sample_size'] >= 10 else 'low'),
        axis=1
    )
    
    # Adiciona informação sobre força da correlação
    correlations_df['correlation_strength'] = correlations_df['correlation'].abs().apply(
        lambda x: 'strong' if x >= 0.7 else
                 ('moderate' if x >= 0.3 else 'weak')
    )
    
    # Salva o cache, se solicitado
    if use_cache and cache_file:
        try:
            correlations_df.to_parquet(cache_file)
            logger.info(f"Resultados salvos em cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Extração de correlações intra-fase concluída em {elapsed_time:.2f} segundos. Retornando {correlations_df.shape[0]} registros.")
    
    return correlations_df

def analyze_correlation_stability(
    phase_correlations_df: pd.DataFrame,
    min_rounds: int = 2,
    correlation_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Analisa a estabilidade das correlações entre tenants ao longo dos rounds.
    
    Args:
        phase_correlations_df: DataFrame resultado da função extract_phase_correlations
        min_rounds: Número mínimo de rounds para considerar uma correlação estável
        correlation_threshold: Limiar mínimo de correlação para considerar significativa
        
    Returns:
        Dict com resultados da análise de estabilidade
    """
    if phase_correlations_df.empty:
        logger.warning("DataFrame de correlações vazio. Não é possível analisar estabilidade.")
        return {}
    
    # Filtra correlações significativas pelo threshold
    significant_corrs = phase_correlations_df[phase_correlations_df['correlation'].abs() >= correlation_threshold]
    
    if significant_corrs.empty:
        logger.warning(f"Nenhuma correlação acima do threshold de {correlation_threshold}.")
        return {
            'stable_correlations': {},
            'correlation_variability': {},
            'summary': {
                'total_pairs': 0,
                'stable_pairs': 0,
                'unstable_pairs': 0
            }
        }
    
    # Agrupa por métrica, fase e par de tenants
    grouped = significant_corrs.groupby(['metric_name', 'experimental_phase', 'tenant_pair'])
    
    # Calcula estatísticas de estabilidade
    stability_stats = grouped.agg({
        'correlation': ['mean', 'std', 'count', 'min', 'max'],
        'round_id': lambda x: list(sorted(x.unique())),
    })
    
    # Renomeia as colunas para facilitar o acesso
    stability_stats.columns = ['corr_mean', 'corr_std', 'round_count', 'corr_min', 'corr_max', 'rounds']
    stability_stats = stability_stats.reset_index()
    
    # Identifica pares estáveis (presentes em pelo menos min_rounds rounds)
    stability_stats['is_stable'] = stability_stats['round_count'] >= min_rounds
    
    # Calcula o coeficiente de variação para estimar a variabilidade relativa
    # Evita divisão por zero
    stability_stats['corr_cv'] = np.where(
        stability_stats['corr_mean'] != 0,
        stability_stats['corr_std'] / stability_stats['corr_mean'].abs(),
        np.nan
    )
    
    # Categoriza a variabilidade
    stability_stats['variability'] = stability_stats['corr_cv'].apply(
        lambda cv: 'low' if pd.isna(cv) or cv < 0.1 else
                  ('medium' if cv < 0.25 else 'high')
    )
    
    # Cria um dicionário de correlações estáveis
    stable_correlations = {}
    for _, row in stability_stats[stability_stats['is_stable']].iterrows():
        metric = row['metric_name']
        phase = row['experimental_phase']
        tenant_pair = row['tenant_pair']
        
        key = (metric, phase)
        if key not in stable_correlations:
            stable_correlations[key] = []
        
        stable_correlations[key].append({
            'tenant_pair': tenant_pair,
            'mean_correlation': row['corr_mean'],
            'std_correlation': row['corr_std'],
            'round_count': row['round_count'],
            'variability': row['variability'],
            'rounds': row['rounds']
        })
    
    # Calcula variabilidade geral por métrica e fase
    variability = stability_stats.groupby(['metric_name', 'experimental_phase']).agg({
        'corr_std': 'mean',
        'corr_cv': 'mean',
        'is_stable': 'mean'  # Proporção de pares estáveis
    }).reset_index()
    
    variability.columns = ['metric_name', 'experimental_phase', 'mean_std', 'mean_cv', 'stable_ratio']
    
    # Retorna resultados
    return {
        'stable_correlations': stable_correlations,
        'correlation_variability': variability.to_dict(orient='records'),
        'summary': {
            'total_pairs': len(stability_stats),
            'stable_pairs': stability_stats['is_stable'].sum(),
            'unstable_pairs': len(stability_stats) - stability_stats['is_stable'].sum()
        }
    }
