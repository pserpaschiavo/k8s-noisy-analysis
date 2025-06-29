import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import multiprocessing
from functools import lru_cache
from pathlib import Path
import time

logger = logging.getLogger(__name__)

def cohens_d(group1, group2):
    """
    Calcula o tamanho de efeito (Cohen's d) entre dois grupos.
    
    Args:
        group1: Array-like com valores do grupo 1
        group2: Array-like com valores do grupo 2
        
    Returns:
        float: Tamanho de efeito (Cohen's d)
    """
    # Verifica se os grupos têm dados suficientes
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    
    # Calcula médias e desvios padrão para ambos os grupos
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Se o desvio padrão for zero em ambos os grupos, retorna NaN
    if std1 == 0 and std2 == 0:
        return np.nan
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Prevenção de divisão por zero
    if pooled_std == 0:
        return np.nan
    
    # Cohen's d
    d = (mean2 - mean1) / pooled_std
    return d

def eta_squared(group1, group2):
    """
    Calcula o tamanho de efeito Eta-squared entre dois grupos.
    
    Args:
        group1: Array-like com valores do grupo 1
        group2: Array-like com valores do grupo 2
        
    Returns:
        float: Tamanho de efeito (Eta-squared)
    """
    # Verifica se os grupos têm dados suficientes
    if len(group1) < 2 or len(group2) < 2:
        return np.nan
    
    # Combina os grupos em um único array
    all_data = np.concatenate([group1, group2])
    
    # Se todos os valores são iguais, retorna NaN (não há variação)
    if np.all(all_data == all_data[0]):
        return np.nan
    
    # Calcula a soma dos quadrados total
    ss_total = np.sum((all_data - np.mean(all_data))**2)
    
    # Prevenção de divisão por zero
    if ss_total == 0:
        return np.nan
    
    # Calcula a soma dos quadrados entre grupos
    mean1, mean2 = np.mean(group1), np.mean(group2)
    ss_between = len(group1) * (mean1 - np.mean(all_data))**2 + len(group2) * (mean2 - np.mean(all_data))**2
    
    # Eta-squared
    eta_sq = ss_between / ss_total
    return eta_sq

def t_test(group1, group2):
    """
    Realiza o t-test independente entre dois grupos.
    
    Args:
        group1: Array-like com valores do grupo 1
        group2: Array-like com valores do grupo 2
        
    Returns:
        Tuple[float, float]: Estatística t e p-valor
    """
    # Verifica se os grupos têm dados suficientes
    if len(group1) < 2 or len(group2) < 2:
        return np.nan, np.nan
    
    # Realiza o t-test
    try:
        t_stat, p_val = stats.ttest_ind(group1, group2, equal_var=False)  # Welch's t-test
        return t_stat, p_val
    except:
        return np.nan, np.nan

def _process_effect_size(args):
    """
    Função auxiliar para processamento paralelo que calcula tamanhos de efeito
    para uma combinação específica de round, métrica, fase e tenant.
    
    Args:
        args: Tupla com argumentos (df_long, round_id, metric_name, phase, tenant_id, baseline_phase)
        
    Returns:
        Dict: Dicionário com os resultados calculados
    """
    df_long, round_id, metric_name, phase, tenant_id, baseline_phase = args
    
    # Pula a baseline comparada com ela mesma
    if phase == baseline_phase:
        return None
    
    # Filtra os dados para o grupo experimental
    exp_data = df_long[(df_long['round_id'] == round_id) & 
                      (df_long['metric_name'] == metric_name) & 
                      (df_long['experimental_phase'] == phase) & 
                      (df_long['tenant_id'] == tenant_id)]
    
    # Filtra os dados para o grupo baseline
    base_data = df_long[(df_long['round_id'] == round_id) & 
                       (df_long['metric_name'] == metric_name) & 
                       (df_long['experimental_phase'] == baseline_phase) & 
                       (df_long['tenant_id'] == tenant_id)]
    
    # Se não houver dados suficientes, retorna None
    if exp_data.empty or base_data.empty:
        return None
    
    # Extrai os valores métricos para ambos os grupos
    exp_values = exp_data['metric_value'].values
    base_values = base_data['metric_value'].values
    
    # Calcula estatísticas
    effect_size = cohens_d(base_values, exp_values)
    eta_sq = eta_squared(base_values, exp_values)
    t_stat, p_value = t_test(base_values, exp_values)
    
    # Retorna um dicionário com os resultados
    return {
        'round_id': round_id,
        'metric_name': metric_name,
        'experimental_phase': phase,
        'tenant_id': tenant_id,
        'baseline_phase': baseline_phase,
        'effect_size': effect_size,
        'p_value': p_value,
        't_statistic': t_stat,
        'eta_squared': eta_sq,
        'sample_size_baseline': len(base_values),
        'sample_size_experimental': len(exp_values)
    }

def extract_effect_sizes(
    df_long: pd.DataFrame,
    rounds: List[str],
    metrics: List[str],
    phases: List[str],
    tenants: List[str],
    baseline_phase: str = "1 - Baseline",
    use_cache: bool = True,
    parallel: bool = False,
    cache_dir: Optional[str] = None
) -> pd.DataFrame:
    """
    Extrai estatísticas de tamanho de efeito (Cohen's d) e p-valores para
    comparações de fase vs. baseline para cada métrica, tenant e round.
    
    Args:
        df_long: DataFrame em formato longo com todos os dados
        rounds: Lista de rounds para análise
        metrics: Lista de métricas para análise
        phases: Lista de fases para análise
        tenants: Lista de tenants para análise
        baseline_phase: Nome da fase de baseline (default: "1 - Baseline")
        use_cache: Se True, usa cache para evitar recálculos
        parallel: Se True, paraleliza o processamento
        cache_dir: Diretório para armazenar o cache (opcional)
        
    Returns:
        DataFrame com colunas: round_id, metric_name, experimental_phase, 
        tenant_id, baseline_phase, effect_size, p_value, eta_squared, etc.
    """
    start_time = time.time()
    logger.info("Iniciando extração de tamanhos de efeito...")
    
    # Configurar cache, se solicitado
    cache_file = None
    if use_cache and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "effect_sizes_cache.parquet")
        
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
                if phase == baseline_phase:
                    continue  # Pula comparação da baseline com ela mesma
                for tenant_id in tenants:
                    args_list.append((df_long, round_id, metric_name, phase, tenant_id, baseline_phase))
    
    # Processa os dados (em paralelo ou sequencial)
    results = []
    if parallel and len(args_list) > 10:  # Só usa paralelismo se tiver muitos itens
        logger.info(f"Processando {len(args_list)} combinações em paralelo...")
        with multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1) as pool:
            results = pool.map(_process_effect_size, args_list)
    else:
        logger.info(f"Processando {len(args_list)} combinações sequencialmente...")
        results = [_process_effect_size(args) for args in args_list]
    
    # Filtra resultados None
    results = [r for r in results if r is not None]
    
    # Converte resultados para DataFrame
    if not results:
        logger.warning("Nenhum resultado calculado. Verifique os dados de entrada.")
        return pd.DataFrame()
    
    effect_sizes_df = pd.DataFrame(results)
    
    # Adiciona avaliação de qualidade dos resultados
    effect_sizes_df['result_quality'] = effect_sizes_df.apply(
        lambda row: 'high' if (not pd.isna(row['effect_size']) and 
                              not pd.isna(row['p_value']) and 
                              row['sample_size_baseline'] >= 30 and 
                              row['sample_size_experimental'] >= 30) else
                   ('medium' if (not pd.isna(row['effect_size']) and 
                                not pd.isna(row['p_value']) and 
                                row['sample_size_baseline'] >= 10 and 
                                row['sample_size_experimental'] >= 10) else 'low'),
        axis=1
    )
    
    # Salva o cache, se solicitado
    if use_cache and cache_file:
        try:
            effect_sizes_df.to_parquet(cache_file)
            logger.info(f"Resultados salvos em cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Erro ao salvar cache: {e}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Extração de tamanhos de efeito concluída em {elapsed_time:.2f} segundos. Retornando {effect_sizes_df.shape[0]} registros.")
    
    return effect_sizes_df
