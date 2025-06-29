import os
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

def combine_pvalues_fisher(p_values):
    """
    Combina múltiplos p-valores usando o método de Fisher.
    
    Args:
        p_values: Lista de p-valores a combinar
        
    Returns:
        float: p-valor combinado
    """
    # Substituir valores zero ou nan para evitar problemas
    p_values = np.array(p_values)
    p_values = np.nan_to_num(p_values, nan=0.5)  # Substitui NaN por 0.5 (neutro)
    p_values = np.clip(p_values, 1e-10, 1.0)  # Limita entre 1e-10 e 1.0
    
    # Estatística de teste de Fisher
    chi_square = -2 * np.sum(np.log(p_values))
    
    # Graus de liberdade = 2 * número de p-valores
    df = 2 * len(p_values)
    
    # p-valor combinado
    combined_p = 1.0 - stats.chi2.cdf(chi_square, df)
    return combined_p

def combine_pvalues_stouffer(p_values, weights=None):
    """
    Combina múltiplos p-valores usando o método de Stouffer.
    
    Args:
        p_values: Lista de p-valores a combinar
        weights: Pesos para cada p-valor (opcional)
        
    Returns:
        float: p-valor combinado
    """
    # Substituir valores problemáticos
    p_values = np.array(p_values)
    p_values = np.nan_to_num(p_values, nan=0.5)  # Substitui NaN por 0.5 (neutro)
    p_values = np.clip(p_values, 1e-10, 1.0)  # Limita entre 1e-10 e 1.0
    
    # Converter p-valores para z-scores
    z_scores = stats.norm.ppf(1 - p_values)
    
    if weights is None:
        weights = np.ones_like(z_scores)
    weights = np.array(weights)
    
    # Z-score combinado
    z_combined = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
    
    # p-valor combinado
    combined_p = 1 - stats.norm.cdf(z_combined)
    return combined_p

def calculate_confidence_interval(values, confidence_level=0.95):
    """
    Calcula o intervalo de confiança usando a distribuição t.
    
    Args:
        values: Array de valores
        confidence_level: Nível de confiança (default: 0.95)
        
    Returns:
        Tuple[float, float]: Limites inferior e superior do intervalo de confiança
    """
    n = len(values)
    if n < 2:
        return np.nan, np.nan
    
    mean = np.mean(values)
    std_err = stats.sem(values)
    h = std_err * stats.t.ppf((1 + confidence_level) / 2, n - 1)
    
    return mean - h, mean + h

def bootstrap_confidence_interval(values, confidence_level=0.95, n_resamples=1000, random_state=None):
    """
    Calcula o intervalo de confiança usando bootstrapping.
    
    Args:
        values: Array de valores
        confidence_level: Nível de confiança (default: 0.95)
        n_resamples: Número de reamostragens (default: 1000)
        random_state: Semente para reprodutibilidade (opcional)
        
    Returns:
        Tuple[float, float]: Limites inferior e superior do intervalo de confiança
    """
    if len(values) < 2:
        return np.nan, np.nan
    
    # Configurar gerador de números aleatórios para reprodutibilidade
    rng = np.random.RandomState(random_state)
    
    # Realizar bootstrapping
    bootstrap_means = []
    for _ in range(n_resamples):
        resample = rng.choice(values, size=len(values), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    # Calcular percentis para o intervalo de confiança
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    
    return lower, upper

def calculate_stability_metrics(values):
    """
    Calcula métricas de estabilidade para um conjunto de valores.
    
    Args:
        values: Array de valores
        
    Returns:
        Dict[str, float]: Dicionário com métricas de estabilidade
    """
    if len(values) < 2:
        return {
            'coefficient_of_variation': np.nan,
            'range_ratio': np.nan,
            'stability_score': np.nan
        }
    
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    
    # Coeficiente de variação (CV)
    cv = std / mean if mean != 0 else np.nan
    
    # Range ratio (max - min) / media
    range_ratio = (np.max(values) - np.min(values)) / mean if mean != 0 else np.nan
    
    # Pontuação de estabilidade (inverso do CV normalizado)
    stability_score = 1 / (1 + abs(cv)) if not np.isnan(cv) else np.nan
    
    return {
        'coefficient_of_variation': cv,
        'range_ratio': range_ratio,
        'stability_score': stability_score
    }

def aggregate_effect_sizes(
    effect_sizes_df: pd.DataFrame,
    alpha: float = 0.05,
    p_value_method: str = 'fisher',
    confidence_level: float = 0.95,
    use_bootstrap: bool = True,
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Agrega os tamanhos de efeito através dos rounds, calculando médias,
    desvios padrão, intervalos de confiança e p-valores combinados.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito por round × métrica × fase × tenant
        alpha: Nível de significância (default: 0.05)
        p_value_method: Método para combinar p-valores ('fisher', 'stouffer')
        confidence_level: Nível de confiança para IC (default: 0.95)
        use_bootstrap: Se True, usa bootstrapping para IC; caso contrário, usa t-distribution
        n_bootstrap: Número de reamostras para bootstrapping
        random_state: Semente para reprodutibilidade
        
    Returns:
        DataFrame com estatísticas agregadas
    """
    start_time = time.time()
    logger.info("Iniciando agregação de tamanhos de efeito...")
    
    if effect_sizes_df.empty:
        logger.warning("DataFrame de entrada vazio. Retornando DataFrame vazio.")
        return pd.DataFrame()
    
    # Verificar colunas necessárias
    required_cols = ['round_id', 'metric_name', 'experimental_phase', 'tenant_id', 
                     'effect_size', 'p_value']
    
    missing_cols = [col for col in required_cols if col not in effect_sizes_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return pd.DataFrame()
    
    # Agrupar por métrica, fase experimental e tenant
    grouped = effect_sizes_df.groupby(['metric_name', 'experimental_phase', 'tenant_id', 'baseline_phase'])
    
    results = []
    for (metric, phase, tenant, baseline), group in grouped:
        # Extrair valores para cálculos
        effect_sizes = group['effect_size'].values
        p_values = group['p_value'].values
        eta_squared_values = group['eta_squared'].values if 'eta_squared' in group.columns else None
        
        # Verificar se há dados suficientes
        if len(effect_sizes) < 1:
            logger.warning(f"Dados insuficientes para {metric}, {phase}, {tenant}. Pulando.")
            continue
        
        # Calcular estatísticas básicas
        mean_effect = np.mean(effect_sizes)
        std_effect = np.std(effect_sizes, ddof=1) if len(effect_sizes) > 1 else np.nan
        
        # Calcular intervalo de confiança (IC)
        if use_bootstrap and len(effect_sizes) >= 3:
            ci_lower, ci_upper = bootstrap_confidence_interval(
                effect_sizes, confidence_level, n_bootstrap, random_state
            )
        else:
            ci_lower, ci_upper = calculate_confidence_interval(effect_sizes, confidence_level)
        
        # Combinar p-valores
        if p_value_method == 'fisher':
            combined_p = combine_pvalues_fisher(p_values)
        elif p_value_method == 'stouffer':
            combined_p = combine_pvalues_stouffer(p_values)
        else:
            logger.warning(f"Método de combinação de p-valores '{p_value_method}' não reconhecido. Usando Fisher.")
            combined_p = combine_pvalues_fisher(p_values)
        
        # Cálculos de estabilidade
        stability_metrics = calculate_stability_metrics(effect_sizes)
        
        # Estatísticas para eta_squared se disponíveis
        if eta_squared_values is not None:
            mean_eta = np.mean(eta_squared_values)
            std_eta = np.std(eta_squared_values, ddof=1) if len(eta_squared_values) > 1 else np.nan
        else:
            mean_eta = np.nan
            std_eta = np.nan
        
        # Avaliar significância e magnitude do efeito
        is_significant = combined_p < alpha
        if np.isnan(mean_effect):
            effect_magnitude = "unknown"
        elif abs(mean_effect) < 0.2:
            effect_magnitude = "negligible"
        elif abs(mean_effect) < 0.5:
            effect_magnitude = "small"
        elif abs(mean_effect) < 0.8:
            effect_magnitude = "medium"
        else:
            effect_magnitude = "large"
        
        # Número de rounds com valor significativo
        if not np.isnan(p_values).all():
            significant_count = np.sum(p_values < alpha)
            total_count = np.sum(~np.isnan(p_values))
            significance_ratio = significant_count / total_count if total_count > 0 else np.nan
        else:
            significant_count = 0
            total_count = 0
            significance_ratio = np.nan
        
        # Pontuação composta de confiabilidade
        if np.isnan(stability_metrics['stability_score']) or np.isnan(significance_ratio):
            reliability_score = np.nan
        else:
            reliability_score = (stability_metrics['stability_score'] + significance_ratio) / 2
        
        # Classificação de confiabilidade
        if np.isnan(reliability_score):
            reliability_category = "unknown"
        elif reliability_score >= 0.7:
            reliability_category = "high"
        elif reliability_score >= 0.4:
            reliability_category = "medium"
        else:
            reliability_category = "low"
        
        # Adicionar resultado ao array
        result = {
            'metric_name': metric,
            'experimental_phase': phase,
            'tenant_id': tenant,
            'baseline_phase': baseline,
            'mean_effect_size': mean_effect,
            'std_effect_size': std_effect,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_eta_squared': mean_eta,
            'std_eta_squared': std_eta,
            'combined_p_value': combined_p,
            'coefficient_of_variation': stability_metrics['coefficient_of_variation'],
            'range_ratio': stability_metrics['range_ratio'],
            'stability_score': stability_metrics['stability_score'],
            'is_significant': is_significant,
            'effect_magnitude': effect_magnitude,
            'rounds_count': len(effect_sizes),
            'significant_rounds': significant_count,
            'significance_ratio': significance_ratio,
            'reliability_score': reliability_score,
            'reliability_category': reliability_category
        }
        results.append(result)
    
    # Converter lista de resultados para DataFrame
    if not results:
        logger.warning("Nenhum resultado agregado calculado.")
        return pd.DataFrame()
    
    aggregated_df = pd.DataFrame(results)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Agregação de tamanhos de efeito concluída em {elapsed_time:.2f} segundos.")
    logger.info(f"Gerados {len(aggregated_df)} resultados agregados.")
    
    return aggregated_df
