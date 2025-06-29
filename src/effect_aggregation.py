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
        p_values: Lista ou array de p-valores a combinar
        
    Returns:
        float: p-valor combinado
    """
    # Converter para numpy array para processamento mais eficiente
    p_values = np.asarray(p_values, dtype=float)
    
    # Remover valores NaN
    p_values = p_values[~np.isnan(p_values)]
    
    # Verificar se há p-valores válidos
    if len(p_values) == 0:
        return np.nan
    
    # Substituir valores zero ou muito próximos por um valor mínimo
    # para evitar problemas com o logaritmo
    min_p = 1e-10
    p_values = np.clip(p_values, min_p, 1.0)
    
    try:
        # Estatística de teste de Fisher
        chi_square = -2 * np.sum(np.log(p_values))
        
        # Graus de liberdade = 2 * número de p-valores
        df = 2 * len(p_values)
        
        # p-valor combinado
        combined_p = 1.0 - stats.chi2.cdf(chi_square, df)
        
        # Verificar resultado
        if np.isnan(combined_p) or np.isinf(combined_p):
            return min_p if chi_square > 0 else 1.0
        
        return combined_p
    except Exception as e:
        logger.warning(f"Erro ao combinar p-valores com método de Fisher: {e}")
        return np.nan

def combine_pvalues_stouffer(p_values, weights=None):
    """
    Combina múltiplos p-valores usando o método de Stouffer.
    
    Args:
        p_values: Lista ou array de p-valores a combinar
        weights: Pesos para cada p-valor (opcional)
        
    Returns:
        float: p-valor combinado
    """
    # Converter para numpy array para processamento mais eficiente
    p_values = np.asarray(p_values, dtype=float)
    
    # Remover valores NaN
    mask = ~np.isnan(p_values)
    p_values = p_values[mask]
    
    # Verificar se há p-valores válidos
    if len(p_values) == 0:
        return np.nan
    
    # Ajustar pesos se fornecidos
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        weights = weights[mask]  # Aplicar mesma máscara usada para p_values
    else:
        weights = np.ones_like(p_values)
    
    # Substituir valores problemáticos
    min_p = 1e-10
    max_p = 1.0 - min_p
    p_values = np.clip(p_values, min_p, max_p)
    
    try:
        # Converter p-valores para z-scores
        # Usamos o inverso da CDF da distribuição normal
        z_scores = stats.norm.ppf(1 - p_values)
        
        # Z-score combinado
        z_combined = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
        
        # p-valor combinado
        combined_p = 1 - stats.norm.cdf(z_combined)
        
        # Verificar resultado
        if np.isnan(combined_p) or np.isinf(combined_p):
            return min_p if z_combined > 0 else 1.0
        
        return combined_p
    except Exception as e:
        logger.warning(f"Erro ao combinar p-valores com método de Stouffer: {e}")
        return np.nan

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
    # Verificações iniciais
    values = np.array(values)
    if len(values) < 2:
        return np.nan, np.nan
    
    # Verifica valores ausentes
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return np.nan, np.nan
    
    # Tratamento de outliers extremos (opcional)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 3.0 * iqr
    upper_bound = q3 + 3.0 * iqr
    filtered_values = values[(values >= lower_bound) & (values <= upper_bound)]
    
    # Se o filtro removeu todos ou quase todos os dados, usar os originais
    if len(filtered_values) < 2 or len(filtered_values) < 0.5 * len(values):
        filtered_values = values
    
    # Configurar gerador de números aleatórios para reprodutibilidade
    rng = np.random.RandomState(random_state)
    
    # Realizar bootstrapping
    bootstrap_means = []
    for _ in range(n_resamples):
        # Garantir que o resampling é feito com replacement
        resample = rng.choice(filtered_values, size=len(filtered_values), replace=True)
        bootstrap_means.append(np.mean(resample))
    
    # Calcular percentis para o intervalo de confiança
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(bootstrap_means, 100 * alpha)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha))
    
    # Validação final dos resultados
    if np.isinf(lower) or np.isinf(upper) or np.isnan(lower) or np.isnan(upper):
        # Fallback para método não paramétrico mais simples
        lower, upper = calculate_confidence_interval(filtered_values, confidence_level)
    
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
        # Extrair valores para cálculos e convertê-los para arrays numpy
        effect_sizes = np.asarray(group['effect_size'].values, dtype=float)
        p_values = np.asarray(group['p_value'].values, dtype=float)
        
        # Extrair eta_squared se disponível
        if 'eta_squared' in group.columns:
            eta_squared_values = np.asarray(group['eta_squared'].values, dtype=float)
        else:
            eta_squared_values = None
        
        # Verificar se há dados suficientes
        if len(effect_sizes) < 1:
            logger.warning(f"Dados insuficientes para {metric}, {phase}, {tenant}. Pulando.")
            continue
        
        # Calcular estatísticas básicas
        mean_effect = np.nanmean(effect_sizes)
        std_effect = np.nanstd(effect_sizes, ddof=1) if len(effect_sizes) > 1 else np.nan
        
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
            mean_eta = np.nanmean(eta_squared_values)
            std_eta = np.nanstd(eta_squared_values, ddof=1) if len(eta_squared_values) > 1 else np.nan
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
        valid_p_values = ~np.isnan(p_values)
        if np.any(valid_p_values):
            significant_count = np.sum(p_values[valid_p_values] < alpha)
            total_count = np.sum(valid_p_values)
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
