#!/usr/bin/env python3
"""
Module: utils_timeseries.py
Description: Utilitários para análise de séries temporais.

Este módulo contém funções auxiliares para manipulação e processamento de séries temporais,
incluindo verificação de estacionariedade, diferenciação e outras transformações comuns.
"""

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Union
import warnings

# Configurar logging
logger = logging.getLogger(__name__)

def check_and_transform_timeseries(
    data: pd.DataFrame, 
    max_lag: int = 5,
    min_points: int = 10,
    difference_if_nonstationary: bool = True
) -> Tuple[pd.DataFrame, bool]:
    """
    Verifica e transforma uma série temporal para análise causal.
    
    Args:
        data: DataFrame com duas colunas (target e possível causa)
        max_lag: Lag máximo para análise (requer dados suficientes)
        min_points: Número mínimo de pontos necessários após transformações
        difference_if_nonstationary: Se True, aplica diferenciação em séries não-estacionárias
    
    Returns:
        Tuple[pd.DataFrame, bool]: DataFrame transformado e flag indicando se é válido para análise
    """
    # Verificar se há dados suficientes
    if len(data) < max_lag + 5:  # Requer pelo menos 5 pontos além do maxlag
        logger.warning(f"Dados insuficientes: {len(data)} pontos, mínimo {max_lag+5}")
        return data, False
    
    # Copia para evitar modificar o DataFrame original
    transformed_data = data.copy()
    
    # Verificar estacionariedade se solicitado
    if difference_if_nonstationary:
        try:
            from statsmodels.tsa.stattools import adfuller
            # Diferenciação de primeira ordem se não for estacionário
            for col in range(transformed_data.shape[1]):
                adf_result = adfuller(transformed_data.iloc[:, col], autolag='AIC')
                if adf_result[1] > 0.05:  # p-valor > 0.05 indica não-estacionariedade
                    # Aplicamos diferenciação
                    transformed_data.iloc[:, col] = transformed_data.iloc[:, col].diff().dropna()
            
            # Remove valores NaN após diferenciação
            transformed_data = transformed_data.dropna()
        except Exception as e:
            logger.warning(f"Erro ao verificar estacionariedade: {str(e)}")
    
    # Verifica novamente tamanho após transformações
    if len(transformed_data) <= max_lag + 3:
        logger.warning(f"Dados insuficientes após transformações: {len(transformed_data)} pontos")
        return transformed_data, False
    
    return transformed_data, True

def resample_and_align_timeseries(
    data: pd.DataFrame,
    freq: str = '1min',
    fill_method: str = 'interpolate',
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Reamostra e alinha séries temporais com frequência uniforme.
    
    Args:
        data: DataFrame com índice de timestamp
        freq: Frequência de reamostragem (ex: '1min', '5min')
        fill_method: Método para preencher valores ausentes ('interpolate', 'ffill', etc)
        min_periods: Número mínimo de observações para interpolação válida
    
    Returns:
        DataFrame com séries temporais alinhadas e reamostradas
    """
    # Verifica se o índice é datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        logger.warning("Índice não é do tipo datetime, tentando converter")
        try:
            data.index = pd.to_datetime(data.index)
        except:
            logger.error("Não foi possível converter índice para datetime")
            return data
    
    # Reamostra para frequência regular
    resampled = data.resample(freq).mean()
    
    # Preenche valores ausentes com o método escolhido
    if fill_method == 'interpolate':
        filled = resampled.interpolate(method='time', limit_area='inside', min_periods=min_periods)
    elif fill_method == 'ffill':
        filled = resampled.ffill()
    elif fill_method == 'bfill':
        filled = resampled.bfill()
    else:
        filled = resampled.interpolate(method='time').ffill().bfill()
    
    return filled

def run_granger_causality_test(
    data: pd.DataFrame, 
    maxlag: int = 3,
    verbose: bool = False
) -> dict:
    """
    Executa teste de causalidade de Granger com tratamento de erro robusto.
    
    Args:
        data: DataFrame com duas colunas (target e possível causa)
        maxlag: Número máximo de lags para o teste
        verbose: Se True, imprime resultados detalhados
    
    Returns:
        Dicionário com resultados do teste ou None se falhar
    """
    if data.shape[1] != 2:
        logger.error(f"DataFrame deve ter exatamente 2 colunas, encontrado {data.shape[1]}")
        return None
        
    # Verificar e transformar os dados
    transformed_data, valid = check_and_transform_timeseries(
        data, 
        max_lag=maxlag,
        min_points=maxlag + 5,
        difference_if_nonstationary=True
    )
    
    if not valid:
        logger.warning("Dados insuficientes ou inválidos para teste de Granger")
        return None
        
    # Executa o teste com tratamento de exceções
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            
            test_results = grangercausalitytests(
                transformed_data, 
                maxlag=maxlag, 
                verbose=verbose
            )
            
            # Extrai o menor p-valor entre todos os lags
            p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
            min_p_value = min(p_values) if p_values else np.nan
            
            return {
                'p_values': p_values,
                'min_p_value': min_p_value,
                'results': test_results
            }
            
    except Exception as e:
        logger.warning(f"Erro ao executar teste de Granger: {str(e)}")
        return None
