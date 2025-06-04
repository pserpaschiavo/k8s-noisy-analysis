"""
Module: analysis_correlation.py
Description: Correlation analysis utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from statsmodels.tsa.stattools import acf

# Configuração de logging
logger = logging.getLogger(__name__)

plt.style.use('tableau-colorblind10')

def compute_correlation_matrix(df_identifier, metric: str, phase: str, round_id: str, method: str = 'pearson') -> pd.DataFrame:
    """
    Computes the correlation matrix (Pearson or Spearman) between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    
    Args:
        df_identifier: Identificador do DataFrame ou o próprio DataFrame
        metric: Nome da métrica para análise
        phase: Fase experimental para filtrar
        round_id: ID do round para filtrar
        method: Método de correlação ('pearson', 'kendall', ou 'spearman')
        
    Returns:
        DataFrame com matriz de correlação
    """
    # Handle DataFrame identification for caching purposes
    if isinstance(df_identifier, pd.DataFrame):
        df = df_identifier
    else:
        # If it's a string identifier, try to load from parquet or other source
        df = pd.read_parquet(df_identifier) if isinstance(df_identifier, str) else df_identifier
    
    # Validate correlation method
    valid_methods = ['pearson', 'kendall', 'spearman']
    if method not in valid_methods:
        logger.warning(f"Método de correlação inválido: {method}. Usando 'pearson'.")
        method = 'pearson'
    
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"Sem dados para cálculo de correlação: {metric}, {phase}, {round_id}")
        return pd.DataFrame()
    
    # Pivot to wide format: index=timestamp, columns=tenant_id, values=metric_value
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    
    # Check for missing data and handle it
    if wide.isna().any().any():
        logger.info(f"Dados faltantes detectados na matriz de correlação. Aplicando interpolação linear.")
        wide = wide.interpolate(method='linear')
    
    # Contornando o problema de tipo executando a função de correlação com o tipo correto
    if method == 'pearson':
        corr = wide.corr(method='pearson')
    elif method == 'kendall':
        corr = wide.corr(method='kendall')
    elif method == 'spearman':
        corr = wide.corr(method='spearman')
    else:
        # Usa pearson como padrão
        logger.warning(f"Método {method} desconhecido, usando pearson")
        corr = wide.corr(method='pearson')
        
    logger.info(f"Matriz de correlação ({method}) calculada para {metric}, {phase}, {round_id}: {corr.shape}")
    return corr

def plot_correlation_heatmap(corr_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str, method: str = 'pearson') -> str | None:
    """
    Plots a heatmap of the correlation matrix between tenants for a given metric, phase, and round.
    
    Args:
        corr_matrix: Matriz de correlação calculada
        metric: Nome da métrica
        phase: Fase experimental 
        round_id: ID do round
        out_dir: Diretório de saída
        method: Método de correlação usado ('pearson', 'kendall', ou 'spearman')
        
    Returns:
        Caminho para o gráfico gerado ou None se não houver dados
    """
    if corr_matrix.empty:
        logger.warning(f"Matriz de correlação vazia. Não foi possível gerar heatmap para {metric}, {phase}, {round_id}")
        return None
        
    plt.figure(figsize=(10, 8))
    
    # Melhorar a visualização com máscaras para o triângulo superior
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask, 1)] = True
    
    # Melhorar a estética do heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='vlag', 
        vmin=-1, 
        vmax=1, 
        center=0, 
        square=True, 
        linewidths=0.5, 
        mask=mask,
        cbar_kws={"label": f"{method.title()} correlation", "shrink": 0.8}
    )
    
    plt.title(f'{method.title()} correlation between tenants\n{metric} - {phase} - {round_id}', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    # Formatação corrigida para corresponder ao padrão dos plots de covariância
    out_path = os.path.join(out_dir, f"correlation_heatmap_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info(f"Heatmap de correlação salvo em {out_path}")
    return out_path

def compute_covariance_matrix(df: pd.DataFrame, metric: str, phase: str, round_id: str) -> pd.DataFrame:
    """
    Computes the covariance matrix between tenants for a given metric, phase, and round.
    Returns a DataFrame with tenants as both rows and columns.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        return pd.DataFrame()
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    cov = wide.cov()
    return cov

def plot_covariance_heatmap(cov_matrix: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """
    Plots a heatmap of the covariance matrix between tenants for a given metric, phase, and round.
    """
    if cov_matrix.empty:
        return None
    plt.figure(figsize=(8, 6))
    sns.heatmap(cov_matrix, annot=True, cmap='crest', center=0, square=True, linewidths=0.5, cbar_kws={"label": "Covariance"})
    plt.title(f'Covariance between tenants\n{metric} - {phase} - {round_id}')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"covariance_heatmap_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path

def compute_cross_correlation(df: pd.DataFrame, metric: str, phase: str, round_id: str, tenants: list[str] | None = None, max_lag: int = 20) -> dict:
    """
    Calcula a correlação cruzada (CCF) entre pares de tenants com defasagem.
    
    Args:
        df: DataFrame em formato long
        metric: Nome da métrica
        phase: Fase experimental
        round_id: ID do round
        tenants: Lista de tenants a considerar (se None, usa todos)
        max_lag: Número máximo de defasagens a considerar
        
    Returns:
        Dicionário com pares de tenants como chaves e arrays de CCF como valores
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"Sem dados para cálculo de CCF: {metric}, {phase}, {round_id}")
        return {}
    
    # Convert to wide format
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    if tenants is None:
        tenants = wide.columns.tolist()
    
    # Check if we have at least 2 tenants
    if len(tenants) < 2:
        logger.warning(f"Menos de 2 tenants disponíveis para CCF: {tenants}")
        return {}
    
    # Handle missing data
    if wide.isna().any().any():
        wide = wide.interpolate(method='linear')
    
    ccf_results = {}
    for i, tenant1 in enumerate(tenants):
        for tenant2 in tenants[i+1:]:  # Evitar duplicação
            if tenant1 not in wide.columns or tenant2 not in wide.columns:
                continue
                
            # Get both time series
            ts1 = wide[tenant1].fillna(wide[tenant1].mean())
            ts2 = wide[tenant2].fillna(wide[tenant2].mean())
            
            # Normalizar dados (importante para CCF)
            ts1_norm = (ts1 - ts1.mean()) / ts1.std()
            ts2_norm = (ts2 - ts2.mean()) / ts2.std()
            
            # Calcular CCF para lags positivos (tenant1 -> tenant2)
            ccf_vals = []
            for lag in range(max_lag+1):
                if lag == 0:
                    # Para lag=0, correlação é simétrica
                    corr = np.corrcoef(ts1_norm, ts2_norm)[0, 1]
                    ccf_vals.append(corr)
                else:
                    # Para lag > 0, calcular correlação com deslocamento
                    corr = np.corrcoef(ts1_norm[:-lag], ts2_norm[lag:])[0, 1]
                    ccf_vals.append(corr)
            
            # Calcular CCF para lags negativos (tenant2 -> tenant1) e reverter a ordem
            neg_ccf_vals = []
            for lag in range(1, max_lag+1):
                corr = np.corrcoef(ts1_norm[lag:], ts2_norm[:-lag])[0, 1]
                neg_ccf_vals.append(corr)
            
            # Combine negativo (invertido) + zero + positivo
            full_ccf = neg_ccf_vals[::-1] + ccf_vals
            ccf_results[(tenant1, tenant2)] = full_ccf
    
    return ccf_results


def plot_ccf(ccf_dict: dict, metric: str, phase: str, round_id: str, out_dir: str, max_lag: int = 20) -> list:
    """
    Plota correlação cruzada (CCF) para pares de tenants.
    
    Args:
        ccf_dict: Dicionário com resultados de CCF
        metric: Nome da métrica
        phase: Fase experimental
        round_id: ID do round
        out_dir: Diretório de saída
        max_lag: Número máximo de lags usado no cálculo
        
    Returns:
        Lista de caminhos para os gráficos gerados
    """
    if not ccf_dict:
        logger.warning(f"Sem dados de CCF para plotar: {metric}, {phase}, {round_id}")
        return []
    
    out_paths = []
    for (tenant1, tenant2), ccf_vals in ccf_dict.items():
        plt.figure(figsize=(12, 6))
        
        # Ajustar lags para centralizar em zero
        lags = np.arange(-max_lag, max_lag + 1)
        plt.stem(lags, ccf_vals, linefmt='b-', markerfmt='bo', basefmt='r-')
        
        # Adicionar linhas de referência
        plt.axhline(y=0, color='gray', linestyle='-', alpha=0.6)
        plt.axvline(x=0, color='gray', linestyle='-', alpha=0.6)
        
        # Encontrar o lag com maior correlação (em valor absoluto)
        max_idx = np.argmax(np.abs(ccf_vals))
        max_lag_val = lags[max_idx]
        max_corr = ccf_vals[max_idx]
        
        # Marcar o pico de correlação
        plt.plot(max_lag_val, max_corr, 'ro', markersize=10)
        plt.annotate(f'Max: {max_corr:.3f} @ lag {max_lag_val}', 
                   xy=(max_lag_val, max_corr),
                   xytext=(max_lag_val + 1, max_corr + 0.05),
                   arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                   fontsize=10)
        
        # Adicionar interpretação direcional
        direction = ""
        if max_lag_val > 0:
            direction = f"{tenant1} → {tenant2}"
        elif max_lag_val < 0:
            direction = f"{tenant2} → {tenant1}"
        else:
            direction = "Contemporânea"
        
        plt.title(f'Correlação cruzada entre {tenant1} e {tenant2}\n{metric} - {phase} - {round_id}\nRelação: {direction}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Defasagem (lag)')
        plt.ylabel('Correlação cruzada')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Adicionar bandas de confiança (aproximação com 1.96/sqrt(N))
        n_samples = 2 * max_lag + 1  # Aproximação
        conf_interval = 1.96 / np.sqrt(n_samples)
        plt.axhspan(-conf_interval, conf_interval, alpha=0.2, color='gray')
        
        plt.tight_layout()
        
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"ccf_{tenant1}_{tenant2}_{metric}_{phase}_{round_id}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        
        logger.info(f"Gráfico CCF salvo em {out_path}")
        out_paths.append(out_path)
    
    return out_paths


def plot_lag_scatter(df: pd.DataFrame, metric: str, phase: str, round_id: str, tenant1: str, tenant2: str, lag: int, out_dir: str) -> str | None:
    """
    Plota diagrama de dispersão (scatter) com defasagem entre dois tenants.
    
    Args:
        df: DataFrame em formato long
        metric: Nome da métrica
        phase: Fase experimental
        round_id: ID do round
        tenant1: Primeiro tenant (eixo X)
        tenant2: Segundo tenant (eixo Y)
        lag: Defasagem em períodos (>0: tenant1 lidera, <0: tenant2 lidera)
        out_dir: Diretório de saída
        
    Returns:
        Caminho para o gráfico gerado ou None
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"Sem dados para lag scatter: {metric}, {phase}, {round_id}")
        return None
    
    wide = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
    if tenant1 not in wide.columns or tenant2 not in wide.columns:
        logger.warning(f"Tenants {tenant1} ou {tenant2} não encontrados nos dados")
        return None
    
    # Handle missing data
    if wide.isna().any().any():
        wide = wide.interpolate(method='linear')
    
    ts1 = wide[tenant1]
    ts2 = wide[tenant2]
    
    if lag > 0:
        # tenant1 lidera (tenant1 em t-lag influencia tenant2 em t)
        paired_data = pd.DataFrame({
            tenant1: ts1[:-lag].values,
            tenant2: ts2[lag:].values
        })
        title = f'Lag scatter ({lag} períodos): {tenant1} → {tenant2}'
    elif lag < 0:
        # tenant2 lidera (tenant2 em t+lag influencia tenant1 em t)
        lag_abs = abs(lag)
        paired_data = pd.DataFrame({
            tenant1: ts1[lag_abs:].values,
            tenant2: ts2[:-lag_abs].values
        })
        title = f'Lag scatter ({abs(lag)} períodos): {tenant2} → {tenant1}'
    else:
        # Contemporâneo (lag = 0)
        paired_data = pd.DataFrame({
            tenant1: ts1.values,
            tenant2: ts2.values
        })
        title = f'Scatter contemporâneo: {tenant1} vs {tenant2}'
    
    plt.figure(figsize=(10, 8))
    
    # Calcular correlação nos dados emparelhados
    correlation = paired_data.corr().iloc[0, 1]
    
    # Scatter plot com linha de tendência
    sns.regplot(x=tenant1, y=tenant2, data=paired_data, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
    
    plt.title(f'{title}\n{metric} - {phase} - {round_id}\nCorrelação: {correlation:.3f}', fontsize=12, fontweight='bold')
    plt.xlabel(f'{tenant1} (t{"-"+str(lag) if lag > 0 else ""})')
    plt.ylabel(f'{tenant2} (t{"+"+str(abs(lag)) if lag < 0 else ""})')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"lag_scatter_{tenant1}_{tenant2}_{lag}_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    logger.info(f"Gráfico lag scatter salvo em {out_path}")
    return out_path
