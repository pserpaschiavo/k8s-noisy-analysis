"""
Module: analysis_descriptive.py
Description: Descriptive statistics and plotting utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
# Removendo configuração local do matplotlib
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from functools import lru_cache
import logging

from src.utils import configure_matplotlib
from src.pipeline_stage import PipelineStage
from src.config import PipelineConfig
from typing import Dict, Any, List
from src.visualization.descriptive_plots import (
    plot_metric_timeseries_multi_tenant_all_phases,
    plot_metric_barplot_by_phase,
    plot_metric_boxplot
)

# Configuração centralizada do matplotlib
configure_matplotlib()

# Setup logging
logger = logging.getLogger(__name__)

# Use the style from config
# plt.style.use('tableau-colorblind10') # Removed to use centralized config


def compute_descriptive_stats(df, groupby_cols=None) -> pd.DataFrame:
    """
    Compute descriptive statistics (count, mean, std, min, max, skewness, kurtosis) for metric_value,
    grouped by the specified columns.
    
    Args:
        df: DataFrame com os dados a serem analisados
        groupby_cols: List of columns to group by
        
    Returns:
        DataFrame with descriptive statistics
    """
    
    if groupby_cols is None:
        groupby_cols = ['tenant_id', 'metric_name', 'experimental_phase', 'round_id']
    
    # More comprehensive statistics including skewness and kurtosis
    stats = df.groupby(groupby_cols)['metric_value'].agg([
        'count', 'mean', 'std', 'min', 'max',
        ('skewness', lambda x: x.skew()),
        ('kurtosis', lambda x: x.kurtosis())
    ]).reset_index()
    
    logger.info(f"Computed descriptive stats for {len(groupby_cols)} groups with {len(stats)} rows")
    return stats


class DescriptiveAnalysisStage(PipelineStage):
    """
    Pipeline stage for descriptive analysis.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("descriptive_analysis", "Descriptive statistics and anomaly detection")
        self.config = config

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the descriptive analysis stage.
        """
        self.logger.info("Starting descriptive analysis...")
        df = context.get('data')
        if df is None or df.empty:
            self.logger.error("DataFrame 'data' not available for descriptive analysis.")
            return context

        # Compute descriptive stats
        descriptive_stats = compute_descriptive_stats(df)
        context['descriptive_stats'] = descriptive_stats
        
        # Save to CSV
        output_dir = self.config.get_output_dir("descriptive_analysis")
        csv_path = os.path.join(output_dir, "descriptive_stats.csv")
        descriptive_stats.to_csv(csv_path, index=False)
        self.logger.info(f"Descriptive stats saved to {csv_path}")

        # --- Reintegrar a geração de plots ---
        plot_paths = []
        
        # Extrair round_id e métricas do contexto/config
        if 'round_id' in df.columns and not df['round_id'].empty:
            round_id = df['round_id'].unique()[0]
        else:
            self.logger.warning("Could not determine round_id from DataFrame. Skipping plot generation.")
            return context

        selected_metrics = self.config.get_selected_metrics()
        if not selected_metrics:
            self.logger.warning("No selected metrics found in config. Skipping plot generation.")
            return context

        for metric in selected_metrics:
            self.logger.info(f"Generating descriptive plots for metric: {metric} in round: {round_id}")
            
            # Plot: Time series com todas as fases
            path = plot_metric_timeseries_multi_tenant_all_phases(df, metric, round_id, output_dir)
            if path:
                plot_paths.append(path)

            # Plot: Barplot por fase
            path = plot_metric_barplot_by_phase(df, metric, round_id, output_dir)
            if path:
                plot_paths.append(path)

            # Plot: Boxplot por fase
            path = plot_metric_boxplot(df, metric, round_id, output_dir)
            if path:
                plot_paths.append(path)
        
        context['descriptive_plots'] = plot_paths
        self.logger.info(f"Generated {len(plot_paths)} descriptive plots.")
        # --- Fim da reintegração ---

        self.logger.info("Descriptive analysis completed.")
        return context


def detect_anomalies(df: pd.DataFrame, metric: str, phase: str, round_id: str, window_size: int = 10, threshold: float = 2.0) -> pd.DataFrame:
    """
    Detecta anomalias nas séries temporais usando rolling window e Z-score.
    
    Args:
        df: DataFrame em formato long
        metric: Nome da métrica para analisar
        phase: Fase experimental para filtrar
        round_id: ID do round para filtrar
        window_size: Tamanho da janela para médias móveis
        threshold: Limiar de Z-score para considerar um ponto como anomalia
        
    Returns:
        DataFrame com as anomalias detectadas
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)].copy()
    if subset.empty:
        logger.warning(f"Sem dados para detecção de anomalias: {metric}, {phase}, {round_id}")
        return pd.DataFrame()
    
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Ordenar por timestamp para rolling window
    subset = subset.sort_values(['tenant_id', 'timestamp'])
    
    # Inicializar DataFrame para resultados
    anomalies = pd.DataFrame()
    
    # Processar cada tenant separadamente
    for tenant, group in subset.groupby('tenant_id'):
        # Calcular estatísticas rolling (média e desvio padrão)
        rolling_mean = group['metric_value'].rolling(window=window_size, center=True).mean()
        rolling_std = group['metric_value'].rolling(window=window_size, center=True).std()
        
        # Substituir NaN no início/fim do rolling por médias globais
        rolling_mean = rolling_mean.fillna(group['metric_value'].mean())
        rolling_std = rolling_std.fillna(group['metric_value'].std())
        rolling_std = rolling_std.replace(0, group['metric_value'].std())  # Evitar divisão por zero
        
        # Calcular Z-score
        z_scores = np.abs((group['metric_value'] - rolling_mean) / rolling_std)
        
        # Identificar anomalias
        is_anomaly = z_scores > threshold
        
        # Filtrar anomalias e adicionar ao DataFrame de resultados
        tenant_anomalies = group[is_anomaly].copy()
        if not tenant_anomalies.empty:
            tenant_anomalies['z_score'] = z_scores[is_anomaly]
            anomalies = pd.concat([anomalies, tenant_anomalies])
    
    # Ordenar por gravidade (Z-score)
    if not anomalies.empty:
        anomalies = anomalies.sort_values('z_score', ascending=False)
        logger.info(f"Detectadas {len(anomalies)} anomalias para {metric} em {phase}, {round_id}")
    else:
        logger.info(f"Nenhuma anomalia detectada para {metric} em {phase}, {round_id}")
    
    return anomalies
