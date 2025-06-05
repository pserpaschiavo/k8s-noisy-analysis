"""
Module: analysis_anomaly.py
Description: Implementa métodos para detecção de anomalias em séries temporais.

Este módulo define funções para identificar observações anômalas em séries temporais
usando diferentes técnicas, como detecção baseada em distribuição estatística,
decomposição temporal e métodos de aprendizado de máquina.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import IsolationForest
from scipy import stats

# Usando import condicional para evitar dependência circular
try:
    from src.pipeline import PipelineStage
    pipeline_available = True
except ImportError:
    pipeline_available = False
    # Classe base mock para quando pipeline.py não pode ser importado
    class PipelineStage:
        def __init__(self, *args, **kwargs):
            pass

# Configuração de logging
logger = logging.getLogger("analysis_anomaly")

class AnomalyDetectionStage(PipelineStage):
    """
    Estágio do pipeline para detectar anomalias nas séries temporais.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            name="Anomaly Detection", 
            description="Detecção de anomalias em séries temporais multi-tenant"
        )
        self.output_dir = output_dir
    
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementação da detecção de anomalias.
        
        Args:
            context: Contexto atual do pipeline com resultados anteriores.
            
        Returns:
            Contexto atualizado com resultados da detecção de anomalias.
        """
        self.logger.info("Iniciando detecção de anomalias")
        
        # Verificar se temos dados para analisar
        df_long = context.get('df_long')
        if df_long is None:
            self.logger.warning("DataFrame principal não encontrado no contexto")
            context['error'] = "DataFrame principal não disponível para detecção de anomalias"
            return context
        
        # Diretório de saída
        output_dir = self.output_dir or context.get('output_dir')
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), 'outputs', 'plots', 'anomaly_detection')
        else:
            output_dir = os.path.join(output_dir, 'plots', 'anomaly_detection')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extrair dados necessários
        experiment_id = context.get('experiment_id', df_long['experiment_id'].iloc[0])
        metrics = context.get('selected_metrics', df_long['metric_name'].unique())
        tenants = context.get('selected_tenants', df_long['tenant_id'].unique())
        rounds = df_long['round_id'].unique()
        
        anomaly_metrics = {}
        visualization_paths = []
        
        # Para cada combinação de métrica/round
        for metric in metrics:
            anomalies_df = []
            
            for round_id in rounds:
                for phase in df_long['experimental_phase'].unique():
                    # Processar cada tenant
                    for tenant in tenants:
                        # Filtrar dados
                        tenant_data = df_long[(df_long['metric_name'] == metric) & 
                                             (df_long['round_id'] == round_id) & 
                                             (df_long['experimental_phase'] == phase) &
                                             (df_long['tenant_id'] == tenant)]
                        
                        if len(tenant_data) < 10:
                            continue
                        
                        try:
                            # Detectar anomalias usando Z-score
                            anomalies = detect_anomalies_zscore(
                                tenant_data, z_threshold=3.0
                            )
                            
                            if not anomalies.empty:
                                anomalies['tenant_id'] = tenant
                                anomalies['round_id'] = round_id
                                anomalies['experimental_phase'] = phase
                                anomalies_df.append(anomalies)
                            
                            # Gerar visualização
                            fig_path = plot_anomalies(
                                tenant_data, anomalies, 
                                metric, tenant, round_id, phase,
                                output_dir
                            )
                            visualization_paths.append(fig_path)
                            
                        except Exception as e:
                            self.logger.error(f"Erro ao detectar anomalias para {tenant}, {metric}, {round_id}: {str(e)}")
            
            # Consolida todas as anomalias desta métrica
            if anomalies_df:
                anomaly_metrics[metric] = pd.concat(anomalies_df, ignore_index=True)
        
        # Armazenar resultados no contexto
        context['anomaly_metrics'] = anomaly_metrics
        context['anomaly_visualization_paths'] = visualization_paths
        
        self.logger.info(f"Detecção de anomalias concluída. {len(visualization_paths)} visualizações geradas.")
        
        return context


def detect_anomalies_zscore(
    df: pd.DataFrame,
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detecta anomalias usando o método de Z-score.
    
    Args:
        df: DataFrame com dados de um tenant para uma métrica
        z_threshold: Limiar de Z-score para considerar um ponto como anomalia
        
    Returns:
        DataFrame com as anomalias detectadas
    """
    # Verifica se há dados suficientes
    if len(df) < 4:
        logger.warning("Dados insuficientes para detecção de anomalias")
        return pd.DataFrame()
    
    # Calcula estatísticas
    mean_val = df['metric_value'].mean()
    std_val = df['metric_value'].std()
    
    # Evita divisão por zero
    if std_val == 0:
        logger.warning("Desvio padrão zero detectado, não é possível calcular z-score")
        return pd.DataFrame()
    
    # Calcula z-score
    df['z_score'] = (df['metric_value'] - mean_val) / std_val
    
    # Identifica anomalias
    anomalies = df[abs(df['z_score']) > z_threshold].copy()
    
    return anomalies


def plot_anomalies(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    metric: str,
    tenant: str,
    round_id: str,
    phase: str,
    output_dir: str
) -> str:
    """
    Gera visualização de série temporal com anomalias destacadas.
    
    Args:
        df: DataFrame com série temporal completa
        anomalies: DataFrame com pontos anômalos
        metric: Nome da métrica
        tenant: ID do tenant
        round_id: ID do round
        phase: Fase experimental
        output_dir: Diretório para salvar a visualização
        
    Returns:
        Caminho para o arquivo de imagem gerado
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Converter timestamp para datetime se necessário
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Encontrar o timestamp inicial da fase
    phase_start = df['timestamp'].min()
    
    # Calcular a duração total da fase em segundos
    total_duration = (df['timestamp'].max() - phase_start).total_seconds()
    
    # Sempre usar segundos para consistência
    time_unit = 1  # Usar sempre segundos
    x_label = "Segundos desde o início da fase"
    
    # Calcular tempos relativos
    elapsed = (df['timestamp'] - phase_start).dt.total_seconds() / time_unit
    
    # Plot da série temporal
    ax.plot(elapsed, df['metric_value'], 'b-', label='Série Temporal')
    
    # Destaca anomalias
    if not anomalies.empty:
        # Garantir formato datetime
        if not pd.api.types.is_datetime64_any_dtype(anomalies['timestamp']):
            anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'], errors='coerce')
        
        # Calcular tempo relativo para anomalias
        anomaly_elapsed = (anomalies['timestamp'] - phase_start).dt.total_seconds() / time_unit
        ax.scatter(anomaly_elapsed, anomalies['metric_value'], color='red', s=80, alpha=0.6, label='Anomalias')
    
    # Adiciona linha média e bandas de confiança
    mean_val = df['metric_value'].mean()
    std_val = df['metric_value'].std()
    
    ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label='Média')
    ax.axhline(y=mean_val + 3*std_val, color='orange', linestyle=':', alpha=0.5, label='Limiar (3σ)')
    ax.axhline(y=mean_val - 3*std_val, color='orange', linestyle=':', alpha=0.5)
    
    # Formatação do gráfico
    ax.set_title(f'Detecção de Anomalias: {metric} - {tenant}', fontweight='bold')
    plt.suptitle(f'Round: {round_id}, Fase: {phase}', fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'Valor ({metric})')
    
    # Formata timestamps
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    # Adiciona legenda
    plt.legend(loc='best')
    
    # Salva o gráfico
    safe_metric = metric.replace('/', '_').replace(' ', '_')
    safe_tenant = tenant.replace('-', '_')
    safe_phase = phase.replace(' ', '_')
    file_name = f"anomaly_detection_{safe_metric}_{safe_tenant}_{safe_phase}_{round_id}.png"
    fig_path = os.path.join(output_dir, file_name)
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig_path
