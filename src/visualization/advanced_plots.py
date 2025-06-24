"""
Module: src.visualization.advanced_plots
Description: Advanced visualization functions for scientific-grade multi-round analysis
"""

import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import warnings

from src.visualization_config import PUBLICATION_CONFIG

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)

# Apply global style settings
plt.rcParams.update(PUBLICATION_CONFIG.get('figure_style', {}))

def generate_consolidated_timeseries(
    df_long: pd.DataFrame,
    metric: str,
    output_dir: str,
    rounds: Optional[List[str]] = None,
    tenants: Optional[List[str]] = None,
    normalize_time: bool = True,
    add_confidence_bands: bool = True,
    add_phase_annotations: bool = True
) -> Optional[str]:
    """
    Gera time series consolidados agregando todos os rounds para uma métrica específica.
    
    Args:
        df_long: DataFrame em formato long com dados de todos os rounds
        metric: Nome da métrica a ser plotada
        output_dir: Diretório de saída para o plot
        rounds: Lista de rounds a incluir (se None, usa todos)
        tenants: Lista de tenants a incluir (se None, usa todos)
        normalize_time: Se True, usa tempo relativo desde início de cada fase
        add_confidence_bands: Se True, adiciona bandas de confiança
        add_phase_annotations: Se True, adiciona anotações das fases experimentais
        
    Returns:
        Caminho do arquivo gerado ou None em caso de erro
    """
    logger.info(f"Gerando time series consolidado para métrica: {metric}")
    
    # Filtrar dados para a métrica especificada
    metric_df = df_long[df_long['metric_name'] == metric].copy()
    
    if metric_df.empty:
        logger.warning(f"Nenhum dado encontrado para métrica '{metric}'")
        return None
    
    # Filtrar rounds e tenants se especificados
    if rounds:
        metric_df = metric_df[metric_df['round_id'].isin(rounds)]
    if tenants:
        metric_df = metric_df[metric_df['tenant_id'].isin(tenants)]
    
    if metric_df.empty:
        logger.warning(f"Nenhum dado após filtros para métrica '{metric}'")
        return None
    
    # Preparar configuração da figura
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Time Series Consolidado - {metric.replace("_", " ").title()}', 
                fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    # Obter listas únicas de rounds, tenants e fases
    unique_rounds = sorted(metric_df['round_id'].unique())
    unique_tenants = sorted(metric_df['tenant_id'].unique())
    unique_phases = sorted(metric_df['experimental_phase'].unique())
    
    # Configurar paletas de cores
    round_colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, len(unique_rounds)))
    tenant_colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(unique_tenants)))
    
    # Plot 1: Time series por Round (todos os tenants agregados)
    ax1 = axes[0]
    xlabel = 'Timestamp'  # Valor padrão
    
    for i, round_id in enumerate(unique_rounds):
        round_data = metric_df[metric_df['round_id'] == round_id]
        
        if normalize_time and 'relative_time' in round_data.columns:
            time_col = 'relative_time'
            xlabel = 'Tempo Relativo (segundos)'
        else:
            time_col = 'timestamp'
            xlabel = 'Timestamp'
        
        # Agregar por timestamp (média entre tenants)
        aggregated = round_data.groupby(time_col)['metric_value'].agg(['mean', 'std']).reset_index()
        
        # Plot da média
        ax1.plot(aggregated[time_col], aggregated['mean'], 
                color=round_colors[i], label=f'Round {round_id}', 
                linewidth=2, alpha=0.8)
        
        # Banda de confiança se solicitada
        if add_confidence_bands and not aggregated['std'].isna().all():
            ax1.fill_between(aggregated[time_col], 
                           aggregated['mean'] - aggregated['std'],
                           aggregated['mean'] + aggregated['std'],
                           color=round_colors[i], alpha=0.2)
    
    ax1.set_title('Evolução Temporal por Round (Média entre Tenants)')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Time series por Tenant (todos os rounds sobrepostos)
    ax2 = axes[1]
    for i, tenant in enumerate(unique_tenants):
        tenant_data = metric_df[metric_df['tenant_id'] == tenant]
        
        for j, round_id in enumerate(unique_rounds):
            round_tenant_data = tenant_data[tenant_data['round_id'] == round_id]
            
            if not round_tenant_data.empty:
                if normalize_time and 'relative_time' in round_tenant_data.columns:
                    time_col = 'relative_time'
                else:
                    time_col = 'timestamp'
                
                # Estilo diferente para cada round
                linestyle = ['-', '--', '-.', ':'][j % 4]
                alpha = 0.7 if j == 0 else 0.5
                
                ax2.plot(round_tenant_data[time_col], round_tenant_data['metric_value'],
                        color=tenant_colors[i], linestyle=linestyle, alpha=alpha,
                        label=f'{tenant} (R{round_id})' if j == 0 else None,
                        linewidth=1.5)
    
    ax2.set_title('Evolução por Tenant (Todos os Rounds)')
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Média móvel suavizada por Round
    ax3 = axes[2]
    window_size = max(5, len(metric_df) // 100)  # Window adaptativo
    
    for i, round_id in enumerate(unique_rounds):
        round_data = metric_df[metric_df['round_id'] == round_id]
        
        # Agregar e calcular média móvel
        if normalize_time and 'relative_time' in round_data.columns:
            time_col = 'relative_time'
        else:
            time_col = 'timestamp'
        
        aggregated = round_data.groupby(time_col)['metric_value'].mean().reset_index()
        aggregated = aggregated.sort_values(time_col)
        
        # Média móvel
        aggregated['rolling_mean'] = aggregated['metric_value'].rolling(
            window=window_size, center=True, min_periods=1).mean()
        
        ax3.plot(aggregated[time_col], aggregated['rolling_mean'],
                color=round_colors[i], label=f'Round {round_id}',
                linewidth=3, alpha=0.9)
    
    ax3.set_title(f'Tendências Suavizadas (Média Móvel - Janela: {window_size})')
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel(f'{metric.replace("_", " ").title()}')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribuição por Fase Experimental
    ax4 = axes[3]
    
    # Preparar dados para boxplot por fase e round
    plot_data = []
    for round_id in unique_rounds:
        for phase in unique_phases:
            phase_data = metric_df[
                (metric_df['round_id'] == round_id) & 
                (metric_df['experimental_phase'] == phase)
            ]
            if not phase_data.empty:
                plot_data.extend([{
                    'Round': f'R{round_id}',
                    'Phase': phase,
                    'Value': value,
                    'Round-Phase': f'R{round_id}-{phase}'
                } for value in phase_data['metric_value']])
    
    if plot_data:
        plot_df = pd.DataFrame(plot_data)
        
        # Boxplot com rounds separados
        sns.boxplot(data=plot_df, x='Phase', y='Value', hue='Round', ax=ax4)
        ax4.set_title('Distribuição por Fase Experimental e Round')
        ax4.set_xlabel('Fase Experimental')
        ax4.set_ylabel(f'{metric.replace("_", " ").title()}')
        ax4.tick_params(axis='x', rotation=45)
    
    # Anotações de fases se solicitadas
    if add_phase_annotations and normalize_time:
        _add_phase_annotations(axes[:3], metric_df, unique_phases)
    
    # Ajustar layout
    plt.tight_layout()
    
    # Salvar figura
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'consolidated_timeseries_{metric}.png')
    
    try:
        dpi = PUBLICATION_CONFIG.get('figure_style', {}).get('figure.dpi', 300)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        logger.info(f"Time series consolidado salvo em: {output_path}")
        plt.close(fig)
        return output_path
    except Exception as e:
        logger.error(f"Erro ao salvar time series consolidado: {e}")
        plt.close(fig)
        return None


def _add_phase_annotations(axes: List, df: pd.DataFrame, phases: List[str]) -> None:
    """
    Adiciona anotações visuais para marcar transições entre fases experimentais.
    """
    try:
        # Encontrar timestamps de transição entre fases
        phase_transitions = {}
        for phase in phases:
            phase_data = df[df['experimental_phase'] == phase]
            if not phase_data.empty and 'relative_time' in phase_data.columns:
                phase_transitions[phase] = {
                    'start': phase_data['relative_time'].min(),
                    'end': phase_data['relative_time'].max()
                }
        
        # Adicionar linhas verticais para transições
        phase_colors = ['red', 'orange', 'green', 'blue', 'purple']
        
        for ax in axes:
            for i, (phase, times) in enumerate(phase_transitions.items()):
                color = phase_colors[i % len(phase_colors)]
                
                # Linha de início da fase
                ax.axvline(x=times['start'], color=color, linestyle='--', 
                          alpha=0.6, linewidth=1)
                
                # Anotação da fase
                ax.annotate(phase, xy=(times['start'], ax.get_ylim()[1]), 
                           xytext=(5, -5), textcoords='offset points',
                           ha='left', va='top', fontsize=8, color=color,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                   edgecolor=color, alpha=0.8))
                
    except Exception as e:
        logger.warning(f"Erro ao adicionar anotações de fase: {e}")


def generate_all_consolidated_timeseries(
    df_long: pd.DataFrame,
    output_dir: str,
    **kwargs
) -> Dict[str, str]:
    """
    Gera time series consolidados para todas as métricas disponíveis.
    
    Args:
        df_long: DataFrame em formato long
        output_dir: Diretório de saída
        **kwargs: Argumentos adicionais para generate_consolidated_timeseries
        
    Returns:
        Dicionário mapeando métrica -> caminho do arquivo gerado
    """
    logger.info("Gerando time series consolidados para todas as métricas...")
    
    timeseries_dir = os.path.join(output_dir, 'timeseries')
    os.makedirs(timeseries_dir, exist_ok=True)
    
    results = {}
    metrics = sorted(df_long['metric_name'].unique())
    
    for metric in metrics:
        try:
            output_path = generate_consolidated_timeseries(
                df_long=df_long,
                metric=metric,
                output_dir=timeseries_dir,
                **kwargs
            )
            if output_path:
                results[metric] = output_path
                logger.info(f"✅ Time series gerado para {metric}")
            else:
                logger.warning(f"❌ Falha ao gerar time series para {metric}")
                
        except Exception as e:
            logger.error(f"❌ Erro ao processar métrica {metric}: {e}")
    
    logger.info(f"Time series consolidados concluídos: {len(results)} métricas processadas")
    return results
