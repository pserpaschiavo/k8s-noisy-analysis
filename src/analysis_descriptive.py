"""
Module: analysis_descriptive.py
Description: Descriptive statistics and plotting utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
from functools import lru_cache
import logging

# Setup logging
logger = logging.getLogger(__name__)

# Use the style from config
plt.style.use('tableau-colorblind10')


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


def plot_metric_timeseries_multi_tenant(df: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """
    Plota séries temporais para todos os tenants de uma métrica, fase e round, cada tenant com uma cor.
    Eixo X: tempo relativo em segundos; Y: valor da métrica; Cores: tenants.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"Sem dados para {metric}, {phase}, {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Encontrar o timestamp inicial da fase
    phase_start = subset['timestamp'].min()
    
    # Calcular a duração total da fase em segundos
    total_duration = (subset['timestamp'].max() - phase_start).total_seconds()
    
    # Sempre usar segundos para consistência
    time_unit = 1  # Usar sempre segundos
    x_label = "Segundos desde o início da fase"
    
    plt.figure(figsize=(12, 5))
    for tenant, group in subset.groupby('tenant_id'):
        group = group.sort_values('timestamp')
        # Calcular o tempo relativo desde o início da fase (não apenas do tenant)
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds() / time_unit
        plt.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', label=tenant)
    plt.title(f"Série temporal - {metric} - {phase} - {round_id}")
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.legend(title='Tenant')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_metric_barplot_by_phase(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Barplot: eixo X = fases, barras = média da métrica, cores = tenants, erro = desvio padrão.
    """
    import numpy as np
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"Sem dados para {metric} em {round_id}")
        return None
    phases = subset['experimental_phase'].unique()
    tenants = subset['tenant_id'].unique()
    means = {tenant: [] for tenant in tenants}
    stds = {tenant: [] for tenant in tenants}
    for phase in phases:
        for tenant in tenants:
            vals = subset[(subset['experimental_phase'] == phase) & (subset['tenant_id'] == tenant)]['metric_value']
            means[tenant].append(vals.mean())
            stds[tenant].append(vals.std())
    x = np.arange(len(phases))
    width = 0.8 / len(tenants)
    plt.figure(figsize=(10, 6))
    for i, tenant in enumerate(tenants):
        plt.bar(x + i*width, means[tenant], width, yerr=stds[tenant], label=tenant, capsize=5)
    plt.xticks(x + width*(len(tenants)-1)/2, phases.tolist() if isinstance(phases, np.ndarray) else phases) # Convert to list
    plt.xlabel('Fase experimental')
    plt.ylabel(f'{metric} (média ± desvio padrão)')
    plt.title(f'Comparação de {metric} por fase e tenant - {round_id}')
    plt.legend(title='Tenant')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"barplot_{metric}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_metric_timeseries_multi_tenant_all_phases(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Plota séries temporais para todos os tenants e todas as fases em um único plot, separando as fases com cores de fundo e anotações.
    Eixo X: tempo relativo (segundos desde início do round, fases em sequência); Y: valor da métrica; Cores: tenants.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)].copy()
    if subset.empty:
        logger.warning(f"Sem dados para {metric} em {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Calcular a duração total do round em segundos
    round_start = subset['timestamp'].min()
    total_duration = (subset['timestamp'].max() - round_start).total_seconds()
    
    # Sempre usar segundos para consistência
    time_unit = 1  # Usar sempre segundos
    x_label = "Segundos desde o início do round (fases em sequência)"
        
    # Ordenar por fase e timestamp
    phase_order = sorted(subset['experimental_phase'].unique())
    subset['phase_order'] = pd.Categorical(subset['experimental_phase'], categories=phase_order, ordered=True)
    subset = subset.sort_values(['phase_order', 'timestamp'])
    phases = phase_order
    tenants = subset['tenant_id'].unique()
    color_map = cm.get_cmap('tab10')
    plt.figure(figsize=(14, 6))
    # Gera cores de fundo para fases dinamicamente
    if len(phases) > 3:
        phase_colors = sns.color_palette('pastel', len(phases)).as_hex()
    else:
        phase_colors = ['#f0f0f0', '#ffcccc', '#e0e0ff']
    legend_handles = []
    t_offset = 0
    ax = plt.gca()
    for i, phase in enumerate(phases):
        phase_data = subset[subset['experimental_phase'] == phase]
        if phase_data.empty:
            continue
        t0 = phase_data.groupby('tenant_id')['timestamp'].min().min()
        t_max = 0
        for j, tenant in enumerate(tenants):
            group = phase_data[phase_data['tenant_id'] == tenant].sort_values('timestamp')
            if group.empty:
                continue
            # Calcular tempo relativo em segundos
            elapsed = (group['timestamp'] - t0).dt.total_seconds() / time_unit + t_offset
            plt.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', label=tenant if i == 0 else "_nolegend_", color=color_map(j))
            if i == 0:
                legend_handles.append(mpatches.Patch(color=color_map(j), label=tenant))
            t_max = max(t_max, elapsed.max() if not elapsed.empty else 0)
        # shade da fase
        if t_max > t_offset:
            plt.axvspan(t_offset, t_max, color=phase_colors[i % len(phase_colors)], alpha=0.25 if i != 1 else 0.45, zorder=0)
            # Anotação de fase em coordenadas de eixos (sempre topo)
            ax.annotate(
                phase,
                xy=((t_offset + t_max)/2, 1.01),
                xycoords=("data", "axes fraction"),
                ha='center', va='bottom', fontsize=11, color='gray', alpha=0.8, fontweight='bold'
            )
        t_offset = t_max
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.title(f'Série temporal multi-tenant - {metric} - {round_id} (todas as fases)')
    plt.legend(handles=legend_handles, title='Tenant', loc='best')
    plt.grid(True, axis='both')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"timeseries_multi_{metric}_allphases_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_metric_boxplot(df: pd.DataFrame, metric: str, round_id: str, out_dir: str):
    """
    Grouped boxplot: eixo X = fase, cada box = tenant (hue), y = valor da métrica.
    """
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    if subset.empty:
        return None
    plt.figure(figsize=(12, 6))
    ax = sns.boxplot(
        data=subset,
        x='experimental_phase',
        y='metric_value',
        hue='tenant_id',
        palette='tab10',
        showfliers=False
    )
    plt.xlabel('Fase experimental')
    plt.ylabel(metric)
    plt.title(f'Boxplot de {metric} por fase e tenant - {round_id}')
    plt.legend(title='Tenant', loc='best')
    plt.grid(True, axis='y')
    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"boxplot_{metric}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    return out_path


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


def plot_anomalies(df: pd.DataFrame, anomalies: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str) -> str | None:
    """
    Plota série temporal com anomalias destacadas.
    
    Args:
        df: DataFrame completo em formato long
        anomalies: DataFrame com anomalias detectadas
        metric: Nome da métrica
        phase: Fase experimental
        round_id: ID do round
        out_dir: Diretório de saída para o gráfico
        
    Returns:
        Caminho para o gráfico gerado ou None se não houver dados/anomalias
    """
    if anomalies.empty:
        logger.info(f"Sem anomalias para plotar: {metric}, {phase}, {round_id}")
        return None
    
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        logger.warning(f"Sem dados para plotar anomalias: {metric}, {phase}, {round_id}")
        return None
    
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    
    # Encontrar o timestamp inicial da fase
    phase_start = subset['timestamp'].min()
    
    # Calcular a duração total da fase em segundos
    total_duration = (subset['timestamp'].max() - phase_start).total_seconds()
    
    # Sempre usar segundos para consistência
    time_unit = 1  # Usar sempre segundos
    x_label = "Segundos desde o início da fase"
    
    plt.figure(figsize=(14, 7))
    color_map = cm.get_cmap('tab10')
    
    # Plotar séries temporais normais
    for i, (tenant, group) in enumerate(subset.groupby('tenant_id')):
        group = group.sort_values('timestamp')
        # Calcular o tempo relativo desde o início da fase
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds() / time_unit
        plt.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', 
                label=tenant, color=color_map(i), alpha=0.7)
    
    # Destacar anomalias
    for i, (tenant, group) in enumerate(anomalies.groupby('tenant_id')):
        # Usar o mesmo timestamp de início da fase para calcular o tempo relativo
        elapsed = (group['timestamp'] - phase_start).dt.total_seconds() / time_unit
        plt.scatter(elapsed, group['metric_value'], color='red', s=100, marker='X', 
                   label=f"{tenant} (anomalias)" if i == 0 else "_nolegend_", zorder=10)
    
    plt.title(f"Série temporal com anomalias - {metric} - {phase} - {round_id}")
    plt.xlabel(x_label)
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3)
    plt.legend(title='Tenant')
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"anomalies_{metric}_{phase}_{round_id}.png")
    plt.savefig(out_path)
    plt.close()
    
    logger.info(f"Gráfico de anomalias salvo em {out_path}")
    return out_path
