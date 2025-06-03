"""
Module: analysis_descriptive.py
Description: Descriptive statistics and plotting utilities for multi-tenant time series analysis.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
plt.style.use('tableau-colorblind10')


def compute_descriptive_stats(df: pd.DataFrame, groupby_cols=None) -> pd.DataFrame:
    """
    Compute descriptive statistics (count, mean, std, min, max) for metric_value,
    grouped by the specified columns (e.g., ['tenant_id', 'metric_name', 'experimental_phase', 'round_id']).
    """
    if groupby_cols is None:
        groupby_cols = ['tenant_id', 'metric_name', 'experimental_phase', 'round_id']
    stats = df.groupby(groupby_cols)['metric_value'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
    return stats


def plot_metric_timeseries_multi_tenant(df: pd.DataFrame, metric: str, phase: str, round_id: str, out_dir: str):
    """
    Plota séries temporais para todos os tenants de uma métrica, fase e round, cada tenant com uma cor.
    Eixo X: tempo relativo em segundos; Y: valor da métrica; Cores: tenants.
    """
    subset = df[(df['metric_name'] == metric) & (df['experimental_phase'] == phase) & (df['round_id'] == round_id)]
    if subset.empty:
        print(f"[DEBUG] Sem dados para {metric}, {phase}, {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
    plt.figure(figsize=(12, 5))
    for tenant, group in subset.groupby('tenant_id'):
        group = group.sort_values('timestamp')
        t0 = group['timestamp'].iloc[0]
        elapsed = (group['timestamp'] - t0).dt.total_seconds()
        plt.plot(elapsed, group['metric_value'], marker='o', markersize=3, linestyle='-', label=tenant)
    plt.title(f"Série temporal - {metric} - {phase} - {round_id}")
    plt.xlabel("Segundos desde o início da fase/round")
    plt.ylabel(metric)
    plt.legend(title='Tenant')
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
        print(f"[DEBUG] Sem dados para {metric} em {round_id}")
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
    plt.xticks(x + width*(len(tenants)-1)/2, phases)
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
        print(f"[DEBUG] Sem dados para {metric} em {round_id}")
        return None
    if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
        subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
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
            elapsed = (group['timestamp'] - t0).dt.total_seconds() + t_offset
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
    plt.xlabel('Segundos desde o início do round (fases em sequência)')
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
