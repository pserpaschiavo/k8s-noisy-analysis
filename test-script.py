from __future__ import annotations

import argparse
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math


# ===================== Configuração global de estilo =====================
def configure_style():
    sns.set_theme(style="whitegrid", context="talk")
    sns.set_palette("colorblind")  # Okabe–Ito-friendly
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })


# ===================== Utilidades =====================
def first_or_none(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(path: Path, tight_rect=None):
    if tight_rect:
        plt.tight_layout(rect=tight_rect)
    else:
        plt.tight_layout()
    plt.savefig(path)
    plt.close()


def safe_filename(text: str) -> str:
    import re
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text))
    s = re.sub(r"_+", "_", s).strip("._-")
    return s if s else "plot"


# Pretty metric mapping used across analyses
METRIC_NAME_MAP = {
    'cpu_usage': "CPU Usage",
    'memory_usage': "Memory Usage",
    'network_receive': "Network RX",
    'network_transmit': "Network TX",
    'disk_io_total': "Disk I/O",
}


def _pretty_metric(m):
    if pd.isna(m):
        return m
    key = str(m)
    return METRIC_NAME_MAP.get(key, m)


def _phase_sort_key(phase: str) -> tuple:
    if isinstance(phase, str) and '-' in phase:
        head, tail = phase.split('-', 1)
        try:
            return (int(head), tail)
        except ValueError:
            return (999, phase)
    return (999, str(phase))


def _grid_dims(n: int, max_cols: int = 4) -> tuple[int, int]:
    if n <= 0:
        return (1, 1)
    cols = min(max_cols, max(1, n))
    rows = int(math.ceil(n / cols))
    return rows, cols


# ===================== Análise Descritiva =====================
def analyze_descriptive(parquet_path: str | None, figs_dir: Path, csv_dir: Path, max_points: int = 200_000):
    print("[DESCRIPTIVE] Starting descriptive analysis...")
    df = None
    if parquet_path and Path(parquet_path).exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  WARNING: Failed to read parquet '{parquet_path}': {e}")

    if df is None or df.empty:
        print("  WARNING: Parquet not available. Skipping raw descriptive plots.")
        return

    # Amostragem para evitar gráficos pesados
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).sort_values("timestamp")

    # Distribuição por fase e métrica (ECDF)
    try:
        from matplotlib.lines import Line2D
        phases_all = sorted(df['experimental_phase'].dropna().unique().tolist(), key=_phase_sort_key)
        for metric, sub in df.groupby("metric_name", sort=False):
            mlabel = str(_pretty_metric(metric))
            present = [ph for ph in phases_all if ph in set(sub['experimental_phase'].dropna().unique())]
            if not present:
                continue
            plt.figure(figsize=(12, 7))
            sns.ecdfplot(
                data=sub,
                x="metric_value",
                hue="experimental_phase",
                stat="proportion",
                hue_order=present,
                linewidth=1.6,
            )
            plt.title(f"ECDF by Phase — {mlabel}")
            plt.xlabel("Metric Value"); plt.ylabel("Proportion")
            plt.ylim(0, 1)
            # Manual legend in numeric phase order
            palette = sns.color_palette(n_colors=len(present))
            proxies = [Line2D([0], [0], color=palette[i], lw=2) for i in range(len(present))]
            plt.legend(proxies, present, title='Phase', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
            save_fig(figs_dir / f"figure_desc_ecdf_{safe_filename(mlabel)}.png", tight_rect=[0, 0, 0.85, 1])

            # CSV: ECDF points per phase
            try:
                records = []
                for ph in present:
                    filt = sub.loc[sub['experimental_phase'] == ph]
                    ser = pd.to_numeric(filt['metric_value'], errors='coerce').dropna().to_numpy()
                    if ser.size == 0:
                        continue
                    x = np.sort(ser.astype(float, copy=False))
                    y = np.arange(1, x.size + 1) / x.size
                    for xv, yv in zip(x, y):
                        records.append({
                            'metric_name': metric,
                            'metric_pretty': mlabel,
                            'phase': ph,
                            'value': float(xv),
                            'proportion': float(yv),
                        })
                if records:
                    ecdf_df = pd.DataFrame.from_records(records)
                    ecdf_path = csv_dir / f"ecdf_{safe_filename(mlabel)}.csv"
                    ecdf_df.to_csv(ecdf_path, index=False)
            except Exception as e:
                print(f"  WARNING: ECDF CSV export failed for {metric}: {e}")
    except Exception as e:
        print(f"  WARNING: ECDF plotting failed: {e}")

    # Boxplot por fase (agregado por métrica)
    try:
        for metric, sub in df.groupby("metric_name", sort=False):
            mlabel = str(_pretty_metric(metric))
            plt.figure(figsize=(12, 6))
            phases = sorted(sub['experimental_phase'].dropna().unique().tolist(), key=_phase_sort_key)
            sns.boxplot(data=sub, x="experimental_phase", y="metric_value", order=phases)
            plt.title(f"Distribution by Phase — {mlabel}")
            plt.xlabel("Phase"); plt.ylabel(str(mlabel))
            plt.xticks(rotation=20)
            save_fig(figs_dir / f"figure_desc_box_{safe_filename(mlabel)}.png")

            # CSV: Box stats per phase
            try:
                def q1(s):
                    return s.quantile(0.25)
                def q3(s):
                    return s.quantile(0.75)
                stats = sub.groupby('experimental_phase')['metric_value'].agg(
                    count='count', mean='mean', std='std', min='min', q1=q1, median='median', q3=q3, max='max'
                ).reset_index().rename(columns={'experimental_phase': 'phase'})
                stats.insert(0, 'metric_name', metric)
                stats.insert(1, 'metric_pretty', mlabel)
                box_path = csv_dir / f"box_stats_{safe_filename(mlabel)}.csv"
                stats.to_csv(box_path, index=False)
            except Exception as e:
                print(f"  WARNING: Box stats CSV export failed for {metric}: {e}")
    except Exception as e:
        print(f"  WARNING: Boxplot plotting failed: {e}")


# ===================== Modo compacto — Descritivo =====================
def analyze_descriptive_compact(parquet_path: str | None, paper_dir: Path, max_points: int = 200_000):
    print("[DESCRIPTIVE-COMPACT] Generating ECDF+Box combined plots...")
    df = None
    if parquet_path and Path(parquet_path).exists():
        try:
            df = pd.read_parquet(parquet_path)
        except Exception as e:
            print(f"  WARNING: Failed to read parquet '{parquet_path}': {e}")
    if df is None or df.empty:
        print("  WARNING: Parquet not available. Skipping compact descriptive plots.")
        return

    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42).sort_values("timestamp")

    phases_all = sorted(df['experimental_phase'].dropna().unique().tolist(), key=_phase_sort_key)
    for metric, sub in df.groupby("metric_name", sort=False):
        mlabel = str(_pretty_metric(metric))
        present = [ph for ph in phases_all if ph in set(sub['experimental_phase'].dropna().unique())]
        if not present:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        # ECDF
        sns.ecdfplot(
            data=sub,
            x="metric_value",
            hue="experimental_phase",
            stat="proportion",
            hue_order=present,
            linewidth=1.6,
            ax=axes[0],
        )
        axes[0].set_title("ECDF by Phase")
        axes[0].set_xlabel("Metric Value"); axes[0].set_ylabel("Proportion")
        axes[0].set_ylim(0, 1)
        # Box
        phases = present
        sns.boxplot(data=sub, x="experimental_phase", y="metric_value", order=phases, ax=axes[1])
        axes[1].set_title("Distribution by Phase")
        axes[1].set_xlabel("Phase"); axes[1].set_ylabel(str(mlabel)); axes[1].tick_params(axis='x', rotation=20)
        # Legend outside (from ECDF)
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, title='Phase', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        fig.suptitle(f"Distribution by Phase — {mlabel}", y=1.02, weight='bold')
        plt.tight_layout(rect=[0, 0, 0.85, 0.96])
        out = paper_dir / f"desc_{safe_filename(mlabel)}.png"
        plt.savefig(out)
        plt.close()
        print(f"  SUCCESS: {out.name} generated.")


# ===================== Impacto (Cohen d e % Δ) =====================
def analyze_impact(figs_dir: Path, outputs_root_glob: str, csv_dir: Path):
    print("[IMPACT] Starting multi-round impact analysis...")

    files = sorted(glob.glob(f"{outputs_root_glob}/round-*/impact_analysis/csv/impact_analysis_summary_round-*.csv"))
    if not files:
        print("  WARNING: No impact CSVs found.")
        return

    df_full = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_full.rename(columns={
        'experimental_phase': 'phase',
        'percentage_change': 'impact_pct',
        'cohen_d': 'effect_size'
    }, inplace=True)

    # Filter only noise phases (exclude baseline and recovery)
    df_noise = df_full[~df_full['phase'].str.contains('Recovery|Baseline', case=False, na=False)].copy()

    # Pretty labels for metrics
    df_noise['metric_pretty'] = df_noise['metric_name'].map(_pretty_metric)

    # Compare tenants per phase for each metric (use original phase)
    agg_stats = df_noise.groupby(['metric_pretty', 'phase', 'tenant_id']).agg(
        mean_impact=('impact_pct', 'mean'),
        std_impact=('impact_pct', 'std'),
        mean_effect_size=('effect_size', 'mean')
    ).reset_index()
    agg_stats['std_impact'] = agg_stats['std_impact'].fillna(0)

    # Orders
    metric_order_labels = [METRIC_NAME_MAP.get(k, k) for k in METRIC_NAME_MAP.keys()]
    phase_order = sorted(agg_stats['phase'].dropna().unique(), key=_phase_sort_key)
    tenant_order = sorted(agg_stats['tenant_id'].dropna().unique())

    # Export aggregated stats CSV (combined)
    try:
        out_csv = csv_dir / "impact_aggregated_stats.csv"
        agg_stats.rename(columns={'metric_pretty': 'metric'}, inplace=False).to_csv(out_csv, index=False)
    except Exception as e:
        print(f"  WARNING: Failed to export impact aggregated stats CSV: {e}")

    # Create one figure per metric
    metrics_in_data = [m for m in metric_order_labels if m in set(agg_stats['metric_pretty'])]
    for metric in metrics_in_data:
        data = agg_stats[agg_stats['metric_pretty'] == metric]
        present_phases = [ph for ph in phase_order if ph in set(data['phase'])]
        if not present_phases:
            continue
        plt.figure(figsize=(12, 6))
        ax = plt.gca()
        sns.barplot(
            data=data,
            x='phase', y='mean_impact', hue='tenant_id',
            ax=ax, dodge=True, capsize=.05, palette="colorblind",
            order=present_phases, hue_order=tenant_order, errorbar=None
        )
        # Manual error bars only (no effect size labels)
        for i, ph in enumerate(present_phases):
            for j, tn in enumerate(tenant_order):
                row = data[(data['phase'] == ph) & (data['tenant_id'] == tn)]
                if row.empty:
                    continue
                r = row.iloc[0]
                mean_val = float(r['mean_impact']) if pd.notna(r['mean_impact']) else 0.0
                std_val = float(r['std_impact']) if ('std_impact' in r and pd.notna(r['std_impact'])) else 0.0
                xloc = i + (j - (len(tenant_order)-1)/2) * 0.8/len(tenant_order)
                ax.errorbar(x=xloc, y=mean_val, yerr=std_val, fmt='none', capsize=4, color='black', elinewidth=1.2)
        ax.axhline(0, color='black', lw=1.2, linestyle='--')
        ax.set_title(f"Aggregated Impact by Phase — {metric}", weight='bold')
        ax.set_xlabel('Phase'); ax.set_ylabel('Mean Percent Impact (%)')
        ax.tick_params(axis='x', rotation=20)
        ax.legend(title='Tenant', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
        out_name = f"impact_aggregated_{safe_filename(str(metric))}.png"
        save_fig(figs_dir / out_name, tight_rect=[0, 0, 0.85, 1])
    print(f"  SUCCESS: {out_name} generated.")


# ===================== Modo compacto — Impacto =====================
def analyze_impact_compact(paper_dir: Path, outputs_root_glob: str):
    print("[IMPACT-COMPACT] Generating multi-panel impact plot...")
    files = sorted(glob.glob(f"{outputs_root_glob}/round-*/impact_analysis/csv/impact_analysis_summary_round-*.csv"))
    if not files:
        print("  WARNING: No impact CSVs found.")
        return
    df_full = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df_full.rename(columns={
        'experimental_phase': 'phase',
        'percentage_change': 'impact_pct',
        'cohen_d': 'effect_size'
    }, inplace=True)
    df_noise = df_full[~df_full['phase'].str.contains('Recovery|Baseline', case=False, na=False)].copy()
    df_noise['metric_pretty'] = df_noise['metric_name'].map(_pretty_metric)
    agg_stats = df_noise.groupby(['metric_pretty', 'phase', 'tenant_id']).agg(
        mean_impact=('impact_pct', 'mean')
    ).reset_index()
    metric_order_labels = [METRIC_NAME_MAP.get(k, k) for k in METRIC_NAME_MAP.keys()]
    metrics = [m for m in metric_order_labels if m in set(agg_stats['metric_pretty'])]
    if not metrics:
        print("  WARNING: No metrics to plot for impact.")
        return
    rows, cols = _grid_dims(len(metrics), max_cols=3)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols+2, 4.6*rows), squeeze=False)
    phase_order = sorted(agg_stats['phase'].dropna().unique(), key=_phase_sort_key)
    tenant_order = sorted(agg_stats['tenant_id'].dropna().unique())
    for idx, metric in enumerate(metrics):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        data = agg_stats[agg_stats['metric_pretty'] == metric]
        present_phases = [ph for ph in phase_order if ph in set(data['phase'])]
        sns.barplot(
            data=data,
            x='phase', y='mean_impact', hue='tenant_id', ax=ax,
            order=present_phases, hue_order=tenant_order, errorbar=None
        )
        ax.axhline(0, color='black', lw=1.0, linestyle='--')
        ax.set_title(metric)
        ax.set_xlabel('Phase'); ax.set_ylabel('Mean % Impact')
        ax.tick_params(axis='x', rotation=20)
        if r != 0:
            ax.set_title(metric, fontsize=12)
    # Remove empty axes
    for j in range(len(metrics), rows*cols):
        r, c = divmod(j, cols)
        axes[r][c].axis('off')
    # Shared legend
    handles, labels = axes[0][0].get_legend_handles_labels()
    for axrow in axes:
        for ax in axrow:
            ax.legend_.remove() if ax.get_legend() else None
    if handles:
        fig.legend(handles, labels, title='Tenant', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.suptitle('Aggregated Impact by Phase — All Metrics', y=1.02, weight='bold')
    plt.tight_layout(rect=[0, 0, 0.85, 0.96])
    out = paper_dir / 'impact_all_metrics.png'
    plt.savefig(out)
    plt.close()
    print(f"  SUCCESS: {out.name} generated.")


# ===================== Correlação (gerar suspeitos) =====================
def analyze_correlation(figs_dir: Path, outputs_root_glob: str, phases: list[str] | None, csv_dir: Path):
    print("[CORRELATION] Starting correlation analysis...")
    corr_path = first_or_none(f"{outputs_root_glob}/multi_round_analysis/multi_round_correlation_all.csv")
    if not corr_path:
        print("  WARNING: Consolidated correlation file not found.")
        return

    df_corr = pd.read_csv(corr_path)
    phase_list = phases or sorted(df_corr['phase'].dropna().unique().tolist())
    metric_col = 'metric_name' if 'metric_name' in df_corr.columns else ('metric' if 'metric' in df_corr.columns else None)
    metric_values = sorted(df_corr[metric_col].dropna().unique().tolist()) if metric_col else [None]

    for metric_val in metric_values:
        mlabel = _pretty_metric(metric_val) if metric_val is not None else 'All Metrics'
        for ph in phase_list:
            out_name = ""
            sel = (df_corr['phase'] == ph)
            if metric_val is not None:
                sel &= (df_corr[metric_col] == metric_val)
            df_phase = df_corr[sel].copy()
            if df_phase.empty:
                print(f"  WARNING: No correlation data for phase '{ph}' and metric '{metric_val}'.")
                continue

            all_tenants = sorted(list(set(df_phase['tenant1']) | set(df_phase['tenant2'])))
            corr_pivot = df_phase.pivot_table(index='tenant1', columns='tenant2', values='mean_correlation')
            corr_pivot = corr_pivot.reindex(index=all_tenants, columns=all_tenants)
            symmetric = corr_pivot.combine_first(corr_pivot.T).fillna(0)
            np.fill_diagonal(symmetric.values, 1)

            plt.figure(figsize=(12, 10))
            sns.heatmap(symmetric, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)
            plt.title(f'Mean Correlation Heatmap — {mlabel} (Phase: {ph})', weight='bold', pad=18)
            plt.xlabel('Tenant'); plt.ylabel('Tenant')
            mslug = safe_filename(str(mlabel)) if metric_val is not None else 'all'
            out_name = f"correlation_heatmap_{mslug}_{ph}.png"
            save_fig(figs_dir / out_name)
            print(f"  SUCCESS: {out_name} generated.")

            # CSV: correlation symmetric matrix
            try:
                corr_csv = csv_dir / f"correlation_matrix_{mslug}_{ph}.csv"
                symmetric.to_csv(corr_csv)
            except Exception as e:
                print(f"  WARNING: Failed to export correlation matrix CSV for phase {ph}, metric {metric_val}: {e}")


# ===================== Modo compacto — Correlação =====================
def analyze_correlation_compact(paper_dir: Path, outputs_root_glob: str, phases: list[str] | None):
    print("[CORRELATION-COMPACT] Generating per-metric phase grids...")
    corr_path = first_or_none(f"{outputs_root_glob}/multi_round_analysis/multi_round_correlation_all.csv")
    if not corr_path:
        print("  WARNING: Consolidated correlation file not found.")
        return
    df_corr = pd.read_csv(corr_path)
    phase_list = phases or sorted(df_corr['phase'].dropna().unique().tolist(), key=_phase_sort_key)
    metric_col = 'metric_name' if 'metric_name' in df_corr.columns else ('metric' if 'metric' in df_corr.columns else None)
    metric_values = sorted(df_corr[metric_col].dropna().unique().tolist()) if metric_col else [None]
    for metric_val in metric_values:
        mlabel = _pretty_metric(metric_val) if metric_val is not None else 'All Metrics'
        n = len(phase_list)
        rows, cols = _grid_dims(n, max_cols=4)
        fig, axes = plt.subplots(rows, cols, figsize=(4.8*cols+2, 4.4*rows), squeeze=False)
        # Shared vmin/vmax
        vmin, vmax = -1, 1
        plotted = 0
        for i, ph in enumerate(phase_list):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            sel = (df_corr['phase'] == ph)
            if metric_val is not None:
                sel &= (df_corr[metric_col] == metric_val)
            df_phase = df_corr[sel].copy()
            if df_phase.empty:
                ax.axis('off'); continue
            all_tenants = sorted(list(set(df_phase['tenant1']) | set(df_phase['tenant2'])))
            corr_pivot = df_phase.pivot_table(index='tenant1', columns='tenant2', values='mean_correlation')
            corr_pivot = corr_pivot.reindex(index=all_tenants, columns=all_tenants)
            symmetric = corr_pivot.combine_first(corr_pivot.T).fillna(0)
            np.fill_diagonal(symmetric.values, 1)
            hm = sns.heatmap(symmetric, cmap='coolwarm', vmin=vmin, vmax=vmax, cbar=False, ax=ax)
            ax.set_title(ph, fontsize=11)
            ax.set_xlabel(''); ax.set_ylabel('')
            plotted += 1
        # hide unused axes
        for j in range(n, rows*cols):
            r, c = divmod(j, cols)
            axes[r][c].axis('off')
        # Shared colorbar
        mappable = axes[0][0].collections[0] if axes[0][0].collections else None
        if mappable is not None:
            cbar = fig.colorbar(mappable, ax=axes, orientation='vertical', shrink=0.9)
            cbar.set_label('Mean Correlation')
        fig.suptitle(f"Mean Correlation — {mlabel}", y=1.02, weight='bold')
        plt.tight_layout(rect=[0, 0, 0.93, 0.96])
        out = paper_dir / f"corr_{safe_filename(str(mlabel))}_grid.png"
        plt.savefig(out)
        plt.close()
        print(f"  SUCCESS: {out.name} generated.")


# ===================== Causalidade (provar direção) =====================
def analyze_causality(figs_dir: Path, outputs_root_glob: str, phases: list[str] | None, target_metric: str, csv_dir: Path):
    print("[CAUSALITY] Starting causality analysis...")
    caus_path = first_or_none(f"{outputs_root_glob}/multi_round_analysis/multi_round_causality_all.csv")
    if not caus_path:
        print("  WARNING: Consolidated causality file not found.")
        return

    df_caus = pd.read_csv(caus_path)
    phase_list = phases or sorted(df_caus['phase'].dropna().unique().tolist())
    # Determine which metrics to plot: 'all' means iterate over all present
    if target_metric and target_metric.lower() != 'all':
        metrics_to_plot = [target_metric]
    else:
        metrics_to_plot = sorted(df_caus['metric'].dropna().unique().tolist())

    for metric_val in metrics_to_plot:
        mlabel = _pretty_metric(metric_val)
        for ph in phase_list:
            df_phase = df_caus[(df_caus['phase'] == ph) & (df_caus['metric'] == metric_val)].copy()
            if df_phase.empty:
                print(f"  WARNING: No causality data for phase '{ph}' and metric '{metric_val}'.")
                continue

            caus_agg = df_phase.groupby(['source', 'target'])['score'].mean().reset_index()
            all_tenants = sorted(list(set(caus_agg['source']) | set(caus_agg['target'])))
            caus_pivot = caus_agg.pivot_table(index='target', columns='source', values='score').fillna(0)
            caus_pivot = caus_pivot.reindex(index=all_tenants, columns=all_tenants).fillna(0)

            plt.figure(figsize=(12, 10))
            sns.heatmap(caus_pivot, annot=True, cmap='rocket_r', fmt=".3f", linewidths=.5)
            plt.title(f'Mean Causality Heatmap — {mlabel} (Phase: {ph})', weight='bold', pad=18)
            plt.xlabel('Source'); plt.ylabel('Target')
            out_name = f"causality_heatmap_{safe_filename(str(mlabel))}_{ph}.png"
            save_fig(figs_dir / out_name)
            print(f"  SUCCESS: {out_name} generated.")

            # CSV: causality matrix (target rows × source cols)
            try:
                caus_csv = csv_dir / f"causality_matrix_{safe_filename(str(mlabel))}_{ph}.csv"
                caus_pivot.to_csv(caus_csv)
            except Exception as e:
                print(f"  WARNING: Failed to export causality matrix CSV for phase {ph}, metric {metric_val}: {e}")

    # Optional: top links by consistency across rounds
    freq_path = first_or_none(f"{outputs_root_glob}/multi_round_analysis/multi_round_causality_frequency.csv")
    if freq_path:
        freq_df = pd.read_csv(freq_path)
        freq_df['pair'] = freq_df['source'] + '→' + freq_df['target']
        top_k = freq_df.sort_values('consistency_rate', ascending=False).head(15)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_k, x='consistency_rate', y='pair', orient='h')
        plt.title('Top-15 Causal Links by Consistency')
        plt.xlabel('Consistency Rate'); plt.ylabel('Pair (source→target)')
        save_fig(figs_dir / 'causality_consistency_top15.png')
        print("  SUCCESS: causality_consistency_top15.png generated.")

        # CSV: top-15 consistency
        try:
            top_csv = csv_dir / 'causality_consistency_top15.csv'
            top_k.to_csv(top_csv, index=False)
        except Exception as e:
            print(f"  WARNING: Failed to export causality consistency top-15 CSV: {e}")


# ===================== Modo compacto — Causalidade =====================
def analyze_causality_compact(paper_dir: Path, outputs_root_glob: str, phases: list[str] | None, target_metric: str):
    print("[CAUSALITY-COMPACT] Generating per-metric grids with key phases and consistency...")
    caus_path = first_or_none(f"{outputs_root_glob}/multi_round_analysis/multi_round_causality_all.csv")
    if not caus_path:
        print("  WARNING: Consolidated causality file not found.")
        return
    df_caus = pd.read_csv(caus_path)
    # Determine metrics
    if target_metric and target_metric.lower() != 'all':
        metrics_to_plot = [target_metric]
    else:
        metrics_to_plot = sorted(df_caus['metric'].dropna().unique().tolist())
    # Determine key phases
    all_phases = phases or sorted(df_caus['phase'].dropna().unique().tolist(), key=_phase_sort_key)
    keys = []
    desired = ['Baseline', 'CPU-Noise', 'Disk-Noise', 'Combined-Noise', 'Recovery']
    for name in desired:
        pick = next((ph for ph in all_phases if name.lower() in ph.lower()), None)
        if pick:
            keys.append(pick)
    # Fallback if any missing
    if not keys:
        keys = all_phases[:5]

    # Consistency table
    freq_path = first_or_none(f"{outputs_root_glob}/multi_round_analysis/multi_round_causality_frequency.csv")
    freq_df = pd.read_csv(freq_path) if freq_path else None

    for metric_val in metrics_to_plot:
        mlabel = _pretty_metric(metric_val)
        rows, cols = 2, 3  # 5 phases + 1 consistency panel
        fig, axes = plt.subplots(rows, cols, figsize=(5.2*cols+1, 4.6*rows), squeeze=False)
        # Establish vmax across selected phases for consistent color scale
        vmax = 0.0
        symmetric_list = []
        for ph in keys:
            df_phase = df_caus[(df_caus['phase'] == ph) & (df_caus['metric'] == metric_val)].copy()
            if df_phase.empty:
                symmetric_list.append((ph, None))
                continue
            caus_agg = df_phase.groupby(['source', 'target'])['score'].mean().reset_index()
            all_tenants = sorted(list(set(caus_agg['source']) | set(caus_agg['target'])))
            pivot = caus_agg.pivot_table(index='target', columns='source', values='score').reindex(index=all_tenants, columns=all_tenants).fillna(0)
            vmax = max(vmax, float(pivot.to_numpy().max(initial=0)))
            symmetric_list.append((ph, pivot))
        if vmax <= 0:
            vmax = 1.0
        # Plot the 5 phases
        for i, (ph, pivot) in enumerate(symmetric_list[:5]):
            r, c = divmod(i, cols)
            ax = axes[r][c]
            if pivot is None:
                ax.text(0.5, 0.5, f"No data\n{ph}", ha='center', va='center')
                ax.axis('off'); continue
            sns.heatmap(pivot, cmap='rocket_r', vmin=0, vmax=vmax, cbar=False, ax=ax)
            ax.set_title(ph, fontsize=11)
            ax.set_xlabel(''); ax.set_ylabel('')
        # Consistency panel
        axc = axes[1][2]
        if freq_df is not None and not freq_df.empty:
            top_k = freq_df.sort_values('consistency_rate', ascending=False).head(10)
            top_k = top_k.copy()
            top_k['pair'] = top_k['source'] + '→' + top_k['target']
            sns.barplot(data=top_k, x='consistency_rate', y='pair', orient='h', ax=axc)
            axc.set_title('Top-10 Consistency')
            axc.set_xlabel('Consistency Rate'); axc.set_ylabel('Pair')
        else:
            axc.axis('off')
        # Shared colorbar
        ref_ax = axes[0][0]
        if ref_ax.collections:
            mappable = ref_ax.collections[0]
            cbar = fig.colorbar(mappable, ax=axes, orientation='vertical', shrink=0.9)
            cbar.set_label('Causality Score')
        fig.suptitle(f"Causality — {mlabel} (Key Phases)", y=1.02, weight='bold')
        plt.tight_layout(rect=[0, 0, 0.93, 0.96])
        out = paper_dir / f"caus_{safe_filename(str(mlabel))}_grid.png"
        plt.savefig(out)
        plt.close()
        print(f"  SUCCESS: {out.name} generated.")


# ===================== CLI / Main =====================
def main():
    parser = argparse.ArgumentParser(description="Deep noisy neighbor analysis for publications.")
    parser.add_argument("--root", default="outputs/*/*", help="Outputs root glob (default: outputs/*/*)")
    parser.add_argument("--phases", default="all", help="Comma-separated phases or 'all'")
    parser.add_argument("--phase", default=None, help="Deprecated: single phase (use --phases)")
    parser.add_argument("--metric", default="all", help="Causality target metric or 'all' for all metrics")
    parser.add_argument("--parquet", default="data/processed/sfi2_long.parquet", help="Parquet path for descriptive analysis")
    parser.add_argument("--figs-dir", default="figs", help="Output directory for figures")
    parser.add_argument("--csv-dir", default="figs/csv", help="Output directory for CSV exports")
    parser.add_argument("--compact", action='store_true', help="Generate compact aggregated figures for paper")
    parser.add_argument("--paper-dir", default="figs/paper", help="Output directory for compact figures")
    args = parser.parse_args()

    configure_style()
    figs_dir = ensure_dir(Path(args.figs_dir))
    csv_dir = ensure_dir(Path(args.csv_dir))
    paper_dir = ensure_dir(Path(args.paper_dir))

    print("Full Analysis Pipeline Started.")
    print("="*40)

    analyze_descriptive(args.parquet, figs_dir, csv_dir)
    analyze_impact(figs_dir, args.root, csv_dir)

    # Determine phases list
    phases = None
    if args.phase:
        phases = [args.phase]
    elif args.phases and args.phases.lower() != 'all':
        phases = [p.strip() for p in args.phases.split(',') if p.strip()]

    analyze_correlation(figs_dir, args.root, phases, csv_dir)
    analyze_causality(figs_dir, args.root, phases, args.metric, csv_dir)

    # Compact figures for paper
    if args.compact:
        analyze_descriptive_compact(args.parquet, paper_dir)
        analyze_impact_compact(paper_dir, args.root)
        analyze_correlation_compact(paper_dir, args.root, phases)
        analyze_causality_compact(paper_dir, args.root, phases, args.metric)

    print("\n"+"="*40)
    print("Full Analysis Pipeline Finished.")


if __name__ == "__main__":
    main()