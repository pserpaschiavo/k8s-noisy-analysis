#!/usr/bin/env python3
"""Generate impact signature heatmaps organized by metric.

Produces one heatmap per metric (tenants x phases) and a 2x2 consolidated panel
showing up to the first four metrics with a shared colorbar.
"""

from __future__ import annotations

import os
import logging
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from src.visualization.utils.plot_helpers import (
    create_subplots,
    add_shared_colorbar,
    save_figure,
)

logger = logging.getLogger("impact_signature_by_metric")


# ---------------------------------------------------------------------------
# Local formatting helpers (duplicated in small form to avoid circular import)
# ---------------------------------------------------------------------------
def _format_metric_name(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return s
    import re as _re
    s = s.replace('__', '_')
    s = _re.sub(r'[\-_]+', ' ', s.lower())
    s = _re.sub(r'\s+', ' ', s).strip()
    reps = {
        'cpu usage': 'CPU Usage',
        'memory usage': 'Memory Usage',
        'disk io total': 'Disk I/O Total',
        'network receive': 'Network Receive',
        'network transmit': 'Network Transmit',
        'network throughput': 'Network Throughput',
    }
    return reps.get(s, s.title())


def _format_phase_name(raw: str) -> str:
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return s
    import re as _re
    m = _re.match(r'^(\d+)[-_](.+)$', s)
    if m:
        prefix, rest = m.group(1), m.group(2)
        rest = _re.sub(r'[\-_]+', ' ', rest).strip()
        reps = {
            'cpu noise': 'CPU Noise',
            'memory noise': 'Memory Noise',
            'network noise': 'Network Noise',
            'disk noise': 'Disk Noise',
            'combined noise': 'Combined Noise',
            'recovery': 'Recovery',
            'baseline': 'Baseline',
        }
        rest_norm = reps.get(rest.lower(), rest.title())
        return f"{prefix}-{rest_norm}"
    # Fallback mapping
    reps2 = {
        'baseline': 'Baseline',
        'cpu noise': 'CPU Noise',
        'memory noise': 'Memory Noise',
        'network noise': 'Network Noise',
        'disk noise': 'Disk Noise',
        'combined noise': 'Combined Noise',
        'recovery': 'Recovery',
    }
    return reps2.get(s.lower(), s.title())


# ---------------------------------------------------------------------------
# Core generation function
# ---------------------------------------------------------------------------
def generate_impact_signatures_by_metric(out_dir: str, exclude_baseline: bool = True):
    """Generate per-metric and consolidated impact signature heatmaps.

    Parameters
    ----------
    out_dir : str
        Root output directory produced by previous analysis steps.
    exclude_baseline : bool, default True
        If True, remove baseline phase rows before plotting.
    """
    impact_csv = os.path.join(out_dir, 'impact', 'impact_aggregated_stats.csv')
    if not os.path.exists(impact_csv):
        logger.warning("Aggregated impact file not found; skipping metric-based impact signatures.")
        return None

    try:
        df = pd.read_csv(impact_csv)
    except Exception as e:  # pragma: no cover - defensive
        logger.error(f"Failed to read aggregated impact: {e}")
        return None

    required_cols = {'metric_name', 'tenant_id', 'experimental_phase', 'mean_percentage_change'}
    if not required_cols.issubset(df.columns) or df.empty:
        logger.warning("Impact DataFrame missing required columns or empty.")
        return None

    if exclude_baseline:
        df = df[~df['experimental_phase'].str.contains('baseline', case=False, na=False)]
        if df.empty:
            logger.warning("No data after baseline exclusion.")
            return None

    metrics = sorted(df['metric_name'].unique())
    if not metrics:
        return None

    sig_dir = os.path.join(out_dir, 'impact')
    os.makedirs(sig_dir, exist_ok=True)

    # --------------------------- Individual panels ---------------------------
    for metric in metrics:
        mdf = df[df['metric_name'] == metric]
        if mdf.empty:
            continue
        pivot = pd.pivot_table(
            mdf,
            values='mean_percentage_change',
            index='experimental_phase',
            columns='tenant_id',
            aggfunc='mean'
        )
        if pivot.empty:
            continue
        pivot_fmt = pivot.rename(index=lambda x: _format_phase_name(str(x)))
        center_val = 0 if (pivot_fmt.values.min() < 0 and pivot_fmt.values.max() > 0) else None
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.heatmap(pivot_fmt, annot=True, fmt='.1f', cmap='viridis', linewidths=.5,
                    center=center_val, ax=ax)
        ax.set_title(f'Impact Signature - {_format_metric_name(metric)}')
        ax.set_xlabel('Victim Tenant')
        ax.set_ylabel('Experimental Phase' + (' (no baseline)' if exclude_baseline else ''))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        fig.tight_layout()
        save_figure(fig, os.path.join(sig_dir, f'impact_signature_metric_{metric}.png'))
        plt.close(fig)

    # ------------------------- Consolidated 2x2 panel ------------------------
    n_cols = 2
    n_rows = 2
    fig, axes = create_subplots(n_rows, n_cols, figsize=(20, 12))
    global_min = None
    global_max = None
    plotted = 0
    for i, metric in enumerate(metrics[: n_rows * n_cols]):
        mdf = df[df['metric_name'] == metric]
        if mdf.empty:
            continue
        pivot = pd.pivot_table(
            mdf,
            values='mean_percentage_change',
            index='experimental_phase',
            columns='tenant_id',
            aggfunc='mean'
        )
        if pivot.empty:
            continue
        pivot_fmt = pivot.rename(index=lambda x: _format_phase_name(str(x)))
        r, c = i // n_cols, i % n_cols
        ax = axes[r, c]
        vals = pivot_fmt.values
        numeric = vals[~np.isnan(vals)]
        if numeric.size:
            vmin = numeric.min()
            vmax = numeric.max()
            global_min = vmin if global_min is None else min(global_min, vmin)
            global_max = vmax if global_max is None else max(global_max, vmax)
        sns.heatmap(pivot_fmt, annot=True, fmt='.1f', cmap='viridis', linewidths=.5,
                    ax=ax, cbar=False)
        ax.set_title(_format_metric_name(metric))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        # Only leftmost column shows y tick labels
        if c > 0:
            ax.tick_params(axis='y', labelleft=False)
        # Only bottom row shows x tick labels
        if r < n_rows - 1:
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_xlabel('')
        ax.set_ylabel('')
        plotted += 1

    # Hide any unused axes
    for j in range(plotted, n_rows * n_cols):
        r, c = j // n_cols, j % n_cols
        axes[r, c].axis('off')

    if global_min is not None and global_max is not None:
        add_shared_colorbar(fig, vmin=global_min, vmax=global_max, cmap='viridis',
                            axes=axes.ravel(), label='Mean % Change')

    fig.suptitle('Impact Signatures by Metric and Tenant', fontsize=16, y=0.96)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    consolidated_path = os.path.join(sig_dir, 'impact_signatures_by_metric_consolidated_panel.png')
    save_figure(fig, consolidated_path)
    plt.close(fig)
    logger.info(f"Consolidated impact signatures by metric panel saved at {consolidated_path}")
    return consolidated_path


def _cli():  # pragma: no cover - CLI convenience
    parser = argparse.ArgumentParser(description="Generate impact signature panels by metric")
    parser.add_argument("--out-dir", required=True, help="Output directory root")
    parser.add_argument("--include-baseline", action="store_true", help="Include baseline phase")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    generate_impact_signatures_by_metric(args.out_dir, exclude_baseline=not args.include_baseline)


if __name__ == "__main__":  # pragma: no cover
    _cli()
