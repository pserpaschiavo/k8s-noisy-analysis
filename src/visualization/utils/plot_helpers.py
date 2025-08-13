"""Shared plotting helper utilities for k8s-noisy-analysis.

Centralizes common figure creation, legends, colorbars and bar annotations
used across multi-round and impact signature visualizations.

Safe to import anywhere; keeps external dependencies limited to matplotlib & seaborn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Tuple, Optional, Dict, Any, Literal

import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

logger = logging.getLogger(__name__)

# Default style setup guard (idempotent)
_DEF_STYLE_APPLIED = False

def apply_default_style(palette: str = 'viridis', style: Literal['white','dark','whitegrid','darkgrid','ticks'] = 'whitegrid'):
    global _DEF_STYLE_APPLIED
    if _DEF_STYLE_APPLIED:
        return
    # Cast style to satisfy static type checkers (runtime accepts str literal)
    sns.set_theme(style=style)
    sns.set_palette(palette)
    _DEF_STYLE_APPLIED = True


def create_subplots(nrows: int, ncols: int, figsize=(12, 6), sharex=False, sharey=False, constrained=True):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False,
                             constrained_layout=constrained, sharex=sharex, sharey=sharey)
    return fig, axes


def add_shared_legend(fig, handles, labels, title: str, ncol: int, anchor=(0.5, -0.02)):
    if not handles or not labels:
        return None
    lg = fig.legend(handles, labels, title=title, loc='lower center',
                    bbox_to_anchor=anchor, ncol=ncol, fontsize=9, title_fontsize=10, frameon=False)
    return lg


def add_shared_colorbar(fig, vmin, vmax, cmap, axes, label: str, fraction=0.03, pad=0.02):
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=fraction, pad=pad)
    cbar.set_label(label)
    return cbar


def annotate_bars(ax, fmt="{:.1f}", fontsize=7, min_display=0.0, offset_pts=1):
    for p in ax.patches:
        if not isinstance(p, Rectangle):
            continue
        height = getattr(p, 'get_height', lambda: None)()
        if height is None or height != height or abs(height) < min_display:  # NaN check height!=height
            continue
        x = p.get_x() + p.get_width() / 2
        vertical_offset = offset_pts if height >= 0 else -offset_pts
        ax.annotate(fmt.format(height), (x, height), ha='center',
                    va='bottom' if height >= 0 else 'top', fontsize=fontsize,
                    xytext=(0, vertical_offset), textcoords='offset points')


def save_figure(fig, path: str, dpi: int = 300, tight: bool = True):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kwargs: Dict[str, Any] = {'dpi': dpi}
    if tight:
        kwargs['bbox_inches'] = 'tight'  # type: ignore[assignment]
    fig.savefig(path, **kwargs)
    logger.info(f"Saved figure: {path}")


@dataclass
class GridSpec:
    n_items: int
    max_cols: int = 2

    @property
    def n_cols(self) -> int:
        return min(self.max_cols, max(1, self.n_items))

    @property
    def n_rows(self) -> int:
        if self.n_items == 0:
            return 0
        return (self.n_items + self.n_cols - 1) // self.n_cols


def compute_grid(n_items: int, max_cols: int = 2) -> Tuple[int, int]:
    gs = GridSpec(n_items=n_items, max_cols=max_cols)
    return gs.n_rows, gs.n_cols


__all__ = [
    'apply_default_style', 'create_subplots', 'add_shared_legend', 'add_shared_colorbar',
    'annotate_bars', 'save_figure', 'compute_grid'
]
