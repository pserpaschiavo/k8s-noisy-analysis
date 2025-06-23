# -*- coding: utf-8 -*-
"""
Module: visualization_config.py
Description: Centralized configuration for academic publication-quality visualizations.

This module defines standardized settings for colors, labels, fonts, and other visual
elements to ensure consistency and readability across all generated plots.
"""

# Standard configuration for academic publications
PUBLICATION_CONFIG = {
    # Consistent color scheme for tenants (colorblind-friendly)
    'tenant_colors': {
        'tenant-cpu': '#1f77b4',      # Muted Blue
        'tenant-mem': '#ff7f0e',  # Safety Orange
        'tenant-ntk': '#2ca02c', # Cooked Asparagus Green
        'tenant-dsk': '#d62728', # Brick Red
        'tenant-nsy': '#9467bd',   # Muted Purple
        'default': '#8c564b',        # Chestnut
    },
    
    # Consistent marker styles for tenants
    'tenant_markers': {
        'tenant-cpu': 'o',
        'tenant-mem': 's',
        'tenant-ntk': '^',
        'tenant-dsk': 'D',
        'tenant-nsy': 'v',
        'default': 'x',
    },

    # Standardized display names for tenants
    'tenant_display_names': {
        'tenant-cpu': 'CPU Tenant',
        'tenant-mem': 'Memory Tenant',
        'tenant-ntk': 'Network Tenant',
        'tenant-dsk': 'Storage Tenant',
        'tenant-nsy': 'Noisy Tenant',
    },
    
    # Standardized color scheme for experimental phases
    'phase_colors': {
        'baseline': '#7f7f7f',       # Medium Gray
        'cpu-noise': '#e41a1c',      # Red
        'memory-noise': '#377eb8',   # Blue
        'network-noise': '#4daf4a',  # Green
        'disk-noise': '#984ea3',     # Purple
        'combined-noise': '#ff7f00', # Orange
        'recovery': '#a65628',       # Brown
    },
    
    # Standardized display names in English for phases
    'phase_display_names': {
        'baseline': 'Baseline',
        'cpu-noise': 'CPU Noise',
        'memory-noise': 'Memory Noise',
        'network-noise': 'Network Noise',
        'disk-noise': 'Disk I/O Noise',
        'combined-noise': 'Combined Noise',
        'recovery': 'Recovery',
    },

    # Centralized colormaps for heatmaps
    'heatmap_colormaps': {
        'correlation': 'vlag',
        'covariance': 'crest',
        'p_value': 'viridis_r',
        'score': 'viridis',
    },

    # Standardization of metric names and units
    'metric_display_names': {
        'cpu_usage': {'name': 'CPU Usage', 'unit': 'Cores'},
        'memory_usage': {'name': 'Memory Usage', 'unit': 'GB'},
        'network_io': {'name': 'Network I/O', 'unit': 'MB/s'},
        'disk_io': {'name': 'Disk I/O', 'unit': 'MB/s'},
        'network_usage': {'name': 'Network Usage', 'unit': 'MB/s'},
        'disk_usage': {'name': 'Disk Usage', 'unit': 'MB/s'},
        'memory_usage_bytes': {'name': 'Memory Usage', 'unit': 'GB'},
        'network_transmit_bytes_per_second': {'name': 'Network Transmit', 'unit': 'MB/s'},
        'network_receive_bytes_per_second': {'name': 'Network Receive', 'unit': 'MB/s'},
        'disk_read_bytes_per_second': {'name': 'Disk Read', 'unit': 'MB/s'},
        'disk_write_bytes_per_second': {'name': 'Disk Write', 'unit': 'MB/s'},
    },
    
    # Matplotlib style configuration for high-quality figures
    'figure_style': {
        'figure.dpi': 300,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'axes.linewidth': 1.0,
        'grid.color': 'grey',
        'grid.linestyle': '--',
        'grid.linewidth': 0.5,
        'savefig.transparent': False,
        'savefig.format': 'png',
    }
}
