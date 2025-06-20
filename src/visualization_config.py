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
        'CPU Tenant': '#1f77b4',      # Muted Blue
        'Memory Tenant': '#ff7f0e',  # Safety Orange
        'Network Tenant': '#2ca02c', # Cooked Asparagus Green
        'Storage Tenant': '#d62728', # Brick Red
        'Noisy Tenant': '#9467bd',   # Muted Purple
        'default': '#8c564b',        # Chestnut
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
        '1-Baseline': '#7f7f7f',       # Medium Gray
        '2-CPU-Noise': '#e41a1c',      # Red
        '3-Memory-Noise': '#377eb8',   # Blue
        '4-Network-Noise': '#4daf4a',  # Green
        '5-Disk-Noise': '#984ea3',     # Purple
        '6-Combined-Noise': '#ff7f00', # Orange
        '7-Recovery': '#a65628',       # Brown
    },
    
    # Standardized display names in English for phases
    'phase_display_names': {
        '1-Baseline': 'Baseline',
        '2-CPU-Noise': 'CPU Noise',
        '3-Memory-Noise': 'Memory Noise',
        '4-Network-Noise': 'Network Noise',
        '5-Disk-Noise': 'Disk I/O Noise',
        '6-Combined-Noise': 'Combined Noise',
        '7-Recovery': 'Recovery',
    },

    # Standardization of metric names and units
    'metric_display_names': {
        'cpu_usage': {'name': 'CPU Usage', 'unit': 'Cores'},
        'memory_usage': {'name': 'Memory Usage', 'unit': 'GB'},
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
