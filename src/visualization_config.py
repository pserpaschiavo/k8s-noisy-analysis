# -*- coding: utf-8 -*-
"""
Module: visualization_config.py
Description: Centralized configuration for academic publication-quality visualizations.

This module defines standardized settings for colors, labels, fonts, and other visual
elements to ensure consistency and readability across all generated plots.
"""

# Standard configuration for academic publications
PUBLICATION_CONFIG = {
    # Consistent color scheme for tenants (Okabe–Ito: colorblind-friendly)
    # Palette reference: https://jfly.uni-koeln.de/color/
    'tenant_colors': {
        'tenant-cpu': '#0072B2',  # Blue
        'tenant-mem': '#E69F00',  # Orange
        'tenant-ntk': '#009E73',  # Bluish green
        'tenant-dsk': '#CC79A7',  # Reddish purple
        'tenant-nsy': '#D55E00',  # Vermillion
        'default':   '#56B4E9',   # Sky blue
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
    
    # Standardized color scheme for experimental phases (Okabe–Ito, neutral gray for baseline)
    'phase_colors': {
        '1 - Baseline': '#9A9A9A',  # Neutral Gray
        '2 - CPU Noise': '#D55E00',  # Vermillion
        '3 - Memory Noise': '#009E73',  # Bluish green
        '4 - Network Noise': '#56B4E9',  # Sky blue
        '5 - Disk Noise': '#CC79A7',  # Reddish purple
        '6 - Combined Noise': '#E69F00',  # Orange
        '7 - Recovery': '#0072B2',  # Blue
    },

    # Optional hatch patterns per phase to aid in grayscale / colorblind contexts
    # Valid patterns examples: '/', '\\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
    'phase_hatches': {
        '1 - Baseline': '',
        '2 - CPU Noise': '///',
        '3 - Memory Noise': '\\\\\\',
        '4 - Network Noise': '++',
        '5 - Disk Noise': 'xx',
        '6 - Combined Noise': '..',
        '7 - Recovery': '--',
    },

    # Standardized display names for metrics
    'metric_display_names': {
        'cpu_usage': {'name': 'CPU Usage', 'unit': '%'},
        'memory_usage': {'name': 'Memory Usage', 'unit': 'Bytes'},
        'network_io': {'name': 'Network I/O', 'unit': 'Bytes/s'},
        'disk_io': {'name': 'Disk I/O', 'unit': 'Bytes/s'},
        'network_usage': {'name': 'Network Usage', 'unit': 'Bytes/s'},
        'disk_usage': {'name': 'Disk Usage', 'unit': 'Bytes/s'},
        'memory_usage_bytes': {'name': 'Memory Usage', 'unit': 'Bytes'},
    },

    # Standardized display names for experimental phases
    'phase_display_names': {
        '1 - Baseline': 'Baseline',
        '2 - CPU Noise': 'CPU Noise',
        '3 - Memory Noise': 'Memory Noise',
        '4 - Network Noise': 'Network Noise',
        '5 - Disk Noise': 'Disk Noise',
        '6 - Combined Noise': 'Combined Noise',
        '7 - Recovery': 'Recovery',
    },
    
    # Colormap configuration for heatmap visualizations
    'heatmap_colormaps': {
        'correlation': 'coolwarm',
        'covariance': 'viridis',
        'p_value': 'viridis_r',
        'score': 'plasma'
    },
    
    # Font and style settings for figures - updated to use universally available fonts
    'figure_style': {
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 'FreeSans', 'sans-serif'],
        'font.serif': ['DejaVu Serif', 'Liberation Serif', 'FreeSerif', 'serif'],
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
