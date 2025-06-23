"""
Centralized configuration for visualizations.
"""

# Color palette for tenants
TENANT_COLORS = {
    'tenant-a': 'blue',
    'tenant-b': 'green',
    'tenant-c': 'red',
    'default': 'black'
}

# Marker styles for tenants
TENANT_MARKERS = {
    'tenant-a': 'o',
    'tenant-b': 's',
    'tenant-c': '^',
    'default': 'x'
}

# Line styles for different analysis phases
PHASE_LINESTYLES = {
    'phase-1': '-',
    'phase-2': '--',
    'phase-3': ':',
    'default': '-.'
}

# Colormaps for heatmaps
HEATMAP_COLORMAPS = {
    'p_value': 'viridis_r',
    'score': 'viridis'
}
