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

# Configuration for publication-quality plots
PUBLICATION_CONFIG = {
    "metric_display_names": {
        "cpu_usage": {"name": "CPU Usage", "unit": "%"},
        "memory_usage": {"name": "Memory Usage", "unit": "MiB"},
        "network_io": {"name": "Network I/O", "unit": "bytes/s"},
        "disk_io": {"name": "Disk I/O", "unit": "bytes/s"}
    },
    "phase_display_names": {
        "1 - Baseline": "Baseline",
        "2 - CPU Noise": "CPU Noise",
        "3 - Memory Noise": "Memory Noise",
        "4 - Network Noise": "Network Noise",
        "5 - Disk Noise": "Disk Noise",
        "6 - Combined Noise": "Combined Noise",
        "7 - Recovery": "Recovery"
    },
    "tenant_display_names": {
        "tenant-cpu": "CPU-Bound",
        "tenant-mem": "Memory-Bound",
        "tenant-dsk": "Disk-Bound",
        "tenant-ntk": "Network-Bound",
        "tenant-nsy": "Noisy-Neighbor"
    },
    "heatmap_colormaps": {
        "correlation": "coolwarm",
        "covariance": "viridis",
        "p_value": "viridis_r",
        "score": "plasma"
    }
}
