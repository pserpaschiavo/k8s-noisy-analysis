"""
Module: src.visualization.config
Description: Centralized configuration for all visualizations in the pipeline.

This module provides consistent styling and naming for plots, ensuring a uniform
look and feel across all generated artifacts. It defines color palettes,
marker styles, line styles, and display names for various elements like
tenants, phases, and metrics.
"""

# Color palette for distinguishing tenants in plots.
# Using a dictionary allows for consistent color mapping for each tenant.
TENANT_COLORS = {
    'tenant-a': 'blue',
    'tenant-b': 'green',
    'tenant-c': 'red',
    'default': 'black'  # Fallback color for any tenant not explicitly defined.
}

# Marker styles for tenants, useful for scatter plots or line plots with markers.
# This helps differentiate tenants in black-and-white publications.
TENANT_MARKERS = {
    'tenant-a': 'o',  # Circle
    'tenant-b': 's',  # Square
    'tenant-c': '^',  # Triangle
    'default': 'x'   # Cross for undefined tenants.
}

# Line styles for different analysis phases.
# Helps distinguish between different experimental phases in time-series plots.
PHASE_LINESTYLES = {
    'phase-1': '-',   # Solid line
    'phase-2': '--',  # Dashed line
    'phase-3': ':',   # Dotted line
    'default': '-.' # Dash-dot line for any other phase.
}

# Recommended colormaps for heatmaps to ensure perceptual uniformity and clarity.
HEATMAP_COLORMAPS = {
    'p_value': 'viridis_r',  # Reversed viridis for p-values (lower is better).
    'score': 'viridis'       # Standard viridis for scores (higher is better).
}

# Configuration for generating publication-quality plots.
# This section centralizes display names and other settings to ensure
# all plots are ready for scientific papers or presentations.
PUBLICATION_CONFIG = {
    # Defines human-readable names and units for metrics.
    "metric_display_names": {
        "cpu_usage": {"name": "CPU Usage", "unit": "%"},
        "memory_usage": {"name": "Memory Usage", "unit": "MiB"},
        "network_io": {"name": "Network I/O", "unit": "bytes/s"},
        "disk_io": {"name": "Disk I/O", "unit": "bytes/s"}
    },
    # Defines human-readable names for experimental phases.
    # These names will be used in plot titles, legends, and axes.
    "phase_display_names": {
        "1 - Baseline": "Baseline",
        "2 - CPU Noise": "CPU Noise",
        "3 - Memory Noise": "Memory Noise",
        "4 - Network Noise": "Network Noise",
        "5 - Disk Noise": "Disk Noise",
        "6 - Combined Noise": "Combined Noise",
        "7 - Recovery": "Recovery"
    },
    # Defines human-readable names for tenants, often describing their workload.
    "tenant_display_names": {
        "tenant-cpu": "CPU-Bound",
        "tenant-mem": "Memory-Bound",
        "tenant-dsk": "Disk-Bound",
        "tenant-ntk": "Network-Bound",
        "tenant-nsy": "Noisy-Neighbor"
    },
    # Defines specific colormaps for different types of heatmaps for consistency.
    "heatmap_colormaps": {
        "correlation": "coolwarm",   # Diverging map for correlations (-1 to 1).
        "covariance": "viridis",     # Sequential map for covariance.
        "p_value": "viridis_r",      # Reversed sequential for p-values.
        "score": "plasma"            # Sequential map for scores.
    }
}


def format_metric_name(raw: str) -> str:
    """Normalize raw metric identifiers into a human-friendly display string.

    Rules / heuristics:
      - Safe cast to str, strip
      - Replace underscores/dashes with spaces
      - Collapse multiple spaces
      - Title-case then apply explicit replacements for canonical forms
    """
    if raw is None:
        return ""
    import re as _re
    s = str(raw).strip()
    if not s:
        return s
    s = _re.sub(r'[\-_]+', ' ', s)
    s = _re.sub(r'\s+', ' ', s).strip().lower()
    replacements = {
        'cpu usage': 'CPU Usage',
        'memory usage': 'Memory Usage',
        'disk io total': 'Disk I/O Total',
        'disk io': 'Disk I/O',
        'network receive': 'Network Receive',
        'network transmit': 'Network Transmit',
        'network throughput': 'Network Throughput',
    }
    return replacements.get(s, s.title())
