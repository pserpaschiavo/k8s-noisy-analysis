"""
Module: correlation_plots
Description: Módulo para visualizações de correlações intra-fase.
"""

from .correlation_plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_ccf,
    plot_lag_scatter
)

from .descriptive_plots import (
    plot_metric_timeseries_multi_tenant,
    plot_metric_timeseries_multi_tenant_all_phases,
    plot_metric_barplot_by_phase,
    plot_metric_boxplot,
    plot_anomalies
)

from .impact_plots import (
    plot_impact_summary
)

from .causality_plots import (
    plot_causality_graph,
    plot_causality_heatmap
)

from .phase_comparison_plots import (
    plot_phase_comparison,
    compare_correlation_matrices,
    compare_causality_graphs
)

__all__ = [
    'plot_correlation_heatmap',
    'plot_covariance_heatmap',
    'plot_ccf',
    'plot_lag_scatter',
    'plot_metric_timeseries_multi_tenant',
    'plot_metric_timeseries_multi_tenant_all_phases',
    'plot_metric_barplot_by_phase',
    'plot_metric_boxplot',
    'plot_anomalies',
    'plot_impact_summary',
    'plot_causality_graph',
    'plot_causality_heatmap',
    'plot_phase_comparison',
    'compare_correlation_matrices',
    'compare_causality_graphs'
]
