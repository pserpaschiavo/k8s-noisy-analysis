# Multi-Tenant Kubernetes Time Series Analysis Presentation Guide

## Overview

This document provides a guide for your presentation on multi-tenant time series data analysis from Kubernetes experiments. All visualizations have been organized in the `/home/phil/Projects/k8s-noisy-analysis/outputs/presentation/` directory.

## Structure of Presentation

### 1. Introduction (2-3 minutes)
- Briefly explain the goal of the analysis: understanding how different tenants behave under normal and attack conditions
- Mention that we're analyzing time series data across baseline, attack, and recovery phases
- Key metrics analyzed: CPU usage, memory usage, disk I/O, and network bandwidth

### 2. Phase Comparisons (5 minutes)
- Location: `01_phase_comparisons/`
- These visualizations compare how each tenant's metrics change across the three experimental phases
- Key observations to highlight:
  - Tenant-A shows high variability across metrics
  - Tenant-D was most affected during the attack phase
  - Tenant-D showed the best recovery pattern

### 3. Anomaly Detection (5 minutes)
- Location: `02_anomaly_detection/`
- Focus on how anomalies were detected for each tenant during the attack phase
- Compare with recovery phase to show persistence or resolution of anomalies
- Key observations to highlight:
  - Tenant-A showed significant anomalies during the attack phase
  - Some anomalies persisted into the recovery phase
  - Network bandwidth showed the most pronounced anomalies

### 4. Correlation Analysis (5 minutes)
- Location: `03_correlation_analysis/`
- These heatmaps show the inter-tenant correlations during each phase
- Key observations to highlight:
  - Changes in correlation patterns between baseline and attack
  - Recovery phase correlation compared to baseline
  - Persistent changes in relationships between certain tenants

### 5. Key Insights (3 minutes)
- Reference the summary document (`00_presentation_summary.md`) for key insights
- Main takeaways:
  - Tenant-A is identified as the noisiest tenant with a score of 2.275
  - Tenant-D showed significant changes during attack phase but recovered well
  - Network bandwidth and CPU usage were the most sensitive metrics to attacks

### 6. Recommendations for Further Work (2 minutes)
- Deeper analysis of inter-tenant relationships during attacks
- Metric sensitivity analysis to identify early warning indicators
- Development of recovery pattern classification for resilience assessment

## Tips for Presenting Visualizations
- When showing the phase comparison plots, focus on the right side which shows percentage changes relative to baseline
- For anomaly detection, highlight the red regions that indicate detected anomalies
- When discussing correlation heatmaps, point out the strongest correlations (dark blue/red) and how they change

## Additional Notes
- All code and analysis can be reproduced using the unified pipeline with: `python -m src.run_unified_pipeline --config config/pipeline_config.yaml`
- The insight generation can be reproduced with: `python generate_presentation_insights.py`
