# Multi-Tenant Time Series Analysis - Presentation Summary

*Generated on: 2025-06-05 03:37:20*

## Tenant Behavior Summary

| Tenant | Classification | Description | Noisy Score |
|--------|---------------|-------------|------------|
| tenant-d | Highly variable | tenant-d shows high variability across metrics, suggesting it may be a noisy tenant. | 1.459 |
| tenant-c | Highly variable | tenant-c shows high variability across metrics, suggesting it may be a noisy tenant. | 1.178 |
| tenant-a | Moderately variable | tenant-a shows moderate variability in its metrics. | 0.758 |
| tenant-b | Moderately variable | tenant-b shows moderate variability in its metrics. | 0.599 |

## Phase Comparison Insights

### Cpu Usage Variability

- **tenant-b**: tenant-b's cpu_usage_variability increased by 66.6% during the recovery phase.
- **tenant-d**: tenant-d's cpu_usage_variability decreased by 64.5% during the attack phase.
- **tenant-c**: tenant-c's cpu_usage_variability decreased by 58.1% during the attack phase.
- **tenant-a**: tenant-a's cpu_usage_variability decreased by 42.5% during the attack phase.
- **tenant-c**: tenant-c's cpu_usage_variability increased by 17.2% during the recovery phase.

## Anomaly Detection Highlights

### Cpu

- **usage_variability**: Anomalies detected in usage_variability's cpu during the attack phase.
- **usage_variability**: Anomalies in usage_variability's cpu persisted into recovery phase.

## Key Takeaways

- tenant-d shows the highest variability with a noisy score of 1.459.
- tenant-d was most affected during the attack phase with significant changes in 1 metrics.
- usage_variability exhibited the most anomalies with unusual behavior detected in 2 metrics.

## Recommendations for Further Analysis

1. **Deeper focus on inter-tenant relationships**: Analyze correlation patterns between tenants during attack phases to better understand impact propagation.
2. **Metric sensitivity analysis**: Identify which metrics are most sensitive to attacks and should be monitored closely.
3. **Recovery pattern classification**: Develop a classification system for different types of recovery patterns to predict system resilience.
