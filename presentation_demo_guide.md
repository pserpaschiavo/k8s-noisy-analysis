# Presentation Demo Guide

This guide provides step-by-step instructions for running the demo for the presentation on June 5, 2025.

## Prerequisites

- Ensure all required Python packages are installed: `pip install -r requirements.txt`
- Make sure you have the demo data in `/home/phil/Projects/k8s-noisy-analysis/data/processed/demo_exp_1_round.parquet`

## Step 1: Generate All Results (if needed)

If you need to regenerate all analysis results from scratch:

```bash
# Run the full pipeline with the demo experiment
python -m src.run_unified_pipeline --config config/pipeline_config.yaml
```

This will take approximately 5-10 minutes and will generate all plots, insights, and analysis results in the `/home/phil/Projects/k8s-noisy-analysis/outputs/demo-experiment-1-round/` directory.

## Step 2: Generate Presentation Insights

Generate insights specifically for the presentation:

```bash
# Generate presentation insights
python generate_presentation_insights.py
```

## Step 3: Organize Visualizations for Presentation

Organize all visualizations in a presentation-friendly format:

```bash
# Organize visualization materials
python organize_presentation_visualizations.py
```

This will create a well-structured presentation directory at `/home/phil/Projects/k8s-noisy-analysis/outputs/presentation/`.

## Step 4: Review Presentation Materials

1. Open the presentation summary:
   ```bash
   less /home/phil/Projects/k8s-noisy-analysis/outputs/presentation/00_presentation_summary.md
   ```

2. Review the presentation guide:
   ```bash
   less /home/phil/Projects/k8s-noisy-analysis/outputs/presentation/presentation_guide.md
   ```

3. Explore the organized visualization directories:
   - Phase comparisons: `/home/phil/Projects/k8s-noisy-analysis/outputs/presentation/01_phase_comparisons/`
   - Anomaly detection: `/home/phil/Projects/k8s-noisy-analysis/outputs/presentation/02_anomaly_detection/`
   - Correlation analysis: `/home/phil/Projects/k8s-noisy-analysis/outputs/presentation/03_correlation_analysis/`
   - Cross-correlation analysis: `/home/phil/Projects/k8s-noisy-analysis/outputs/presentation/03_correlation_analysis/cross_correlation/`

## Step 5: During the Presentation

1. Use the `presentation_guide.md` for the structure and key points to highlight
2. Show visualizations in the following order:
   - Introduction (no visuals needed)
   - Phase comparisons from `01_phase_comparisons/`
   - Anomaly detection from `02_anomaly_detection/`
   - Correlation heatmaps from `03_correlation_analysis/`
   - **NEW**: Cross-correlation plots from `03_correlation_analysis/cross_correlation/`
   - Key insights from `00_presentation_summary.md`
   - Recommendations for further work

3. For cross-correlation plots:
   - Explain the X-axis represents time lag between tenant behaviors
   - Point out the peak correlation and its corresponding lag
   - Explain that positive lag means first tenant influences second tenant
   - Note the direction indicator (A → B) in the plot title
   - Highlight the gray band showing statistical significance threshold

## Follow-up Questions

Be prepared to answer the following questions:
1. How does cross-correlation help identify noisy tenants?
2. What's the difference between regular correlation and cross-correlation?
3. How reliable are the lag indicators in determining causality?
4. How can this analysis be applied to real-world Kubernetes environments?

## Demo Script for Cross-Correlation

"Here we have our new cross-correlation analysis that shows temporal relationships between tenants. Unlike regular correlation that only shows if metrics move together, cross-correlation shows us if one tenant's behavior precedes and potentially causes another tenant's behavior.

Looking at this plot, we can see that tenant-A's CPU usage spikes tend to precede similar spikes in tenant-B with a lag of about X time units. This suggests a causal relationship where tenant-A is likely influencing tenant-B, not just correlating with it.

The red dot shows the point of maximum correlation, and the title indicates the direction of influence. The gray band represents statistical significance - correlations outside this band are considered meaningful.

This analysis is particularly valuable for identifying noisy neighbors in multi-tenant environments because it can reveal which tenants are consistently affecting others with a time delay, which is stronger evidence of causality than simple correlation."
