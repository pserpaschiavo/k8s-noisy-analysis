"""
Module: insight_aggregation.py
Description: Functions for inter-tenant insight aggregation and comparison methodologies.

This module complements report_generation.py by providing specific functionalities for:
1. Aggregation of insights about each tenant.
2. Inter-tenant comparison tables.
3. Comparative visualizations for analysis.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import matplotlib.lines as mlines

# Configure plots
plt.style.use('tableau-colorblind10')
logger = logging.getLogger("insight_aggregation")

def aggregate_tenant_insights(
    tenant_metrics: Optional[pd.DataFrame] = None,
    phase_comparison_results: Optional[Dict[str, pd.DataFrame]] = None,
    granger_matrices: Optional[Dict[str, pd.DataFrame]] = None,
    te_matrices: Optional[Dict[str, pd.DataFrame]] = None,
    correlation_matrices: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    anomaly_metrics: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Any]:
    """
    Aggregates insights about each tenant from all available analyses.
    Adapts to the 7-phase experimental model.

    Args:
        tenant_metrics: DataFrame with metrics for each tenant.
        phase_comparison_results: Results of the comparison between phases.
        granger_matrices: Granger causality matrices.
        te_matrices: Transfer Entropy matrices.
        correlation_matrices: Correlation matrices.
        anomaly_metrics: Dictionary with information about detected anomalies.

    Returns:
        Dictionary with consolidated insights for each tenant or an error message.
    """
    insights: Dict[str, Dict[str, Any]] = {}

    if tenant_metrics is None or tenant_metrics.empty:
        logger.error("tenant_metrics DataFrame is missing or empty.")
        return {"error_message": "Required data for insight aggregation (tenant_metrics) not available."}

    if 'tenant_id' not in tenant_metrics.columns:
        logger.error("'tenant_id' column is missing from tenant_metrics DataFrame.")
        return {"error_message": "'tenant_id' column missing from tenant_metrics."}

    # Standardize tenant_id to string type to prevent key errors
    tenant_metrics['tenant_id'] = tenant_metrics['tenant_id'].astype(str)

    # Standardize keys and indices in all input data structures
    phase_comparison_results = phase_comparison_results or {}
    for df in phase_comparison_results.values():
        if 'tenant_id' in df.columns:
            df['tenant_id'] = df['tenant_id'].astype(str)

    granger_matrices = {k: v.rename(columns=str, index=str) for k, v in (granger_matrices or {}).items()}
    te_matrices = {k: v.rename(columns=str, index=str) for k, v in (te_matrices or {}).items()}

    anomaly_metrics = anomaly_metrics or {}
    for df in anomaly_metrics.values():
        if 'tenant_id' in df.columns:
            df['tenant_id'] = df['tenant_id'].astype(str)

    # Initialize insights for each tenant
    insights: Dict[str, Dict[str, Any]] = {}
    for tenant in tenant_metrics['tenant_id'].unique():
        insights[tenant] = {
            'name': tenant,
            'rank': 0,
            'noisy_score': 0.0,
            'is_noisy_tenant': False,
            'is_victim_tenant': False,
            'main_impacted_tenants': [],
            'main_impact_sources': [],
            'anomalous_metrics': [],
            'noise_sensitivity': {},
            'correlation_patterns': [],
            'recommendations': []
        }

    # Fill in basic ranking and score information
    sorted_metrics = tenant_metrics.sort_values(by='noisy_score', ascending=False).reset_index(drop=True)
    for idx in sorted_metrics.index:
        row = sorted_metrics.loc[idx]
        tenant = row['tenant_id']
        insights[tenant]['rank'] = idx + 1
        insights[tenant]['noisy_score'] = row['noisy_score']

        # Determine if it is a "noisy tenant" (top 25%)
        if idx < len(sorted_metrics) // 4:
            insights[tenant]['is_noisy_tenant'] = True

    # Analyze causality to determine impact relationships
    for metric_name, matrix in te_matrices.items():
        if matrix.empty:
            continue

        for tenant in insights.keys():
            if tenant not in matrix.columns:
                continue

            # Identify the main tenants impacted by this tenant
            te_values = matrix[tenant].drop(tenant, errors='ignore')
            if not te_values.empty:
                significant_te = te_values[te_values > 0.05]
                if not significant_te.empty:
                    top_impacted = significant_te.nlargest(2)
                    for impacted_tenant_raw, te_val in top_impacted.items():
                        impacted_tenant = str(impacted_tenant_raw)
                        insights[tenant]['main_impacted_tenants'].append({
                            'tenant': impacted_tenant,
                            'score': float(te_val),
                            'metric': metric_name
                        })
                        if impacted_tenant in insights:
                            insights[impacted_tenant]['is_victim_tenant'] = True
                            insights[impacted_tenant]['main_impact_sources'].append({
                                'tenant': tenant,
                                'score': float(te_val),
                                'metric': metric_name
                            })

    # Analyze phase comparison to identify patterns during noise phases
    noise_phases = ['CPU-Noise', 'Memory-Noise', 'Network-Noise', 'Disk-Noise', 'Combined-Noise']
    for metric_name, stats_df in phase_comparison_results.items():
        if stats_df.empty or 'tenant_id' not in stats_df.columns:
            continue

        for _, row in stats_df.iterrows():
            tenant = row.get('tenant_id')
            if tenant not in insights:
                continue

            for phase in noise_phases:
                col_name = f"{phase}_vs_baseline_pct"
                if col_name in row and not pd.isna(row[col_name]):
                    variation = row[col_name]
                    if abs(variation) > 25:  # Sensitivity threshold: 25% variation
                        if phase not in insights[tenant]['noise_sensitivity']:
                            insights[tenant]['noise_sensitivity'][phase] = []
                        insights[tenant]['noise_sensitivity'][phase].append({
                            'metric': metric_name,
                            'variation_pct': float(variation)
                        })

    # Analyze anomalies, if available
    if anomaly_metrics:
        for metric_name, anomalies_df in anomaly_metrics.items():
            if anomalies_df.empty or 'tenant_id' not in anomalies_df.columns:
                continue

            for tenant in anomalies_df['tenant_id'].unique():
                if tenant not in insights:
                    continue
                tenant_anomalies = anomalies_df[anomalies_df['tenant_id'] == tenant]
                if not tenant_anomalies.empty:
                    insights[tenant]['anomalous_metrics'].append({
                        'metric': metric_name,
                        'anomaly_count': len(tenant_anomalies),
                        'max_zscore': float(tenant_anomalies['z_score'].max()) if 'z_score' in tenant_anomalies.columns else 0.0
                    })

    # Generate specific recommendations for each tenant
    for tenant, tenant_insight in insights.items():
        recommendations = []
        if tenant_insight['is_noisy_tenant']:
            recommendations.append("Consider adjusting resource limits to avoid impacting other tenants.")
            if tenant_insight['anomalous_metrics']:
                metrics_list = [m['metric'] for m in tenant_insight['anomalous_metrics']]
                recommendations.append(f"Investigate anomalous usage spikes in metrics: {', '.join(metrics_list)}.")

        if tenant_insight['is_victim_tenant'] and tenant_insight['main_impact_sources']:
            impact_sources = list(set([s['tenant'] for s in tenant_insight['main_impact_sources']]))
            recommendations.append(f"Consider isolating from impacting tenants: {', '.join(impact_sources)}.")

        if tenant_insight['noise_sensitivity']:
            sensitive_phases = tenant_insight['noise_sensitivity'].keys()
            recommendations.append(f"Shows sensitivity to: {', '.join(sensitive_phases)}. Monitor relevant metrics during these phases.")

        tenant_insight['recommendations'] = recommendations

    return insights

def generate_comparative_table(tenant_insights: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates a final comparative table for inter-tenant analysis.

    Args:
        tenant_insights: Dictionary with aggregated insights for each tenant.

    Returns:
        DataFrame with a detailed comparative table.
    """
    table_data = []
    for tenant, insights in tenant_insights.items():
        table_data.append({
            'Rank': insights['rank'],
            'Tenant': tenant,
            'Noisy Score': insights['noisy_score'],
            'Is Noisy?': 'Yes' if insights['is_noisy_tenant'] else 'No',
            'Is Victim?': 'Yes' if insights['is_victim_tenant'] else 'No',
            'Impacts On': ", ".join(list(set([i['tenant'] for i in insights['main_impacted_tenants']]))),
            'Impacted By': ", ".join(list(set([s['tenant'] for s in insights['main_impact_sources']]))),
            'Sensitive To': ", ".join(insights['noise_sensitivity'].keys()),
            'Recommendations': " | ".join(insights['recommendations'])
        })
    return pd.DataFrame(table_data).sort_values(by='Rank')

def plot_insight_matrix(
    insights: Dict[str, Dict[str, Any]],
    output_path: str
) -> None:
    """
    Generates a visual matrix of insights for a quick overview.

    Args:
        insights: Dictionary with aggregated insights.
        output_path: Path to save the plot.
    """
    if not insights:
        logger.warning("Cannot generate insight matrix: insights dictionary is empty.")
        return

    tenants = sorted(insights.keys(), key=lambda t: insights[t]['rank'])
    data = []
    for tenant in tenants:
        insight = insights[tenant]
        data.append({
            'Noisy': 1 if insight['is_noisy_tenant'] else 0,
            'Victim': 1 if insight['is_victim_tenant'] else 0,
            'Has Anomalies': 1 if insight['anomalous_metrics'] else 0,
            'CPU Sensitive': 1 if 'CPU-Noise' in insight['noise_sensitivity'] else 0,
            'Memory Sensitive': 1 if 'Memory-Noise' in insight['noise_sensitivity'] else 0,
            'Network Sensitive': 1 if 'Network-Noise' in insight['noise_sensitivity'] else 0,
            'Disk Sensitive': 1 if 'Disk-Noise' in insight['noise_sensitivity'] else 0,
            'Combined Sensitive': 1 if 'Combined-Noise' in insight['noise_sensitivity'] else 0
        })
    
    df = pd.DataFrame(data, index=tenants)
    
    plt.figure(figsize=(12, max(6, len(tenants) * 0.6)))
    sns.heatmap(df, annot=True, cmap=["#e0e0e0", "#1f77b4"], cbar=False, linewidths=.5, linecolor='white', fmt='d')
    plt.title('Tenant Insight Matrix', fontsize=16)
    plt.xlabel('Insight Category', fontsize=12)
    plt.ylabel('Tenant ID', fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Insight matrix plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving insight matrix plot: {e}")
    finally:
        plt.close()

def run_insight_aggregation(
    context: Dict[str, Any],
    out_dir: str
) -> Dict[str, Any]:
    """
    Main function to run the insight aggregation stage.

    Args:
        context: Dictionary with results from previous stages.
        out_dir: Output directory for reports and plots.

    Returns:
        Dictionary with paths to the generated artifacts.
    """
    logger.info("Starting insight aggregation stage...")
    
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Extract data from context
    tenant_metrics_df = context.get('tenant_metrics')
    phase_comp_results = context.get('phase_comparison_results', {}).get('stats_by_metric', {})
    granger_matrices = context.get('granger_results', {}).get('causality_matrices', {})
    te_matrices = context.get('te_results', {}).get('causality_matrices', {})
    correlation_matrices = context.get('correlation_results', {}).get('correlation_matrices', {})
    anomaly_metrics = context.get('anomaly_results', {}).get('anomaly_metrics', {})
    
    # Aggregate insights
    aggregated_insights = aggregate_tenant_insights(
        tenant_metrics=tenant_metrics_df,
        phase_comparison_results=phase_comp_results,
        granger_matrices=granger_matrices,
        te_matrices=te_matrices,
        correlation_matrices=correlation_matrices,
        anomaly_metrics=anomaly_metrics
    )
    
    if "error_message" in aggregated_insights:
        logger.error(f"Failed to aggregate insights: {aggregated_insights['error_message']}")
        return {}

    # Generate comparative table
    comparative_table = generate_comparative_table(aggregated_insights)
    table_path = os.path.join(reports_dir, "comparative_insights_table.csv")
    comparative_table.to_csv(table_path, index=False)
    logger.info(f"Comparative insights table saved to {table_path}")

    # Generate insight matrix plot
    plot_path = os.path.join(reports_dir, "insight_matrix.png")
    plot_insight_matrix(aggregated_insights, plot_path)
    
    logger.info("Insight aggregation stage completed.")
    
    return {
        "aggregated_insights": aggregated_insights,
        "comparative_table_path": table_path,
        "insight_matrix_plot_path": plot_path
    }
