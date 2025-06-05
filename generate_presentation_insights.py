#!/usr/bin/env python3
"""
Script: generate_presentation_insights.py
Description: Generate insights from the analysis results for presentation purposes.

This script aggregates results from different analysis stages and creates a summary
of insights focused on phase comparisons and anomaly detection for presentation.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger("insight_generation")

# Style configuration for plots
plt.style.use('tableau-colorblind10')

def load_pipeline_data(output_dir: str) -> Dict[str, Any]:
    """
    Load the necessary data from the pipeline outputs.
    
    Args:
        output_dir: The base output directory of the pipeline run
    
    Returns:
        Dictionary containing the loaded data
    """
    data = {
        "phase_comparison_results": {},
        "anomaly_metrics": {},
        "correlation_matrices": {}
    }
    
    # Load parquet data
    parquet_path = "/home/phil/Projects/k8s-noisy-analysis/data/processed/demo_exp_1_round.parquet"
    if os.path.exists(parquet_path):
        logger.info(f"Loading parquet data from {parquet_path}")
        df_long = pd.read_parquet(parquet_path)
        data["df_long"] = df_long
        
        # Extract tenant metrics
        tenant_metrics = []
        for tenant_id in df_long['tenant_id'].unique():
            tenant_data = df_long[df_long['tenant_id'] == tenant_id]
            
            # Calculate a noisy score based on coefficient of variation
            metric_scores = {}
            for metric in df_long['metric_name'].unique():
                metric_data = tenant_data[tenant_data['metric_name'] == metric]
                if not metric_data.empty:
                    cv = metric_data['metric_value'].std() / max(abs(metric_data['metric_value'].mean()), 0.001)
                    metric_scores[metric] = cv
            
            # Average score across metrics
            if metric_scores:
                noisy_score = sum(metric_scores.values()) / len(metric_scores)
            else:
                noisy_score = 0.0
                
            tenant_metrics.append({
                "tenant_id": tenant_id,
                "noisy_score": noisy_score
            })
            
        data["tenant_metrics"] = pd.DataFrame(tenant_metrics)
    else:
        logger.warning(f"Parquet file not found at {parquet_path}")
    
    # Load or generate phase comparison results
    phase_comparison_plots = Path(output_dir) / "plots" / "phase_comparison"
    if phase_comparison_plots.exists():
        logger.info("Phase comparison plots are available")
        
        # Generate simple phase comparison results from data
        if "df_long" in data:
            for metric in data["df_long"]['metric_name'].unique():
                # Extract and calculate stats for each phase
                phase_stats = {}
                
                for phase in data["df_long"]['experimental_phase'].unique():
                    phase_data = data["df_long"][(data["df_long"]['metric_name'] == metric) & 
                                                 (data["df_long"]['experimental_phase'] == phase)]
                    
                    if not phase_data.empty:
                        stats_by_tenant = {}
                        for tenant in phase_data['tenant_id'].unique():
                            tenant_data = phase_data[phase_data['tenant_id'] == tenant]
                            stats_by_tenant[tenant] = {
                                'mean': tenant_data['metric_value'].mean(),
                                'std': tenant_data['metric_value'].std(),
                                'max': tenant_data['metric_value'].max(),
                                'min': tenant_data['metric_value'].min()
                            }
                        phase_stats[phase] = stats_by_tenant
                
                data["phase_comparison_results"][metric] = phase_stats
    else:
        logger.warning(f"Phase comparison plots not found at {phase_comparison_plots}")
    
    # Load information about anomalies
    anomaly_plots = Path(output_dir) / "plots" / "anomaly_detection"
    if anomaly_plots.exists():
        logger.info("Anomaly detection plots are available")
        
        # Extract anomaly information from filenames
        for file_path in anomaly_plots.glob("*.png"):
            file_name = file_path.name
            parts = file_name.replace(".png", "").split("_")
            
            # Extract components from filename pattern: anomaly_detection_metric_tenant_phase_round
            if len(parts) >= 5 and parts[0] == "anomaly" and parts[1] == "detection":
                metric = parts[2]
                tenant = parts[3] + "_" + parts[4]  # tenant_a, tenant_b, etc.
                
                # Extract phase (which might contain spaces)
                phase_parts = []
                for i in range(5, len(parts) - 1):  # Exclude the last part (round-1)
                    phase_parts.append(parts[i])
                phase = "_".join(phase_parts)
                
                # Store information
                if metric not in data["anomaly_metrics"]:
                    data["anomaly_metrics"][metric] = {}
                    
                if tenant not in data["anomaly_metrics"][metric]:
                    data["anomaly_metrics"][metric][tenant] = []
                    
                data["anomaly_metrics"][metric][tenant].append({
                    "phase": phase,
                    "file_name": str(file_path)
                })
    else:
        logger.warning(f"Anomaly detection plots not found at {anomaly_plots}")
        
    # Load correlation matrices
    correlation_plots = Path(output_dir) / "plots" / "correlation"
    if correlation_plots.exists():
        logger.info("Correlation plots are available")
        
        # Extract correlation information from filenames
        for file_path in correlation_plots.glob("*.png"):
            file_name = file_path.name
            
            # Check if it's a correlation heatmap
            if file_name.startswith("correlation_heatmap_"):
                parts = file_name.replace(".png", "").split("_")
                
                # Extract components: correlation_heatmap_metric_phase_round
                if len(parts) >= 4:
                    method = "correlation"
                    metric = parts[2]
                    
                    # Extract phase (which might contain spaces)
                    phase_parts = []
                    for i in range(3, len(parts) - 1):  # Exclude the last part (round-1)
                        phase_parts.append(parts[i])
                    phase = "_".join(phase_parts)
                    
                    # Store information
                    correlation_key = f"{method}_{phase}"
                    if correlation_key not in data["correlation_matrices"]:
                        data["correlation_matrices"][correlation_key] = {}
                    
                    data["correlation_matrices"][correlation_key][metric] = str(file_path)
    else:
        logger.warning(f"Correlation plots not found at {correlation_plots}")
    
    return data

def generate_insights(data: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """
    Generate insights from the loaded data.
    
    Args:
        data: Dictionary containing the loaded analysis data
        output_dir: The directory where to save outputs
        
    Returns:
        Dictionary containing insights
    """
    insights = {
        "phase_comparisons": {},
        "anomaly_detections": {},
        "tenant_behaviors": {}
    }
    
    # Create the insights directory if it doesn't exist
    insights_dir = Path(output_dir) / "insights"
    insights_dir.mkdir(exist_ok=True)
    
    # Generate phase comparison insights
    if "phase_comparison_results" in data and data["phase_comparison_results"]:
        for metric, phases in data["phase_comparison_results"].items():
            metric_insights = []
            
            # Compare baseline with attack phase
            if "1 - Baseline" in phases and "2 - Attack" in phases:
                baseline = phases["1 - Baseline"]
                attack = phases["2 - Attack"]
                
                for tenant in baseline:
                    if tenant in attack:
                        # Calculate percentage change
                        baseline_mean = baseline[tenant]['mean']
                        attack_mean = attack[tenant]['mean']
                        
                        if baseline_mean != 0:
                            pct_change = ((attack_mean - baseline_mean) / abs(baseline_mean)) * 100
                            significance = abs(pct_change)
                            
                            # Add insight if change is significant
                            if significance > 15:  # More than 15% change
                                direction = "increased" if pct_change > 0 else "decreased"
                                metric_insights.append({
                                    "tenant": tenant,
                                    "description": f"{tenant}'s {metric} {direction} by {abs(pct_change):.1f}% during the attack phase.",
                                    "significance": significance
                                })
            
            # Compare attack with recovery phase
            if "2 - Attack" in phases and "3 - Recovery" in phases:
                attack = phases["2 - Attack"]
                recovery = phases["3 - Recovery"]
                
                for tenant in attack:
                    if tenant in recovery:
                        # Calculate percentage change
                        attack_mean = attack[tenant]['mean']
                        recovery_mean = recovery[tenant]['mean']
                        
                        if attack_mean != 0:
                            pct_change = ((recovery_mean - attack_mean) / abs(attack_mean)) * 100
                            significance = abs(pct_change)
                            
                            # Add insight if change is significant
                            if significance > 15:  # More than 15% change
                                direction = "increased" if pct_change > 0 else "decreased"
                                metric_insights.append({
                                    "tenant": tenant,
                                    "description": f"{tenant}'s {metric} {direction} by {abs(pct_change):.1f}% during the recovery phase.",
                                    "significance": significance
                                })
                                
                                # Check if returned to baseline
                                if "1 - Baseline" in phases and tenant in phases["1 - Baseline"]:
                                    baseline_mean = phases["1 - Baseline"][tenant]['mean']
                                    recovery_vs_baseline = ((recovery_mean - baseline_mean) / abs(baseline_mean)) * 100
                                    
                                    if abs(recovery_vs_baseline) < 10:  # Within 10% of baseline
                                        metric_insights.append({
                                            "tenant": tenant,
                                            "description": f"{tenant}'s {metric} returned to near-baseline levels during recovery.",
                                            "significance": 50  # High significance for recovery
                                        })
            
            # Sort insights by significance
            metric_insights.sort(key=lambda x: x["significance"], reverse=True)
            insights["phase_comparisons"][metric] = metric_insights
    
    # Generate anomaly detection insights
    if "anomaly_metrics" in data and data["anomaly_metrics"]:
        for metric, tenants in data["anomaly_metrics"].items():
            metric_insights = []
            
            for tenant, phases in tenants.items():
                # Check if anomalies were detected in attack phase
                attack_phases = [p for p in phases if "attack" in p["phase"].lower()]
                if attack_phases:
                    metric_insights.append({
                        "tenant": tenant,
                        "description": f"Anomalies detected in {tenant}'s {metric} during the attack phase.",
                        "file_name": attack_phases[0]["file_name"]  # Reference to the plot
                    })
                    
                # Check if anomalies were detected in recovery but not in baseline
                recovery_phases = [p for p in phases if "recovery" in p["phase"].lower()]
                baseline_phases = [p for p in phases if "baseline" in p["phase"].lower()]
                
                if recovery_phases and baseline_phases:
                    metric_insights.append({
                        "tenant": tenant,
                        "description": f"Anomalies in {tenant}'s {metric} persisted into recovery phase.",
                        "file_name": recovery_phases[0]["file_name"]  # Reference to the plot
                    })
            
            insights["anomaly_detections"][metric] = metric_insights
    
    # Generate tenant behavior insights
    if "tenant_metrics" in data and not data["tenant_metrics"].empty:
        tenant_df = data["tenant_metrics"]
        
        # Sort tenants by noisy score
        sorted_tenants = tenant_df.sort_values(by="noisy_score", ascending=False)
        
        for _, row in sorted_tenants.iterrows():
            tenant = row["tenant_id"]
            noisy_score = row["noisy_score"]
            
            # Classify tenant behavior
            if noisy_score > 1.0:
                insights["tenant_behaviors"][tenant] = {
                    "classification": "Highly variable",
                    "description": f"{tenant} shows high variability across metrics, suggesting it may be a noisy tenant.",
                    "noisy_score": noisy_score
                }
            elif noisy_score > 0.5:
                insights["tenant_behaviors"][tenant] = {
                    "classification": "Moderately variable",
                    "description": f"{tenant} shows moderate variability in its metrics.",
                    "noisy_score": noisy_score
                }
            else:
                insights["tenant_behaviors"][tenant] = {
                    "classification": "Stable",
                    "description": f"{tenant} shows stable behavior across metrics.",
                    "noisy_score": noisy_score
                }
    
    # Save insights to file
    insights_path = insights_dir / "presentation_insights.json"
    with open(insights_path, "w") as f:
        json.dump(insights, f, indent=2)
    
    logger.info(f"Insights saved to {insights_path}")
    return insights

def generate_presentation_summary(insights: Dict[str, Any], output_dir: str) -> None:
    """
    Generate a markdown summary for presentation.
    
    Args:
        insights: Dictionary containing generated insights
        output_dir: The directory where to save outputs
    """
    insights_dir = Path(output_dir) / "insights"
    insights_dir.mkdir(exist_ok=True)
    
    summary_path = insights_dir / "presentation_summary.md"
    
    with open(summary_path, "w") as f:
        f.write("# Multi-Tenant Time Series Analysis - Presentation Summary\n\n")
        f.write(f"*Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
        
        # Tenant behavior summary
        f.write("## Tenant Behavior Summary\n\n")
        
        if "tenant_behaviors" in insights and insights["tenant_behaviors"]:
            # Create a markdown table for tenant behaviors
            f.write("| Tenant | Classification | Description | Noisy Score |\n")
            f.write("|--------|---------------|-------------|------------|\n")
            
            # Sort tenants by noisy score (highest first)
            sorted_tenants = sorted(
                insights["tenant_behaviors"].items(),
                key=lambda x: x[1]["noisy_score"],
                reverse=True
            )
            
            for tenant, behavior in sorted_tenants:
                f.write(f"| {tenant} | {behavior['classification']} | {behavior['description']} | {behavior['noisy_score']:.3f} |\n")
            
            f.write("\n")
        else:
            f.write("No tenant behavior data available.\n\n")
        
        # Phase comparison insights
        f.write("## Phase Comparison Insights\n\n")
        
        if "phase_comparisons" in insights and insights["phase_comparisons"]:
            for metric, metric_insights in insights["phase_comparisons"].items():
                f.write(f"### {metric.replace('_', ' ').title()}\n\n")
                
                if metric_insights:
                    # Take the top 5 most significant insights
                    top_insights = sorted(metric_insights, key=lambda x: x["significance"], reverse=True)[:5]
                    
                    for insight in top_insights:
                        f.write(f"- **{insight['tenant']}**: {insight['description']}\n")
                    
                    f.write("\n")
                else:
                    f.write("No significant insights found for this metric.\n\n")
        else:
            f.write("No phase comparison insights available.\n\n")
        
        # Anomaly detection insights
        f.write("## Anomaly Detection Highlights\n\n")
        
        if "anomaly_detections" in insights and insights["anomaly_detections"]:
            for metric, metric_insights in insights["anomaly_detections"].items():
                f.write(f"### {metric.replace('_', ' ').title()}\n\n")
                
                if metric_insights:
                    for insight in metric_insights:
                        f.write(f"- **{insight['tenant']}**: {insight['description']}\n")
                    
                    f.write("\n")
                else:
                    f.write("No anomalies detected for this metric.\n\n")
        else:
            f.write("No anomaly detection insights available.\n\n")
        
        # Conclusion and recommendations
        f.write("## Key Takeaways\n\n")
        
        # Generate takeaways based on available insights
        takeaways = []
        
        # Find most variable tenant
        if "tenant_behaviors" in insights and insights["tenant_behaviors"]:
            most_noisy = max(
                insights["tenant_behaviors"].items(),
                key=lambda x: x[1]["noisy_score"]
            )
            takeaways.append(f"- {most_noisy[0]} shows the highest variability with a noisy score of {most_noisy[1]['noisy_score']:.3f}.")
        
        # Find patterns in phase comparisons
        if "phase_comparisons" in insights and insights["phase_comparisons"]:
            # Count tenants with significant changes in attack phase
            attack_changes = {}
            for metric, metric_insights in insights["phase_comparisons"].items():
                for insight in metric_insights:
                    if "attack phase" in insight["description"]:
                        tenant = insight["tenant"]
                        attack_changes[tenant] = attack_changes.get(tenant, 0) + 1
            
            if attack_changes:
                most_affected = max(attack_changes.items(), key=lambda x: x[1])
                takeaways.append(f"- {most_affected[0]} was most affected during the attack phase with significant changes in {most_affected[1]} metrics.")
                
            # Count tenants that recovered well
            recovery_metrics = {}
            for metric, metric_insights in insights["phase_comparisons"].items():
                for insight in metric_insights:
                    if "returned to near-baseline" in insight["description"]:
                        tenant = insight["tenant"]
                        recovery_metrics[tenant] = recovery_metrics.get(tenant, 0) + 1
            
            if recovery_metrics:
                best_recovery = max(recovery_metrics.items(), key=lambda x: x[1])
                takeaways.append(f"- {best_recovery[0]} showed the best recovery pattern with {best_recovery[1]} metrics returning to baseline levels.")
        
        # Add insights from anomaly detection
        if "anomaly_detections" in insights and insights["anomaly_detections"]:
            # Count anomalies by tenant
            anomaly_counts = {}
            for metric, metric_insights in insights["anomaly_detections"].items():
                for insight in metric_insights:
                    tenant = insight["tenant"]
                    anomaly_counts[tenant] = anomaly_counts.get(tenant, 0) + 1
            
            if anomaly_counts:
                most_anomalies = max(anomaly_counts.items(), key=lambda x: x[1])
                takeaways.append(f"- {most_anomalies[0]} exhibited the most anomalies with unusual behavior detected in {most_anomalies[1]} metrics.")
        
        # Write takeaways
        if takeaways:
            for takeaway in takeaways:
                f.write(f"{takeaway}\n")
        else:
            f.write("- No significant patterns were identified in the data.\n")
        
        f.write("\n## Recommendations for Further Analysis\n\n")
        f.write("1. **Deeper focus on inter-tenant relationships**: Analyze correlation patterns between tenants during attack phases to better understand impact propagation.\n")
        f.write("2. **Metric sensitivity analysis**: Identify which metrics are most sensitive to attacks and should be monitored closely.\n")
        f.write("3. **Recovery pattern classification**: Develop a classification system for different types of recovery patterns to predict system resilience.\n")
    
    logger.info(f"Presentation summary saved to {summary_path}")

def main():
    """Main function to generate insights and summary for presentation."""
    # Define base output directory
    output_dir = "/home/phil/Projects/k8s-noisy-analysis/outputs/demo-experiment-1-round"
    
    # Add print statements for debugging
    print(f"Starting insight generation for {output_dir}")
    
    # Load data from pipeline outputs
    print(f"Loading data from {output_dir}")
    logger.info(f"Loading data from {output_dir}")
    data = load_pipeline_data(output_dir)
    print(f"Data loaded. Keys: {list(data.keys())}")
    
    # Generate insights
    print("Generating insights from analysis results")
    logger.info("Generating insights from analysis results")
    insights = generate_insights(data, output_dir)
    print(f"Insights generated. Keys: {list(insights.keys())}")
    
    # Create presentation summary
    print("Creating presentation summary")
    logger.info("Creating presentation summary")
    generate_presentation_summary(insights, output_dir)
    
    print("Insights generation completed successfully!")
    logger.info("Insights generation completed successfully!")

if __name__ == "__main__":
    main()
