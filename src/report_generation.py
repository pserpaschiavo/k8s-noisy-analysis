"""
Module: report_generation.py
Description: Generation of reports and aggregation of insights for multi-tenant analysis.

This module implements functionalities for:
1. Consolidation of results from all analyses.
2. Identification of "noisy tenants" based on objective metrics.
3. Generation of a final report and a comparative inter-tenant table.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from matplotlib.patches import Rectangle

# Configure plots
plt.style.use('tableau-colorblind10')
logger = logging.getLogger("report_generation")

def generate_tenant_metrics(
    granger_matrices: Dict[str, pd.DataFrame],
    te_matrices: Dict[str, pd.DataFrame],
    correlation_matrices: Dict[str, Dict[str, pd.DataFrame]],
    phase_comparison_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Generates impact metrics for each tenant based on the analyses.
    Adapts to the 7-phase experimental model by calculating the maximum variation
    across all noise phases.

    Args:
        granger_matrices: Granger causality matrices.
        te_matrices: Transfer Entropy matrices.
        correlation_matrices: Correlation matrices by metric and phase.
        phase_comparison_results: Results of the comparison between phases.

    Returns:
        DataFrame with metrics and "noisy" ranking for each tenant.
    """
    # Structure to store metrics for each tenant
    tenant_metrics = {}

    # Collect all tenants present in any matrix
    all_tenants = set()
    for matrix_dict in [granger_matrices, te_matrices]:
        for matrix in matrix_dict.values():
            if not matrix.empty:
                all_tenants.update(matrix.index)

    for corr_dict in correlation_matrices.values():
        for matrix in corr_dict.values():
            if not matrix.empty:
                all_tenants.update(matrix.index)

    # Initialize metrics for each tenant
    for tenant in all_tenants:
        tenant_metrics[tenant] = {
            'causality_impact_score': 0.0,  # How much this tenant impacts others
            'causality_affected_score': 0.0, # How much this tenant is affected by others
            'correlation_strength': 0.0,  # Average strength of correlations with other tenants
            'total_phase_variation': 0.0,  # Maximum average variation between noise phases and baseline
            'cpu_noise_variation': 0.0,
            'memory_noise_variation': 0.0,
            'network_noise_variation': 0.0,
            'disk_noise_variation': 0.0,
            'combined_noise_variation': 0.0,
            'noisy_score': 0.0,  # Final score to rank "noisy" tenants
            'metrics_count': 0,  # Counter for normalization
            'variation_metrics_count': 0
        }

    # 1. Granger causality analysis
    for key, matrix in granger_matrices.items():
        if matrix.empty:
            continue

        # For each tenant as a source of causality
        for source in matrix.columns:
            if source not in tenant_metrics:
                continue

            # Low p-values = high causality
            # We transform to 1 - p_value so that high values = more causality
            causal_values = 1.0 - matrix[source].to_numpy()
            # Remove diagonal and NaNs
            causal_values = [v for v in causal_values if not np.isnan(v) and v < 1.0]

            if causal_values:
                tenant_metrics[source]['causality_impact_score'] += np.mean(causal_values)
                tenant_metrics[source]['metrics_count'] += 1

    # 2. Transfer Entropy analysis
    for key, matrix in te_matrices.items():
        if matrix.empty:
            continue

        for source in matrix.columns:
            if source not in tenant_metrics:
                continue

            # TE: higher values = more causality
            te_values = matrix[source].to_numpy()
            # Remove diagonal and NaNs
            te_values = [v for v in te_values if not np.isnan(v) and v > 0]

            if te_values:
                tenant_metrics[source]['causality_impact_score'] += np.mean(te_values) * 5  # Higher weight for TE
                tenant_metrics[source]['metrics_count'] += 1

    # 3. Correlation analysis
    for metric_key, phase_dict in correlation_matrices.items():
        for phase, matrix in phase_dict.items():
            if matrix.empty:
                continue

            for tenant in matrix.index:
                if tenant not in tenant_metrics:
                    continue

                # Absolute correlation values (ignoring auto-correlation)
                corr_values = matrix.loc[tenant].abs().values
                corr_values = [v for v in corr_values if not np.isnan(v) and v < 1.0]

                if corr_values:
                    tenant_metrics[tenant]['correlation_strength'] += np.mean(corr_values)
                    tenant_metrics[tenant]['metrics_count'] += 1

    # 4. Analysis of variation between phases
    for key, stats_df in phase_comparison_results.items():
        if stats_df.empty:
            continue

        for _, row in stats_df.iterrows():
            tenant = row['tenant_id']
            if tenant not in tenant_metrics:
                continue

            # Calculate variation for each noise type
            noise_phases = ['CPU-Noise', 'Memory-Noise', 'Network-Noise', 'Disk-Noise', 'Combined-Noise']
            for phase in noise_phases:
                col_name = f"{phase}_vs_baseline_pct"
                if col_name in row.index:
                    variation = pd.to_numeric(row[col_name], errors='coerce')
                    if not pd.isna(variation):
                        variation_abs = abs(variation)
                        phase_key = f"{phase.lower().replace('-', '_')}_variation"
                        tenant_metrics[tenant][phase_key] += variation_abs
                        tenant_metrics[tenant]['total_phase_variation'] += variation_abs
            
            tenant_metrics[tenant]['variation_metrics_count'] += 1


    # Normalize and calculate final score
    for tenant, metrics in tenant_metrics.items():
        # Avoid division by zero
        count = max(metrics['metrics_count'], 1)
        var_count = max(metrics['variation_metrics_count'], 1)

        # Normalization
        metrics['causality_impact_score'] /= count
        metrics['correlation_strength'] /= count
        metrics['total_phase_variation'] /= var_count
        metrics['cpu_noise_variation'] /= var_count
        metrics['memory_noise_variation'] /= var_count
        metrics['network_noise_variation'] /= var_count
        metrics['disk_noise_variation'] /= var_count
        metrics['combined_noise_variation'] /= var_count


        # Calculate the final score (weighted)
        metrics['noisy_score'] = (
            metrics['causality_impact_score'] * 0.5 +  # 50% for causality
            metrics['correlation_strength'] * 0.3 +    # 30% for correlation
            metrics['total_phase_variation'] * 0.2           # 20% for phase variation
        )

    # Convert to DataFrame, sorted by total score
    df = pd.DataFrame.from_dict(tenant_metrics, orient='index')
    df.index.name = 'tenant_id'
    df = df.reset_index()
    df = df.sort_values(by='noisy_score', ascending=False)

    return df

def generate_markdown_report(
    tenant_metrics: pd.DataFrame,
    context: Dict[str, Any],
    rank_plot_path: str,
    metrics_table_path: str,
    phase_variation_plot_path: str,
    out_dir: str
) -> str:
    """
    Generates a final report in Markdown format.

    Args:
        tenant_metrics: DataFrame with metrics for each tenant.
        context: Context with results from all analyses.
        rank_plot_path: Path to the tenant ranking plot.
        metrics_table_path: Path to the complete metrics table.
        phase_variation_plot_path: Path to the phase variation plot.
        out_dir: Output directory for the report.

    Returns:
        Path to the generated report file.
    """
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_path = os.path.join(out_dir, f"{report_filename}.md")

    with open(report_path, 'w') as f:
        f.write("# Multi-Tenant Analysis Report\n\n")
        f.write(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Noisy tenants section
        f.write("## Identification of Tenants with the Greatest Impact\n\n")

        # Identify the top noisy tenant
        if not tenant_metrics.empty:
            top_tenant = tenant_metrics.iloc[0]['tenant_id']
            top_score = tenant_metrics.iloc[0]['noisy_score']
            f.write(f"**Tenant with the greatest impact:** `{top_tenant}` (score: {top_score:.2f})\n\n")
            f.write(f"![Tenant Ranking]({os.path.basename(rank_plot_path)})\n\n")

            # Comparative table of tenants
            f.write("### Tenant Comparative Table\n\n")
            f.write("| Tenant | Total Score | Causal Impact | Correlation | Total Variation | CPU Var. | Memory Var. | Network Var. | Disk Var. | Combined Var. |\n")
            f.write("|---|---|---|---|---|---|---|---|---|---|\n")

            for _, row in tenant_metrics.iterrows():
                f.write(f"| {row['tenant_id']} | {row['noisy_score']:.2f} | {row['causality_impact_score']:.2f} | ")
                f.write(f"{row['correlation_strength']:.2f} | {row['total_phase_variation']:.2f} | ")
                f.write(f"{row['cpu_noise_variation']:.2f} | {row['memory_noise_variation']:.2f} | ")
                f.write(f"{row['network_noise_variation']:.2f} | {row['disk_noise_variation']:.2f} | ")
                f.write(f"{row['combined_noise_variation']:.2f} |\n")

            f.write("\n*Full table available at:* ")
            f.write(f"`{os.path.basename(metrics_table_path)}`\n\n")

        # Visualizations section
        f.write("## Generated Visualizations\n\n")

        # Phase variation plot
        if phase_variation_plot_path and os.path.exists(phase_variation_plot_path):
            f.write(f"### Phase Variation Analysis\n\n")
            f.write(f"![Phase Variation by Tenant]({os.path.basename(phase_variation_plot_path)})\n\n")


        # Descriptive plots
        desc_plots = context.get('descriptive_plot_paths', [])
        if desc_plots:
            f.write(f"### Descriptive Analysis ({len(desc_plots)} visualizations)\n\n")
            for i, path in enumerate(desc_plots[:3]):  # Limit to 3 examples
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(desc_plots) > 3:
                f.write(f"- *...and {len(desc_plots) - 3} more visualizations*\n")
            f.write("\n")

        # Correlation plots
        corr_plots = context.get('correlation_plot_paths', [])
        if corr_plots:
            f.write(f"### Correlation Analysis ({len(corr_plots)} visualizations)\n\n")
            for i, path in enumerate(corr_plots[:3]):
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(corr_plots) > 3:
                f.write(f"- *...and {len(corr_plots) - 3} more visualizations*\n")
            f.write("\n")

        # Causality plots
        causality_plots = context.get('causality_plot_paths', [])
        if causality_plots:
            f.write(f"### Causality Analysis ({len(causality_plots)} visualizations)\n\n")
            for i, path in enumerate(causality_plots[:3]):
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(causality_plots) > 3:
                f.write(f"- *...and {len(causality_plots) - 3} more visualizations*\n")
            f.write("\n")

        # Phase comparison plots
        phase_plots = context.get('phase_comparison_plot_paths', [])
        if phase_plots:
            f.write(f"### Phase Comparison ({len(phase_plots)} visualizations)\n\n")
            for i, path in enumerate(phase_plots[:3]):
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(phase_plots) > 3:
                f.write(f"- *...and {len(phase_plots) - 3} more visualizations*\n")
            f.write("\n")

        # Methodology
        f.write("## Methodology\n\n")
        f.write("This report uses a multi-dimensional analysis methodology to identify tenants with the greatest impact:\n\n")
        f.write("1. **Causality Analysis**:\n")
        f.write("   - Granger Causality: Tests whether past values of one tenant help predict future values of another.\n")
        f.write("   - Transfer Entropy: Quantifies the directional information transfer between time series.\n\n")
        f.write("2. **Correlation Analysis**:\n")
        f.write("   - Measures the strength of the linear relationship between metrics of different tenants.\n")
        f.write("   - Higher values indicate greater interdependence.\n\n")
        f.write("3. **Phase Variation**:\n")
        f.write("   - Quantifies the magnitude of change in metrics during various noise phases (CPU, Memory, etc.) compared to the baseline.\n")
        f.write("   - Tenants with higher variation are considered to have a greater impact or be more sensitive to the noisy environment.\n\n")

        f.write("**The final score is calculated as a weighted average:**\n")
        f.write("- 50% Causal Impact (with higher weight for causality detected via Transfer Entropy)\n")
        f.write("- 30% Correlation Strength\n")
        f.write("- 20% Phase Variation\n\n")

        f.write("### Limitations of the Methodology\n\n")
        f.write("- Statistical causality does not necessarily imply direct physical causality.\n")
        f.write("- Correlation does not imply causation; it may reflect common external factors.\n")
        f.write("- The analysis assumes that the time series are adequately sampled and stationary.\n")

    return report_path

def generate_tenant_ranking_plot(tenant_metrics: pd.DataFrame, output_path: str) -> None:
    """
    Generates a visualization of the tenant ranking by impact.

    Args:
        tenant_metrics: DataFrame with tenant metrics.
        output_path: Path to save the plot.
    """
    if tenant_metrics.empty:
        logger.warning("Cannot generate tenant ranking plot: tenant_metrics is empty.")
        return

    plt.figure(figsize=(12, 7))

    # Main plot of the total score
    ax = sns.barplot(
        x='tenant_id',
        y='noisy_score',
        data=tenant_metrics,
        palette="viridis",
        hue='tenant_id',
        legend=False
    )

    # Add values above each bar
    for p in ax.patches:
        if isinstance(p, Rectangle):
            ax.annotate(f"{p.get_height():.2f}", 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 9), 
                        textcoords='offset points')

    plt.title('Tenant Ranking by Noisy Score', fontsize=16)
    plt.xlabel('Tenant ID', fontsize=12)
    plt.ylabel('Noisy Score (Impact)', fontsize=12)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.tight_layout()
    
    # Save the plot
    try:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Tenant ranking plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving tenant ranking plot to {output_path}: {e}")
    finally:
        plt.close()


def generate_phase_variation_plot(tenant_metrics: pd.DataFrame, output_path: str) -> None:
    """
    Generates a grouped bar chart of phase variations for each tenant.

    Args:
        tenant_metrics: DataFrame with tenant metrics.
        output_path: Path to save the plot.
    """
    if tenant_metrics.empty:
        logger.warning("Cannot generate phase variation plot: tenant_metrics is empty.")
        return

    plot_data = tenant_metrics[[
        'tenant_id',
        'cpu_noise_variation',
        'memory_noise_variation',
        'network_noise_variation',
        'disk_noise_variation',
        'combined_noise_variation'
    ]].copy()

    plot_data.set_index('tenant_id').plot(
        kind='bar',
        figsize=(15, 8),
        width=0.8
    )

    plt.title('Phase Variation by Tenant and Noise Type', fontsize=16)
    plt.xlabel('Tenant ID', fontsize=12)
    plt.ylabel('Average Metric Variation (%)', fontsize=12)
    plt.xticks(rotation=45, horizontalalignment="right")
    plt.legend(title='Noise Type')
    plt.tight_layout()

    # Save the plot
    try:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Phase variation plot saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving phase variation plot to {output_path}: {e}")
    finally:
        plt.close()


def save_metrics_table(tenant_metrics: pd.DataFrame, output_path: str) -> None:
    """
    Saves the complete tenant metrics table to a CSV file.

    Args:
        tenant_metrics: DataFrame with tenant metrics.
        output_path: Path to save the CSV file.
    """
    try:
        tenant_metrics.to_csv(output_path, index=False)
        logger.info(f"Tenant metrics table saved to {output_path}")
    except Exception as e:
        logger.error(f"Error saving metrics table to {output_path}: {e}")

def run_report_generation(
    context: Dict[str, Any],
    out_dir: str
) -> Dict[str, Any]:
    """
    Main function to run the report generation stage.

    Args:
        context: Dictionary with results from previous stages.
        out_dir: Output directory for reports and plots.

    Returns:
        Dictionary with paths to the generated artifacts.
    """
    logger.info("Starting report generation stage...")
    
    # Create output directory if it doesn't exist
    reports_dir = os.path.join(out_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    
    # Extract results from context
    granger_results = context.get('granger_results', {})
    te_results = context.get('te_results', {})
    correlation_results = context.get('correlation_results', {})
    phase_comparison_stats = context.get('phase_comparison_stats', {}) # Corrected key

    # Generate tenant impact metrics
    tenant_metrics_df = generate_tenant_metrics(
        granger_matrices=granger_results.get('causality_matrices', {}),
        te_matrices=te_results.get('causality_matrices', {}),
        correlation_matrices=correlation_results.get('correlation_matrices', {}),
        phase_comparison_results=phase_comparison_stats.get('stats_by_metric', {}) # Use the corrected variable
    )
    
    # Define paths for artifacts
    metrics_table_path = os.path.join(reports_dir, "tenant_impact_metrics.csv")
    rank_plot_path = os.path.join(reports_dir, "tenant_ranking.png")
    phase_variation_plot_path = os.path.join(reports_dir, "phase_variation_by_tenant.png")

    # Save artifacts
    save_metrics_table(tenant_metrics_df, metrics_table_path)
    generate_tenant_ranking_plot(tenant_metrics_df, rank_plot_path)
    generate_phase_variation_plot(tenant_metrics_df, phase_variation_plot_path)

    # Generate Markdown report
    report_path = generate_markdown_report(
        tenant_metrics=tenant_metrics_df,
        context=context,
        rank_plot_path=rank_plot_path,
        metrics_table_path=metrics_table_path,
        phase_variation_plot_path=phase_variation_plot_path,
        out_dir=reports_dir
    )

    logger.info(f"Report generation stage completed. Report saved to {report_path}")

    return {
        "report_path": report_path,
        "metrics_table_path": metrics_table_path,
        "rank_plot_path": rank_plot_path,
        "phase_variation_plot_path": phase_variation_plot_path
    }
