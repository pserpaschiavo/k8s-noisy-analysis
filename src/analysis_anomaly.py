"""
Module: analysis_anomaly.py
Description: Implements methods for anomaly detection in time series.

This module defines functions to identify anomalous observations in time series
using different techniques, such as statistical distribution-based detection,
temporal decomposition, and machine learning methods.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import IsolationForest
from scipy import stats

# Using conditional import to avoid circular dependency
try:
    from src.pipeline import PipelineStage
    pipeline_available = True
except ImportError:
    pipeline_available = False
    # Mock base class for when pipeline.py cannot be imported
    class PipelineStage:
        def __init__(self, *args, **kwargs):
            pass

# Logging setup
logger = logging.getLogger("analysis_anomaly")

class AnomalyDetectionStage(PipelineStage):
    """
    Pipeline stage to detect anomalies in time series.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(
            name="Anomaly Detection", 
            description="Anomaly detection in multi-tenant time series"
        )
        self.output_dir = output_dir
    
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementation of anomaly detection.
        
        Args:
            context: Current pipeline context with previous results.
            
        Returns:
            Updated context with anomaly detection results.
        """
        self.logger.info("Starting anomaly detection")
        
        # Check if we have data to analyze
        df_long = context.get('df_long')
        if df_long is None:
            self.logger.warning("Main DataFrame not found in context")
            context['error'] = "Main DataFrame not available for anomaly detection"
            return context
        
        # Output directory
        output_dir = self.output_dir or context.get('output_dir')
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), 'outputs', 'plots', 'anomaly_detection')
        else:
            output_dir = os.path.join(output_dir, 'plots', 'anomaly_detection')
            
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract necessary data
        experiment_id = context.get('experiment_id', df_long['experiment_id'].iloc[0])
        metrics = context.get('selected_metrics', df_long['metric_name'].unique())
        tenants = context.get('selected_tenants', df_long['tenant_id'].unique())
        rounds = df_long['round_id'].unique()
        
        anomaly_metrics = {}
        visualization_paths = []
        
        # For each metric/round combination
        for metric in metrics:
            anomalies_df = []
            
            for round_id in rounds:
                for phase in df_long['experimental_phase'].unique():
                    # Process each tenant
                    for tenant in tenants:
                        # Filter data
                        tenant_data = df_long[(df_long['metric_name'] == metric) & 
                                             (df_long['round_id'] == round_id) & 
                                             (df_long['experimental_phase'] == phase) &
                                             (df_long['tenant_id'] == tenant)]
                        
                        if len(tenant_data) < 10:
                            continue
                        
                        try:
                            # Detect anomalies using Z-score
                            anomalies = detect_anomalies_zscore(
                                tenant_data, z_threshold=3.0
                            )
                            
                            if not anomalies.empty:
                                anomalies['tenant_id'] = tenant
                                anomalies['round_id'] = round_id
                                anomalies['experimental_phase'] = phase
                                anomalies_df.append(anomalies)
                            
                            # Generate visualization
                            fig_path = plot_anomalies(
                                tenant_data, anomalies, 
                                metric, tenant, round_id, phase,
                                output_dir
                            )
                            visualization_paths.append(fig_path)
                            
                        except Exception as e:
                            self.logger.error(f"Error detecting anomalies for {tenant}, {metric}, {round_id}: {str(e)}")
            
            # Consolidate all anomalies for this metric
            if anomalies_df:
                anomaly_metrics[metric] = pd.concat(anomalies_df, ignore_index=True)
        
        # Store results in context
        context['anomaly_metrics'] = anomaly_metrics
        context['anomaly_visualization_paths'] = visualization_paths
        
        self.logger.info(f"Anomaly detection completed. {len(visualization_paths)} visualizations generated.")
        
        return context


def detect_anomalies_zscore(
    df: pd.DataFrame,
    z_threshold: float = 3.0
) -> pd.DataFrame:
    """
    Detects anomalies using the Z-score method.
    
    Args:
        df: DataFrame with data for one tenant for one metric
        z_threshold: Z-score threshold to consider a point as an anomaly
        
    Returns:
        DataFrame with the detected anomalies
    """
    # Check if there is enough data
    if len(df) < 4:
        logger.warning("Insufficient data for anomaly detection")
        return pd.DataFrame()
    
    # Calculate statistics
    mean_val = df['metric_value'].mean()
    std_val = df['metric_value'].std()
    
    # Avoid division by zero
    if std_val == 0:
        logger.warning("Zero standard deviation detected, cannot calculate z-score")
        return pd.DataFrame()
    
    # Calculate z-score
    df['z_score'] = (df['metric_value'] - mean_val) / std_val
    
    # Identify anomalies
    anomalies = df[abs(df['z_score']) > z_threshold].copy()
    
    return anomalies


def plot_anomalies(
    df: pd.DataFrame,
    anomalies: pd.DataFrame,
    metric: str,
    tenant: str,
    round_id: str,
    phase: str,
    output_dir: str
) -> str:
    """
    Generates a time series visualization with highlighted anomalies.
    
    Args:
        df: DataFrame with the complete time series
        anomalies: DataFrame with anomalous points
        metric: Metric name
        tenant: Tenant ID
        round_id: Round ID
        phase: Experimental phase
        output_dir: Directory to save the visualization
        
    Returns:
        Path to the generated image file
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Convert timestamp to datetime if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Find the initial timestamp of the phase
    phase_start = df['timestamp'].min()
    
    # Calculate the total duration of the phase in seconds
    total_duration = (df['timestamp'].max() - phase_start).total_seconds()
    
    # Always use seconds for consistency
    time_unit = 1  # Always use seconds
    x_label = "Seconds since phase start"
    
    # Calculate relative times
    elapsed = (df['timestamp'] - phase_start).dt.total_seconds() / time_unit
    
    # Plot the time series
    ax.plot(elapsed, df['metric_value'], 'b-', label='Time Series')
    
    # Highlight anomalies
    if not anomalies.empty:
        # Ensure datetime format
        if not pd.api.types.is_datetime64_any_dtype(anomalies['timestamp']):
            anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'], errors='coerce')
        
        # Calculate relative time for anomalies
        anomaly_elapsed = (anomalies['timestamp'] - phase_start).dt.total_seconds() / time_unit
        ax.scatter(anomaly_elapsed, anomalies['metric_value'], color='red', s=80, alpha=0.6, label='Anomalies')
    
    # Add mean line and confidence bands
    mean_val = df['metric_value'].mean()
    std_val = df['metric_value'].std()
    
    ax.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label='Mean')
    ax.axhline(y=mean_val + 3*std_val, color='orange', linestyle=':', alpha=0.5, label='Threshold (3σ)')
    ax.axhline(y=mean_val - 3*std_val, color='orange', linestyle=':', alpha=0.5)
    
    # Chart formatting
    ax.set_title(f'Anomaly Detection: {metric} - {tenant}', fontweight='bold')
    plt.suptitle(f'Round: {round_id}, Phase: {phase}', fontsize=10)
    ax.set_xlabel(x_label)
    ax.set_ylabel(f'Value ({metric})')
    
    # Format timestamps
    plt.xticks(rotation=45)
    fig.tight_layout()
    
    # Add legend
    plt.legend(loc='best')
    
    # Save the chart
    safe_metric = metric.replace('/', '_').replace(' ', '_')
    safe_tenant = tenant.replace('-', '_')
    safe_phase = phase.replace(' ', '_')
    file_name = f"anomaly_detection_{safe_metric}_{safe_tenant}_{safe_phase}_{round_id}.png"
    fig_path = os.path.join(output_dir, file_name)
    
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    # plt.close() # Comentado para permitir a exibição do gráfico
    plt.show()
    
    return fig_path
