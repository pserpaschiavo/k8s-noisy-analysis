"""
Module: data_ingestion.py
Description: Functions for ingesting, validating, and consolidating experiment data into a long-format DataFrame.
"""
import os
import pandas as pd
from typing import List, Optional, Dict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def list_experiments(data_root: str) -> List[str]:
    """List all experiment directories under the data root."""
    return [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

def list_rounds(experiment_path: str) -> List[str]:
    """List all round directories under an experiment."""
    return [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) if os.path.isdir(os.path.join(experiment_path, d))]

def list_phases(round_path: str) -> List[str]:
    """List all phase directories (Baseline, Attack, Recovery) under a round."""
    return [os.path.join(round_path, d) for d in os.listdir(round_path) if os.path.isdir(os.path.join(round_path, d))]

def list_tenants(phase_path: str, tenant_prefix: str = "tenant-") -> List[str]:
    """List all tenant directories under a phase, filtering by prefix."""
    return [os.path.join(phase_path, d) for d in os.listdir(phase_path) if d.startswith(tenant_prefix) and os.path.isdir(os.path.join(phase_path, d))]

def list_metric_files(tenant_path: str) -> List[str]:
    """List all CSV metric files in a tenant directory."""
    return [os.path.join(tenant_path, f) for f in os.listdir(tenant_path) if f.endswith('.csv')]

def ingest_experiment_data(
    data_root: str,
    selected_metrics: Optional[List[str]] = None,
    selected_tenants: Optional[List[str]] = None,
    selected_rounds: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ingest all experiment data into a long-format DataFrame, with optional filtering by metrics, tenants, and rounds.
    """
    records = []
    for experiment_path in list_experiments(data_root):
        experiment_id = os.path.basename(experiment_path)
        for round_path in list_rounds(experiment_path):
            round_id = os.path.basename(round_path)
            if selected_rounds and round_id not in selected_rounds:
                continue
            for phase_path in list_phases(round_path):
                experimental_phase = os.path.basename(phase_path)
                for tenant_path in list_tenants(phase_path):
                    tenant_id = os.path.basename(tenant_path)
                    if selected_tenants and tenant_id not in selected_tenants:
                        continue
                    for metric_file in list_metric_files(tenant_path):
                        metric_name = os.path.splitext(os.path.basename(metric_file))[0]
                        if selected_metrics and metric_name not in selected_metrics:
                            continue
                        try:
                            df = pd.read_csv(metric_file)
                            if 'timestamp' not in df.columns or 'value' not in df.columns:
                                logging.warning(f"File {metric_file} missing required columns. Skipping.")
                                continue
                            for _, row in df.iterrows():
                                records.append({
                                    'timestamp': row['timestamp'],
                                    'metric_value': row['value'],
                                    'metric_name': metric_name,
                                    'tenant_id': tenant_id,
                                    'experimental_phase': experimental_phase,
                                    'round_id': round_id,
                                    'experiment_id': experiment_id
                                })
                        except Exception as e:
                            logging.error(f"Error reading {metric_file}: {e}")
    df_long = pd.DataFrame.from_records(records)
    # Type conversions and cleaning
    if not df_long.empty:
        # Conversão correta do timestamp do Prometheus
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
        df_long['metric_value'] = pd.to_numeric(df_long['metric_value'], errors='coerce')
    return df_long
