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

# Define the official order of experimental phases to ensure correct processing
PHASE_ORDER = [
    '1 - Baseline',
    '2 - CPU-Noise',
    '3 - Memory-Noise',
    '4 - Network-Noise',
    '5 - Disk-Noise',
    '6 - Combined-Noise',
    '7 - Recovery'
]

def list_experiments(data_root: str) -> List[str]:
    """List all experiment directories under the data root.
    
    If the data_root points directly to an experiment directory (containing rounds),
    return just that directory. Otherwise list all subdirectories as experiments.
    """
    # Check if data_root already points to an experiment directory (contains round directories)
    has_rounds = any(d.startswith("round-") for d in os.listdir(data_root) 
                     if os.path.isdir(os.path.join(data_root, d)))
    
    if has_rounds:
        # This is already an experiment directory, return it as the only experiment
        return [data_root]
    else:
        # This is a parent directory containing multiple experiments
        return [os.path.join(data_root, d) for d in os.listdir(data_root) 
                if os.path.isdir(os.path.join(data_root, d))]

def list_rounds(experiment_path: str) -> List[str]:
    """List all round directories under an experiment, filtering for 'round-*' prefix."""
    try:
        return [os.path.join(experiment_path, d) for d in os.listdir(experiment_path) 
                if os.path.isdir(os.path.join(experiment_path, d)) and d.startswith("round-")]
    except FileNotFoundError:
        logging.error(f"Experiment path not found: {experiment_path}")
        return []

def list_phases(round_path: str) -> List[str]:
    """List all phase directories under a round, sorted in the correct logical order.
    
    The function now validates directories against a predefined phase order and
    logs a warning for any unexpected directories, which are then ignored.
    """
    try:
        dirs = [d for d in os.listdir(round_path) if os.path.isdir(os.path.join(round_path, d))]
    except FileNotFoundError:
        logging.error(f"Round path not found: {round_path}")
        return []

    # Sort the found directories based on the predefined PHASE_ORDER
    def sort_key(d):
        try:
            return PHASE_ORDER.index(d)
        except ValueError:
            return float('inf')  # Place unknown phases at the end

    sorted_dirs = sorted(dirs, key=sort_key)
    
    known_phases = [d for d in sorted_dirs if d in PHASE_ORDER]
    unknown_dirs = [d for d in sorted_dirs if d not in PHASE_ORDER]
    
    if unknown_dirs:
        logging.warning(f"Found and ignored unexpected phase directories in {round_path}: {unknown_dirs}")

    return [os.path.join(round_path, d) for d in known_phases]

def list_tenants(phase_path: str, tenant_prefix: str = "tenant-") -> List[str]:
    """List all tenant directories under a phase, filtering by prefix."""
    return [os.path.join(phase_path, d) for d in os.listdir(phase_path) if d.startswith(tenant_prefix) and os.path.isdir(os.path.join(phase_path, d))]

def list_metric_files(tenant_path: str) -> List[str]:
    """List all CSV metric files in a tenant directory."""
    return [os.path.join(tenant_path, f) for f in os.listdir(tenant_path) if f.endswith('.csv')]

def load_from_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Load data directly from an existing parquet file.
    
    Args:
        parquet_path: Path to the parquet file to load
        
    Returns:
        Pandas DataFrame with the loaded data
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
        
    logging.info(f"Loading data from existing parquet file: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    logging.info(f"Loaded {len(df)} records from parquet file")
    
    return df

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
                        
                        # Correctly associate the metric with its tenant
                        # The tenant_id should be derived from the folder name, not the metric file
                        
                        try:
                            df = pd.read_csv(metric_file)
                            if 'timestamp' not in df.columns or 'value' not in df.columns:
                                logging.warning(f"File {metric_file} missing required columns. Skipping.")
                                continue
                            
                            # The metric name should be clean (e.g., 'cpu_usage'), and the tenant_id
                            # should be what is specified in the directory structure.
                            
                            for _, row in df.iterrows():
                                records.append({
                                    'timestamp': row['timestamp'],
                                    'metric_value': row['value'],
                                    'metric_name': metric_name,
                                    'tenant_id': tenant_id, # This is the fix
                                    'experimental_phase': experimental_phase,
                                    'round_id': round_id,
                                    'experiment_id': experiment_id
                                })
                        except Exception as e:
                            logging.error(f"Error reading {metric_file}: {e}")
    
    if not records:
        logging.warning("No records were ingested. Please check data paths and selections in config.")
        return pd.DataFrame()
    
    df_long = pd.DataFrame.from_records(records)
    # Type conversions and cleaning
    if not df_long.empty:
        # Correctly convert Prometheus Unix timestamps (assuming float/int) to datetime objects
        df_long['timestamp'] = pd.to_datetime(df_long['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
        df_long['metric_value'] = pd.to_numeric(df_long['metric_value'], errors='coerce')
    return df_long
