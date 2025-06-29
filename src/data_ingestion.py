"""
Module: data_ingestion.py
Description: Functions for ingesting, validating, and consolidating experiment data into a long-format DataFrame.
"""
import os
import pandas as pd
from typing import List, Optional, Dict
import logging

from src.utils import normalize_phase_name

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

# Define the official order of experimental phases to ensure correct processing
PHASE_ORDER = [
    '1 - Baseline',
    '2 - CPU Noise',
    '3 - Memory Noise',
    '4 - Network Noise',
    '5 - Disk Noise',
    '6 - Combined Noise',
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

    # Normalize phase names before sorting and returning
    normalized_dirs = {}
    for d in dirs:
        normalized_name = normalize_phase_name(d)
        if normalized_name:
            if normalized_name not in normalized_dirs: # Keep the first encountered directory for a given canonical name
                normalized_dirs[normalized_name] = d
        else:
            logging.warning(f"Found and ignored unexpected phase directory in {round_path}: {d}")

    # Sort the found directories based on the predefined PHASE_ORDER
    def sort_key(normalized_name):
        try:
            return PHASE_ORDER.index(normalized_name)
        except ValueError:
            return float('inf')  # Place unknown phases at the end

    sorted_normalized_names = sorted(normalized_dirs.keys(), key=sort_key)

    # Return the original directory names in the correct order
    return [os.path.join(round_path, normalized_dirs[name]) for name in sorted_normalized_names]


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
    selected_phases: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Ingest all experiment data into a long-format DataFrame, with optional filtering.
    This version handles combined metrics (network_io) and filename mappings (disk_io).
    """
    records = []
    if selected_metrics is None:
        selected_metrics = []

    for experiment_path in list_experiments(data_root):
        experiment_id = os.path.basename(experiment_path)
        for round_path in list_rounds(experiment_path):
            round_id = os.path.basename(round_path)
            if selected_rounds and round_id not in selected_rounds:
                continue
            for phase_path in list_phases(round_path):
                experimental_phase = os.path.basename(phase_path)
                normalized_phase_name = experimental_phase.split(' - ', 1)[-1].lower().replace(' ', '-')
                if selected_phases and normalized_phase_name not in selected_phases:
                    continue

                for tenant_path in list_tenants(phase_path):
                    tenant_id = os.path.basename(tenant_path)
                    if selected_tenants and tenant_id not in selected_tenants:
                        continue
                    
                    # Handle network_io as a special case by combining two files
                    if 'network_io' in selected_metrics:
                        rx_file = os.path.join(tenant_path, 'network_receive.csv')
                        tx_file = os.path.join(tenant_path, 'network_transmit.csv')
                        
                        if os.path.exists(rx_file) and os.path.exists(tx_file):
                            try:
                                rx_df = pd.read_csv(rx_file, usecols=['timestamp', 'value'])
                                tx_df = pd.read_csv(tx_file, usecols=['timestamp', 'value'])
                                
                                rx_df['timestamp'] = pd.to_datetime(rx_df['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
                                tx_df['timestamp'] = pd.to_datetime(tx_df['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
                                
                                rx_df.dropna(subset=['timestamp'], inplace=True)
                                tx_df.dropna(subset=['timestamp'], inplace=True)

                                # Merge based on the nearest timestamp
                                merged_df = pd.merge_asof(
                                    rx_df.sort_values('timestamp'), 
                                    tx_df.sort_values('timestamp'), 
                                    on='timestamp', 
                                    direction='nearest', 
                                    tolerance=pd.Timedelta('2s')
                                ).fillna(0)
                                
                                merged_df['metric_value'] = pd.to_numeric(merged_df['value_x'], errors='coerce') + pd.to_numeric(merged_df['value_y'], errors='coerce')
                                merged_df.dropna(subset=['metric_value'], inplace=True)

                                for _, row in merged_df.iterrows():
                                    records.append({
                                        'timestamp': row['timestamp'],
                                        'metric_value': row['metric_value'],
                                        'metric_name': 'network_io',
                                        'tenant_id': tenant_id,
                                        'experimental_phase': experimental_phase,
                                        'round_id': round_id,
                                        'experiment_id': experiment_id
                                    })
                            except Exception as e:
                                logging.error(f"Error processing network_io for {tenant_path}: {e}")
                        else:
                            logging.warning(f"Skipping network_io for {tenant_path}: one or both files are missing.")

                    # Process other metrics
                    for metric_file in list_metric_files(tenant_path):
                        metric_name_from_file = os.path.splitext(os.path.basename(metric_file))[0]
                        
                        if 'network_receive' in metric_name_from_file or 'network_transmit' in metric_name_from_file:
                            continue

                        current_metric_name = None
                        if metric_name_from_file == 'disk_io_total' and 'disk_io' in selected_metrics:
                            current_metric_name = 'disk_io'
                        elif metric_name_from_file in selected_metrics:
                            current_metric_name = metric_name_from_file
                        
                        if not current_metric_name:
                            continue

                        try:
                            df = pd.read_csv(metric_file, on_bad_lines='warn')
                            if 'timestamp' not in df.columns or 'value' not in df.columns:
                                logging.warning(f"File {metric_file} missing required columns. Skipping.")
                                continue

                            df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S', errors='coerce')
                            df['metric_value'] = pd.to_numeric(df['value'], errors='coerce')

                            invalid_timestamps = df['timestamp'].isnull().sum()
                            invalid_values = df['metric_value'].isnull().sum()
                            if invalid_timestamps > 0 or invalid_values > 0:
                                logging.warning(f"Found {invalid_timestamps} invalid timestamps and {invalid_values} invalid values in {metric_file}. Dropping rows.")
                                df.dropna(subset=['timestamp', 'metric_value'], inplace=True)

                            for _, row in df.iterrows():
                                records.append({
                                    'timestamp': row['timestamp'],
                                    'metric_value': row['metric_value'],
                                    'metric_name': current_metric_name,
                                    'tenant_id': tenant_id,
                                    'experimental_phase': experimental_phase,
                                    'round_id': round_id,
                                    'experiment_id': experiment_id
                                })
                        except Exception as e:
                            logging.error(f"Error processing file {metric_file}: {e}")
    
    if not records:
        logging.warning("No records were ingested. Please check data paths and selections in config.")
        return pd.DataFrame()
    
    return pd.DataFrame.from_records(records)
