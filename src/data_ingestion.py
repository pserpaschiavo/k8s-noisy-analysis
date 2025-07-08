"""
Module: data_ingestion.py
Description: Functions for ingesting, validating, and consolidating experiment data into a long-format DataFrame.
"""
import os
import pandas as pd
from typing import List, Optional, Dict
import logging

from src.utils import normalize_phase_name
from src.pipeline_stage import PipelineStage
from src.config import PipelineConfig


class DataIngestionStage(PipelineStage):
    """Pipeline stage for ingesting and loading experiment data."""

    def __init__(self, config: PipelineConfig):
        """
        Initializes the DataIngestionStage.

        Args:
            config: The pipeline configuration object.
        """
        super().__init__("Data Ingestion", "Ingests and loads experiment data from raw files or a pre-processed Parquet file.")
        self.config = config

    def _execute_implementation(self, context: dict) -> dict:
        """
        Executes the data ingestion logic.

        This method checks if a pre-processed Parquet file is specified and exists.
        If so, it loads the data from there. Otherwise, it ingests the data from
        the raw experiment files. The resulting DataFrame is stored in the context.

        Args:
            context: The pipeline context dictionary.

        Returns:
            The updated context dictionary with the ingested data.
        """
        logging.info("Executing Data Ingestion Stage...")
        
        processed_data_path = self.config.get_processed_data_path()

        if processed_data_path and os.path.exists(processed_data_path):
            logging.info(f"Loading data from existing parquet file: {processed_data_path}")
            try:
                df = load_from_parquet(processed_data_path)
                context['data'] = df
                logging.info("Successfully loaded data from parquet.")
            except Exception as e:
                logging.error(f"Failed to load from parquet file {processed_data_path}: {e}")
                raise
        else:
            logging.info("Ingesting data from raw experiment files...")
            try:
                df = ingest_experiment_data(
                    data_root=self.config.get_data_root(),
                    selected_metrics=self.config.get_selected_metrics(),
                    selected_tenants=self.config.get_selected_tenants(),
                    selected_rounds=self.config.get_selected_rounds(),
                    selected_phases=self.config.get_selected_phases(),
                )
                context['data'] = df
                logging.info("Successfully ingested data from raw files.")

                # Save the processed data if a path is provided in the config
                if processed_data_path:
                    logging.info(f"Saving ingested data to {processed_data_path}...")
                    try:
                        # Ensure the directory exists
                        os.makedirs(os.path.dirname(processed_data_path), exist_ok=True)
                        df.to_parquet(processed_data_path)
                        logging.info("Successfully saved data to parquet file.")
                    except Exception as e:
                        logging.error(f"Failed to save data to parquet file {processed_data_path}: {e}")

            except Exception as e:
                logging.error(f"Data ingestion from raw files failed: {e}")
                raise

        if 'data' not in context or context['data'].empty:
            logging.error("Data ingestion resulted in an empty DataFrame. Halting pipeline.")
            raise ValueError("Data ingestion failed, resulting in no data.")

        logging.info("Data Ingestion Stage completed.")
        return context

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

    # Map filename stems to their canonical metric name.
    # If a metric's filename (e.g., 'cpu_usage.csv') is the same as its desired
    # metric name, it does not need to be in this mapping.
    METRIC_FILENAME_TO_NAME = {
        'disk_io_total': 'disk_io',
        # Add other mappings here if filenames differ from metric names
    }


    # Normalize selected phases for consistent matching
    normalized_selected_phases = [
        phase.split(' - ', 1)[-1].lower().replace(' ', '-') 
        for phase in selected_phases
    ] if selected_phases else None

    for experiment_path in list_experiments(data_root):
        experiment_id = os.path.basename(experiment_path)
        for round_path in list_rounds(experiment_path):
            round_id = os.path.basename(round_path)
            if selected_rounds and round_id not in selected_rounds:
                continue
            for phase_path in list_phases(round_path):
                experimental_phase = os.path.basename(phase_path)
                normalized_phase_name = experimental_phase.split(' - ', 1)[-1].lower().replace(' ', '-')
                
                # Use the normalized list for filtering
                if normalized_selected_phases and normalized_phase_name not in normalized_selected_phases:
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
                        filename_stem = os.path.splitext(os.path.basename(metric_file))[0]
                        
                        # Skip files used for combined metrics
                        if 'network_receive' in filename_stem or 'network_transmit' in filename_stem:
                            continue

                        # Determine the canonical metric name
                        # Use the mapping if available, otherwise use the filename stem
                        metric_name = METRIC_FILENAME_TO_NAME.get(filename_stem, filename_stem)

                        # Process only if the metric is in the selected list
                        if metric_name not in selected_metrics:
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
                                    'metric_name': metric_name,
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
