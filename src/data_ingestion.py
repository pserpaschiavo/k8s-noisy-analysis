"""
Module: data_ingestion.py
Description: Handles the ingestion of raw experimental data from various sources.
"""
import os
import pandas as pd
import logging
from typing import List, Optional, Dict, Any
import glob

from .pipeline_stage import PipelineStage
from .config import PipelineConfig

# Setup logging
logger = logging.getLogger(__name__)

def get_metric_name_from_filename(filename: str) -> str:
    """
    Extracts the metric name from the filename.
    Example: 'cpu_usage_total.csv' -> 'cpu_usage_total'
    """
    return os.path.splitext(os.path.basename(filename))[0]

def load_and_process_file(filepath: str, metric_name: str, experimental_phase: str, tenant_id: str) -> Optional[pd.DataFrame]:
    """
    Loads a single CSV file, processes it, and returns a DataFrame.
    """
    try:
        df = pd.read_csv(filepath)
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='%Y%m%d_%H%M%S')
        df['metric_name'] = metric_name
        df['experimental_phase'] = experimental_phase
        df['tenant_id'] = tenant_id
        
        # Rename 'value' column to 'metric_value'
        if 'value' in df.columns:
            df.rename(columns={'value': 'metric_value'}, inplace=True)
        else:
            logger.warning(f"File {filepath} does not have a 'value' column. Skipping.")
            return None

        # Ensure metric_value is numeric, coercing errors
        df['metric_value'] = pd.to_numeric(df['metric_value'], errors='coerce')
        
        # Drop rows where metric_value could not be converted
        df.dropna(subset=['metric_value'], inplace=True)

        return df[['timestamp', 'metric_name', 'experimental_phase', 'tenant_id', 'metric_value']]
    except Exception as e:
        logger.error(f"Error processing file {filepath}: {e}")
        return None

def ingest_data_for_round(round_path: str, round_id: str, selected_metrics: Optional[List[str]] = None, selected_phases: Optional[List[str]] = None) -> Optional[pd.DataFrame]:
    """
    Ingests all metric data for a specific round by searching recursively.
    """
    logger.info(f"Ingesting data from round path: {round_path} for round: {round_id}")
    
    # Find all CSV files recursively in the directory
    search_pattern = os.path.join(round_path, '**', '*.csv')
    metric_files = glob.glob(search_pattern, recursive=True)
    
    if not metric_files:
        logger.warning(f"No CSV files found recursively in {round_path}")
        return None
        
    all_dfs = []
    for filepath in metric_files:
        try:
            # Extract experimental phase from the path.
            # Example path: .../round-1/1 - Baseline/tenant-cpu/cpu_usage.csv
            # We want to extract "1 - Baseline" and "tenant-cpu"
            relative_path = os.path.relpath(filepath, round_path)
            path_parts = relative_path.split(os.sep)
            experimental_phase = path_parts[0]
            tenant_id = path_parts[1]

            # Filter by selected phases if provided
            if selected_phases and experimental_phase not in selected_phases:
                logger.debug(f"Skipping phase '{experimental_phase}' as it is not in the selected list.")
                continue

            metric_name = get_metric_name_from_filename(filepath)
            
            # Filter by selected metrics if provided
            if selected_metrics and metric_name not in selected_metrics:
                logger.debug(f"Skipping metric '{metric_name}' as it is not in the selected list.")
                continue
                
            logger.info(f"Processing file: {filepath} for metric: {metric_name}, phase: {experimental_phase}, tenant: {tenant_id}")
            df_long = load_and_process_file(filepath, metric_name, experimental_phase, tenant_id)
            
            if df_long is not None:
                all_dfs.append(df_long)

        except (IndexError, ValueError) as e:
            logger.warning(f"Could not determine experimental phase or tenant for file {filepath}. Skipping. Error: {e}")
            
    if not all_dfs:
        logger.warning(f"No data could be ingested from {round_path} after filtering.")
        return None
        
    # Concatenate all DataFrames
    round_df = pd.concat(all_dfs, ignore_index=True)
    round_df['round_id'] = round_id  # Add round_id column
    logger.info(f"Successfully ingested {len(round_df)} records from {len(all_dfs)} files in {round_path}")
    
    return round_df

def normalize_phase_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the 'experimental_phase' column by removing round-specific prefixes.
    Example: 'round-1-phase-1' -> 'phase-1'
    """
    if 'experimental_phase' in df.columns:
        # The regex extracts the part of the string that follows a pattern like 'round-X-'
        df['experimental_phase'] = df['experimental_phase'].str.replace(r'round-\d+-', '', regex=True)
        logger.info("Normalized 'experimental_phase' names.")
    return df

class DataIngestionStage(PipelineStage):
    """
    A pipeline stage for ingesting data from raw files into a structured format.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("data_ingestion", "Ingests raw data from files.")
        self.config = config

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Executes the data ingestion process for a specific round.
        """
        self.logger.info(f"Starting data ingestion for round: {round_id}...")
        
        data_root = self.config.get_data_root()
        round_path = os.path.join(data_root, round_id)

        if not os.path.isdir(round_path):
            self.logger.error(f"Round directory not found: {round_path}. Aborting ingestion for this round.")
            return {"ingested_data": pd.DataFrame(), "summary": "Round directory not found"}

        selected_metrics = self.config.get_selected_metrics()
        selected_phases = self.config.get_selected_phases()

        self.logger.info(f"Ingesting data for round: {round_id}")
        round_df = ingest_data_for_round(
            round_path,
            round_id,
            selected_metrics=selected_metrics,
            selected_phases=selected_phases
        )

        if round_df is None or round_df.empty:
            self.logger.warning(f"No data ingested for round: {round_id}")
            return {"ingested_data": pd.DataFrame(), "summary": "No data ingested"}

        # Normalizing phase names can be useful if there are round-specific prefixes
        # that need to be removed for consistent analysis across rounds.
        # round_df = normalize_phase_names(round_df)
        
        summary = {
            "records_ingested": len(round_df),
            "metrics": list(round_df['metric_name'].unique()),
            "phases": list(round_df['experimental_phase'].unique()),
            "tenants": list(round_df['tenant_id'].unique())
        }
        
        self.logger.info(f"Data ingestion for round {round_id} complete. Ingested {summary['records_ingested']} records.")
        
        # The primary data is returned separately from the results dictionary
        # as per the PipelineStage.execute contract.
        return {"ingested_data": round_df, "summary": summary}
