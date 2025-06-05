"""
Basic data ingestion plugin to demonstrate the plugin-based architecture.
"""
import os
from typing import Dict, Any, List
import logging
import pandas as pd
import glob
from pathlib import Path

from src.pipeline_core import PipelinePlugin
from src.parse_config import get_data_root, get_processed_data_dir, get_experiment_folder
from src.config import DEFAULT_EXPERIMENT_FOLDER

class DataIngestionPlugin(PipelinePlugin):
    """
    Plugin responsible for loading data from CSV files or parquet files.
    Converts the first implementation of DataIngestionStage into the plugin architecture.
    """
    
    @classmethod
    def get_id(cls) -> str:
        """Return the unique identifier for this plugin."""
        return "data_ingestion"
    
    @classmethod
    def get_dependencies(cls) -> List[str]:
        """This plugin has no dependencies."""
        return []
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the data ingestion logic.
        
        Args:
            context: The current pipeline context
            
        Returns:
            Dict[str, Any]: The updated context with loaded data
        """
        self.logger.info("Starting data ingestion")
        
        # Get configuration parameters
        data_root = get_data_root(self.config)
        processed_data_dir = get_processed_data_dir(self.config)
        experiment_folder = get_experiment_folder(self.config) or DEFAULT_EXPERIMENT_FOLDER
        
        # Construct experiment directory path
        experiment_dir = os.path.join(data_root, experiment_folder)
        self.logger.info(f"Using experiment directory: {experiment_dir}")
        
        # Check if we should load from parquet
        input_parquet_path = self.config.get('input_parquet_path')
        if input_parquet_path and os.path.exists(input_parquet_path):
            self.logger.info(f"Loading data from specified parquet file: {input_parquet_path}")
            df = self._load_from_parquet(input_parquet_path)
            context['dataframe'] = df
            return context
        
        # Check if a consolidated parquet already exists
        output_parquet_name = self.config.get('output_parquet_name', 'consolidated_long.parquet')
        default_parquet_path = os.path.join(processed_data_dir, output_parquet_name)
        if os.path.exists(default_parquet_path):
            self.logger.info(f"Loading data from existing consolidated parquet: {default_parquet_path}")
            df = self._load_from_parquet(default_parquet_path)
            context['dataframe'] = df
            return context
        
        # Otherwise, process raw data
        self.logger.info("No parquet file found, processing raw data")
        df = self._process_raw_data(experiment_dir)
        
        # Save the processed DataFrame to parquet
        os.makedirs(processed_data_dir, exist_ok=True)
        parquet_path = os.path.join(processed_data_dir, output_parquet_name)
        self.logger.info(f"Saving processed data to parquet: {parquet_path}")
        df.to_parquet(parquet_path, index=False)
        
        # Update context and return
        context['dataframe'] = df
        context['experiment_folder_applied'] = True
        return context
    
    def _load_from_parquet(self, parquet_path: str) -> pd.DataFrame:
        """
        Load data from a parquet file.
        
        Args:
            parquet_path: Path to the parquet file
            
        Returns:
            pd.DataFrame: The loaded DataFrame
        """
        self.logger.info(f"Loading data from parquet: {parquet_path}")
        df = pd.read_parquet(parquet_path)
        self.logger.info(f"Loaded DataFrame with shape: {df.shape}")
        return df
    
    def _process_raw_data(self, experiment_dir: str) -> pd.DataFrame:
        """
        Process raw data from CSV files.
        
        Args:
            experiment_dir: Path to the experiment directory
            
        Returns:
            pd.DataFrame: The processed DataFrame in long format
        """
        self.logger.info(f"Processing raw data from: {experiment_dir}")
        
        # Simplified implementation for demonstration
        # In a real implementation, we would:
        # 1. Traverse the directory structure
        # 2. Load CSV files for each tenant/metric
        # 3. Extract metadata (round, phase, tenant, etc.)
        # 4. Consolidate into a long-format DataFrame
        
        # Placeholder implementation
        rounds = sorted([d for d in os.listdir(experiment_dir) if d.startswith('round-')])
        
        if not rounds:
            self.logger.error(f"No round directories found in {experiment_dir}")
            raise ValueError(f"No round directories found in {experiment_dir}")
        
        dfs = []
        
        for round_id in rounds:
            round_dir = os.path.join(experiment_dir, round_id)
            phases = sorted([d for d in os.listdir(round_dir) if os.path.isdir(os.path.join(round_dir, d))])
            
            for phase in phases:
                phase_dir = os.path.join(round_dir, phase)
                tenants = [d for d in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, d))]
                
                for tenant in tenants:
                    tenant_dir = os.path.join(phase_dir, tenant)
                    csv_files = glob.glob(os.path.join(tenant_dir, "*.csv"))
                    
                    for csv_path in csv_files:
                        metric_name = Path(csv_path).stem
                        
                        try:
                            # Load CSV data
                            metric_df = pd.read_csv(csv_path)
                            
                            # Add metadata
                            metric_df['metric_name'] = metric_name
                            metric_df['tenant_id'] = tenant
                            metric_df['experimental_phase'] = phase
                            metric_df['round_id'] = round_id
                            
                            # Rename columns
                            metric_df = metric_df.rename(columns={'value': 'metric_value'})
                            
                            # Append to list of DataFrames
                            dfs.append(metric_df)
                        except Exception as e:
                            self.logger.error(f"Error processing {csv_path}: {e}")
        
        # Concatenate all DataFrames
        if not dfs:
            self.logger.error("No data was loaded")
            raise ValueError("No data was loaded")
        
        df = pd.concat(dfs, ignore_index=True)
        self.logger.info(f"Processed DataFrame with shape: {df.shape}")
        
        return df
