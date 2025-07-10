"""
Module: data_export.py
Description: Handles the export of DataFrames to various file formats.
"""
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional

from .pipeline_stage import PipelineStage
from .config import PipelineConfig

# Setup logging
logger = logging.getLogger(__name__)

def save_dataframe(df: pd.DataFrame, path: str, file_format: str = 'parquet'):
    """
    Saves a DataFrame to a specified path and format.
    """
    try:
        if file_format == 'parquet':
            df.to_parquet(path, index=False)
        elif file_format == 'csv':
            df.to_csv(path, index=False)
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return
        logger.info(f"DataFrame successfully saved to {path}")
    except Exception as e:
        logger.error(f"Error saving DataFrame to {path}: {e}")

def load_dataframe(path: str, file_format: str = 'parquet') -> Optional[pd.DataFrame]:
    """
    Loads a DataFrame from a specified path and format.
    """
    try:
        if file_format == 'parquet':
            df = pd.read_parquet(path)
        elif file_format == 'csv':
            df = pd.read_csv(path)
        else:
            logger.error(f"Unsupported file format: {file_format}")
            return None
        logger.info(f"DataFrame successfully loaded from {path}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found at {path}")
        return None
    except Exception as e:
        logger.error(f"Error loading DataFrame from {path}: {e}")
        return None

class DataExportStage(PipelineStage):
    """
    Pipeline stage for exporting the processed DataFrame.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("data_export", "Export processed data")
        self.config = config

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Exports the processed DataFrame to a Parquet file.
        This stage runs after data ingestion and processing for a specific round.
        """
        self.logger.info(f"Starting data export for round '{round_id}'...")
        if data is None or data.empty:
            self.logger.error("Input DataFrame 'data' is not available for export.")
            return {}

        # The data passed to this stage is already for a specific round
        round_df = data
        
        # Define the output path for the round-specific Parquet file
        output_dir = self.config.get_output_dir_for_round("data_export", round_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # Use a descriptive filename for the round's data
        output_path = os.path.join(output_dir, f"processed_data_{round_id}.parquet")
        
        # Save the DataFrame
        save_dataframe(round_df, output_path, file_format='parquet')
        
        self.logger.info(f"Data for round '{round_id}' exported successfully to {output_path}.")
        
        return {"processed_data_path": output_path}
