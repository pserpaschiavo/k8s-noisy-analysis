"""
Module: data_parquet_utils.py
Description: Utilities for handling Parquet data files for analysis results
"""
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional, List, Union
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path

# Setup logging
logger = logging.getLogger(__name__)

class ParquetDataManager:
    """
    A class that manages Parquet data files for various analysis results.
    Provides utilities for saving, loading, and managing analysis results in Parquet format.
    """
    
    def __init__(self, base_output_dir: str = None):
        """
        Initialize the ParquetDataManager.
        
        Args:
            base_output_dir: Base directory for output Parquet files. If None, uses './data/parquet_outputs'
        """
        self.base_output_dir = base_output_dir or "./data/parquet_outputs"
        os.makedirs(self.base_output_dir, exist_ok=True)
        logger.info(f"ParquetDataManager initialized with base directory: {self.base_output_dir}")
    
    def ensure_path_exists(self, filepath: str) -> None:
        """
        Ensure that the directory path for a file exists.
        
        Args:
            filepath: Path to the file
        """
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
    
    def save_analysis_result(self, 
                            df: pd.DataFrame, 
                            analysis_type: str, 
                            round_id: Optional[str] = None,
                            sub_type: Optional[str] = None,
                            custom_name: Optional[str] = None) -> str:
        """
        Save analysis results to a Parquet file with organized directory structure.
        
        Args:
            df: DataFrame with analysis results
            analysis_type: Type of analysis (e.g., 'descriptive', 'impact', 'correlation')
            round_id: Optional round identifier for round-specific results
            sub_type: Optional sub-type of analysis (e.g., 'heatmap', 'time_series')
            custom_name: Optional custom name for the file
            
        Returns:
            Path to the saved Parquet file
        """
        if df is None or df.empty:
            logger.warning(f"Cannot save empty DataFrame for {analysis_type} analysis.")
            return ""
            
        # Create directory structure
        analysis_dir = os.path.join(self.base_output_dir, analysis_type)
        
        if round_id:
            analysis_dir = os.path.join(analysis_dir, round_id)
        
        if sub_type:
            analysis_dir = os.path.join(analysis_dir, sub_type)
            
        os.makedirs(analysis_dir, exist_ok=True)
        
        # Create filename
        if custom_name:
            filename = f"{custom_name}.parquet"
        else:
            if round_id and sub_type:
                filename = f"{analysis_type}_{round_id}_{sub_type}.parquet"
            elif round_id:
                filename = f"{analysis_type}_{round_id}.parquet"
            elif sub_type:
                filename = f"{analysis_type}_{sub_type}.parquet"
            else:
                filename = f"{analysis_type}.parquet"
        
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            # Handle types not supported by Parquet
            df = self._handle_unsupported_types(df)
            
            # Save DataFrame to Parquet
            self.ensure_path_exists(filepath)
            df.to_parquet(filepath, index=False)
            logger.info(f"Successfully saved {analysis_type} analysis results to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving {analysis_type} analysis to {filepath}: {e}")
            return ""
    
    def _handle_unsupported_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle data types not supported by Parquet.
        
        Args:
            df: DataFrame to check for unsupported types
            
        Returns:
            DataFrame with types fixed for Parquet compatibility
        """
        # Create a copy to avoid modifying the original
        df = df.copy()
        
        for col in df.columns:
            # Convert lists and other complex objects to strings
            if df[col].dtype == 'object':
                # Check if the column contains lists or other non-string objects
                if any(isinstance(x, (list, dict, tuple)) for x in df[col].dropna()):
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else None)
            
            # Convert any unsupported dtypes to string
            if not pa.types.is_primitive(pa.infer_type(df[col].iloc[0]) if len(df) > 0 else pa.string()):
                df[col] = df[col].astype(str)
                
        return df
    
    def load_analysis_result(self, 
                           analysis_type: str, 
                           round_id: Optional[str] = None,
                           sub_type: Optional[str] = None,
                           custom_name: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Load analysis results from a Parquet file.
        
        Args:
            analysis_type: Type of analysis (e.g., 'descriptive', 'impact', 'correlation')
            round_id: Optional round identifier for round-specific results
            sub_type: Optional sub-type of analysis (e.g., 'heatmap', 'time_series')
            custom_name: Optional custom name for the file
            
        Returns:
            DataFrame with analysis results or None if file not found
        """
        # Create directory structure
        analysis_dir = os.path.join(self.base_output_dir, analysis_type)
        
        if round_id:
            analysis_dir = os.path.join(analysis_dir, round_id)
        
        if sub_type:
            analysis_dir = os.path.join(analysis_dir, sub_type)
            
        # Create filename
        if custom_name:
            filename = f"{custom_name}.parquet"
        else:
            if round_id and sub_type:
                filename = f"{analysis_type}_{round_id}_{sub_type}.parquet"
            elif round_id:
                filename = f"{analysis_type}_{round_id}.parquet"
            elif sub_type:
                filename = f"{analysis_type}_{sub_type}.parquet"
            else:
                filename = f"{analysis_type}.parquet"
        
        filepath = os.path.join(analysis_dir, filename)
        
        try:
            if not os.path.exists(filepath):
                logger.warning(f"Analysis result file not found at {filepath}")
                return None
                
            df = pd.read_parquet(filepath)
            logger.info(f"Successfully loaded {analysis_type} analysis results from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading {analysis_type} analysis from {filepath}: {e}")
            return None
    
    def save_consolidated_results(self, dfs: Dict[str, pd.DataFrame], output_path: Optional[str] = None) -> str:
        """
        Save multiple DataFrames as a partitioned Parquet dataset.
        Each DataFrame will be saved as a separate partition.
        
        Args:
            dfs: Dictionary of DataFrames where keys are partition names
            output_path: Optional custom output path
            
        Returns:
            Path to the saved Parquet dataset
        """
        if not dfs:
            logger.warning("No DataFrames provided for consolidated saving.")
            return ""
            
        # Create output path
        output_path = output_path or os.path.join(self.base_output_dir, "consolidated")
        os.makedirs(output_path, exist_ok=True)
        
        try:
            # Save each DataFrame as a separate partition
            for name, df in dfs.items():
                if df is None or df.empty:
                    logger.warning(f"Skipping empty DataFrame for {name}")
                    continue
                    
                partition_path = os.path.join(output_path, f"{name}.parquet")
                
                # Handle types not supported by Parquet
                df = self._handle_unsupported_types(df)
                
                # Save DataFrame to Parquet
                self.ensure_path_exists(partition_path)
                df.to_parquet(partition_path, index=False)
                logger.info(f"Saved partition {name} to {partition_path}")
                
            logger.info(f"Successfully saved consolidated results to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving consolidated results to {output_path}: {e}")
            return ""
    
    def list_available_analyses(self) -> Dict[str, List[str]]:
        """
        List all available analysis results in the base directory.
        
        Returns:
            Dictionary mapping analysis types to lists of available files
        """
        results = {}
        
        try:
            # Check if base directory exists
            if not os.path.exists(self.base_output_dir):
                logger.warning(f"Base directory {self.base_output_dir} does not exist.")
                return {}
                
            # Iterate through subdirectories (analysis types)
            for analysis_type in os.listdir(self.base_output_dir):
                analysis_path = os.path.join(self.base_output_dir, analysis_type)
                
                if os.path.isdir(analysis_path):
                    # Find all Parquet files for this analysis type
                    parquet_files = []
                    
                    for root, _, files in os.walk(analysis_path):
                        for file in files:
                            if file.endswith(".parquet"):
                                # Get relative path from analysis_path
                                rel_path = os.path.relpath(
                                    os.path.join(root, file), 
                                    analysis_path
                                )
                                parquet_files.append(rel_path)
                    
                    results[analysis_type] = parquet_files
            
            return results
        except Exception as e:
            logger.error(f"Error listing available analyses: {e}")
            return {}


def fix_parquet_generation(config, df: pd.DataFrame = None) -> Optional[pd.DataFrame]:
    """
    Fix the issue with Parquet file generation mentioned in ROADMAP.
    This function ensures that the Parquet file exists and is properly generated.
    
    Args:
        config: PipelineConfig object with configuration settings
        df: DataFrame to save (if None, attempts to load and return from disk)
    
    Returns:
        DataFrame that was saved or loaded from the Parquet file
    """
    processed_data_path = config.get_processed_data_path()
    
    if processed_data_path is None:
        logger.error("Processed data path is not defined in configuration")
        return None
    
    # Ensure directory exists
    data_dir = os.path.dirname(processed_data_path)
    os.makedirs(data_dir, exist_ok=True)
    
    if df is not None:
        # Save DataFrame to Parquet
        try:
            df.to_parquet(processed_data_path, index=False)
            logger.info(f"Successfully saved DataFrame to {processed_data_path}")
            return df
        except Exception as e:
            logger.error(f"Error saving DataFrame to {processed_data_path}: {e}")
            return None
    else:
        # Try to load DataFrame from Parquet
        try:
            if os.path.exists(processed_data_path):
                df = pd.read_parquet(processed_data_path)
                logger.info(f"Successfully loaded DataFrame from {processed_data_path}")
                return df
            else:
                logger.warning(f"Parquet file {processed_data_path} does not exist and no DataFrame was provided to save")
                return None
        except Exception as e:
            logger.error(f"Error loading DataFrame from {processed_data_path}: {e}")
            return None


def export_analysis_to_parquet(df: pd.DataFrame, 
                             analysis_name: str, 
                             output_dir: str,
                             round_id: Optional[str] = None,
                             include_timestamp: bool = True) -> Optional[str]:
    """
    Export analysis results to a Parquet file.
    Utility function for direct use in analysis modules.
    
    Args:
        df: DataFrame with analysis results
        analysis_name: Name of the analysis
        output_dir: Output directory
        round_id: Optional round identifier
        include_timestamp: Whether to include a timestamp in the filename
        
    Returns:
        Path to the saved Parquet file
    """
    import datetime
    
    if df is None or df.empty:
        logger.warning(f"Cannot export empty DataFrame for {analysis_name} analysis")
        return None
        
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = ""
    if include_timestamp:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
    if round_id:
        filename = f"{analysis_name}_{round_id}_{timestamp}.parquet" if timestamp else f"{analysis_name}_{round_id}.parquet"
    else:
        filename = f"{analysis_name}_{timestamp}.parquet" if timestamp else f"{analysis_name}.parquet"
        
    output_path = os.path.join(output_dir, filename)
    
    try:
        # Handle complex types before saving
        for col in df.columns:
            if df[col].dtype == 'object':
                # Convert complex objects to strings
                if any(isinstance(x, (list, dict, tuple)) for x in df[col].dropna()):
                    df[col] = df[col].apply(lambda x: str(x) if x is not None else None)
                    
        df.to_parquet(output_path, index=False)
        logger.info(f"Successfully exported {analysis_name} analysis to {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Error exporting {analysis_name} analysis to {output_path}: {e}")
        return None
