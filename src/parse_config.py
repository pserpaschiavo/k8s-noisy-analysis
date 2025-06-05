"""
Module: parse_config.py
Description: Utilities for user-customized parsing/selection of metrics, tenants, and rounds for targeted analysis.
"""
from typing import List, Optional
import yaml
import os

def load_parse_config(config_path: str) -> dict:
    """Load user parse configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def get_selected_metrics(config: dict) -> Optional[List[str]]:
    return config.get('selected_metrics')

def get_selected_tenants(config: dict) -> Optional[List[str]]:
    return config.get('selected_tenants')

def get_selected_rounds(config: dict) -> Optional[List[str]]:
    return config.get('selected_rounds')

def get_data_root(config: dict) -> Optional[str]:
    """Get the data root directory from config, or return None if not set."""
    return config.get('data_root')

def get_experiment_folder(config: dict) -> Optional[str]:
    """Get the experiment folder name from config, or return None if not set."""
    return config.get('experiment_folder')

def get_experiment_dir(config: dict) -> Optional[str]:
    """
    Get the full experiment directory path by combining data_root with experiment_folder.
    If experiment_folder is not set, return data_root as it might already point to an experiment.
    """
    data_root = get_data_root(config)
    experiment_folder = get_experiment_folder(config)
    
    if not data_root:
        from src import config
        data_root = config.DATA_ROOT
        
    if experiment_folder:
        return os.path.join(data_root, experiment_folder)
    return data_root

def get_processed_data_dir(config: dict) -> Optional[str]:
    """Get the processed data output directory from config, or return None if not set."""
    return config.get('processed_data_dir')

def get_input_parquet_path(config: dict) -> Optional[str]:
    """Get the input parquet path from config, or return None if not set."""
    return config.get('input_parquet_path')

def get_output_parquet_name(config: dict) -> Optional[str]:
    """Get the output parquet name from config, or return None if not set."""
    return config.get('output_parquet_name', 'consolidated_long.parquet')
