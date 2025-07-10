"""
Module: config.py
Description: Central configuration for pipeline paths, default selections, and parameters.
"""
import os
import yaml
from typing import Dict, Any, Optional, List

class PipelineConfig:
    """
    Loads and provides access to the pipeline configuration from a YAML file.
    """
    def __init__(self, config_path: str):
        """
        Initializes the configuration object.

        Args:
            config_path (str): The path to the YAML configuration file.
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config_data: Dict[str, Any] = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Gets a value from the configuration.
        """
        return self.config_data.get(key, default)

    def get_base_output_dir(self) -> str:
        """
        Returns the base output directory defined in the configuration.
        """
        return self.get('output_dir', 'outputs')

    def get_experiment_name(self) -> str:
        """
        Returns the name of the experiment.
        """
        return self.get('experiment_name', 'default_experiment')

    def get_output_dir(self, stage_name: Optional[str] = None) -> str:
        """
        Creates and returns the output directory for a specific pipeline stage.
        If stage_name is None, returns the base directory for the experiment.
        """
        base_output_dir = self.get_base_output_dir()
        experiment_name = self.get_experiment_name()
        
        if stage_name:
            stage_output_dir = os.path.join(base_output_dir, experiment_name, stage_name)
        else:
            stage_output_dir = os.path.join(base_output_dir, experiment_name)
            
        os.makedirs(stage_output_dir, exist_ok=True)
        return stage_output_dir

    def get_output_dir_for_round(self, stage_name: str, round_id: str) -> str:
        """
        Creates and returns the output directory for a specific stage within a round.
        The structure will be: <base_output_dir>/<experiment_name>/<round_id>/<stage_name>
        """
        experiment_output_dir = self.get_output_dir() # Returns the base experiment folder
        round_stage_dir = os.path.join(experiment_output_dir, round_id, stage_name)
        os.makedirs(round_stage_dir, exist_ok=True)
        return round_stage_dir

    def get_processed_data_path(self) -> Optional[str]:
        """
        Builds and returns the full path for the processed data file (Parquet)
        from the directory and filename defined in the configuration.
        Returns None if the keys are not defined.
        """
        processed_dir = self.get('processed_data_dir')
        parquet_name = self.get('output_parquet_name')
        
        if processed_dir and parquet_name:
            return os.path.join(processed_dir, parquet_name)
        
        return None

    def get_data_root(self) -> str:
        """
        Returns the root directory for the raw experiment data.
        """
        return self.get('data_root', 'exp_data')

    def get_selected_metrics(self) -> Optional[List[str]]:
        """
        Returns the list of selected metrics.
        """
        return self.get('selected_metrics')

    def get_selected_tenants(self) -> Optional[List[str]]:
        """
        Returns the list of selected tenants.
        """
        return self.get('selected_tenants')

    def get_selected_rounds(self) -> Optional[List[str]]:
        """
        Returns the list of selected rounds.
        """
        return self.get('selected_rounds')

    def get_selected_phases(self) -> Optional[List[str]]:
        """
        Returns the list of selected phases.
        """
        return self.get('selected_phases')

    def get_metric_display_names(self) -> Dict[str, str]:
        """
        Returns the mapping of metric names to friendly display names.
        Returns an empty dictionary if not found.
        """
        visualization_config = self.get('visualization', {})
        return visualization_config.get('metric_display_names', {})

# Root directories
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo-data')
DEFAULT_EXPERIMENT_FOLDER = 'demo-experiment-1-round'
EXPERIMENT_DIR = os.path.join(DATA_ROOT, DEFAULT_EXPERIMENT_FOLDER)
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs') # Added OUTPUT_DIR

# Default output paths
CONSOLIDATED_LONG_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_long.parquet')

# Default parse selections (can be overridden by user config)
DEFAULT_SELECTED_METRICS = None  # or list of metric names
DEFAULT_SELECTED_TENANTS = None  # or list of tenant names
DEFAULT_SELECTED_ROUNDS = None   # or list of round names

# Utility to ensure output directories exist
def ensure_directories():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure OUTPUT_DIR is created

