"""
Module: config.py
Description: Central configuration for pipeline paths, default selections, and parameters.
"""
import os

# Root directories
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo-data')
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')

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

