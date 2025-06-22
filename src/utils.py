#!/usr/bin/env python3
"""
Module: utils.py
Description: General utilities for the analysis pipeline.
"""

import warnings
import contextlib
import logging

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_statsmodels_warnings():
    """
    Context manager to suppress repetitive statsmodels warnings.
    Filters specific warnings while keeping other important ones.
    """
    with warnings.catch_warnings():
        # Filter common statsmodels warnings
        warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', 'Value in x_0 detected')
        warnings.filterwarnings('ignore', message='.*flat prior.*')
        warnings.filterwarnings('ignore', message='.*distribution is not normalized.*')
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')
        
        # Suppress convergence warnings
        warnings.filterwarnings('ignore', message='.*Maximum Likelihood optimization failed.*')
        warnings.filterwarnings('ignore', message='.*The iteration limit.*')
        
        # Other common statistical warnings
        warnings.filterwarnings('ignore', message='.*p-value.*')
        
        yield


def validate_data_availability(config: dict) -> bool:
    """
    Performs a pre-flight check to validate that raw data files exist.

    This function checks for the existence of .csv files based on the pipeline
    configuration. It assumes a directory structure of:
    <data_root>/<round>/<phase>/<tenant>/<metric>.csv

    Args:
        config: The pipeline configuration dictionary.

    Returns:
        True if all expected files are found, False otherwise. The function
        logs warnings for missing files but does not raise an exception.
    """
    import os
    logger.info("Performing pre-flight check for raw data files...")
    
    data_root = config.get("data_root")
    if not data_root:
        logger.error("Pre-flight check failed: 'data_root' not specified in configuration.")
        return False

    selected_metrics = config.get("selected_metrics", [])
    selected_tenants = config.get("selected_tenants", [])
    selected_rounds = config.get("selected_rounds", [])

    # Phase names are based on docs/adaptation_plan_new_phases_en.md
    # The exact directory names are an assumption.
    phase_names = [
        "baseline", "cpu-noise", "memory-noise", "network-noise", 
        "disk-noise", "combined-noise", "recovery"
    ]

    if not all([selected_metrics, selected_tenants, selected_rounds]):
        logger.warning("Pre-flight check skipped: Missing one or more required fields in config: selected_metrics, selected_tenants, selected_rounds.")
        return True # Skip validation if config is incomplete

    missing_files = []
    for round_name in selected_rounds:
        for phase_name in phase_names:
            for tenant_name in selected_tenants:
                for metric_name in selected_metrics:
                    # Path assumption: <data_root>/<round>/<phase>/<tenant>/<metric>.csv
                    file_path = os.path.join(data_root, round_name, phase_name, tenant_name, f"{metric_name}.csv")
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)

    if missing_files:
        logger.warning("Pre-flight check found missing data files:")
        for f in missing_files:
            logger.warning(f"  - {f}")
        logger.warning("Pipeline will continue, but ingestion may be incomplete.")
        return False
    else:
        logger.info("Pre-flight check successful. All expected raw data files were found.")
        return True
