#!/usr/bin/env python3
"""
Script to run the multi-tenant time series analysis pipeline.

Usage:
    ./run_pipeline.py [--config CONFIG_PATH] 
                     [--data-root DATA_ROOT] 
                     [--output-dir OUTPUT_DIR]
                     [--selected-metrics METRICS [METRICS ...]]
                     [--selected-tenants TENANTS [TENANTS ...]]
                     [--selected-rounds ROUNDS [ROUNDS ...]]

Examples:
    ./run_pipeline.py --config config/pipeline_config.yaml
    ./run_pipeline.py --selected-metrics cpu_usage memory_usage --selected-tenants tenant-a tenant-b
"""
import sys
from src.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
