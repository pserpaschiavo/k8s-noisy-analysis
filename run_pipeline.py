#!/usr/bin/env python3
"""
Script para executar o pipeline de análise de séries temporais multi-tenant.

Uso:
    ./run_pipeline.py [--config CONFIG_PATH] 
                     [--data-root DATA_ROOT] 
                     [--output-dir OUTPUT_DIR]
                     [--selected-metrics METRICS [METRICS ...]]
                     [--selected-tenants TENANTS [TENANTS ...]]
                     [--selected-rounds ROUNDS [ROUNDS ...]]

Exemplos:
    ./run_pipeline.py --config config/pipeline_config.yaml
    ./run_pipeline.py --selected-metrics cpu_usage memory_usage --selected-tenants tenant-a tenant-b
"""
import sys
from src.pipeline import main

if __name__ == "__main__":
    sys.exit(main())
