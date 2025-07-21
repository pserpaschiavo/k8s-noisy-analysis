#!/usr/bin/env python3
"""
Main entry point for executing the multi-tenant time series analysis pipeline.

This script orchestrates the execution of various analysis stages, from data
ingestion to final report generation. It supports processing multiple experimental
rounds, running analysis for each round, and then consolidating the results.

Usage:
    ./run_pipeline.py --config <path_to_config.yaml>

Example:
    ./run_pipeline.py --config config/pipeline_config_sfi2.yaml
"""
import sys
import logging
import argparse
from typing import Dict, Any
import os
import pandas as pd

# Import all pipeline stage classes
from src.data_ingestion import DataIngestionStage
from src.data_segment import DataSegmentationStage
from src.data_export import DataExportStage
from src.analysis_descriptive import DescriptiveAnalysisStage
from src.analysis_correlation import CorrelationAnalysisStage
from src.analysis_causality import CausalityAnalysisStage
from src.analysis_impact import ImpactAnalysisStage
from src.analysis_phase_comparison import PhaseComparisonStage
from src.analysis_multi_round import MultiRoundAnalysisStage
from src.analysis_fault_tolerance import FaultToleranceAnalysisStage
from src.report_generation import ReportGenerationStage
from src.analysis_export import AnalysisExportStage, export_analysis_results_for_external_tools
from src.config import PipelineConfig
from src.utils import configure_matplotlib, validate_data_availability

def main():
    """
    Main entry point for pipeline execution.
    """
    parser = argparse.ArgumentParser(description="Run the k8s-noisy data analysis pipeline.")
    parser.add_argument('--config', type=str, required=True, help='Path to the main YAML configuration file.')
    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Load main configuration
    config = PipelineConfig(args.config)

    # Configure Matplotlib for non-interactive plotting
    configure_matplotlib()

    # --- Per-Round Analysis ---
    logger = logging.getLogger(__name__)
    selected_rounds = config.get_selected_rounds()

    if not selected_rounds:
        # Try to infer rounds from the data directory structure if not in config
        data_root = config.get_data_root()
        if os.path.isdir(data_root):
            subdirs = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d)) and d.startswith('round-')]
            if subdirs:
                logger.info(f"Rounds not specified in config, inferred from data directory: {subdirs}")
                selected_rounds = sorted(subdirs)

    if not selected_rounds:
        logger.error("No rounds specified in config and could not infer any from the data directory. Aborting.")
        return

    all_rounds_results = {}
    all_pipeline_results = {}

    # Per-round analysis
    for round_id in selected_rounds:
        logging.info(f"Processing round: {round_id}")
        current_round_results = {}
        
        # Data Ingestion
        data_ingestion_stage = DataIngestionStage(config)
        # The first stage receives no prior data or results.
        ingestion_results = data_ingestion_stage.execute(data=None, all_results={}, round_id=round_id)
        ingested_data = ingestion_results.get("ingested_data")
        current_round_results[data_ingestion_stage.stage_name] = ingestion_results

        if ingested_data is None or ingested_data.empty:
            logging.warning(f"Skipping round {round_id} due to no data being ingested.")
            continue

        # Data Segmentation
        data_segmentation_stage = DataSegmentationStage(config)
        segmentation_results = data_segmentation_stage.execute(data=ingested_data, all_results=current_round_results, round_id=round_id)
        segmented_data = segmentation_results.get("segmented_data")
        current_round_results[data_segmentation_stage.stage_name] = segmentation_results

        if segmented_data is None:
            logging.warning(f"Skipping analysis for round {round_id} due to segmentation failure.")
            continue

        # Data Export - Salvar os dados processados em formato parquet
        data_export_stage = DataExportStage(config)
        export_results = data_export_stage.execute(data=segmented_data, all_results=current_round_results, round_id=round_id)
        current_round_results[data_export_stage.stage_name] = export_results
        
        # Também salvar para posterior consolidação
        # Vamos armazenar os dados segmentados para cada round em uma lista
        if 'all_segmented_data' not in all_pipeline_results:
            all_pipeline_results['all_segmented_data'] = []
        
        # Adicionar os dados do round atual (já inclui a coluna round_id)
        all_pipeline_results['all_segmented_data'].append(segmented_data)
        logging.info(f"Stored segmented data for round {round_id} for later consolidation")

        # Standard Analysis Stages
        analysis_stages = [
            DescriptiveAnalysisStage(config),
            ImpactAnalysisStage(config),
            CorrelationAnalysisStage(config),
            CausalityAnalysisStage(config),
            PhaseComparisonStage(config),
            FaultToleranceAnalysisStage(config)
        ]

        for stage in analysis_stages:
            logging.info(f"Executing stage: {stage.stage_name} for round: {round_id}")
            stage_results = stage.execute(data=segmented_data, all_results=current_round_results, round_id=round_id)
            current_round_results[stage.stage_name] = stage_results
            
        # Export Analysis Results to Parquet for external tools
        export_stage = AnalysisExportStage(config)
        export_results = export_stage.execute(data=segmented_data, all_results=current_round_results, round_id=round_id)
        current_round_results[export_stage.stage_name] = export_results
        logger.info(f"Analysis results for round {round_id} exported to Parquet tables.")

        all_rounds_results[round_id] = current_round_results

    # Consolidate results for multi-round and report generation
    all_pipeline_results['per_round'] = all_rounds_results

    # --- Consolidar e salvar dados de todos os rounds ---
    if 'all_segmented_data' in all_pipeline_results and all_pipeline_results['all_segmented_data']:
        logger.info("Consolidating data from all rounds...")
        # Concatenar todos os DataFrames com dados de todos os rounds
        consolidated_df = pd.concat(all_pipeline_results['all_segmented_data'], ignore_index=True)
        
        # Usar o novo utilitário para garantir a geração correta do arquivo Parquet
        from src.data_parquet_utils import fix_parquet_generation
        fix_parquet_generation(config, consolidated_df)
        logger.info(f"Saved consolidated data from all rounds to {config.get_processed_data_path()}")
        
        # Armazenar o DataFrame consolidado nos resultados do pipeline
        all_pipeline_results['consolidated_df'] = consolidated_df
    
    # --- Multi-Round Analysis ---
    if len(selected_rounds) > 1:
        logger.info("Executing multi-round analysis stage...")
        multi_round_analysis_stage = MultiRoundAnalysisStage(config)
        multi_round_results = multi_round_analysis_stage.execute(
            data=None, 
            all_results=all_pipeline_results, 
            round_id='multi-round'
        )
        all_pipeline_results['multi_round_analysis'] = multi_round_results
    else:
        logger.info("Skipping multi-round analysis as only one round was processed.")

    # --- Report Generation ---
    logger.info("Executing report generation stage...")
    report_generation_stage = ReportGenerationStage(config)
    report_generation_stage.execute(
        all_results=all_pipeline_results
    )
    
    # --- Final Export of All Results for External Tools ---
    logger.info("Exporting consolidated analysis results for external tools...")
    exported_tables = export_analysis_results_for_external_tools(config, all_pipeline_results)
    logger.info(f"Exported {len(exported_tables)} tables for external analysis tools.")
    all_pipeline_results['exported_tables'] = exported_tables

    logger.info("Pipeline execution completed successfully.")


if __name__ == '__main__':
    main()
