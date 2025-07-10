#!/usr/bin/env python3
"""
Module: pipeline.py
Description: Defines the main pipeline for data analysis, orchestrating various stages.
"""
import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Union, Tuple
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import yaml
import networkx as nx

# Project module imports
from src.data_ingestion import ingest_experiment_data
from src.data_export import save_dataframe, load_dataframe
from src.data_segment import filter_long_df, get_wide_format_for_analysis
from src.analysis_descriptive import compute_descriptive_stats
from src.visualization.descriptive_plots import (
    plot_metric_timeseries_multi_tenant,
    plot_metric_barplot_by_phase,
    plot_metric_boxplot,
    plot_metric_timeseries_multi_tenant_all_phases
)
from src.analysis_correlation import compute_correlation_matrix, compute_covariance_matrix, compute_cross_correlation
from src.report_generation import generate_tenant_metrics, generate_tenant_ranking_plot, generate_markdown_report
from src import config
from src.parse_config import load_parse_config, get_selected_metrics, get_selected_tenants, get_selected_rounds
from src.parse_config import get_data_root, get_processed_data_dir, get_experiment_folder, get_experiment_dir
from src.utils import validate_data_availability as preflight_check
from src.visualization.correlation_plots import (
    plot_correlation_heatmap,
    plot_covariance_heatmap,
    plot_ccf
)
from src.pipeline_stage import PipelineStage
from src.analysis_phase_comparison import PhaseComparisonStage as AnalysisPhaseComparisonStage
from src.analysis_impact import ImpactAnalysisStage

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline")


def validate_data_availability(df: pd.DataFrame, config_dict: Dict[str, Any], min_data_points: int = 10) -> Tuple[bool, Dict[str, List[str]]]:
    """
    Validates if there is enough data for the selected metrics, tenants, and rounds.

    Args:
        df: The long format DataFrame.
        config_dict: The configuration dictionary.
        min_data_points: The minimum number of data points required.

    Returns:
        A tuple containing:
        - A boolean indicating if the data is valid.
        - A dictionary with details about missing or insufficient data.
    """
    logger.info("Starting data availability validation...")
    is_valid = True
    report = {
        "missing_metrics": [],
        "insufficient_data_metrics": [],
        "missing_tenants": [],
        "available_metrics": list(df['metric_name'].unique()),
        "available_tenants": list(df['tenant_id'].unique()),
    }

    selected_metrics = get_selected_metrics(config_dict)
    selected_tenants = get_selected_tenants(config_dict)
    
    # Validate metrics
    if selected_metrics:
        for metric in selected_metrics:
            if metric not in report['available_metrics']:
                report['missing_metrics'].append(metric)
                is_valid = False
            else:
                # Check for sufficient data points
                metric_df = df[df['metric_name'] == metric]
                if len(metric_df) < min_data_points:
                    report['insufficient_data_metrics'].append(f"{metric} (found {len(metric_df)} points)")
                    is_valid = False

    # Validate tenants
    if selected_tenants: # If specific tenants are selected
        for tenant in selected_tenants:
            if tenant not in report['available_tenants']:
                report['missing_tenants'].append(tenant)
                is_valid = False

    if not is_valid:
        logger.warning("Data validation failed.")
        if report['missing_metrics']:
            logger.warning(f"Missing metrics: {report['missing_metrics']}")
            logger.warning(f"Available metrics in data: {report['available_metrics']}")
        if report['insufficient_data_metrics']:
            logger.warning(f"Insufficient data for metrics: {report['insufficient_data_metrics']}")
        if report['missing_tenants']:
            logger.warning(f"Missing tenants: {report['missing_tenants']}")
            logger.warning(f"Available tenants in data: {report['available_tenants']}")
    else:
        logger.info("Data availability validation passed successfully.")
        
    return is_valid, report


class DataIngestionStage(PipelineStage):
    """Stage for ingesting raw data and consolidating into a long DataFrame."""
    
    def __init__(self):
        super().__init__("data_ingestion", "Data ingestion and consolidation")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes data ingestion according to configuration.
        
        Args:
            context: Pipeline context with configurations.
            
        Returns:
            Updated context with long DataFrame.
        """
        # Get configurations
        config_dict = context.get('config', {})
        data_root = context.get('data_root', config.DATA_ROOT)
        # Check if experiment_folder was applied by the wrapper or patch
        applied_by_wrapper = context.get('experiment_folder_applied', False)
        
        # If experiment_folder is defined and not applied by the wrapper
        experiment_folder = config_dict.get('experiment_folder')
        if experiment_folder and not applied_by_wrapper:
            # Check if data_root already has experiment_folder (to avoid duplication)
            if not data_root.endswith(experiment_folder):
                data_root = os.path.join(data_root, experiment_folder)
                self.logger.info(f"Using experiment path: {data_root} (data_root + experiment_folder)")
            else:
                self.logger.info(f"Experiment folder '{experiment_folder}' has already been applied to data_root: {data_root}")
        
        # Propagate experiment_folder_applied to the context for other stages
        context['experiment_folder_applied'] = True
        
        selected_metrics = config_dict.get('selected_metrics')
        selected_tenants = config_dict.get('selected_tenants')
        selected_rounds = config_dict.get('selected_rounds')
        force_reprocess = context.get('force_reprocess', False)
        
        processed_data_dir = context.get('processed_data_dir')
        if processed_data_dir is None:
            processed_data_dir = config_dict.get('processed_data_dir', config.PROCESSED_DATA_DIR)
        
        # Check for a specified input parquet path in the config
        input_parquet_path = config_dict.get('input_parquet_path')
        
        # Determine the output parquet file name
        output_parquet_name = config_dict.get('output_parquet_name', 'consolidated_long.parquet')
        
        # Ensure the directory exists before use
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # If output_parquet_name is already an absolute path, use it as consolidated_long_path
        if os.path.isabs(output_parquet_name):
            consolidated_long_path = output_parquet_name
        else:
            consolidated_long_path = os.path.join(processed_data_dir, output_parquet_name)
        
        # Case 1: Specified input parquet file - load directly
        if input_parquet_path and os.path.exists(input_parquet_path) and not force_reprocess:
            self.logger.info(f"Using specific input parquet: {input_parquet_path}")
            
            try:
                from src.data_ingestion import load_from_parquet
                df_long = load_from_parquet(input_parquet_path)
                self.logger.info(f"Data loaded successfully. Total records: {len(df_long)}")
                
                # Add to context
                context['data'] = df_long
                context['consolidated_long_path'] = input_parquet_path
                
                return context
            except Exception as e:
                self.logger.error(f"Error loading input parquet file: {e}")
                self.logger.info("Continuing with check for consolidated data or reprocessing...")
        
        # Case 2: Check if a consolidated parquet file already exists and we are not forcing reprocessing
        if os.path.exists(consolidated_long_path) and not force_reprocess:
            self.logger.info(f"Consolidated data file found: {consolidated_long_path}")
            self.logger.info("Loading already processed data... (use --force-reprocess to reprocess)")
            
            try:
                df_long = load_dataframe(consolidated_long_path)
                self.logger.info(f"Data loaded successfully. Total records: {len(df_long)}")
                
                # Add to context
                context['data'] = df_long
                context['consolidated_long_path'] = consolidated_long_path
                
                return context
            except Exception as e:
                self.logger.error(f"Error loading existing file: {e}")
                self.logger.info("Continuing with data reprocessing...")
        
        # Case 3: Process raw data
        self.logger.info(f"Starting data ingestion from: {data_root}")
        
        # Ingest data
        df_long = ingest_experiment_data(
            data_root=data_root,
            selected_metrics=selected_metrics,
            selected_tenants=selected_tenants,
            selected_rounds=selected_rounds
        )
        
        self.logger.info(f"Ingestion completed. Total records: {len(df_long)}")
        
        # Add to context
        context['data'] = df_long
        context['consolidated_long_path'] = consolidated_long_path
        
        return context


class DataValidationStage(PipelineStage):
    """Stage for validating data availability before analysis."""

    def __init__(self):
        super().__init__("data_validation", "Data availability validation")

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the data validation.
        """
        df_long = context.get('data')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame not available for validation. Halting pipeline.")
            context['pipeline_halted'] = True
            context['data_validation_report'] = "DataFrame is empty or not available."
            return context

        config_dict = context.get('config', {})
        
        is_valid, validation_report = validate_data_availability(df_long, config_dict)
        
        context['data_validation_report'] = validation_report
        
        if not is_valid:
            self.logger.error("Data validation failed. Halting pipeline.")
            context['pipeline_halted'] = True
        
        return context


class DataExportStage(PipelineStage):
    """Stage for exporting the consolidated DataFrame."""
    
    def __init__(self):
        super().__init__("data_export", "Export of the consolidated DataFrame")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exports the long DataFrame to a file.
        
        Args:
            context: Context with long DataFrame.
            
        Returns:
            Updated context.
        """
        df_long = context.get('data')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame not available for export.")
            return context
            
        # Get configurations
        config_dict = context.get('config', {})
        processed_data_dir = context.get('processed_data_dir', config.PROCESSED_DATA_DIR)
        
        # Use output file name defined in the configuration
        output_parquet_name = config_dict.get('output_parquet_name', 'consolidated_long.parquet')
        consolidated_long_path = os.path.join(processed_data_dir, output_parquet_name)
        
        # Check if we are using a specific input file
        input_parquet_path = config_dict.get('input_parquet_path')
        if input_parquet_path and os.path.exists(input_parquet_path) and context.get('consolidated_long_path') == input_parquet_path:
            self.logger.info(f"Using existing input parquet file: {input_parquet_path}")
            # We don't need to save again if we are using the input file diretamente
            return context
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(consolidated_long_path), exist_ok=True)
        
        # Export
        save_dataframe(df_long, consolidated_long_path, format='parquet')
        self.logger.info(f"DataFrame exported to: {consolidated_long_path}")
        
        # Update context
        context['consolidated_long_path'] = consolidated_long_path
        
        return context


class DescriptiveAnalysisStage(PipelineStage):
    """Stage for descriptive analysis."""
    
    def __init__(self):
        super().__init__("descriptive_analysis", "Descriptive analysis of metrics")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes descriptive analysis and generates visualizations.
        
        Args:
            context: Context with long DataFrame.
            
        Returns:
            Updated context with results and plot paths.
        """
        df_long = context.get('data')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame not available for descriptive analysis.")
            return context
            
        # Configure output directory
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'descriptive')
        os.makedirs(out_dir, exist_ok=True)
        
        # Calculate descriptive statistics
        self.logger.info("Calculating descriptive statistics...")
        stats = compute_descriptive_stats(df_long)
        context['descriptive_stats'] = stats
        
        # Generate plots for each metric/round/phase combination
        self.logger.info("Generating visualizations...")
        
        # Get unique combinations of experiment/round/phase/metric
        experiments = df_long['experiment_id'].unique()
        
        plot_paths = []
        
        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                for metric in round_df['metric_name'].unique():
                    # Generate aggregated plots by phase and the consolidated plot of all phases
                    try:
                        path = plot_metric_barplot_by_phase(round_df, metric, round_id, out_dir)
                        plot_paths.append(path)
                        
                        path = plot_metric_boxplot(round_df, metric, round_id, out_dir)
                        plot_paths.append(path)

                        path = plot_metric_timeseries_multi_tenant_all_phases(round_df, metric, round_id, out_dir)
                        plot_paths.append(path)

                    except Exception as e:
                        self.logger.error(f"Error generating aggregated plots for {metric}, {round_id}: {e}")
                    
                    # Generate plots per individual phase
                    for phase in round_df['experimental_phase'].unique():
                        try:
                            path = plot_metric_timeseries_multi_tenant(
                                round_df, metric, phase, round_id, out_dir
                            )
                            plot_paths.append(path)
                        except Exception as e:
                            self.logger.error(f"Error generating plot for {metric}, {phase}, {round_id}: {e}")
        
        # Update context
        context['descriptive_plot_paths'] = plot_paths
        
        return context


class CorrelationAnalysisStage(PipelineStage):
    """Stage for correlation and covariance analysis."""
    
    def __init__(self):
        super().__init__("correlation_analysis", "Correlation analysis between tenants")
        # Adding a specific logger for this stage to facilitate debugging
        self.stage_logger = logging.getLogger(f"pipeline.CorrelationAnalysisStage")

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes correlation analysis and generates visualizations.

        Args:
            context: Context with long DataFrame.

        Returns:
            Updated context with results and plot paths.
        """
        df_long = context.get('data')
        if df_long is None or df_long.empty:
            self.stage_logger.error("DataFrame not available for correlation analysis.")
            return context

        # Configure output directory
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'correlation')
        os.makedirs(out_dir, exist_ok=True)

        # Structures to store results
        correlation_matrices = {}
        covariance_matrices = {}
        plot_paths = []

        # Get unique combinations of experiment/round/phase/metric
        experiments = df_long['experiment_id'].unique()
        self.stage_logger.debug(f"Experiments found: {experiments}")

        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            self.stage_logger.debug(f"Processing experiment_id: {experiment_id}")
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                self.stage_logger.debug(f"Processing round_id: {round_id} for experiment {experiment_id}")
                for metric in round_df['metric_name'].unique():
                    self.stage_logger.debug(f"Processing metric: {metric} for round {round_id}")
                    metric_key = f"{experiment_id}:{round_id}:{metric}"
                    correlation_matrices[metric_key] = {}
                    covariance_matrices[metric_key] = {}

                    for phase in round_df['experimental_phase'].unique():
                        self.stage_logger.debug(f"Processing phase: {phase} for metric {metric}")
                        phase_key = f"{metric_key}:{phase}"

                        # Calculate correlation
                        try:
                            # Log before calling the function that might cause the error
                            self.stage_logger.info(f"Calculating correlation for metric: {metric}, phase: {phase}, round_id: {round_id}")
                            
                            # Filter DataFrame for the specific phase
                            phase_specific_df = round_df[round_df['experimental_phase'] == phase]

                            if phase_specific_df.empty:
                                self.stage_logger.warning(f"Empty DataFrame for {metric}, {phase}, {round_id} after filtering by phase. Skipping correlation.")
                                continue
                            
                            self.stage_logger.info(f"Tenants for correlation analysis: {phase_specific_df['tenant_id'].unique()}")
                            self.stage_logger.info(f"Data points for correlation analysis: {len(phase_specific_df)}")

                            # Calculate correlation and generate visualization
                            corr = compute_correlation_matrix(phase_specific_df, metric, phase, round_id)
                            correlation_matrices[metric_key][phase] = corr

                            # Check result and generate correlation plot
                            if corr is not None and not corr.empty:
                                try:
                                    self.stage_logger.info(f"Generating correlation plot for {metric}, {phase}, {round_id}")
                                    path = plot_correlation_heatmap(corr, metric, phase, round_id, out_dir)
                                    if path:
                                        plot_paths.append(path)
                                        self.stage_logger.info(f"Correlation plot generated successfully: {path}")
                                    else:
                                        self.stage_logger.warning(f"Failed to generate correlation plot for {metric}, {phase}, {round_id}")
                                except Exception as plot_err:
                                    self.stage_logger.error(f"Error generating correlation plot for {metric}, {phase}, {round_id}: {plot_err}", exc_info=True)
                            elif corr is None:
                                self.stage_logger.warning(f"Correlation matrix returned None for {metric}, {phase}, {round_id}")
                            else: # corr.empty is True
                                self.stage_logger.warning(f"Empty correlation matrix for {metric}, {phase}, {round_id}")

                        except Exception as e:
                            self.stage_logger.error(f"Unexpected error calculating correlation for {metric}, {phase}, {round_id}: {e}", exc_info=True)

                        # Calculate covariance
                        try:
                            self.stage_logger.info(f"Calculating covariance for metric: {metric}, phase: {phase}, round_id: {round_id}.")
                            phase_specific_df_cov = round_df[round_df['experimental_phase'] == phase]
                            if phase_specific_df_cov.empty:
                                self.stage_logger.warning(f"Empty DataFrame for {metric}, {phase}, {round_id} after filtering by phase. Skipping covariance.")
                                continue

                            cov = compute_covariance_matrix(phase_specific_df_cov, metric, phase, round_id)
                            covariance_matrices[metric_key][phase] = cov

                            if cov is not None and not cov.empty:
                                path = plot_covariance_heatmap(cov, metric, phase, round_id, out_dir)
                                plot_paths.append(path)
                            elif cov is None:
                                self.stage_logger.warning(f"Covariance matrix returned None for {metric}, {phase}, {round_id}")
                            else:
                                self.stage_logger.warning(f"Empty covariance matrix for {metric}, {phase}, {round_id}")
                        except Exception as e_cov:
                            self.stage_logger.error(f"Unexpected error calculating covariance for {metric}, {phase}, {round_id}: {e_cov}", exc_info=True)
                        
                        # Calculate cross-correlation (CCF)
                        try:
                            self.stage_logger.info(f"Calculating cross-correlation (CCF) for metric: {metric}, phase: {phase}, round_id: {round_id}")
                            
                            # Filter DataFrame for the specific phase
                            phase_specific_df_ccf = round_df[round_df['experimental_phase'] == phase]
                            
                            # Check if we have enough data
                            if not phase_specific_df_ccf.empty:
                                # Specific directory for CCF plots
                                ccf_dir = os.path.join(out_dir, "cross_correlation")
                                os.makedirs(ccf_dir, exist_ok=True)
                                
                                # Calculate CCF
                                ccf_results = compute_cross_correlation(phase_specific_df_ccf, metric, phase, round_id, max_lag=20)
                                
                                if ccf_results:
                                    # Generate CCF plots
                                    ccf_paths = plot_ccf(ccf_results, metric, phase, round_id, ccf_dir, max_lag=20)
                                    if ccf_paths:
                                        plot_paths.extend(ccf_paths)
                                        self.stage_logger.info(f"Generated {len(ccf_paths)} cross-correlation plots for {metric}, {phase}, {round_id}")
                                else:
                                    self.stage_logger.warning(f"No cross-correlation data for {metric}, {phase}, {round_id}")
                            else:
                                self.stage_logger.warning(f"Empty DataFrame for {metric}, {phase}, {round_id}. Skipping cross-correlation.")
                                
                        except Exception as e_ccf:
                            self.stage_logger.error(f"Error calculating cross-correlation for {metric}, {phase}, {round_id}: {e_ccf}", exc_info=True)
        
        # Update context
        context['correlation_matrices'] = correlation_matrices
        context['covariance_matrices'] = covariance_matrices
        context['correlation_plot_paths'] = plot_paths
        
        self.stage_logger.info(f"Correlation analysis completed. Generated {len(plot_paths)} plots.")
        
        return context


class CausalityAnalysisStage(PipelineStage):
    """
    Stage for causality analysis between time series of different tenants.
    
    Implements causality analyses using:
    - Granger causality test
    - Transfer Entropy (TE)
    
    The analysis is performed for each combination of metric, experimental phase, and round,
    generating causality matrices and graph visualizations.
    """
    
    def __init__(self):
        super().__init__("causality_analysis", "Causality analysis (Granger and Transfer Entropy)")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes causality analysis and generates visualizations.
        
        Args:
            context: Context with long DataFrame.
            
        Returns:
            Updated context with results and plot paths.
        """
        from src.analysis_causality import CausalityAnalyzer
        from tqdm import tqdm
        
        df_long = context.get('data')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame not available for causality analysis.")
            return context
            
        # Configure output directory
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'causality')
        os.makedirs(out_dir, exist_ok=True)
        
        # Configuration parameters
        config = context.get('config', {})
        causality_config = config.get('causality', {})
        granger_threshold = causality_config.get('granger_threshold', 0.05)
        
        # Structures to store results
        granger_matrices = {}
        te_matrices = {}
        plot_paths = []
        
        # Initialize causality analyzer with the output directory
        analyzer = CausalityAnalyzer(df_long, out_dir)
        
        # Process each combination of experiment, round, metric, and phase
        experiments = df_long['experiment_id'].unique()
        
        for experiment_id in experiments:
            self.logger.info(f"Processing experiment: {experiment_id}")
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                
                for metric in tqdm(round_df['metric_name'].unique(), desc=f"Causality Analysis for Metrics in {round_id}"):
                    for phase in round_df['experimental_phase'].unique():
                        self.logger.info(f"Running causality analysis for {metric}, {phase}, {round_id}")
                        try:
                            # Check if there is enough data before calling the analyzer
                            phase_df = round_df[(round_df['experimental_phase'] == phase) & (round_df['metric_name'] == metric)]
                            if phase_df.empty or phase_df['tenant_id'].nunique() < 2:
                                self.logger.warning(f"Insufficient data for {metric}, {phase}, {round_id}. Skipping causality analysis.")
                                continue

                            # Run the full analysis and plotting pipeline for this slice
                            analysis_results = analyzer.run_and_plot_causality_analysis(
                                metric=metric,
                                phase=phase,
                                round_id=round_id,
                                p_value_threshold=granger_threshold
                            )
                            
                            # Store results
                            result_key = f"{experiment_id}:{round_id}:{phase}:{metric}"
                            if not analysis_results["granger_matrix"].empty:
                                granger_matrices[result_key] = analysis_results["granger_matrix"]
                            if not analysis_results["te_matrix"].empty:
                                te_matrices[result_key] = analysis_results["te_matrix"]
                            if analysis_results["plot_paths"]:
                                plot_paths.extend(analysis_results["plot_paths"])

                        except Exception as e:
                            self.logger.error(f"Error processing causality for {metric}, {phase}, {round_id}: {e}", exc_info=True)
        
        # Update context
        context['granger_matrices'] = granger_matrices
        context['te_matrices'] = te_matrices
        context['causality_plot_paths'] = plot_paths
        
        self.logger.info(f"Causality analysis completed. Generated {len(plot_paths)} plots.")
        return context


class ReportGenerationStage(PipelineStage):
    """
    Stage for generating the final consolidated report.
    
    Aggregates insights from all previous stages and generates:
    1. Textual report with main findings
    2. Inter-tenant comparative table
    3. Identification of "noisy tenants" based on objective criteria
    """
    
    def __init__(self):
        super().__init__("report_generation", "Final report generation")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generates the final report with consolidated insights.
        
        Args:
            context: Context with results from all previous stages.
            
        Returns:
            Updated context with the report path and comparative table.
        """
        from src.report_generation import (
            generate_tenant_metrics,
            generate_markdown_report,
            generate_tenant_ranking_plot,
            generate_phase_variation_plot
        )
        
        df_long = context.get('data')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame not available for report generation.")
            return context
        
        # Configure output directory
        report_dir = os.path.join(context.get('output_dir', 'outputs'), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Retrieve relevant information from the context
        granger_matrices = context.get('granger_matrices', {})
        te_matrices = context.get('te_matrices', {})
        correlation_matrices = context.get('correlation_matrices', {})
        phase_comparison_results = context.get('phase_comparison_results', {})
        
        # Generate tenant metrics and ranking
        self.logger.info("Generating tenant metrics and ranking...")
        tenant_metrics = generate_tenant_metrics(
            granger_matrices,
            te_matrices, 
            correlation_matrices,
            phase_comparison_results.get('stats_by_metric', {})
        )
        
        # Log what was generated
        self.logger.info("Tenant metrics generated: type=%s", type(tenant_metrics))
        if isinstance(tenant_metrics, pd.DataFrame):
            self.logger.info(f"Available columns: {list(tenant_metrics.columns)}")
            self.logger.info(f"Number of tenants: {len(tenant_metrics)}")
        
        # Save the metrics table
        metrics_table_path = os.path.join(report_dir, f"{report_filename}_tenant_metrics.csv")
        tenant_metrics.to_csv(metrics_table_path, index=False)
        
        # Create tenant ranking visualization
        rank_plot_path = os.path.join(report_dir, f"{report_filename}_tenant_ranking.png")
        generate_tenant_ranking_plot(tenant_metrics, rank_plot_path)
        
        # Create phase variation visualization
        phase_variation_plot_path = os.path.join(report_dir, f"{report_filename}_phase_variation.png")
        generate_phase_variation_plot(tenant_metrics, phase_variation_plot_path)
        
        # Generate final markdown report
        self.logger.info("Generating markdown report...")
        report_path = generate_markdown_report(
            tenant_metrics=tenant_metrics,
            context=context,
            rank_plot_path=rank_plot_path,
            metrics_table_path=metrics_table_path,
            phase_variation_plot_path=phase_variation_plot_path,
            out_dir=report_dir
        )
        
        # Update context
        context['report_path'] = report_path
        context['tenant_metrics'] = tenant_metrics
        context['tenant_metrics_path'] = metrics_table_path
        context['tenant_ranking_path'] = rank_plot_path
        context['phase_variation_plot_path'] = phase_variation_plot_path
        
        self.logger.info(f"Full report generated at: {report_path}")
        
        return context


class Pipeline:
    """Orchestrates the execution of the analysis pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the pipeline with a given configuration.
        
        Args:
            config: Dictionary with pipeline configurations.
        """
        self.config = config
        self.context: Dict[str, Any] = {'config': config}
        self.stages: List[PipelineStage] = []
        self.logger = logging.getLogger("pipeline.Pipeline")
        self.all_stages: Dict[str, PipelineStage] = {}
        self._setup_stages()
        
    def _setup_stages(self):
        """Initializes and adds all pipeline stages."""
        self.all_stages = {
            "data_ingestion": DataIngestionStage(),
            "data_validation": DataValidationStage(),
            "data_export": DataExportStage(),
            "descriptive_analysis": DescriptiveAnalysisStage(),
            "correlation_analysis": CorrelationAnalysisStage(),
            "causality_analysis": CausalityAnalysisStage(),
            "phase_comparison": AnalysisPhaseComparisonStage(),
            "impact_analysis": ImpactAnalysisStage(),
            "report_generation": ReportGenerationStage(),
        }
        
    def run(self, force_reprocess: bool = False, selected_stages: Optional[List[str]] = None):
        """
        Executes all pipeline stages in order.
        
        Args:
            force_reprocess: If True, forces data reprocessing.
            selected_stages: A list of stage names to execute. If None, all stages will run.
        """
        self.logger.info("Starting pipeline execution...")

        if selected_stages:
            # Garantir que a ingestão e validação de dados sempre ocorram se um estágio de análise for selecionado
            if any(s not in ['data_ingestion', 'data_validation'] for s in selected_stages):
                if 'data_ingestion' not in selected_stages:
                    selected_stages.insert(0, 'data_ingestion')
                    self.logger.info("Adicionando o estágio 'data_ingestion' como pré-requisito.")
                if 'data_validation' not in selected_stages:
                    # Inserir após a ingestão
                    ingestion_index = selected_stages.index('data_ingestion')
                    selected_stages.insert(ingestion_index + 1, 'data_validation')
                    self.logger.info("Adicionando o estágio 'data_validation' como pré-requisito.")

            self.stages = [self.all_stages[name] for name in selected_stages if name in self.all_stages]
            self.logger.info(f"Executing selected stages: {[s.name for s in self.stages]}")
        else:
            self.logger.info("Executing all stages.")
            self.stages = list(self.all_stages.values())

        self.context['force_reprocess'] = force_reprocess
        self.context['output_dir'] = self.config.get('output_dir', 'outputs')
        self.context['data_root'] = self.config.get('data_root', config.DATA_ROOT)
        self.context['processed_data_dir'] = self.config.get('processed_data_dir', config.PROCESSED_DATA_DIR)

        # Perform pre-flight check for raw data files
        if not preflight_check(self.config):
            self.logger.warning("Pre-flight check for raw data files failed. Ingestion may be incomplete.")
            # The pipeline continues as per the action plan, but with a warning.

        for stage in self.stages:
            try:
                # A chave para o sucesso: garantir que o contexto é sempre atualizado
                # e repassado para o próximo estágio.
                updated_context = stage.execute(self.context)
                self.context.update(updated_context)

                if self.context.get('pipeline_halted'):
                    self.logger.error(f"Pipeline execution halted at stage '{stage.name}'. Reason: {self.context.get('data_validation_report', 'Unknown')}")
                    break
                if self.context.get('error') is not None:
                    self.logger.error(f"Pipeline stopped at stage '{stage.name}' due to an error: {self.context.get('error')}")
                    break
            except Exception as e:
                self.logger.error(f"An unexpected error occurred in stage '{stage.name}': {e}", exc_info=True)
                break
        
        self.logger.info("Pipeline execution finished.")

def main():
    """Main function to configure and run the pipeline."""
    parser = argparse.ArgumentParser(description="Run the multi-tenant time series analysis pipeline.")
    parser.add_argument("--config", type=str, default="config/pipeline_config_sfi2.yaml", help="Path to the pipeline configuration file.")
    parser.add_argument("--force-reprocess", action="store_true", help="Force reprocessing of all data, ignoring existing consolidated files.")
    
    # Argumentos para parametrização
    parser.add_argument(
        "--analysis-stages",
        nargs='+',
        choices=['data_ingestion', 'data_validation', 'data_export', 'descriptive_analysis', 'correlation_analysis', 'causality_analysis', 'phase_comparison', 'impact_analysis', 'report_generation'],
        help="Select which analysis stages to run. If not provided, all stages will run."
    )
    parser.add_argument("--selected-rounds", nargs='+', help="Override selected rounds from config file.")
    parser.add_argument("--selected-metrics", nargs='+', help="Override selected metrics from config file.")
    parser.add_argument("--selected-tenants", nargs='+', help="Override selected tenants from config file.")

    args = parser.parse_args()

    # Load configuration
    try:
        config_dict = load_parse_config(args.config)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {args.config}")
        return 1
    except yaml.YAMLError as e:
        logger.error(f"Error parsing configuration file: {e}")
        return 1

    # Override config with command-line arguments if provided
    if args.selected_rounds:
        config_dict['selected_rounds'] = args.selected_rounds
        logger.info(f"Overriding selected_rounds with: {args.selected_rounds}")
    if args.selected_metrics:
        config_dict['selected_metrics'] = args.selected_metrics
        logger.info(f"Overriding selected_metrics with: {args.selected_metrics}")
    if args.selected_tenants:
        config_dict['selected_tenants'] = args.selected_tenants
        logger.info(f"Overriding selected_tenants with: {args.selected_tenants}")

    # Create and run the pipeline
    pipeline = Pipeline(config_dict)
    pipeline.run(force_reprocess=args.force_reprocess, selected_stages=args.analysis_stages)

    logger.info("Pipeline execution finished.")
    return 0


if __name__ == "__main__":
    # This allows running the pipeline directly from this script
    sys.exit(main())
