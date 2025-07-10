# -*- coding: utf-8 -*-

"""
Module for the consolidated analysis of multiple execution rounds.

This module is responsible for aggregating the results of various pipeline
executions, allowing for a more robust statistical analysis and the
generation of visualizations that demonstrate the consistency and
variability of the results.

Main Features:
- Load and consolidate impact results from multiple rounds.
- Calculate aggregated statistical metrics (mean, standard deviation, etc.).
- Prepare data for the generation of multi-round visualizations.

Classes:
- MultiRoundAnalysisStage: Orchestrates the execution of the multi-round analysis.
"""

import pandas as pd
import os
import glob
from typing import Optional, Dict, List, Any
from .pipeline_stage import PipelineStage
from .config import PipelineConfig
from .visualization.multi_round_plots import (
    plot_aggregated_impact_boxplots,
    plot_aggregated_impact_bar_charts,
    plot_correlation_consistency_heatmap,
    plot_causality_consistency_matrix,
    plot_aggregated_causality_graph
)

class MultiRoundAnalysisStage(PipelineStage):
    """
    Represents the consolidated multi-round analysis stage in the pipeline.

    This class inherits from `PipelineStage` and implements the logic to load,
    consolidate, and analyze the results from all execution rounds
    defined in the experiment's configuration.
    """

    def __init__(self, config: PipelineConfig):
        """
        Initializes the multi-round analysis stage.

        Args:
            config (PipelineConfig): The pipeline configuration object.
        """
        super().__init__(
            stage_name="multi_round_analysis", 
            description="Consolidates and analyzes the results of multiple rounds."
        )
        self.config = config
        # The main output directory for the entire experiment
        self.experiment_output_dir = self.config.get_output_dir()
        # The specific directory for this stage's artifacts
        self.stage_output_dir = os.path.join(self.experiment_output_dir, self.stage_name)
        os.makedirs(self.stage_output_dir, exist_ok=True)

    def _find_reports_from_context(self, all_results: Dict[str, Any], analysis_stage_name: str, result_key: str) -> List[pd.DataFrame]:
        """
        Extracts specific results from the context for all rounds.

        Args:
            all_results: The dictionary containing all results from all rounds.
            analysis_stage_name: The name of the analysis stage to get results from (e.g., 'impact_analysis').
            result_key: The key for the specific result DataFrame within the stage's results.

        Returns:
            A list of DataFrames, one for each round.
        """
        dataframes = []
        for round_id, round_results in all_results.items():
            stage_results = round_results.get(analysis_stage_name)
            if stage_results and isinstance(stage_results, dict):
                result_df = stage_results.get(result_key)
                if result_df is not None and not result_df.empty:
                    # Ensure round_id is present for aggregation
                    if 'round_id' not in result_df.columns:
                        result_df['round_id'] = round_id
                    dataframes.append(result_df)
                else:
                    self.logger.warning(f"No '{result_key}' found for stage '{analysis_stage_name}' in round '{round_id}'.")
            else:
                self.logger.warning(f"No results found for stage '{analysis_stage_name}' in round '{round_id}'.")
        
        if not dataframes:
            self.logger.warning(f"No dataframes found for stage '{analysis_stage_name}' with key '{result_key}' across all rounds.")
            
        return dataframes

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Executes the logic of consolidating and analyzing the results from all rounds.

        Args:
            data: Not used in this stage, as it operates on the consolidated results.
            all_results: A dictionary containing all results from all previous stages and rounds.
            round_id: A nominal identifier for this execution (e.g., 'multi-round').

        Returns:
            A dictionary containing the paths to the generated artifacts.
        """
        self.logger.info(f"Starting multi-round analysis stage for experiment: {self.config.get_experiment_name()}")

        # This stage operates on the `all_results` which contains per-round data.
        # The structure is expected to be: {'per_round': {'round-1': {'stage_name': {...}}, 'round-2': ...}}
        per_round_results = all_results.get('per_round', {})
        if not per_round_results:
            self.logger.error("Could not find 'per_round' results in the input data. Aborting multi-round analysis.")
            return {}
        
        # --- Multi-Round Impact Analysis ---
        self.logger.info("Starting multi-round impact analysis...")
        impact_dfs = self._find_reports_from_context(per_round_results, 'impact_analysis', 'impact_metrics')
        if impact_dfs:
            self._process_impact_analysis(impact_dfs)
        else:
            self.logger.warning("No impact reports found. Skipping multi-round impact analysis.")

        # --- Multi-Round Correlation Analysis ---
        self.logger.info("Starting multi-round correlation analysis...")
        correlation_dfs = self._find_reports_from_context(per_round_results, 'correlation_analysis', 'correlation_results')
        if correlation_dfs:
            self._process_correlation_analysis(correlation_dfs)
        else:
            self.logger.warning("No correlation reports found. Skipping multi-round correlation analysis.")

        # --- Multi-Round Causality Analysis ---
        self.logger.info("Starting multi-round causality analysis...")
        causality_dfs = self._find_reports_from_context(per_round_results, 'causality_analysis', 'causality_results')
        if causality_dfs:
            self._process_causality_analysis(causality_dfs)
        else:
            self.logger.warning("No causality reports found. Skipping multi-round causality analysis.")

        self.logger.info(f"Multi-round analysis stage completed. Artifacts saved in: {self.stage_output_dir}")
        
        # The results are the artifacts saved to disk, but we can return a summary.
        return {"multi_round_output_dir": self.stage_output_dir}

    def _process_impact_analysis(self, impact_dfs: List[pd.DataFrame]):
        """
        Processes the impact analysis reports from all rounds.
        """
        consolidated_df = pd.concat(impact_dfs, ignore_index=True)
        if consolidated_df.empty:
            self.logger.error("Consolidation of impact reports failed or resulted in an empty DataFrame.")
            return

        self.logger.info(f"Impact results from {len(impact_dfs)} rounds consolidated successfully.")

        # Logic to calculate aggregated statistics
        aggregated_stats_cohen = self._calculate_aggregated_stats(consolidated_df, 'cohen_d')
        aggregated_stats_perc = self._calculate_aggregated_stats(consolidated_df, 'percentage_change')

        # Merge the aggregated stats
        merge_cols = [col for col in ['metric_name', 'tenant_id', 'phase'] if col in aggregated_stats_cohen.columns and col in aggregated_stats_perc.columns]
        aggregated_stats = pd.merge(aggregated_stats_cohen, aggregated_stats_perc, on=merge_cols)


        # Save the aggregated results to a new CSV
        aggregated_stats_path = os.path.join(self.stage_output_dir, 'multi_round_impact_aggregated_stats.csv')
        aggregated_stats.to_csv(aggregated_stats_path, index=False)
        self.logger.info(f"Aggregated impact stats saved to {aggregated_stats_path}")

        # Generate and save plots
        plot_aggregated_impact_boxplots(consolidated_df, self.stage_output_dir)
        plot_aggregated_impact_bar_charts(consolidated_df, self.stage_output_dir)

    def _consolidate_reports(self, report_paths: List[str], round_id_extractor) -> pd.DataFrame:
        """
        Consolidates multiple CSV reports into a single DataFrame.
        """
        all_dfs = []
        for path in report_paths:
            try:
                df = pd.read_csv(path)
                # Extract round_id from path and add it as a column
                df['round_id'] = round_id_extractor(path)
                all_dfs.append(df)
            except Exception as e:
                self.logger.error(f"Failed to read or process report {path}: {e}")
        
        if not all_dfs:
            return pd.DataFrame()
            
        return pd.concat(all_dfs, ignore_index=True)

    def _calculate_aggregated_stats(self, df: pd.DataFrame, value_col: str) -> pd.DataFrame:
        """
        Calculates aggregated statistics (mean, std, count) for a given value column.
        Groups by metric, tenant, and any other relevant columns.
        """
        grouping_cols = [col for col in ['metric_name', 'tenant_id', 'phase'] if col in df.columns]
        if not grouping_cols:
            self.logger.error("Cannot determine grouping columns for aggregation.")
            return pd.DataFrame()

        self.logger.info(f"Aggregating stats for '{value_col}' grouped by {grouping_cols}")
        
        aggregated = df.groupby(grouping_cols)[value_col].agg(['mean', 'std', 'count']).reset_index()
        aggregated.rename(columns={
            'mean': f'mean_{value_col}',
            'std': f'std_{value_col}',
            'count': 'num_rounds'
        }, inplace=True)
        
        return aggregated

    def _process_correlation_analysis(self, correlation_dfs: List[pd.DataFrame]):
        """
        Processes the correlation analysis reports.
        """
        if not correlation_dfs:
            self.logger.warning("No correlation DataFrames to process.")
            return
            
        consolidated_corr_df = pd.concat(correlation_dfs, ignore_index=True)
        if consolidated_corr_df.empty:
            self.logger.error("Consolidation of correlation reports failed.")
            return

        self.logger.info(f"Correlation results from {len(correlation_dfs)} rounds consolidated.")
        
        # Standardize column names for aggregation
        consolidated_corr_df.rename(columns={
            'metric': 'metric_name'
        }, inplace=True)

        # Calculate mean correlation
        grouping_cols = ['metric_name', 'phase', 'tenant1', 'tenant2']
        # Check for available columns before grouping
        valid_grouping_cols = [col for col in grouping_cols if col in consolidated_corr_df.columns]
        
        if not valid_grouping_cols:
            self.logger.error("Could not find necessary columns to group correlation data.")
            return

        aggregated_corr = consolidated_corr_df.groupby(valid_grouping_cols)['correlation'].agg(['mean', 'std']).reset_index()
        aggregated_corr.rename(columns={'mean': 'mean_correlation', 'std': 'std_correlation'}, inplace=True)

        # Save consolidated data
        consolidated_corr_path = os.path.join(self.stage_output_dir, 'multi_round_correlation_all.csv')
        aggregated_corr.to_csv(consolidated_corr_path, index=False)
        self.logger.info(f"Consolidated correlation data saved to {consolidated_corr_path}")

        # Rename 'metric_name' to 'metric' for compatibility with the plotting function
        if 'metric_name' in aggregated_corr.columns:
            aggregated_corr.rename(columns={'metric_name': 'metric'}, inplace=True)

        # Generate consistency heatmap
        plot_correlation_consistency_heatmap(aggregated_corr, self.stage_output_dir)

    def _process_causality_analysis(self, causality_dfs: List[pd.DataFrame]):
        """
        Processes the causality analysis reports.
        """
        if not causality_dfs:
            self.logger.warning("No causality DataFrames to process.")
            return

        consolidated_causality_df = pd.concat(causality_dfs, ignore_index=True)
        if consolidated_causality_df.empty:
            self.logger.error("Consolidation of causality reports failed.")
            return

        self.logger.info(f"Causality results from {len(causality_dfs)} rounds consolidated.")
        
        # Save consolidated data
        consolidated_causality_path = os.path.join(self.stage_output_dir, 'multi_round_causality_all.csv')
        consolidated_causality_df.to_csv(consolidated_causality_path, index=False)
        self.logger.info(f"Consolidated causality data saved to {consolidated_causality_path}")

        # Calculate frequency and consistency of causal links
        num_rounds = len(causality_dfs)
        causality_frequency = consolidated_causality_df.groupby(['source', 'target']).size().reset_index(name='frequency')
        causality_frequency['consistency_rate'] = causality_frequency['frequency'] / num_rounds
        
        # Save frequency data
        causality_freq_path = os.path.join(self.stage_output_dir, 'multi_round_causality_frequency.csv')
        causality_frequency.to_csv(causality_freq_path, index=False)
        self.logger.info(f"Causality link frequency data saved to {causality_freq_path}")

        # Generate plots
        plot_causality_consistency_matrix(causality_frequency, self.stage_output_dir)
        plot_aggregated_causality_graph(causality_frequency, self.stage_output_dir)

