# -*- coding: utf-8 -*-
"""
Module for generating consolidated analysis reports.
"""
import os
import logging
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime
import glob
import time

from .pipeline_stage import PipelineStage
from .config import PipelineConfig

logger = logging.getLogger(__name__)

class ReportGenerationStage(PipelineStage):
    """
    Generates the final report by consolidating results from all previous stages.
    """
    def __init__(self, config: PipelineConfig):
        """
        Initializes the report generation stage.

        Args:
            config: The pipeline configuration object.
        """
        super().__init__(
            stage_name="report_generation", 
            description="Generates a final consolidated report from all analysis results."
        )
        self.config = config
        self.experiment_output_dir = self.config.get_output_dir()
        self.report_dir = os.path.join(self.experiment_output_dir, "reports")
        os.makedirs(self.report_dir, exist_ok=True)

    def execute(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes the report generation logic. This is a terminal stage and has a custom signature.

        Args:
            all_results: A dictionary containing the full results from all
                         previous pipeline stages, including 'per_round' and
                         'multi_round' keys.

        Returns:
            A dictionary containing the path to the generated report directory.
        """
        self.logger.info(f"Starting final report generation stage.")
        start_time = time.time()
        
        try:
            self._generate_markdown_report(all_results)
        except Exception as e:
            self.logger.error(f"Error executing stage {self.stage_name}: {e}", exc_info=True)
            return {}
            
        elapsed_time = time.time() - start_time
        self.logger.info(f"Stage {self.stage_name} completed in {elapsed_time:.2f} seconds. Report saved in: {self.report_dir}")
        
        return {"report_dir": self.report_dir}

    def _generate_markdown_report(self, all_results: Dict[str, Any]):
        """
        Generates the main report file in Markdown format.

        Args:
            all_results: The dictionary with all pipeline results.
        """
        report_filename = f"final_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(self.report_dir, report_filename)

        per_round_results = all_results.get('per_round', {})
        multi_round_results = all_results.get('multi_round', {})

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Impact Analysis Report - {self.config.get_experiment_name()}\n\n")
            f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## 1. Analysis Summary\n")
            f.write("This report presents the consolidated results of the noisy neighbor impact analysis in a multi-tenant environment. The following sections detail the impact, correlation, causality analyses, and the aggregated results from multiple rounds.\n\n")

            # --- Multi-Round Analysis Section ---
            self._write_multi_round_section(f, multi_round_results)

            # --- Per-Round Analysis Section ---
            self._write_per_round_section(f, per_round_results)

        self.logger.info(f"Markdown report successfully generated at: {report_path}")

    def _write_multi_round_section(self, f, multi_round_results: Dict[str, Any]):
        """
        Writes the multi-round analysis section to the report file.
        """
        f.write("## 2. Consolidated Multi-Round Analysis\n\n")
        if not multi_round_results:
            f.write("No multi-round analysis was performed or results were not found.\n\n")
            return

        f.write("This section aggregates the results from all experimental rounds to provide a more statistically robust view of impact, correlation, and causality patterns.\n\n")

        multi_round_output_dir = multi_round_results.get('multi_round_output_dir')
        if not multi_round_output_dir:
            f.write("Could not determine the output directory for multi-round analysis artifacts.\n\n")
            return

        # --- Aggregated Impact ---
        aggregated_impact_csv = os.path.join(multi_round_output_dir, 'multi_round_impact_aggregated_stats.csv')
        if os.path.exists(aggregated_impact_csv):
            f.write("### Aggregated Impact Statistics\n\n")
            f.write(f"The full table with aggregated impact statistics is available at: `{os.path.relpath(aggregated_impact_csv, self.report_dir)}`\n\n")
            try:
                df_stats = pd.read_csv(aggregated_impact_csv)
                f.write("**Sample of Aggregated Statistics:**\n\n")
                f.write(df_stats.head().to_markdown(index=False))
                f.write("\n\n")
            except Exception as e:
                self.logger.warning(f"Could not read the aggregated statistics file: {e}")

        # --- Causality Frequency ---
        causality_freq_csv = os.path.join(multi_round_output_dir, 'multi_round_causality_frequency.csv')
        if os.path.exists(causality_freq_csv):
            f.write("### Causality Frequency\n\n")
            f.write(f"The full table with the frequency of causal links is available at: `{os.path.relpath(causality_freq_csv, self.report_dir)}`\n\n")
            try:
                df_causality = pd.read_csv(causality_freq_csv)
                f.write("**Sample of Causality Frequency:**\n\n")
                f.write(df_causality.head().to_markdown(index=False))
                f.write("\n\n")
            except Exception as e:
                self.logger.warning(f"Could not read the causality frequency file: {e}")

        # --- Embed Consolidated Plots ---
        f.write("### Consolidated Visualizations\n\n")
        plot_files = glob.glob(os.path.join(multi_round_output_dir, '*.png'))
        if plot_files:
            for plot_path in sorted(plot_files):
                title = os.path.basename(plot_path).replace('_', ' ').replace('.png', '').title()
                relative_path = os.path.relpath(plot_path, self.report_dir)
                f.write(f"#### {title}\n\n")
                f.write(f"![{title}]({relative_path})\n\n")
        else:
            f.write("No consolidated visualizations were found in the multi-round output directory.\n\n")

    def _write_per_round_section(self, f, per_round_results: Dict[str, Any]):
        """
        Writes the per-round analysis summary to the report file.
        """
        f.write("## 3. Per-Round Analysis Summary\n\n")
        if not per_round_results:
            f.write("No per-round analysis results were found.\n\n")
            return

        for round_id, round_results in sorted(per_round_results.items()):
            f.write(f"### Round: {round_id}\n\n")
            
            # Find the main output directory for this round to make relative paths
            round_output_dir = self.config.get_output_dir_for_round("", round_id) # Base dir for the round
            
            # Link to key artifacts and embed key plots
            for stage_name, stage_results in round_results.items():
                if not isinstance(stage_results, dict): continue

                f.write(f"#### Stage: {stage_name}\n\n")
                
                # Link to CSVs
                for key, value in stage_results.items():
                    if isinstance(value, str) and value.endswith('.csv') and os.path.exists(value):
                        relative_path = os.path.relpath(value, self.report_dir)
                        f.write(f"- **{key.replace('_', ' ').title()}:** [`{os.path.basename(value)}`]({relative_path})\n")

                # Embed Plots
                plots = stage_results.get('plots', [])
                if plots:
                    f.write("\n**Visualizations:**\n")
                    for plot_path in plots:
                        if os.path.exists(plot_path):
                            title = os.path.basename(plot_path).replace('_', ' ').replace('.png', '').title()
                            relative_path = os.path.relpath(plot_path, self.report_dir)
                            f.write(f"![{title}]({relative_path})\n")
                        else:
                            self.logger.warning(f"Plot file not found for round {round_id}: {plot_path}")
                f.write("\n---\n")
