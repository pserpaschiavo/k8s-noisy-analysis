"""
Module: analysis_export.py
Description: Handles the export of analysis results to Parquet tables for use in other tools.
"""
import os
import pandas as pd
import logging
from typing import Dict, Any, Optional, List
import numpy as np

from .pipeline_stage import PipelineStage
from .config import PipelineConfig
from .data_parquet_utils import ParquetDataManager

# Setup logging
logger = logging.getLogger(__name__)

class AnalysisExportStage(PipelineStage):
    """
    Pipeline stage for exporting analysis results to Parquet tables.
    This stage provides a way to export different types of analysis results
    to Parquet tables that can be used in other tools like Power BI, Tableau, etc.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("analysis_export", "Exports analysis results to Parquet tables")
        self.config = config
        self.parquet_manager = ParquetDataManager(
            base_output_dir=os.path.join(config.get('output_dir'), 'parquet_tables')
        )
        
    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Exports various analysis results to Parquet tables for the specified round.
        
        Args:
            data: The main DataFrame (unused here as we use results from all_results)
            all_results: Dictionary with results from previous pipeline stages
            round_id: Identifier of the current round
            
        Returns:
            Dictionary with paths to exported Parquet tables
        """
        self.logger.info(f"Starting analysis export for round '{round_id}'...")
        
        exported_tables = {}
        
        # Export Impact Analysis Results
        if 'impact_analysis' in all_results:
            impact_results = all_results['impact_analysis']
            if isinstance(impact_results, dict) and 'summary_df' in impact_results:
                impact_path = self.parquet_manager.save_analysis_result(
                    df=impact_results['summary_df'],
                    analysis_type='impact',
                    round_id=round_id
                )
                exported_tables['impact_analysis'] = impact_path
                
        # Export Correlation Analysis Results
        if 'correlation_analysis' in all_results:
            corr_results = all_results['correlation_analysis']
            if isinstance(corr_results, dict) and 'correlation_df' in corr_results:
                corr_path = self.parquet_manager.save_analysis_result(
                    df=corr_results['correlation_df'],
                    analysis_type='correlation',
                    round_id=round_id
                )
                exported_tables['correlation_analysis'] = corr_path
        
        # Export Descriptive Statistics
        if 'descriptive_analysis' in all_results:
            desc_results = all_results['descriptive_analysis']
            if isinstance(desc_results, dict) and 'summary_stats' in desc_results:
                # Convert the nested dictionary to a DataFrame
                stats_dict = desc_results['summary_stats']
                stats_rows = []
                
                for metric_name, metric_stats in stats_dict.items():
                    for phase, phase_stats in metric_stats.items():
                        for tenant, tenant_stats in phase_stats.items():
                            for stat_name, stat_value in tenant_stats.items():
                                stats_rows.append({
                                    'metric_name': metric_name,
                                    'experimental_phase': phase,
                                    'tenant_id': tenant,
                                    'statistic': stat_name,
                                    'value': stat_value
                                })
                
                if stats_rows:
                    stats_df = pd.DataFrame(stats_rows)
                    desc_path = self.parquet_manager.save_analysis_result(
                        df=stats_df,
                        analysis_type='descriptive',
                        round_id=round_id
                    )
                    exported_tables['descriptive_analysis'] = desc_path
        
        # Export Causality Analysis Results
        if 'causality_analysis' in all_results:
            causality_results = all_results['causality_analysis']
            if isinstance(causality_results, dict) and 'granger_results' in causality_results:
                # Convert the causality results to a DataFrame
                granger_dict = causality_results['granger_results']
                granger_rows = []
                
                for phase, phase_results in granger_dict.items():
                    for cause, effects in phase_results.items():
                        for effect, stats in effects.items():
                            granger_rows.append({
                                'experimental_phase': phase,
                                'cause': cause,
                                'effect': effect,
                                'p_value': stats.get('p_value'),
                                'f_value': stats.get('f_value'),
                                'lag': stats.get('lag')
                            })
                
                if granger_rows:
                    granger_df = pd.DataFrame(granger_rows)
                    causality_path = self.parquet_manager.save_analysis_result(
                        df=granger_df,
                        analysis_type='causality',
                        round_id=round_id,
                        sub_type='granger'
                    )
                    exported_tables['causality_analysis_granger'] = causality_path
        
        # Export Phase Comparison Results
        if 'phase_comparison' in all_results:
            phase_results = all_results['phase_comparison']
            if isinstance(phase_results, dict) and 'comparison_df' in phase_results:
                phase_path = self.parquet_manager.save_analysis_result(
                    df=phase_results['comparison_df'],
                    analysis_type='phase_comparison',
                    round_id=round_id
                )
                exported_tables['phase_comparison'] = phase_path
        
        # Export Multi-Round Analysis Results (if this is a multi-round run)
        if round_id == 'multi-round' and 'multi_round_analysis' in all_results:
            multi_results = all_results['multi_round_analysis']
            
            # Export meta analysis results
            if isinstance(multi_results, dict) and 'meta_analysis_df' in multi_results:
                meta_path = self.parquet_manager.save_analysis_result(
                    df=multi_results['meta_analysis_df'],
                    analysis_type='multi_round',
                    sub_type='meta_analysis'
                )
                exported_tables['multi_round_meta_analysis'] = meta_path
                
            # Export effect size results
            if isinstance(multi_results, dict) and 'effect_sizes_df' in multi_results:
                effect_path = self.parquet_manager.save_analysis_result(
                    df=multi_results['effect_sizes_df'],
                    analysis_type='multi_round',
                    sub_type='effect_sizes'
                )
                exported_tables['multi_round_effect_sizes'] = effect_path
        
        self.logger.info(f"Analysis export for round '{round_id}' complete. Exported {len(exported_tables)} tables.")
        
        return {
            'exported_tables': exported_tables,
            'base_output_dir': self.parquet_manager.base_output_dir
        }


def export_analysis_results_for_external_tools(config: PipelineConfig, all_results: Dict[str, Any]) -> Dict[str, str]:
    """
    Utility function to export all analysis results to Parquet tables for use in external tools.
    This function is designed to be called at the end of the pipeline to export all results
    in a format that can be easily consumed by external tools like Power BI, Tableau, etc.
    
    Args:
        config: Pipeline configuration
        all_results: Dictionary with all results from the pipeline
    
    Returns:
        Dictionary with paths to exported Parquet tables
    """
    logger.info("Exporting analysis results for external tools...")
    
    parquet_manager = ParquetDataManager(
        base_output_dir=os.path.join(config.get('output_dir'), 'external_tables')
    )
    
    exported_tables = {}
    
    # Export consolidated data
    if 'consolidated_df' in all_results:
        consolidated_path = parquet_manager.save_analysis_result(
            df=all_results['consolidated_df'],
            analysis_type='consolidated_data'
        )
        exported_tables['consolidated_data'] = consolidated_path
    
    # Export per-round results
    if 'per_round' in all_results:
        per_round_results = all_results['per_round']
        for round_id, round_results in per_round_results.items():
            # Export summary information for each round
            round_summary = {
                'round_id': round_id
            }
            
            # Collect metadata about each round
            if 'data_ingestion' in round_results and 'summary' in round_results['data_ingestion']:
                ingestion_summary = round_results['data_ingestion']['summary']
                if isinstance(ingestion_summary, dict):
                    round_summary.update({
                        'records_count': ingestion_summary.get('records_ingested', 0),
                        'metrics_count': len(ingestion_summary.get('metrics', [])),
                        'phases_count': len(ingestion_summary.get('phases', [])),
                        'tenants_count': len(ingestion_summary.get('tenants', []))
                    })
            
            # Handle export for specific analysis types
            for analysis_type in ['impact_analysis', 'descriptive_analysis', 'correlation_analysis']:
                if analysis_type in round_results:
                    export_analysis_type(
                        parquet_manager=parquet_manager,
                        analysis_type=analysis_type,
                        round_results=round_results[analysis_type],
                        round_id=round_id,
                        exported_tables=exported_tables
                    )
    
    # Export multi-round results if available
    if 'multi_round_analysis' in all_results:
        multi_round_results = all_results['multi_round_analysis']
        export_multi_round_results(
            parquet_manager=parquet_manager,
            multi_round_results=multi_round_results,
            exported_tables=exported_tables
        )
    
    logger.info(f"Analysis export complete. Exported {len(exported_tables)} tables.")
    
    return exported_tables


def export_analysis_type(
    parquet_manager: ParquetDataManager,
    analysis_type: str,
    round_results: Dict[str, Any],
    round_id: str,
    exported_tables: Dict[str, str]
) -> None:
    """
    Helper function to export a specific analysis type.
    
    Args:
        parquet_manager: ParquetDataManager instance
        analysis_type: Type of analysis
        round_results: Results for the analysis
        round_id: Round identifier
        exported_tables: Dictionary to update with exported table paths
    """
    # Map of result keys to DataFrame names for different analysis types
    result_key_map = {
        'impact_analysis': 'summary_df',
        'descriptive_analysis': 'summary_stats_df',
        'correlation_analysis': 'correlation_df',
        'causality_analysis': 'granger_results_df',
        'phase_comparison': 'comparison_df'
    }
    
    # Get the appropriate result key for this analysis type
    result_key = result_key_map.get(analysis_type)
    
    if result_key and result_key in round_results:
        df = round_results[result_key]
        if isinstance(df, pd.DataFrame) and not df.empty:
            table_path = parquet_manager.save_analysis_result(
                df=df,
                analysis_type=analysis_type.replace('_analysis', ''),
                round_id=round_id
            )
            exported_tables[f"{analysis_type}_{round_id}"] = table_path


def export_multi_round_results(
    parquet_manager: ParquetDataManager,
    multi_round_results: Dict[str, Any],
    exported_tables: Dict[str, str]
) -> None:
    """
    Helper function to export multi-round analysis results.
    
    Args:
        parquet_manager: ParquetDataManager instance
        multi_round_results: Multi-round analysis results
        exported_tables: Dictionary to update with exported table paths
    """
    # Export meta analysis results
    if 'meta_analysis_df' in multi_round_results:
        meta_df = multi_round_results['meta_analysis_df']
        if isinstance(meta_df, pd.DataFrame) and not meta_df.empty:
            meta_path = parquet_manager.save_analysis_result(
                df=meta_df,
                analysis_type='multi_round',
                sub_type='meta_analysis'
            )
            exported_tables['multi_round_meta_analysis'] = meta_path
    
    # Export effect sizes
    if 'effect_sizes_df' in multi_round_results:
        effect_df = multi_round_results['effect_sizes_df']
        if isinstance(effect_df, pd.DataFrame) and not effect_df.empty:
            effect_path = parquet_manager.save_analysis_result(
                df=effect_df,
                analysis_type='multi_round',
                sub_type='effect_sizes'
            )
            exported_tables['multi_round_effect_sizes'] = effect_path
