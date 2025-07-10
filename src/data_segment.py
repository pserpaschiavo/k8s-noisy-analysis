"""
Module: data_segment.py
Description: Utilities for filtering and transforming the consolidated data.
"""
import pandas as pd
import logging
from typing import List, Optional, Dict, Any

from .pipeline_stage import PipelineStage
from .config import PipelineConfig

# Setup logging
logger = logging.getLogger(__name__)

def filter_long_df(df: pd.DataFrame, 
                   selected_metrics: Optional[List[str]] = None, 
                   selected_tenants: Optional[List[str]] = None, 
                   selected_rounds: Optional[List[str]] = None,
                   selected_phases: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Filters the long format DataFrame based on selected metrics, tenants, rounds, and phases.
    """
    df_filtered = df.copy()
    
    if selected_metrics:
        df_filtered = df_filtered[df_filtered['metric_name'].isin(selected_metrics)]
    if selected_tenants:
        df_filtered = df_filtered[df_filtered['tenant_id'].isin(selected_tenants)]
    if selected_rounds:
        df_filtered = df_filtered[df_filtered['round_id'].isin(selected_rounds)]
    if selected_phases:
        df_filtered = df_filtered[df_filtered['experimental_phase'].isin(selected_phases)]
        
    logger.info(f"DataFrame filtered. Original size: {len(df)}, New size: {len(df_filtered)}")
    return df_filtered

def get_wide_format_for_analysis(df: pd.DataFrame, metric: str, phase: str, round_id: str) -> Optional[pd.DataFrame]:
    """
    Transforms a long format DataFrame into a wide format for a specific metric, phase, and round.
    The resulting DataFrame is indexed by timestamp, with tenants as columns.
    """
    logger.debug(f"Creating wide format for metric='{metric}', phase='{phase}', round='{round_id}'")
    
    # Filter data for the specific context
    df_slice = df[
        (df['metric_name'] == metric) & 
        (df['experimental_phase'] == phase) & 
        (df['round_id'] == round_id)
    ]
    
    if df_slice.empty:
        logger.warning(f"No data found for metric='{metric}', phase='{phase}', round='{round_id}'. Cannot create wide format.")
        return None
        
    # Pivot to wide format
    try:
        df_wide = df_slice.pivot_table(
            index='timestamp', 
            columns='tenant_id', 
            values='metric_value'
        )
        
        # Optional: forward-fill missing values to handle asynchronous measurements
        df_wide.ffill(inplace=True)
        df_wide.bfill(inplace=True) # Also back-fill to handle initial NaNs
        
        # Drop any columns that are still all NaN
        df_wide.dropna(axis=1, how='all', inplace=True)
        
        if df_wide.empty:
            logger.warning(f"Wide format is empty after pivoting for metric='{metric}', phase='{phase}', round='{round_id}'.")
            return None

        logger.info(f"Successfully created wide format for '{metric}' with shape {df_wide.shape}")
        return df_wide
        
    except Exception as e:
        logger.error(f"Error pivoting data for metric='{metric}', phase='{phase}', round='{round_id}': {e}")
        return None

class DataSegmentationStage(PipelineStage):
    """
    Pipeline stage for filtering and segmenting data based on the configuration.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__("data_segmentation", "Filter and segment data")
        self.config = config

    def _execute_implementation(self, data: Optional[pd.DataFrame], all_results: Dict[str, Any], round_id: str) -> Dict[str, Any]:
        """
        Applies filters to the DataFrame based on the pipeline configuration.
        """
        self.logger.info(f"Starting data segmentation for round '{round_id}'...")
        if data is None or data.empty:
            self.logger.error("Input DataFrame 'data' is not available for segmentation.")
            return {"segmented_data": None}

        # Get filter criteria from the configuration
        selected_metrics = self.config.get_selected_metrics()
        selected_tenants = self.config.get_selected_tenants()
        selected_phases = self.config.get_selected_phases()

        # Apply the filters
        segmented_data = filter_long_df(
            df=data,
            selected_metrics=selected_metrics,
            selected_tenants=selected_tenants,
            selected_phases=selected_phases
        )

        if segmented_data.empty:
            self.logger.warning(f"Data for round '{round_id}' is empty after segmentation.")
        else:
            self.logger.info(f"Data segmentation for round '{round_id}' completed.")
            
        return {"segmented_data": segmented_data}
