"""
Module: data_segment.py
Description: Utilities for segmenting the long DataFrame and generating wide-format DataFrames for analysis.
"""
import pandas as pd
from typing import Optional, List, Dict

def filter_long_df(
    df: pd.DataFrame,
    phase: Optional[str] = None,
    tenant: Optional[str] = None,
    metric: Optional[str] = None,
    round_id: Optional[str] = None,
    experiment_id: Optional[str] = None
) -> pd.DataFrame:
    """Filter the long DataFrame by phase, tenant, metric, round, and/or experiment."""
    # Create a copy to avoid SettingWithCopyWarning and other display issues
    filtered_df = df.copy()
    
    # Apply filters sequentially instead of using bitwise operators with masks
    if phase:
        filtered_df = filtered_df[filtered_df['experimental_phase'] == phase]
    if tenant:
        filtered_df = filtered_df[filtered_df['tenant_id'] == tenant]
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
    if round_id:
        filtered_df = filtered_df[filtered_df['round_id'] == round_id]
    if experiment_id:
        filtered_df = filtered_df[filtered_df['experiment_id'] == experiment_id]
        
    return filtered_df

def get_wide_format_for_analysis(
    df: pd.DataFrame,
    metric: str,
    phase: Optional[str] = None,
    round_id: Optional[str] = None,
    experiment_id: Optional[str] = None
) -> pd.DataFrame:
    """
    Generate a wide-format DataFrame for a given metric, phase, round, and experiment.
    Rows: timestamps; Columns: tenants; Values: metric_value.
    """
    df_filtered = filter_long_df(df, phase=phase, metric=metric, round_id=round_id, experiment_id=experiment_id)
    wide_df = df_filtered.pivot_table(
        index='timestamp',
        columns='tenant_id',
        values='metric_value',
        aggfunc='mean'  # In case of duplicate timestamps
    )
    wide_df = wide_df.sort_index()
    return wide_df
