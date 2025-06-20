"""
Module: data_export.py
Description: Utility functions for saving and loading processed DataFrames (long/wide) in efficient formats.
"""
import pandas as pd
import os
from typing import Optional

def save_dataframe(df: pd.DataFrame, out_path: str, format: str = 'parquet'):
    """Save DataFrame to disk in the specified format (parquet/csv)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if format == 'parquet':
        df.to_parquet(out_path, index=False)
    elif format == 'csv':
        df.to_csv(out_path, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_dataframe(in_path: str, format: Optional[str] = None) -> pd.DataFrame:
    """Load DataFrame from disk in the specified format (parquet/csv)."""
    if not format:
        if in_path.endswith('.parquet'):
            format = 'parquet'
        elif in_path.endswith('.csv'):
            format = 'csv'
        else:
            raise ValueError("Cannot infer file format from extension.")
    if format == 'parquet':
        return pd.read_parquet(in_path)
    elif format == 'csv':
        return pd.read_csv(in_path)
    else:
        raise ValueError(f"Unsupported format: {format}")

def export_segmented_long(
    df: pd.DataFrame,
    out_dir: str,
    experiment_id: str,
    round_id: str,
    phase: str,
    format: str = 'parquet'
):
    """Exports a filtered long subset to a directory organized by experiment/round/phase."""
    subdir = os.path.join(out_dir, experiment_id, round_id, phase)
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, f"long.{format}")
    save_dataframe(df, out_path, format=format)
    return out_path

def export_segmented_wide(
    df: pd.DataFrame,
    out_dir: str,
    experiment_id: str,
    round_id: str,
    phase: str,
    metric: str,
    format: str = 'parquet'
):
    """Exports a wide subset to a directory organized by experiment/round/phase/metric."""
    subdir = os.path.join(out_dir, experiment_id, round_id, phase, metric)
    os.makedirs(subdir, exist_ok=True)
    out_path = os.path.join(subdir, f"wide.{format}")
    save_dataframe(df, out_path, format=format)
    return out_path
