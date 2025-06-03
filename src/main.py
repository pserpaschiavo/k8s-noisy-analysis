"""
Main entry point for the pipeline. Loads user parse config, ingests data, and saves the consolidated DataFrame.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_ingestion import ingest_experiment_data
from src.data_export import save_dataframe
from src.parse_config import load_parse_config, get_selected_metrics, get_selected_tenants, get_selected_rounds
from src import config

def main():
    # Ensure output/config directories exist
    config.ensure_directories()

    # Load user parse config if available
    config_path = os.path.join(config.CONFIG_DIR, 'parse_config.yaml')
    if os.path.exists(config_path):
        from src.parse_config import get_data_root, get_processed_data_dir
        user_config = load_parse_config(config_path)
        selected_metrics = get_selected_metrics(user_config) or config.DEFAULT_SELECTED_METRICS
        selected_tenants = get_selected_tenants(user_config) or config.DEFAULT_SELECTED_TENANTS
        selected_rounds = get_selected_rounds(user_config) or config.DEFAULT_SELECTED_ROUNDS
        data_root = get_data_root(user_config) or config.DATA_ROOT
        processed_data_dir = get_processed_data_dir(user_config) or config.PROCESSED_DATA_DIR
    else:
        selected_metrics = config.DEFAULT_SELECTED_METRICS
        selected_tenants = config.DEFAULT_SELECTED_TENANTS
        selected_rounds = config.DEFAULT_SELECTED_ROUNDS
        data_root = config.DATA_ROOT
        processed_data_dir = config.PROCESSED_DATA_DIR

    # Ingest data
    df_long = ingest_experiment_data(
        data_root=data_root,
        selected_metrics=selected_metrics,
        selected_tenants=selected_tenants,
        selected_rounds=selected_rounds
    )
    print(f"Loaded {len(df_long)} records.")

    # Save consolidated DataFrame
    consolidated_long_path = os.path.join(processed_data_dir, 'consolidated_long.parquet')
    save_dataframe(df_long, consolidated_long_path, format='parquet')
    print(f"Saved DataFrame to {consolidated_long_path}")

    # Exemplo: exportar subconjunto long e wide para um experimento, round, fase e métrica específicos
    from src.data_segment import filter_long_df, get_wide_format_for_analysis
    from src.data_export import export_segmented_long, export_segmented_wide

    # Parâmetros de exemplo (poderiam vir do config ou CLI)
    example_experiment = df_long['experiment_id'].iloc[0]
    example_round = df_long['round_id'].iloc[0]
    example_phase = df_long['experimental_phase'].iloc[0]
    example_metric = df_long['metric_name'].iloc[0]

    # Filtrar subconjunto long
    df_long_segment = filter_long_df(
        df_long,
        phase=example_phase,
        round_id=example_round,
        experiment_id=example_experiment
    )
    export_segmented_long(
        df_long_segment,
        processed_data_dir,
        experiment_id=example_experiment,
        round_id=example_round,
        phase=example_phase,
        format='parquet'
    )

    # Gerar e exportar wide para uma métrica
    df_wide = get_wide_format_for_analysis(
        df_long,
        metric=example_metric,
        phase=example_phase,
        round_id=example_round,
        experiment_id=example_experiment
    )
    export_segmented_wide(
        df_wide,
        processed_data_dir,
        experiment_id=example_experiment,
        round_id=example_round,
        phase=example_phase,
        metric=example_metric,
        format='parquet'
    )

if __name__ == "__main__":
    main()
