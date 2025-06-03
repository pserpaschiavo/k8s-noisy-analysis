"""
Script de teste para segmentação e exportação de DataFrames long e wide.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_export import save_dataframe, load_dataframe, export_segmented_long, export_segmented_wide
from src.data_segment import filter_long_df, get_wide_format_for_analysis
from src import config

# Caminhos
consolidated_long_path = os.path.join(config.PROCESSED_DATA_DIR, 'consolidated_long.parquet')

# Carregar DataFrame consolidado
print(f"Carregando DataFrame consolidado de {consolidated_long_path}...")
df_long = load_dataframe(consolidated_long_path)
print(f"Shape do DataFrame long: {df_long.shape}")

# Parâmetros de teste (ajuste conforme necessário)
example_experiment = df_long['experiment_id'].iloc[0]
example_round = df_long['round_id'].iloc[0]
example_phase = df_long['experimental_phase'].iloc[0]
example_metric = df_long['metric_name'].iloc[0]
example_tenant = df_long['tenant_id'].iloc[0]

# Teste 1: Filtrar por fase, round, experimento
print("Testando filter_long_df...")
df_segment = filter_long_df(df_long, phase=example_phase, round_id=example_round, experiment_id=example_experiment)
print(f"Shape do segmento filtrado: {df_segment.shape}")

# Teste 2: Filtrar por tenant
print("Testando filter_long_df por tenant...")
df_tenant = filter_long_df(df_long, tenant=example_tenant)
print(f"Shape do segmento por tenant: {df_tenant.shape}")

# Teste 3: Gerar wide para uma métrica
print("Testando get_wide_format_for_analysis...")
df_wide = get_wide_format_for_analysis(df_long, metric=example_metric, phase=example_phase, round_id=example_round, experiment_id=example_experiment)
print(f"Shape do DataFrame wide: {df_wide.shape}")

# Teste 4: Exportar segmento long
print("Testando export_segmented_long...")
long_path = export_segmented_long(df_segment, config.PROCESSED_DATA_DIR, example_experiment, example_round, example_phase, format='parquet')
print(f"Segmento long exportado para: {long_path}")

# Teste 5: Exportar wide
print("Testando export_segmented_wide...")
wide_path = export_segmented_wide(df_wide, config.PROCESSED_DATA_DIR, example_experiment, example_round, example_phase, example_metric, format='parquet')
print(f"Wide exportado para: {wide_path}")

# Teste 6: Carregar de volta e conferir
print("Testando load_dataframe no export long...")
df_long_loaded = load_dataframe(long_path)
print(f"Shape do segmento long recarregado: {df_long_loaded.shape}")

print("Testando load_dataframe no export wide...")
df_wide_loaded = load_dataframe(wide_path)
print(f"Shape do wide recarregado: {df_wide_loaded.shape}")

print("Teste de segmentação e exportação concluído.")
