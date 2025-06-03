"""
Script de teste para análise descritiva: estatísticas e plots.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_export import load_dataframe
from src.analysis_descriptive import compute_descriptive_stats, plot_metric_barplot_by_phase, plot_metric_boxplot, plot_metric_timeseries_multi_tenant, plot_metric_timeseries_multi_tenant_all_phases
from src import config

# Caminho do DataFrame consolidado
df_long = load_dataframe(os.path.join(config.PROCESSED_DATA_DIR, 'consolidated_long.parquet'))

# Estatísticas descritivas agrupadas por tenant, métrica, fase, round
print("Calculando estatísticas descritivas...")
stats = compute_descriptive_stats(df_long)
print(stats.head())

# Diretório de saída para plots
out_dir = os.path.join('outputs', 'plots', 'descriptive')

# Gerar plots multi-tenant para cada métrica do round-1 do experimento demo-experiment-1-round
round_id = 'round-1'
tenants = ['tenant-a', 'tenant-b', 'tenant-c', 'tenant-d']
df_demo = df_long[(df_long['experiment_id'] == 'demo-experiment-1-round') & (df_long['round_id'] == round_id) & (df_long['tenant_id'].isin(tenants))]
metrics = df_demo['metric_name'].unique()
for metric in metrics:
    print(f"Gerando barplot para {metric} em {round_id}...")
    path = plot_metric_barplot_by_phase(df_demo, metric, round_id, out_dir)
    print(f"Barplot salvo em: {path}")
    print(f"Gerando boxplot para {metric} em {round_id}...")
    path = plot_metric_boxplot(df_demo, metric, round_id, out_dir)
    print(f"Boxplot salvo em: {path}")
    print(f"Gerando timeseries multi-tenant agregando todas as fases para {metric}, {round_id}...")
    path = plot_metric_timeseries_multi_tenant_all_phases(df_demo, metric, round_id, out_dir)
    print(f"Timeseries multi-tenant (todas as fases) salvo em: {path}")
    for phase in df_demo['experimental_phase'].unique():
        print(f"Gerando timeseries multi-tenant para {metric}, {phase}, {round_id}...")
        path = plot_metric_timeseries_multi_tenant(df_demo, metric, phase, round_id, out_dir)
        print(f"Timeseries multi-tenant salvo em: {path}")

print("Análise descritiva concluída.")
