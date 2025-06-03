"""
Script de teste para análise de correlação entre tenants.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_export import load_dataframe
from src.analysis_correlation import compute_correlation_matrix, plot_correlation_heatmap, compute_covariance_matrix, plot_covariance_heatmap, compute_cross_correlation, plot_cross_correlation
from src import config

# Caminho do DataFrame consolidado
path = os.path.join(config.PROCESSED_DATA_DIR, 'consolidated_long.parquet')
df_long = load_dataframe(path)

# Diretório de saída para plots
out_dir = os.path.join('outputs', 'plots', 'correlation')

# Parâmetros de teste
round_id = 'round-1'
phases = df_long[(df_long['experiment_id'] == 'demo-experiment-1-round') & (df_long['round_id'] == round_id)]['experimental_phase'].unique()
metrics = df_long[(df_long['experiment_id'] == 'demo-experiment-1-round') & (df_long['round_id'] == round_id)]['metric_name'].unique()

for metric in metrics:
    for phase in phases:
        print(f"Calculando correlação para {metric}, {phase}, {round_id}...")
        corr = compute_correlation_matrix(df_long[df_long['experiment_id'] == 'demo-experiment-1-round'], metric, phase, round_id)
        print(corr)
        path = plot_correlation_heatmap(corr, metric, phase, round_id, out_dir)
        print(f"Heatmap salvo em: {path}")
        # Covariância
        print(f"Calculando covariância para {metric}, {phase}, {round_id}...")
        cov = compute_covariance_matrix(df_long[df_long['experiment_id'] == 'demo-experiment-1-round'], metric, phase, round_id)
        print(cov)
        path = plot_covariance_heatmap(cov, metric, phase, round_id, out_dir)
        print(f"Covariance heatmap salvo em: {path}")
        # Cross-correlation (apenas para os dois primeiros tenants)
        tenants = corr.columns.tolist()
        if len(tenants) >= 2:
            tenant_x, tenant_y = tenants[0], tenants[1]
            print(f"Calculando cross-correlation entre {tenant_x} e {tenant_y} para {metric}, {phase}, {round_id}...")
            crosscorr = compute_cross_correlation(df_long[df_long['experiment_id'] == 'demo-experiment-1-round'], metric, phase, round_id, tenant_x, tenant_y)
            print(crosscorr)
            path = plot_cross_correlation(crosscorr, metric, phase, round_id, tenant_x, tenant_y, out_dir)
            print(f"Cross-correlation plot salvo em: {path}")
print("Teste de correlação concluído.")
