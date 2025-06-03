"""
Script de teste para análise de causalidade (Granger) e visualização por grafo.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_export import load_dataframe
from src.analysis_causality import CausalityModule
from src import config
import numpy as np
import pandas as pd

# Caminho do DataFrame consolidado
path = os.path.join(config.PROCESSED_DATA_DIR, 'consolidated_long.parquet')
df_long = load_dataframe(path)

# Diretório de saída para plots
out_dir = os.path.join('outputs', 'plots', 'causality')

# Parâmetros de teste
round_id = 'round-1'
phases = df_long[(df_long['experiment_id'] == 'demo-experiment-1-round') & (df_long['round_id'] == round_id)]['experimental_phase'].unique()
metrics = ['cpu_usage', 'memory_usage']

# Gera grafos comparativos para cada fase (usando matrizes fake para visualização)
for phase in phases:
    print(f"--- Comparando métricas para {phase}, {round_id} ---")
    tenants = df_long[(df_long['experiment_id'] == 'demo-experiment-1-round') & (df_long['round_id'] == round_id) & (df_long['experimental_phase'] == phase)]['tenant_id'].unique()
    causality_matrices = {}
    for metric in metrics:
        # Matriz fake: p-valor baixo para algumas relações, alto para outras
        mat = pd.DataFrame(0.5, index=tenants, columns=tenants)
        for i, src in enumerate(tenants):
            for j, tgt in enumerate(tenants):
                if src != tgt and (i + j) % 2 == 0:
                    mat.loc[src, tgt] = 0.01  # Relação causal forte
        causality_matrices[metric] = mat
    module = CausalityModule(df_long)
    out_path = os.path.join(out_dir, f"causality_graph_granger_multi_{phase}_{round_id}_fake.png")
    module.visualizer.plot_causality_graph_multi(causality_matrices, out_path, threshold=0.1, directed=True)
    print(f"Causality graph multi-métrica salvo em: {out_path}")

# Exemplo: matriz de p-valores fake para visualização
for metric in metrics:
    for phase in phases:
        tenants = df_long[(df_long['experiment_id'] == 'demo-experiment-1-round') & (df_long['round_id'] == round_id) & (df_long['experimental_phase'] == phase)]['tenant_id'].unique()
        # Matriz fake: p-valor baixo para algumas relações, alto para outras
        mat = pd.DataFrame(0.5, index=tenants, columns=tenants)
        for i, src in enumerate(tenants):
            for j, tgt in enumerate(tenants):
                if src != tgt and (i + j) % 2 == 0:
                    mat.loc[src, tgt] = 0.01  # Relação causal forte
        module = CausalityModule(df_long)
        out_path = os.path.join(out_dir, f"causality_graph_granger_{metric}_{phase}_{round_id}_fake.png")
        module.visualizer.plot_causality_graph(mat, out_path, threshold=0.05, directed=True)
        print(f"Causality graph salvo em: {out_path}")
        # --- Transfer Entropy real ---
        print(f"Calculando Transfer Entropy para {metric}, {phase}, {round_id}...")
        out_path_te = module.run_transfer_entropy_and_plot(metric, phase, round_id, out_dir, bins=8, threshold=0.05)
        print(f"Transfer Entropy graph salvo em: {out_path_te}")
# Gera grafos comparativos de TE para cada fase
for phase in phases:
    print(f"--- Comparando TE para métricas em {phase}, {round_id} ---")
    module = CausalityModule(df_long)
    causality_matrices = {}
    for metric in metrics:
        mat = module.analyzer.compute_transfer_entropy_matrix(metric, phase, round_id, bins=8)
        causality_matrices[metric] = mat
    out_path = os.path.join(out_dir, f"causality_graph_te_multi_{phase}_{round_id}.png")
    # Ajusta threshold para 0.1 para garantir arestas visíveis
    module.visualizer.plot_causality_graph_multi(causality_matrices, out_path, threshold=0.1, directed=True)
    print(f"Causality TE graph multi-métrica salvo em: {out_path}")
print("Teste de visualização de causalidade concluído.")
