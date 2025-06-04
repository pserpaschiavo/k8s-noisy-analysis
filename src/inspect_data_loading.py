"""
Script para diagnosticar a carga de dados do pipeline.
"""
import os
import yaml
from src.data_ingestion import list_experiments, list_rounds, list_phases, list_tenants, list_metric_files

def inspect_data_structure(data_root):
    """Inspeciona a estrutura de diretórios de dados e imprime o que seria carregado."""
    print(f"Inspecionando estrutura de dados em: {data_root}\n")
    
    # Lista experimentos
    experiments = list_experiments(data_root)
    print(f"Encontrados {len(experiments)} experimentos:")
    for exp_path in experiments:
        exp_name = os.path.basename(exp_path)
        print(f"  - Experimento: {exp_name} ({exp_path})")
        
        # Lista rounds
        rounds = list_rounds(exp_path)
        print(f"    Contém {len(rounds)} rounds:")
        for round_path in rounds:
            round_name = os.path.basename(round_path)
            print(f"      - Round: {round_name}")
            
            # Lista fases
            phases = list_phases(round_path)
            print(f"        Contém {len(phases)} fases:")
            for phase_path in phases:
                phase_name = os.path.basename(phase_path)
                print(f"          - Fase: {phase_name}")
                
                # Lista tenants
                tenants = list_tenants(phase_path)
                print(f"            Contém {len(tenants)} tenants:")
                tenant_names = [os.path.basename(t) for t in tenants]
                print(f"              {', '.join(tenant_names)}")
                
                # Para o primeiro tenant, lista métricas disponíveis
                if tenants:
                    metrics = list_metric_files(tenants[0])
                    metric_names = [os.path.splitext(os.path.basename(m))[0] for m in metrics]
                    print(f"            Métricas disponíveis para {os.path.basename(tenants[0])}: {', '.join(metric_names)}")
    
    print("\nInspeção concluída.")

if __name__ == "__main__":
    # Carrega configuração
    config_path = "/home/phil/Projects/gpt-nn-analysis/config/pipeline_config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            data_root = config.get('data_root')
            if data_root:
                inspect_data_structure(data_root)
            else:
                print("data_root não encontrado na configuração")
    else:
        print(f"Arquivo de configuração não encontrado: {config_path}")
