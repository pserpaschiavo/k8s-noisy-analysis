#!/usr/bin/env python3
"""
Script: run_pipeline_3_rounds.py
Descrição: Script conveniente para executar o pipeline com o experimento de 3 rounds.

Este script executa o pipeline configurado para o experimento demo-experiment-3-rounds
usando o suporte a parâmetro experiment_folder.
"""

import os
import sys
import logging
import argparse
from datetime import datetime

# Diretório base do projeto
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_3rounds.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline_3rounds")

def main():
    """Função principal para execução com o experimento de 3 rounds"""
    parser = argparse.ArgumentParser(description="Pipeline para experimento de 3 rounds")
    parser.add_argument("--force-reprocess", action="store_true", help="Força o reprocessamento dos dados brutos")
    parser.add_argument("--no-sliding-window", action="store_true", help="Desativa análise com janelas deslizantes")
    parser.add_argument("--output-dir", help="Diretório personalizado para saída dos resultados")
    
    args = parser.parse_args()
    
    # Verificar se o arquivo de configuração existe
    config_path = os.path.join(PROJECT_DIR, "config", "pipeline_config_3rounds.yaml")
    if not os.path.exists(config_path):
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        sys.exit(1)
    
    # Importar a função wrapper para experimento
    from run_pipeline_with_experiment import get_full_experiment_path
    
    # Obter caminho completo do experimento
    experiment_path = get_full_experiment_path(config_path)
    if not experiment_path:
        logger.error("Não foi possível determinar o caminho do experimento. Abortando.")
        sys.exit(1)
    
    # Importar funcionalidade de pipeline
    from src.run_unified_pipeline import run_unified_pipeline
    
    # Definir diretório de saída padrão se não foi especificado
    output_dir = args.output_dir or os.path.join(PROJECT_DIR, "outputs", "demo-experiment-3-rounds")
    
    # Contexto extra para indicar que o experiment_folder já foi aplicado
    extra_context = {
        'experiment_folder_applied': True  
    }
    
    # Executar o pipeline unificado
    logger.info(f"Iniciando pipeline para experimento de 3 rounds...")
    logger.info(f"- Caminho do experimento: {experiment_path}")
    logger.info(f"- Saída: {output_dir}")
    
    run_unified_pipeline(
        config_path=config_path,
        data_root=experiment_path,
        output_dir=output_dir,
        run_sliding_window=not args.no_sliding_window,
        run_multi_round=True,  # Sempre ativar análise multi-round para o experimento de 3 rounds
        force_reprocess=args.force_reprocess,
        extra_context=extra_context
    )
    
    logger.info("Pipeline para experimento de 3 rounds concluído com sucesso.")

if __name__ == "__main__":
    start_time = datetime.now()
    try:
        main()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Execução total concluída em {duration:.2f} segundos")
    except Exception as e:
        logger.error(f"Erro durante execução: {str(e)}", exc_info=True)
        sys.exit(1)
