#!/usr/bin/env python3
"""
Script: run_pipeline_with_experiment.py
Descrição: Script wrapper para executar o pipeline unificado utilizando o parâmetro experiment_folder.

Este script carrega a configuração, ajusta o data_root com o experiment_folder e executa o pipeline.
"""

import os
import sys
import logging
import argparse
import yaml
from datetime import datetime

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_with_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline_with_experiment")

def get_full_experiment_path(config_path: str):
    """
    Carrega o arquivo de configuração e retorna o caminho completo para o experimento,
    combinando data_root com experiment_folder se disponível.
    """
    
    print(f"DEBUG: Tentando abrir arquivo de configuração: {config_path}")
    print(f"DEBUG: Arquivo existe? {os.path.exists(config_path)}")
    
    try:
        # Carregar configuração
        with open(config_path, 'r') as f:
            config_content = f.read()
            print(f"DEBUG: Conteúdo do arquivo:\n{config_content}")
            config = yaml.safe_load(config_content)
            print(f"DEBUG: Config carregada: {config}")
        
        # Obter data_root e experiment_folder
        data_root = config.get('data_root')
        experiment_folder = config.get('experiment_folder')
        
        print(f"DEBUG: data_root = {data_root}")
        print(f"DEBUG: experiment_folder = {experiment_folder}")
        
        if not data_root:
            logger.warning("data_root não encontrado na configuração.")
            print("DEBUG: data_root não encontrado")
            return None
        
        # Se tiver experiment_folder, combinar com data_root
        if experiment_folder:
            experiment_path = os.path.join(data_root, experiment_folder)
            print(f"DEBUG: Caminho completo = {experiment_path}")
            logger.info(f"Caminho completo do experimento: {experiment_path} (data_root + experiment_folder)")
            return experiment_path
        else:
            print(f"DEBUG: Usando data_root diretamente: {data_root}")
            logger.info(f"experiment_folder não especificado. Usando data_root diretamente: {data_root}")
            return data_root
            
    except Exception as e:
        print(f"DEBUG: Erro ao processar arquivo: {str(e)}")
        logger.error(f"Erro ao processar arquivo de configuração: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(description="Pipeline com suporte a experiment_folder")
    parser.add_argument("--config", default="config/pipeline_config.yaml", help="Caminho para arquivo de configuração YAML")
    parser.add_argument("--no-sliding-window", action="store_true", help="Desativa análise com janelas deslizantes")
    parser.add_argument("--no-multi-round", action="store_true", help="Desativa análise multi-round")
    parser.add_argument("--force-reprocess", action="store_true", help="Força o reprocessamento dos dados brutos")
    parser.add_argument("--input-parquet-path", help="Caminho para um arquivo parquet existente para ser usado diretamente")
    parser.add_argument("--output-dir", help="Diretório para salvar resultados")
    
    args = parser.parse_args()
    
    # Obter caminho completo do experimento
    experiment_path = get_full_experiment_path(args.config)
    
    if not experiment_path:
        logger.error("Não foi possível determinar o caminho do experimento. Abortando.")
        sys.exit(1)
    
    # Importar o pipeline após determinar o caminho do experimento
    from src.run_unified_pipeline import run_unified_pipeline
    
    # Executar o pipeline com o data_root apontando para o caminho completo do experimento
    logger.info(f"Executando pipeline com data_root={experiment_path}")
    
    # Carregar a configuração para passar ao pipeline
    with open(args.config, 'r') as f:
        import yaml
        config = yaml.safe_load(f)
    
    # Criar contexto adicional para indicar que o experiment_folder já foi aplicado
    extra_context = {
        'experiment_folder_applied': True  # Marca que o experiment_folder já foi aplicado ao data_root
    }
    
    run_unified_pipeline(
        config_path=args.config,
        data_root=experiment_path,  # Usar o caminho completo do experimento
        output_dir=args.output_dir,
        run_sliding_window=not args.no_sliding_window,
        run_multi_round=not args.no_multi_round,
        force_reprocess=args.force_reprocess,
        input_parquet_path=args.input_parquet_path,
        extra_context=extra_context  # Passar o contexto adicional
    )
    
    logger.info("Pipeline concluído.")

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
