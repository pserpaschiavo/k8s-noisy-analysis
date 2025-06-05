#!/usr/bin/env python3
"""
Script para debug da implementação do parâmetro experiment_folder
"""
import os
import sys
import yaml
import logging

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Função principal de debug"""
    config_path = "config/pipeline_config.yaml"
    
    # 1. Verificar o arquivo de configuração
    logger.info(f"Verificando arquivo de configuração: {config_path}")
    
    if not os.path.exists(config_path):
        logger.error(f"Arquivo de configuração não encontrado: {config_path}")
        return
        
    # Carregar configuração
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"Configuração carregada: {config}")
    
    # 2. Verificar parâmetros específicos
    data_root = config.get('data_root')
    experiment_folder = config.get('experiment_folder')
    
    logger.info(f"data_root: {data_root}")
    logger.info(f"experiment_folder: {experiment_folder}")
    
    # 3. Verificar caminhos
    if data_root and os.path.exists(data_root):
        logger.info(f"data_root existe: {os.path.exists(data_root)}")
        logger.info(f"Conteúdo de data_root:")
        for item in os.listdir(data_root):
            logger.info(f" - {item}")
    else:
        logger.error(f"data_root não existe: {data_root}")
    
    # 4. Verificar caminho combinado
    if experiment_folder:
        experiment_path = os.path.join(data_root, experiment_folder)
        logger.info(f"Caminho completo: {experiment_path}")
        logger.info(f"Caminho completo existe: {os.path.exists(experiment_path)}")
        
        if os.path.exists(experiment_path):
            logger.info(f"Conteúdo de experiment_path:")
            for item in os.listdir(experiment_path):
                logger.info(f" - {item}")
    
    # 5. Verificar dados em src/config.py
    logger.info("Verificando config.py:")
    from src import config
    logger.info(f"config.DATA_ROOT = {config.DATA_ROOT}")
    logger.info(f"config.DEFAULT_EXPERIMENT_FOLDER = {config.DEFAULT_EXPERIMENT_FOLDER}")
    logger.info(f"config.EXPERIMENT_DIR = {config.EXPERIMENT_DIR}")
    
    # 6. Verificar função auxiliar em parse_config.py
    logger.info("Verificando parse_config.py:")
    from src.parse_config import get_experiment_folder, get_experiment_dir
    
    exp_folder = get_experiment_folder(config)
    exp_dir = get_experiment_dir(config)
    
    logger.info(f"get_experiment_folder resultado: {exp_folder}")
    logger.info(f"get_experiment_dir resultado: {exp_dir}")
    
if __name__ == '__main__':
    main()
