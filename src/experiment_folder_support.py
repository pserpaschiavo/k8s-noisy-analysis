#!/usr/bin/env python3
"""
Script: experiment_folder_support.py
Descrição: Aplica a modificação necessária para suportar o parâmetro experiment_folder.

Este script modifica o arquivo run_unified_pipeline.py para adicionar suporte ao
parâmetro experiment_folder do arquivo de configuração YAML.
"""

import os
import sys
import logging
import yaml

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_experiment_folder_patch():
    """
    Modifica o código para suportar o parâmetro experiment_folder.
    """
    try:
        # 1. Carrega a configuração para verificar se experiment_folder está definido
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'config', 'pipeline_config.yaml')
        
        if not os.path.exists(config_path):
            logger.error(f"Arquivo de configuração não encontrado: {config_path}")
            return False
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        experiment_folder = config.get('experiment_folder')
        if not experiment_folder:
            logger.info("Parâmetro experiment_folder não definido na configuração. Nenhuma mudança necessária.")
            return True
            
        # 2. Aplica a modificação ao arquivo de configuração padrão
        config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
        
        if not os.path.exists(config_file):
            logger.error(f"Arquivo de configuração padrão não encontrado: {config_file}")
            return False
            
        with open(config_file, 'r') as f:
            config_content = f.read()
            
        # Se DEFAULT_EXPERIMENT_FOLDER já existe, atualiza apenas seu valor
        if 'DEFAULT_EXPERIMENT_FOLDER =' in config_content:
            logger.info("DEFAULT_EXPERIMENT_FOLDER já está definido. Atualizando seu valor.")
            import re
            config_content = re.sub(
                r"DEFAULT_EXPERIMENT_FOLDER\s*=\s*['\"](.*)['\"]", 
                f"DEFAULT_EXPERIMENT_FOLDER = '{experiment_folder}'", 
                config_content
            )
        else:
            # Senão, adiciona a nova variável e atualiza EXPERIMENT_DIR
            logger.info("Adicionando suporte para DEFAULT_EXPERIMENT_FOLDER.")
            config_content = config_content.replace(
                "DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo-data')",
                "DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo-data')\n"
                f"DEFAULT_EXPERIMENT_FOLDER = '{experiment_folder}'"
            )
            config_content = config_content.replace(
                "EXPERIMENT_DIR = os.path.join(DATA_ROOT, 'demo-experiment-1-round')",
                "EXPERIMENT_DIR = os.path.join(DATA_ROOT, DEFAULT_EXPERIMENT_FOLDER)"
            )
            
        # Salva o arquivo modificado
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Configuração atualizada com experiment_folder: {experiment_folder}")
        
        # 3. Modifica o arquivo DataIngestionStage do pipeline
        pipeline_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pipeline.py')
        
        if not os.path.exists(pipeline_file):
            logger.error(f"Arquivo do pipeline não encontrado: {pipeline_file}")
            return False
            
        # Lê o conteúdo do arquivo
        with open(pipeline_file, 'r') as f:
            pipeline_content = f.readlines()
        
        # Procura a função _execute_implementation da classe DataIngestionStage
        data_ingestion_found = False
        modified = False
        execute_impl_found = False
        data_root_line_idx = None
        
        for i, line in enumerate(pipeline_content):
            if not data_ingestion_found and 'class DataIngestionStage' in line:
                data_ingestion_found = True
            
            if data_ingestion_found and '_execute_implementation' in line:
                execute_impl_found = True
            
            # Encontra a linha que define data_root
            if execute_impl_found and 'data_root = ' in line and 'context.get' in line:
                data_root_line_idx = i
                break
        
        # Se encontrou a linha, insere código para usar experiment_folder
        if data_root_line_idx is not None:
            # Prepara o novo código para inserir
            new_code = [
                '        data_root = context.get(\'data_root\', config.DATA_ROOT)\n',
                '        # Se tiver experiment_folder definido, usa ele para construir o caminho completo do experimento\n',
                '        experiment_folder = config_dict.get(\'experiment_folder\')\n',
                '        if experiment_folder:\n',
                '            data_root = os.path.join(data_root, experiment_folder)\n',
                '            self.logger.info(f"Usando caminho de experimento: {data_root} (data_root + experiment_folder)")\n',
                '\n'
            ]
            
            # Substitui a linha original pelo novo código
            pipeline_content[data_root_line_idx:data_root_line_idx+1] = new_code
            modified = True
        
        # Salva o arquivo modificado se houve alterações
        if modified:
            with open(pipeline_file, 'w') as f:
                f.writelines(pipeline_content)
            logger.info("Arquivo pipeline.py modificado com sucesso.")
        else:
            logger.warning("Não foi possível localizar ponto de modificação em pipeline.py.")
        
        return True
        
    except Exception as e:
        logger.error(f"Erro ao aplicar patch para experiment_folder: {str(e)}", exc_info=True)
        return False

if __name__ == '__main__':
    success = apply_experiment_folder_patch()
    if success:
        logger.info("Patch aplicado com sucesso!")
    else:
        logger.error("Falha ao aplicar o patch.")
        sys.exit(1)
