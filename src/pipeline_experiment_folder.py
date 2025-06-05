"""
Module: pipeline_experiment_folder.py
Description: Implementa a modificação do pipeline para utilizar o parâmetro experiment_folder.

Este módulo adiciona suporte para utilizar o parâmetro experiment_folder das configurações,
permitindo especificar qual pasta de experimento deve ser usada dentro do diretório de dados.
"""
import os
import sys
import logging
from typing import Dict, Any

# Configuração de logging
logger = logging.getLogger("pipeline_experiment_folder")

def apply_experiment_folder(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Aplica o parâmetro experiment_folder ao contexto, modificando o data_root
    para apontar para o diretório completo do experimento.
    
    Args:
        context: O contexto atual do pipeline
        
    Returns:
        O contexto atualizado com o data_root modificado se experiment_folder estiver definido
    """
    # Obter configurações
    config_dict = context.get('config', {})
    data_root = context.get('data_root')
    
    # Se não tiver data_root definido, tentar obter do config
    if data_root is None:
        data_root = config_dict.get('data_root')
        
    # Se ainda não tiver data_root, usar o padrão do config
    if data_root is None:
        from src import config
        data_root = config.DATA_ROOT
    
    # Verificar se temos experiment_folder definido na configuração
    experiment_folder = config_dict.get('experiment_folder')
    
    # Se temos experiment_folder, combinar com data_root para formar o caminho completo do experimento
    if experiment_folder:
        experiment_dir = os.path.join(data_root, experiment_folder)
        logger.info(f"Usando experiment_folder: {experiment_folder}")
        logger.info(f"Caminho completo do experimento: {experiment_dir}")
        
        # Atualizar data_root no contexto para apontar para o diretório completo do experimento
        context['data_root'] = experiment_dir
    
    return context

def patch_pipeline_run():
    """
    Aplica um monkey patch na função run da classe Pipeline para incluir o processamento
    do parâmetro experiment_folder.
    """
    from src.pipeline import Pipeline
    original_run = Pipeline.run
    
    def patched_run(self):
        """
        Versão modificada da função run que aplica o parâmetro experiment_folder
        antes de executar o pipeline normalmente.
        """
        # Aplicar experiment_folder ao contexto
        self.context = apply_experiment_folder(self.context)
        
        # Executar a função run original
        return original_run(self)
    
    # Aplicar o patch
    Pipeline.run = patched_run
    logger.info("Pipeline.run foi modificada para suportar o parâmetro experiment_folder")
