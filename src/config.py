"""
Module: config.py
Description: Central configuration for pipeline paths, default selections, and parameters.
"""
import os
import yaml
from typing import Dict, Any, Optional, List

class PipelineConfig:
    """
    Classe para carregar e acessar a configuração do pipeline a partir de um arquivo YAML.
    """
    def __init__(self, config_path: str):
        """
        Inicializa o objeto de configuração.

        Args:
            config_path (str): O caminho para o arquivo de configuração YAML.
        """
        with open(config_path, 'r') as f:
            self.config_data: Dict[str, Any] = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """
        Obtém um valor da configuração.
        """
        return self.config_data.get(key, default)

    def get_base_output_dir(self) -> str:
        """
        Retorna o diretório de saída base definido na configuração.
        """
        return self.get('output_dir', 'outputs')

    def get_experiment_name(self) -> str:
        """
        Retorna o nome do experimento.
        """
        return self.get('experiment_name', 'default_experiment')

    def get_output_dir(self, stage_name: Optional[str] = None) -> str:
        """
        Cria e retorna o diretório de saída para um estágio específico do pipeline.
        Se stage_name for None, retorna o diretório base do experimento.
        """
        base_output_dir = self.get_base_output_dir()
        experiment_name = self.get_experiment_name()
        
        if stage_name:
            stage_output_dir = os.path.join(base_output_dir, experiment_name, stage_name)
        else:
            stage_output_dir = os.path.join(base_output_dir, experiment_name)
            
        os.makedirs(stage_output_dir, exist_ok=True)
        return stage_output_dir

    def get_output_dir_for_round(self, stage_name: str, round_id: str) -> str:
        """
        Cria e retorna o diretório de saída para um estágio específico dentro de uma rodada.
        A estrutura será: <base_output_dir>/<experiment_name>/<round_id>/<stage_name>
        """
        experiment_output_dir = self.get_output_dir() # Retorna a pasta base do experimento
        round_stage_dir = os.path.join(experiment_output_dir, round_id, stage_name)
        os.makedirs(round_stage_dir, exist_ok=True)
        return round_stage_dir

    def get_processed_data_path(self) -> Optional[str]:
        """
        Retorna o caminho para o arquivo de dados processados (Parquet).
        """
        return self.get('processed_data_path')

    def get_data_root(self) -> str:
        """
        Retorna o diretório raiz dos dados brutos do experimento.
        """
        return self.get('data_root', 'exp_data')

    def get_selected_metrics(self) -> Optional[List[str]]:
        """
        Retorna a lista de métricas selecionadas.
        """
        return self.get('selected_metrics')

    def get_selected_tenants(self) -> Optional[List[str]]:
        """
        Retorna a lista de tenants selecionados.
        """
        return self.get('selected_tenants')

    def get_selected_rounds(self) -> Optional[List[str]]:
        """
        Retorna a lista de rounds selecionados.
        """
        return self.get('selected_rounds')

    def get_selected_phases(self) -> Optional[List[str]]:
        """
        Retorna a lista de fases selecionadas.
        """
        return self.get('selected_phases')

    def get_metric_display_names(self) -> Dict[str, str]:
        """
        Retorna o mapeamento de nomes de métricas para nomes de exibição amigáveis.
        Retorna um dicionário vazio se não for encontrado.
        """
        visualization_config = self.get('visualization', {})
        return visualization_config.get('metric_display_names', {})

# Root directories
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'demo-data')
DEFAULT_EXPERIMENT_FOLDER = 'demo-experiment-1-round'
EXPERIMENT_DIR = os.path.join(DATA_ROOT, DEFAULT_EXPERIMENT_FOLDER)
PROCESSED_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs') # Added OUTPUT_DIR

# Default output paths
CONSOLIDATED_LONG_PATH = os.path.join(PROCESSED_DATA_DIR, 'consolidated_long.parquet')

# Default parse selections (can be overridden by user config)
DEFAULT_SELECTED_METRICS = None  # or list of metric names
DEFAULT_SELECTED_TENANTS = None  # or list of tenant names
DEFAULT_SELECTED_ROUNDS = None   # or list of round names

# Utility to ensure output directories exist
def ensure_directories():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure OUTPUT_DIR is created

