#!/usr/bin/env python3
"""
Module: utils.py
Description: General utilities for the analysis pipeline.
"""

import warnings
import contextlib
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)

@contextlib.contextmanager
def suppress_statsmodels_warnings():
    """
    Context manager to suppress repetitive statsmodels warnings.
    Filters specific warnings while keeping other important ones.
    """
    with warnings.catch_warnings():
        # Filter common statsmodels warnings
        warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters')
        warnings.filterwarnings('ignore', 'Value in x_0 detected')
        warnings.filterwarnings('ignore', message='.*flat prior.*')
        warnings.filterwarnings('ignore', message='.*distribution is not normalized.*')
        warnings.filterwarnings('ignore', message='.*divide by zero.*')
        warnings.filterwarnings('ignore', message='.*invalid value.*')
        
        # Suppress convergence warnings
        warnings.filterwarnings('ignore', message='.*Maximum Likelihood optimization failed.*')
        warnings.filterwarnings('ignore', message='.*The iteration limit.*')
        
        # Other common statistical warnings
        warnings.filterwarnings('ignore', message='.*p-value.*')
        
        # Matplotlib warnings - Mais agressivo na filtragem de warnings de fonte
        warnings.filterwarnings('ignore', category=UserWarning, message='.*findfont.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*font.*not found.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*family.*not found.*')
        warnings.filterwarnings('ignore', category=UserWarning, message='.*tight_layout.*')
        
        # Pandas warnings
        warnings.filterwarnings('ignore', message='.*A value is trying to be set on a copy of a slice.*')
        warnings.filterwarnings('ignore', message='.*SettingWithCopyWarning.*')
        
        yield

def configure_matplotlib(config=None):
    """
    Centraliza a configuração do matplotlib para evitar inconsistências.
    Deve ser chamado no início de cada script ou módulo que usa visualizações.
    
    Args:
        config: Configuração opcional do pipeline, para ler configurações específicas.
    """
    import matplotlib
    matplotlib.use('Agg')  # Use o backend não interativo Agg
    
    # Fontes padrão que geralmente estão disponíveis em sistemas Linux
    font_family = 'sans-serif'
    font_size = 12
    
    # Leia configurações do arquivo de configuração, se disponível
    if config and 'visualization' in config and 'fonts' in config['visualization']:
        font_config = config['visualization']['fonts']
        font_family = font_config.get('family', font_family)
        font_size = font_config.get('size', font_size)
    
    # Configuração de fontes para evitar warnings
    matplotlib.rcParams['font.family'] = font_family
    
    # Definir listas de fontes disponíveis em praticamente todos os sistemas
    matplotlib.rcParams['font.sans-serif'] = [
        'DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans', 
        'FreeSans', 'Ubuntu', 'Noto Sans', 'sans-serif'
    ]
    
    # Mesmo que usemos 'serif', forneça alternativas para Times New Roman
    matplotlib.rcParams['font.serif'] = [
        'DejaVu Serif', 'Liberation Serif', 'FreeSerif', 
        'Bitstream Vera Serif', 'Noto Serif', 'serif'
    ]
    
    # Configurações adicionais para evitar warnings
    matplotlib.rcParams['figure.max_open_warning'] = 100
    matplotlib.rcParams['font.size'] = font_size
    
    # Configurações para qualidade de saída
    plot_quality = "high"
    if config and 'visualization' in config and 'plot_quality' in config['visualization']:
        plot_quality = config['visualization']['plot_quality']
    
    if plot_quality == "high":
        matplotlib.rcParams['savefig.dpi'] = 300
        matplotlib.rcParams['figure.dpi'] = 120
    elif plot_quality == "medium":
        matplotlib.rcParams['savefig.dpi'] = 200
        matplotlib.rcParams['figure.dpi'] = 100
    else:
        matplotlib.rcParams['savefig.dpi'] = 100
        matplotlib.rcParams['figure.dpi'] = 80
    
    # Configuração para garantir que legendas não causem problemas de layout
    matplotlib.rcParams['legend.loc'] = 'best'


def validate_data_availability(config: dict) -> bool:
    """
    Performs a pre-flight check to validate that raw data files exist.

    This function checks for the existence of .csv files based on the pipeline
    configuration. It assumes a directory structure of:
    <data_root>/<round>/<phase>/<tenant>/<metric>.csv

    Args:
        config: The pipeline configuration dictionary.

    Returns:
        True if all expected files are found, False otherwise. The function
        logs warnings for missing files but does not raise an exception.
    """
    import os
    logger.info("Performing pre-flight check for raw data files...")
    
    data_root = config.get("data_root")
    if not data_root:
        logger.error("Pre-flight check failed: 'data_root' not specified in configuration.")
        return False

    selected_metrics = config.get("selected_metrics", [])
    selected_tenants = config.get("selected_tenants", [])
    selected_rounds = config.get("selected_rounds", [])

    # Phase names are based on docs/adaptation_plan_new_phases_en.md
    # The exact directory names are an assumption.
    phase_names = [
        "baseline", "cpu-noise", "memory-noise", "network-noise", 
        "disk-noise", "combined-noise", "recovery"
    ]

    if not all([selected_metrics, selected_tenants, selected_rounds]):
        logger.warning("Pre-flight check skipped: Missing one or more required fields in config: selected_metrics, selected_tenants, selected_rounds.")
        return True # Skip validation if config is incomplete

    missing_files = []
    total_expected = len(selected_rounds) * len(phase_names) * len(selected_tenants) * len(selected_metrics)
    
    # Usar a estrutura de diretório real em vez de assumir os nomes
    for round_name in selected_rounds:
        round_dir = os.path.join(data_root, round_name)
        if not os.path.exists(round_dir):
            logger.warning(f"Round directory not found: {round_dir}")
            continue
            
        # Buscar fases reais em vez de assumir os nomes
        actual_phases = [d for d in os.listdir(round_dir) 
                        if os.path.isdir(os.path.join(round_dir, d))]
        
        # Mapear fase real para nome canônico
        phase_mapping = {}
        for phase in actual_phases:
            canonical = normalize_phase_name(phase)
            if canonical:
                phase_mapping[canonical] = phase
        
        # Se não houver fases conhecidas, usar as pastas encontradas
        if not phase_mapping:
            phase_mapping = {phase: phase for phase in actual_phases}
            
        for canonical_phase, dir_phase in phase_mapping.items():
            phase_dir = os.path.join(round_dir, dir_phase)
            
            for tenant_name in selected_tenants:
                tenant_dir = os.path.join(phase_dir, tenant_name)
                if not os.path.exists(tenant_dir):
                    continue
                    
                for metric_name in selected_metrics:
                    file_path = os.path.join(tenant_dir, f"{metric_name}.csv")
                    if not os.path.exists(file_path):
                        missing_files.append(file_path)

    if missing_files:
        # Limitar o número de arquivos mostrados para evitar poluição do log
        missing_count = len(missing_files)
        if missing_count > 10:
            sample = missing_files[:5] + ["..."] + missing_files[-5:]
            logger.warning(f"Pre-flight check found {missing_count} missing data files out of {total_expected} expected. Sample:")
            for f in sample:
                logger.warning(f"  - {f}")
        else:
            logger.warning(f"Pre-flight check found {missing_count} missing data files:")
            for f in missing_files:
                logger.warning(f"  - {f}")
        
        logger.warning("Pipeline will continue, but ingestion may be incomplete.")
        return False
    else:
        logger.info("Pre-flight check successful. All expected raw data files were found.")
        return True

# Mapeamento Canônico de Fases
# Este dicionário define o mapeamento de vários possíveis nomes de fase para um único nome canônico.
# O nome canônico é a chave do dicionário.
CANONICAL_PHASE_MAPPING = {
    '1 - Baseline': ['baseline', '1 - baseline', 'Baseline'],
    '2 - CPU Noise': ['cpu-noise', '2 - cpu noise', 'CPU Noise'],
    '3 - Memory Noise': ['memory-noise', '3 - memory noise', 'Memory Noise'],
    '4 - Network Noise': ['network-noise', '4 - network noise', 'Network Noise'],
    '5 - Disk Noise': ['disk-noise', '5 - disk noise', 'Disk Noise', 'Disk I/O Noise'],
    '6 - Combined Noise': ['combined-noise', '6 - combined noise', 'Combined Noise'],
    '7 - Recovery': ['recovery', '7 - recovery', 'Recovery']
}

# Invertendo o mapeamento para busca rápida
_REVERSE_CANONICAL_MAPPING = {alias.lower(): canonical 
                             for canonical, aliases in CANONICAL_PHASE_MAPPING.items() 
                             for alias in aliases + [canonical]}

def normalize_phase_name(name: str) -> Optional[str]:
    """
    Normaliza um nome de fase para o formato canônico.

    Args:
        name: O nome da fase a ser normalizado.

    Returns:
        O nome da fase canônico ou None se não for encontrado.
    """
    return _REVERSE_CANONICAL_MAPPING.get(name.lower())
