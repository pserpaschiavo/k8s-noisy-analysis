"""
Módulo de integração de análise de causalidade melhorada para o pipeline.
Este arquivo fornece um estágio de pipeline melhorado para substituir ou 
aumentar o CausalityAnalysisStage existente.
"""

import os
import logging
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import dos módulos de análise de causalidade
from src.analysis_causality import CausalityAnalyzer, plot_causality_graph
from src.improved_causality_graph import plot_improved_causality_graph, plot_consolidated_causality_graph

# Configurar logging
logger = logging.getLogger(__name__)

def generate_causality_visualizations(
    context: Dict[str, Any], 
    force_regenerate: bool = False
) -> Dict[str, Any]:
    """
    Gera visualizações melhoradas de causalidade a partir do contexto de pipeline existente.
    
    Args:
        context: Contexto do pipeline contendo granger_matrices, te_matrices e configurações
        force_regenerate: Se True, força a regeneração mesmo que as visualizações já existam
        
    Returns:
        Contexto atualizado com novos caminhos de visualizações
    """
    # Verificar se as matrizes de causalidade estão disponíveis
    granger_matrices = context.get('granger_matrices', {})
    te_matrices = context.get('te_matrices', {})
    
    if not granger_matrices and not te_matrices:
        logger.warning("Nenhuma matriz de causalidade encontrada no contexto para gerar visualizações")
        return context
        
    # Configurações
    config = context.get('config', {})
    causality_config = config.get('causality', {})
    granger_threshold = causality_config.get('granger_threshold', 0.05)
    te_threshold = causality_config.get('te_threshold', 0.1)
    
    # Diretório de saída para visualizações
    out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'causality')
    os.makedirs(out_dir, exist_ok=True)
    
    # Diretório específico para visualizações melhoradas
    improved_dir = os.path.join(out_dir, 'improved')
    os.makedirs(improved_dir, exist_ok=True)
    
    # Diretório para visualizações consolidadas
    consolidated_dir = os.path.join(out_dir, 'consolidated')
    os.makedirs(consolidated_dir, exist_ok=True)
    
    # Listas para armazenar caminhos das visualizações
    improved_plot_paths = []
    consolidated_plot_paths = []
    
    # Métricas a consolidar (para grafo multi-métrica)
    metrics_by_experiment_phase_round = {}
    
    # Gerar visualizações melhoradas para matrizes individuais
    logger.info("Gerando visualizações de causalidade melhoradas...")
    
    # Processa matrizes de Granger
    for key, matrix in granger_matrices.items():
        if matrix.empty or matrix.isna().all().all():
            continue
            
        try:
            # Extrair informações do key (experiment_id:round_id:phase:metric)
            parts = key.split(':')
            if len(parts) != 4:
                logger.warning(f"Formato de chave inválido: {key}")
                continue
                
            experiment_id, round_id, phase, metric = parts
            
            # Chave para agrupar métricas para o mesmo experimento/fase/round
            exp_phase_round = f"{experiment_id}:{round_id}:{phase}"
            
            # Adicionar matriz ao dicionário para consolidação posterior
            if exp_phase_round not in metrics_by_experiment_phase_round:
                metrics_by_experiment_phase_round[exp_phase_round] = {}
            metrics_by_experiment_phase_round[exp_phase_round][metric] = matrix
            
            # Caminho para visualização individual melhorada
            improved_out_path = os.path.join(
                improved_dir, 
                f"improved_granger_{metric}_{phase}_{round_id}.png"
            )
            
            # Verificar se a visualização já existe
            if not os.path.exists(improved_out_path) or force_regenerate:
                # Gerar visualização melhorada
                plot_improved_causality_graph(
                    matrix,
                    improved_out_path,
                    threshold=granger_threshold,
                    directed=True,
                    metric=metric
                )
                logger.info(f"Visualização melhorada de Granger gerada: {improved_out_path}")
                improved_plot_paths.append(improved_out_path)
            else:
                logger.info(f"Visualização melhorada já existe: {improved_out_path}")
                improved_plot_paths.append(improved_out_path)
                
        except Exception as e:
            logger.error(f"Erro ao gerar visualização melhorada para {key}: {e}")
            
    # Processa matrizes de Transfer Entropy
    for key, matrix in te_matrices.items():
        if matrix.empty or matrix.isna().all().all():
            continue
            
        try:
            # Extrair informações do key
            parts = key.split(':')
            if len(parts) != 4:
                logger.warning(f"Formato de chave inválido: {key}")
                continue
                
            experiment_id, round_id, phase, metric = parts
            
            # Chave para agrupar métricas
            exp_phase_round = f"{experiment_id}:{round_id}:{phase}"
            
            # Para TE, precisamos inverter a matriz para compatibilidade com plot_improved_causality_graph
            # que espera valores menores = mais causalidade
            te_viz_matrix = 1.0 / (matrix + 1.0)
            
            # Caminho para visualização individual melhorada
            improved_out_path = os.path.join(
                improved_dir, 
                f"improved_te_{metric}_{phase}_{round_id}.png"
            )
            
            # Verificar se a visualização já existe
            if not os.path.exists(improved_out_path) or force_regenerate:
                # Gerar visualização melhorada
                plot_improved_causality_graph(
                    te_viz_matrix,
                    improved_out_path,
                    threshold=0.9,  # Threshold para visualização invertida de TE
                    directed=True,
                    metric=f"{metric} (TE)"
                )
                logger.info(f"Visualização melhorada de TE gerada: {improved_out_path}")
                improved_plot_paths.append(improved_out_path)
            else:
                logger.info(f"Visualização melhorada já existe: {improved_out_path}")
                improved_plot_paths.append(improved_out_path)
                
        except Exception as e:
            logger.error(f"Erro ao gerar visualização melhorada para {key}: {e}")
    
    # Gerar visualizações consolidadas (multi-métrica)
    logger.info("Gerando visualizações consolidadas multi-métrica...")
    for exp_phase_round, metric_matrices in metrics_by_experiment_phase_round.items():
        if not metric_matrices:
            continue
            
        try:
            # Extrair informações do exp_phase_round
            parts = exp_phase_round.split(':')
            if len(parts) != 3:
                logger.warning(f"Formato de chave inválido: {exp_phase_round}")
                continue
                
            experiment_id, round_id, phase = parts
            
            # Caminho para visualização consolidada
            consolidated_out_path = os.path.join(
                consolidated_dir, 
                f"consolidated_{phase}_{round_id}.png"
            )
            
            # Verificar se a visualização já existe
            if not os.path.exists(consolidated_out_path) or force_regenerate:
                # Gerar visualização consolidada
                plot_consolidated_causality_graph(
                    metric_matrices,
                    consolidated_out_path,
                    threshold=granger_threshold,
                    directed=True,
                    phase=phase,
                    round_id=round_id,
                    title_prefix=f'Análise de Causalidade Multi-Métrica'
                )
                logger.info(f"Visualização consolidada gerada: {consolidated_out_path}")
                consolidated_plot_paths.append(consolidated_out_path)
            else:
                logger.info(f"Visualização consolidada já existe: {consolidated_out_path}")
                consolidated_plot_paths.append(consolidated_out_path)
                
        except Exception as e:
            logger.error(f"Erro ao gerar visualização consolidada para {exp_phase_round}: {e}")
    
    # Atualizar contexto com novos caminhos
    context['improved_causality_plot_paths'] = improved_plot_paths
    context['consolidated_causality_plot_paths'] = consolidated_plot_paths
    
    # Adicionar todas as visualizações à lista principal
    causality_plot_paths = context.get('causality_plot_paths', [])
    causality_plot_paths.extend(improved_plot_paths)
    causality_plot_paths.extend(consolidated_plot_paths)
    context['causality_plot_paths'] = causality_plot_paths
    
    return context


class ImprovedCausalityStage:
    """
    Estágio de pipeline melhorado para análise de causalidade.
    Pode ser usado após o CausalityAnalysisStage original para gerar visualizações melhoradas.
    """
    
    def __init__(self, force_regenerate=False):
        """
        Inicializa o estágio.
        
        Args:
            force_regenerate: Se True, força a regeneração de visualizações mesmo que já existam
        """
        self.logger = logging.getLogger(__name__)
        self.force_regenerate = force_regenerate
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa o estágio de visualização de causalidade melhorada.
        
        Args:
            context: Contexto do pipeline
            
        Returns:
            Contexto atualizado
        """
        self.logger.info("Executando estágio de causalidade melhorada...")
        
        try:
            # Gerar visualizações melhoradas
            context = generate_causality_visualizations(context, self.force_regenerate)
            self.logger.info("Estágio de causalidade melhorada concluído com sucesso")
        except Exception as e:
            self.logger.error(f"Erro ao executar estágio de causalidade melhorada: {e}")
            
        return context
