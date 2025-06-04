#!/usr/bin/env python3
"""
Module: pipeline_with_sliding_window.py
Description: Versão customizada do pipeline que inclui análises de janelas deslizantes.

Este módulo expande o pipeline.py padrão adicionando o estágio de análise de janelas deslizantes
para testar e validar sua integração com o fluxo completo de análise.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from typing import Dict, Any, Optional

# Importação do pipeline base
from src.pipeline import Pipeline, PipelineStage, parse_arguments
from src.pipeline import DataIngestionStage, DataExportStage, DescriptiveAnalysisStage
from src.pipeline import CorrelationAnalysisStage, CausalityAnalysisStage, PhaseComparisonStage
from src.pipeline import ReportGenerationStage, InsightAggregationStage

# Importação dos estágios especializados
from src.analysis_sliding_window import SlidingWindowStage
from src.analysis_multi_round import MultiRoundAnalysisStage

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_sliding_window.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def create_pipeline_with_sliding_window() -> Pipeline:
    """
    Cria uma instância de Pipeline com o estágio de janelas deslizantes incluído.
    
    Returns:
        Pipeline configurado com todos os estágios, incluindo janelas deslizantes.
    """
    # Cria uma lista de estágios que inclui o estágio de janelas deslizantes
    stages = [
        DataIngestionStage(),
        DataExportStage(),
        DescriptiveAnalysisStage(),
        CorrelationAnalysisStage(),
        CausalityAnalysisStage(),
        SlidingWindowStage(),  # Adiciona o estágio de janelas deslizantes
        MultiRoundAnalysisStage(),  # Adiciona o estágio de análise multi-round
        PhaseComparisonStage(),
        InsightAggregationStage(),  # Adiciona o estágio de agregação de insights
        ReportGenerationStage()
    ]
    
    # Inicializa o pipeline com os estágios
    pipeline = Pipeline(stages=stages)
    
    return pipeline

def main():
    """Função principal para executar o pipeline com análise de janelas deslizantes."""
    logger = logging.getLogger("pipeline_sliding_window")
    logger.info("Iniciando pipeline com análise de janelas deslizantes")
    
    # Parse argumentos da linha de comando
    parser = argparse.ArgumentParser(description="Pipeline com análise de janelas deslizantes")
    parser.add_argument("--config", help="Caminho para arquivo de configuração YAML")
    parser.add_argument("--data-root", help="Diretório raiz dos dados de experimento")
    parser.add_argument("--output-dir", help="Diretório para salvar resultados")
    parser.add_argument("--selected-metrics", nargs='+', help="Lista de métricas a analisar")
    parser.add_argument("--selected-tenants", nargs='+', help="Lista de tenants a analisar")
    parser.add_argument("--selected-rounds", nargs='+', help="Lista de rounds a analisar")
    
    args = parser.parse_args()
    
    # Criar pipeline customizado com análise de janelas deslizantes
    pipeline = create_pipeline_with_sliding_window()
    
    # Configurar o pipeline conforme os argumentos
    if args.config:
        pipeline.configure_from_yaml(args.config)
    
    # Sobrescrever configurações específicas, se fornecidas
    if args.data_root:
        pipeline.context["config"]["data_root"] = args.data_root
    if args.output_dir:
        pipeline.context["config"]["output_dir"] = args.output_dir
    if args.selected_metrics:
        pipeline.context["config"]["selected_metrics"] = args.selected_metrics
    if args.selected_tenants:
        pipeline.context["config"]["selected_tenants"] = args.selected_tenants
    if args.selected_rounds:
        pipeline.context["config"]["selected_rounds"] = args.selected_rounds
    
    # Executar o pipeline
    results = pipeline.run()
    
    # Verificar se houve erro
    if results and "error" in results:
        logger.error(f"Pipeline falhou: {results['error']}")
        return 1
    
    # Log de estatísticas finais
    logger.info(f"Pipeline concluído em {results.get('elapsed_time', 0) if results else 0} segundos")
    logger.info(f"Diretório de outputs: {pipeline.context['config'].get('output_dir', 'outputs')}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
