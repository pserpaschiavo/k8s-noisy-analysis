#!/usr/bin/env python3
"""
Script to run the multi-tenant time series analysis pipeline.

Usage:
    ./run_pipeline.py [--config CONFIG_PATH] 
                     [--data-root DATA_ROOT] 
                     [--output-dir OUTPUT_DIR]
                     [--selected-metrics METRICS [METRICS ...]]
                     [--selected-tenants TENANTS [TENANTS ...]]
                     [--selected-rounds ROUNDS [ROUNDS ...]]

Examples:
    ./run_pipeline.py --config config/pipeline_config.yaml
    ./run_pipeline.py --selected-metrics cpu_usage memory_usage --selected-tenants tenant-a tenant-b
"""
import sys
import logging
import argparse
import pandas as pd
from typing import Dict, Any
import os
import yaml
from copy import deepcopy

# Importar todas as classes de estágio do pipeline
from src.data_ingestion import DataIngestionStage
from src.analysis_descriptive import DescriptiveAnalysisStage
from src.analysis_correlation import CorrelationAnalysisStage
from src.analysis_causality import CausalityAnalysisStage
from src.analysis_impact import ImpactAnalysisStage
from src.analysis_phase_comparison import PhaseComparisonStage
from src.analysis_multi_round import MultiRoundAnalysisStage
from src.report_generation import ReportGenerationStage
from src.config import PipelineConfig

def main():
    """
    Ponto de entrada principal para a execução do pipeline de análise.
    """
    parser = argparse.ArgumentParser(description="Executa o pipeline de análise de dados do k8s-noisy.")
    parser.add_argument('--config', type=str, required=True, help='Caminho para o arquivo de configuração YAML.')
    args = parser.parse_args()

    try:
        # Configuração do logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )

        # Carregar configuração principal
        main_config = PipelineConfig(args.config)
        logging.info(f"Configuração principal carregada de: {args.config}")

        # Obter rodadas selecionadas
        selected_rounds = main_config.get_selected_rounds()
        if not selected_rounds:
            logging.error("Nenhuma 'selected_rounds' encontrada na configuração. A análise multi-round não pode prosseguir.")
            sys.exit(1)

        # --- Execução Individual por Rodada ---
        for round_id in selected_rounds:
            logging.info(f"--- INICIANDO PROCESSAMENTO DA RODADA: {round_id} ---")
            
            # Criar uma configuração específica para a rodada a partir do dicionário da config principal
            round_config_dict = deepcopy(main_config.config_data)
            
            # Define o diretório de saída completo para a rodada
            round_output_dir = os.path.join(
                main_config.get_output_dir(), 
                main_config.get_experiment_name(), 
                round_id
            )
            round_config_dict['output_dir'] = round_output_dir
            
            # Esvazia o nome do experimento para evitar que os estágios o adicionem novamente
            round_config_dict['experiment_name'] = '' 
            round_config_dict['selected_rounds'] = [round_id]

            # Salvar a configuração da rodada em um arquivo temporário
            temp_config_path = os.path.join('config', f"temp_config_{round_id}.yaml")
            with open(temp_config_path, 'w') as f:
                yaml.dump(round_config_dict, f)

            # Criar uma instância de PipelineConfig para a rodada
            round_config = PipelineConfig(temp_config_path)

            # Inicializar o contexto do pipeline para a rodada
            pipeline_context: Dict[str, Any] = {"config": round_config}

            # Estágios a serem executados para cada rodada individualmente
            single_round_stages = [
                DataIngestionStage(round_config),
                DescriptiveAnalysisStage(round_config),
                ImpactAnalysisStage(round_config),
                CorrelationAnalysisStage(round_config),
                CausalityAnalysisStage(round_config),
                PhaseComparisonStage(round_config),
            ]

            for stage in single_round_stages:
                pipeline_context = stage.execute(pipeline_context)
            
            # Limpar o arquivo de configuração temporário
            os.remove(temp_config_path)
            
            logging.info(f"--- PROCESSAMENTO DA RODADA {round_id} CONCLUÍDO ---")

        # --- Consolidação e Análise Multi-Round ---
        logging.info("--- INICIANDO ANÁLISE MULTI-ROUND CONSOLIDADA ---")
        multi_round_context = {"config": main_config}
        multi_round_stage = MultiRoundAnalysisStage(main_config)
        multi_round_context = multi_round_stage.execute(multi_round_context)

        # --- Geração do Relatório Final ---
        logging.info("--- GERANDO RELATÓRIO FINAL CONSOLIDADO ---")
        report_stage = ReportGenerationStage(
            config=main_config,
            descriptive_stats=pd.DataFrame(),
            impact_results=pd.DataFrame(),
            correlation_results=pd.DataFrame(),
            causality_results=pd.DataFrame(),
            phase_comparison_results=pd.DataFrame(),
            multi_round_stage=multi_round_stage
        )
        report_stage.execute(multi_round_context)

        logging.info("Pipeline multi-round executado com sucesso.")

    except Exception as e:
        logging.error(f"Ocorreu um erro fatal no pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
