#!/usr/bin/env python3
"""
Module: pipeline.py
Description: Sistema de orquestração do pipeline completo de análise de séries temporais multi-tenant.

Este módulo implementa classes e funções para a execução ordenada de todo o fluxo de análise:
1. Ingestão de dados
2. Processamento e exportação de DataFrames
3. Análise descritiva
4. Análise de correlação
5. Análise de causalidade
6. Geração de relatórios

A classe Pipeline é o ponto central de orquestração, enquanto PipelineStage serve como 
base para as diferentes etapas do pipeline.
"""
import os
import sys
import argparse
import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable, Union, Tuple
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import yaml
import networkx as nx

# Importações dos módulos do projeto
from src.data_ingestion import ingest_experiment_data
from src.data_export import save_dataframe, load_dataframe
from src.data_segment import filter_long_df, get_wide_format_for_analysis
from src.analysis_descriptive import compute_descriptive_stats, plot_metric_timeseries_multi_tenant
from src.analysis_descriptive import plot_metric_barplot_by_phase, plot_metric_boxplot
from src.analysis_correlation import compute_correlation_matrix, plot_correlation_heatmap
from src.analysis_correlation import compute_covariance_matrix, plot_covariance_heatmap
from src.analysis_causality import plot_causality_graph
from src.report_generation import generate_tenant_metrics, generate_tenant_ranking_plot, generate_markdown_report
from src.insight_aggregation import aggregate_tenant_insights, generate_comparative_table, plot_comparative_metrics
from src import config
from src.parse_config import load_parse_config, get_selected_metrics, get_selected_tenants, get_selected_rounds
from src.parse_config import get_data_root, get_processed_data_dir

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("pipeline")


class PipelineStage:
    """Classe base para estágios do pipeline."""
    
    def __init__(self, name: str, description: str):
        """
        Inicializa um estágio do pipeline.
        
        Args:
            name: Nome do estágio.
            description: Descrição do propósito do estágio.
        """
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"pipeline.{name}")
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa este estágio do pipeline.
        
        Args:
            context: Dicionário com o contexto atual do pipeline.
                     Contém dados compartilhados entre estágios.
        
        Returns:
            Dicionário atualizado com o resultado deste estágio.
        """
        self.logger.info(f"Iniciando estágio: {self.name}")
        start_time = time.time()
        
        result = self._execute_implementation(context)
        
        elapsed_time = time.time() - start_time
        self.logger.info(f"Estágio {self.name} concluído em {elapsed_time:.2f} segundos")
        
        return result
    
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Implementação específica do estágio. Deve ser sobrescrita por classes derivadas.
        
        Args:
            context: Contexto atual do pipeline.
            
        Returns:
            Contexto atualizado após execução do estágio.
        """
        raise NotImplementedError("Subclasses devem implementar este método.")


class DataIngestionStage(PipelineStage):
    """Estágio para ingestão de dados brutos e consolidação em DataFrame long."""
    
    def __init__(self):
        super().__init__("data_ingestion", "Ingestão e consolidação de dados")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a ingestão de dados conforme configuração.
        
        Args:
            context: Contexto do pipeline com configurações.
            
        Returns:
            Contexto atualizado com DataFrame long.
        """
        # Obter configurações
        config_dict = context.get('config', {})
        data_root = context.get('data_root', config.DATA_ROOT)
        selected_metrics = config_dict.get('selected_metrics')
        selected_tenants = config_dict.get('selected_tenants')
        selected_rounds = config_dict.get('selected_rounds')
        force_reprocess = context.get('force_reprocess', False)
        
        processed_data_dir = context.get('processed_data_dir', config.PROCESSED_DATA_DIR)
        consolidated_long_path = os.path.join(processed_data_dir, 'consolidated_long.parquet')
        
        # Verifica se já existe um arquivo parquet consolidado e se não estamos forçando reprocessamento
        if os.path.exists(consolidated_long_path) and not force_reprocess:
            self.logger.info(f"Arquivo de dados consolidados encontrado: {consolidated_long_path}")
            self.logger.info("Carregando dados já processados... (use --force-reprocess para reprocessar)")
            
            try:
                df_long = load_dataframe(consolidated_long_path)
                self.logger.info(f"Dados carregados com sucesso. Total de registros: {len(df_long)}")
                
                # Adicionar ao contexto
                context['df_long'] = df_long
                context['consolidated_long_path'] = consolidated_long_path
                
                return context
            except Exception as e:
                self.logger.error(f"Erro ao carregar arquivo existente: {e}")
                self.logger.info("Continuando com reprocessamento dos dados...")
        
        self.logger.info(f"Iniciando ingestão de dados de: {data_root}")
        
        # Ingerir dados
        df_long = ingest_experiment_data(
            data_root=data_root,
            selected_metrics=selected_metrics,
            selected_tenants=selected_tenants,
            selected_rounds=selected_rounds
        )
        
        self.logger.info(f"Ingestão concluída. Total de registros: {len(df_long)}")
        
        # Adicionar ao contexto
        context['df_long'] = df_long
        
        return context


class DataExportStage(PipelineStage):
    """Estágio para exportação do DataFrame consolidado."""
    
    def __init__(self):
        super().__init__("data_export", "Exportação do DataFrame consolidado")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Exporta o DataFrame long para arquivo.
        
        Args:
            context: Contexto com DataFrame long.
            
        Returns:
            Contexto atualizado.
        """
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame não disponível para exportação.")
            return context
            
        processed_data_dir = context.get('processed_data_dir', config.PROCESSED_DATA_DIR)
        consolidated_long_path = os.path.join(processed_data_dir, 'consolidated_long.parquet')
        
        # Criar diretório se não existir
        os.makedirs(os.path.dirname(consolidated_long_path), exist_ok=True)
        
        # Exportar
        save_dataframe(df_long, consolidated_long_path, format='parquet')
        self.logger.info(f"DataFrame exportado para: {consolidated_long_path}")
        
        # Atualizar contexto
        context['consolidated_long_path'] = consolidated_long_path
        
        return context


class DescriptiveAnalysisStage(PipelineStage):
    """Estágio para análise descritiva."""
    
    def __init__(self):
        super().__init__("descriptive_analysis", "Análise descritiva das métricas")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa análise descritiva e gera visualizações.
        
        Args:
            context: Contexto com DataFrame long.
            
        Returns:
            Contexto atualizado com resultados e caminhos dos plots.
        """
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame não disponível para análise descritiva.")
            return context
            
        # Configurar diretório de saída
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'descriptive')
        os.makedirs(out_dir, exist_ok=True)
        
        # Calcular estatísticas descritivas
        self.logger.info("Calculando estatísticas descritivas...")
        stats = compute_descriptive_stats(df_long)
        context['descriptive_stats'] = stats
        
        # Gerar plots para cada combinação de métrica/round/fase
        self.logger.info("Gerando visualizações...")
        
        # Obter combinações únicas de experimento/round/fase/métrica
        experiments = df_long['experiment_id'].unique()
        
        plot_paths = []
        
        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                for metric in round_df['metric_name'].unique():
                    # Gerar plots agregados por fase
                    try:
                        path = plot_metric_barplot_by_phase(round_df, metric, round_id, out_dir)
                        plot_paths.append(path)
                        
                        path = plot_metric_boxplot(round_df, metric, round_id, out_dir)
                        plot_paths.append(path)
                    except Exception as e:
                        self.logger.error(f"Erro ao gerar plots agregados para {metric}, {round_id}: {e}")
                    
                    # Gerar plots por fase
                    for phase in round_df['experimental_phase'].unique():
                        try:
                            path = plot_metric_timeseries_multi_tenant(
                                round_df, metric, phase, round_id, out_dir
                            )
                            plot_paths.append(path)
                        except Exception as e:
                            self.logger.error(f"Erro ao gerar plot para {metric}, {phase}, {round_id}: {e}")
        
        # Atualizar contexto
        context['descriptive_plot_paths'] = plot_paths
        
        return context


class CorrelationAnalysisStage(PipelineStage):
    """Estágio para análise de correlação e covariância."""
    
    def __init__(self):
        super().__init__("correlation_analysis", "Análise de correlação entre tenants")
        # Adicionando um logger específico para este estágio para facilitar o debug
        self.stage_logger = logging.getLogger(f"pipeline.CorrelationAnalysisStage")

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa análise de correlação e gera visualizações.

        Args:
            context: Contexto com DataFrame long.

        Returns:
            Contexto atualizado com resultados e caminhos dos plots.
        """
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.stage_logger.error("DataFrame não disponível para análise de correlação.")
            return context

        # Configurar diretório de saída
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'correlation')
        os.makedirs(out_dir, exist_ok=True)

        # Estruturas para armazenar resultados
        correlation_matrices = {}
        covariance_matrices = {}
        plot_paths = []

        # Obter combinações únicas de experimento/round/fase/métrica
        experiments = df_long['experiment_id'].unique()
        self.stage_logger.debug(f"Experimentos encontrados: {experiments}")

        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            self.stage_logger.debug(f"Processando experiment_id: {experiment_id}")
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                self.stage_logger.debug(f"Processando round_id: {round_id} para o experimento {experiment_id}")
                for metric in round_df['metric_name'].unique():
                    self.stage_logger.debug(f"Processando metric: {metric} para o round {round_id}")
                    metric_key = f"{experiment_id}:{round_id}:{metric}"
                    correlation_matrices[metric_key] = {}
                    covariance_matrices[metric_key] = {}

                    for phase in round_df['experimental_phase'].unique():
                        self.stage_logger.debug(f"Processando phase: {phase} para a métrica {metric}")
                        phase_key = f"{metric_key}:{phase}"

                        # Calcular correlação
                        try:
                            # Log antes de chamar a função que pode causar o erro
                            self.stage_logger.info(f"Calculando correlação para metric: {metric}, phase: {phase}, round_id: {round_id}")
                            
                            # Filtrar DataFrame para a fase específica
                            phase_specific_df = round_df[round_df['experimental_phase'] == phase]

                            if phase_specific_df.empty:
                                self.stage_logger.warning(f"DataFrame vazio para {metric}, {phase}, {round_id} após filtrar por fase. Pulando correlação.")
                                continue

                            # Calcular correlação e gerar visualização
                            corr = compute_correlation_matrix(phase_specific_df, metric, phase, round_id)
                            correlation_matrices[metric_key][phase] = corr

                            # Verificar resultado e gerar plot de correlação
                            if corr is not None and not corr.empty:
                                try:
                                    self.stage_logger.info(f"Gerando plot de correlação para {metric}, {phase}, {round_id}")
                                    path = plot_correlation_heatmap(corr, metric, phase, round_id, out_dir)
                                    if path:
                                        plot_paths.append(path)
                                        self.stage_logger.info(f"Plot de correlação gerado com sucesso: {path}")
                                    else:
                                        self.stage_logger.warning(f"Falha ao gerar plot de correlação para {metric}, {phase}, {round_id}")
                                except Exception as plot_err:
                                    self.stage_logger.error(f"Erro ao gerar plot de correlação para {metric}, {phase}, {round_id}: {plot_err}", exc_info=True)
                            elif corr is None:
                                self.stage_logger.warning(f"Matriz de correlação retornou None para {metric}, {phase}, {round_id}")
                            else: # corr.empty é True
                                self.stage_logger.warning(f"Matriz de correlação vazia para {metric}, {phase}, {round_id}")

                        except Exception as e:
                            self.stage_logger.error(f"Erro inesperado ao calcular correlação para {metric}, {phase}, {round_id}: {e}", exc_info=True)

                        # Calcular covariância
                        try:
                            self.stage_logger.info(f"Calculando covariância para metric: {metric}, phase: {phase}, round_id: {round_id}.")
                            phase_specific_df_cov = round_df[round_df['experimental_phase'] == phase]
                            if phase_specific_df_cov.empty:
                                self.stage_logger.warning(f"DataFrame vazio para {metric}, {phase}, {round_id} após filtrar por fase. Pulando covariância.")
                                continue

                            cov = compute_covariance_matrix(phase_specific_df_cov, metric, phase, round_id)
                            covariance_matrices[metric_key][phase] = cov

                            if cov is not None and not cov.empty:
                                path = plot_covariance_heatmap(cov, metric, phase, round_id, out_dir)
                                plot_paths.append(path)
                            elif cov is None:
                                self.stage_logger.warning(f"Matriz de covariância retornou None para {metric}, {phase}, {round_id}")
                            else:
                                self.stage_logger.warning(f"Matriz de covariância vazia para {metric}, {phase}, {round_id}")
                        except Exception as e_cov:
                            self.stage_logger.error(f"Erro inesperado ao calcular covariância para {metric}, {phase}, {round_id}: {e_cov}", exc_info=True)
        
        # Atualizar contexto
        context['correlation_matrices'] = correlation_matrices
        context['covariance_matrices'] = covariance_matrices
        context['correlation_plot_paths'] = plot_paths
        
        self.stage_logger.info(f"Análise de correlação concluída. Gerados {len(plot_paths)} plots.")
        
        return context


class CausalityAnalysisStage(PipelineStage):
    """
    Estágio para análise de causalidade entre séries temporais de diferentes tenants.
    
    Implementa análises de causalidade usando:
    - Teste de causalidade de Granger
    - Transfer Entropy (TE)
    
    A análise é realizada para cada combinação de métrica, fase experimental e round,
    gerando matrizes de causalidade e visualizações em formato de grafo.
    """
    
    def __init__(self):
        super().__init__("causality_analysis", "Análise de causalidade (Granger e Transfer Entropy)")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa análise de causalidade e gera visualizações.
        
        Args:
            context: Contexto com DataFrame long.
            
        Returns:
            Contexto atualizado com resultados e caminhos dos plots.
        """
        from src.analysis_causality import CausalityAnalyzer, plot_causality_graph
        from tqdm import tqdm
        
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame não disponível para análise de causalidade.")
            return context
            
        # Configurar diretório de saída
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'causality')
        os.makedirs(out_dir, exist_ok=True)
        
        # Parâmetros da configuração
        config = context.get('config', {})
        causality_config = config.get('causality', {})
        granger_max_lag = causality_config.get('granger_max_lag', 5)
        granger_threshold = causality_config.get('granger_threshold', 0.05)
        te_bins = causality_config.get('transfer_entropy_bins', 8)
        
        # Estruturas para armazenar resultados
        granger_matrices = {}
        te_matrices = {}
        plot_paths = []
        
        # Inicializa analisador de causalidade
        analyzer = CausalityAnalyzer(df_long)
        
        # Processa cada combinação de experimento, round, métrica e fase
        experiments = df_long['experiment_id'].unique()
        
        for experiment_id in experiments:
            self.logger.info(f"Processando experimento: {experiment_id}")
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            
            for round_id in exp_df['round_id'].unique():
                # Filtra para o round específico
                round_df = exp_df[exp_df['round_id'] == round_id]
                
                for metric in tqdm(round_df['metric_name'].unique(), desc=f"Métricas em {round_id}"):
                    for phase in round_df['experimental_phase'].unique():
                        self.logger.info(f"Calculando causalidade para {metric}, {phase}, {round_id}")
                        try:
                            # Filtrar dados para essa fase/métrica/round
                            phase_df = filter_long_df(
                                round_df,
                                phase=phase,
                                metric=metric,
                                round_id=round_id
                            )
                            
                            # Verificar se há dados suficientes
                            if phase_df.empty or phase_df['tenant_id'].nunique() < 2:
                                self.logger.warning(f"Dados insuficientes para {metric}, {phase}, {round_id}")
                                continue
                                
                            # Calcular causalidade de Granger
                            granger_matrix = analyzer.compute_granger_matrix(
                                metric=metric, 
                                phase=phase, 
                                round_id=round_id,
                                maxlag=granger_max_lag
                            )
                            
                            # Calcular Transfer Entropy
                            te_matrix = analyzer.compute_transfer_entropy_matrix(
                                metric=metric,
                                phase=phase,
                                round_id=round_id,
                                bins=te_bins
                            )
                            
                            # Armazenar resultados
                            result_key = f"{experiment_id}:{round_id}:{phase}:{metric}"
                            granger_matrices[result_key] = granger_matrix
                            te_matrices[result_key] = te_matrix
                            
                            # Gerar visualizações para causalidade de Granger
                            if not granger_matrix.empty and not granger_matrix.isna().all().all():
                                granger_out_path = os.path.join(
                                    out_dir, 
                                    f"causality_graph_granger_{metric}_{phase}_{round_id}.png"
                                )
                                plot_causality_graph(
                                    granger_matrix, 
                                    granger_out_path,
                                    threshold=granger_threshold, 
                                    directed=True,
                                    metric=metric
                                )
                                plot_paths.append(granger_out_path)
                                
                            # Gerar visualizações para Transfer Entropy
                            if not te_matrix.empty and not te_matrix.isna().all().all():
                                # Para TE, valores mais altos = mais causalidade, então usamos threshold inverso
                                te_threshold = 0.1  # Limiar mínimo para considerar relação causal via TE
                                te_out_path = os.path.join(
                                    out_dir, 
                                    f"causality_graph_te_{metric}_{phase}_{round_id}.png"
                                )
                                
                                # Para visualização, invertemos para formato compatível com plot_causality_graph
                                # (que espera valores menores = mais causalidade)
                                te_viz_matrix = 1.0 / (te_matrix + 1.0)
                                
                                plot_causality_graph(
                                    te_viz_matrix,
                                    te_out_path,
                                    threshold=0.9,  # Threshold para visualização (menores valores = mais causalidade)
                                    directed=True,
                                    metric=f"{metric} (TE)"
                                )
                                plot_paths.append(te_out_path)
                        except Exception as e:
                            self.logger.error(f"Erro ao processar causalidade para {metric}, {phase}, {round_id}: {e}")
        
        # Atualizar contexto
        context['granger_matrices'] = granger_matrices
        context['te_matrices'] = te_matrices
        context['causality_plot_paths'] = plot_paths
        
        return context


class PhaseComparisonStage(PipelineStage):
    """
    Estágio para análise comparativa entre diferentes fases experimentais.
    
    Este estágio implementa:
    1. Comparação de métricas entre fases (baseline, ataque, recuperação)
    2. Identificação de mudanças significativas
    3. Visualizações comparativas
    """
    
    def __init__(self):
        super().__init__("phase_comparison", "Análise comparativa entre fases experimentais")
    
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa análise comparativa entre fases experimentais.
        
        Args:
            context: Contexto com DataFrame long e resultados de estágios anteriores
            
        Returns:
            Contexto atualizado com resultados comparativos
        """
        from src.analysis_phase_comparison import PhaseComparisonAnalyzer
        
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame não disponível para análise comparativa de fases.")
            return context
        
        # Configurar diretório de saída
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'phase_comparison')
        os.makedirs(out_dir, exist_ok=True)
        
        # Inicializar analisador de comparação de fases
        analyzer = PhaseComparisonAnalyzer(df_long)
        
        # Estruturas para armazenar resultados
        phase_comparison_results = {}
        plot_paths = []
        
        # Processar cada experimento e round
        experiments = df_long['experiment_id'].unique()
        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                
                # Verificar se há pelo menos 2 fases para comparação
                phases = round_df['experimental_phase'].unique()
                if len(phases) < 2:
                    self.logger.warning(f"Menos de 2 fases disponíveis para {experiment_id}, {round_id}. Pulando comparação.")
                    continue
                
                # Processar cada métrica
                for metric in round_df['metric_name'].unique():
                    try:
                        # Realizar análise comparativa
                        self.logger.info(f"Comparando fases para {metric}, {round_id}")
                        
                        stats_df = analyzer.analyze_metric_across_phases(
                            metric=metric,
                            round_id=round_id,
                            output_dir=out_dir
                        )
                        
                        # Armazenar resultados
                        result_key = f"{experiment_id}:{round_id}:{metric}"
                        phase_comparison_results[result_key] = stats_df
                        
                        # Adicionar caminho do plot gerado
                        plot_path = os.path.join(out_dir, f'phase_comparison_{metric}_{round_id}.png')
                        if os.path.exists(plot_path):
                            plot_paths.append(plot_path)
                            
                    except Exception as e:
                        self.logger.error(f"Erro ao comparar fases para {metric}, {round_id}: {e}", exc_info=True)
        
        # Atualizar contexto
        context['phase_comparison_results'] = phase_comparison_results
        context['phase_comparison_plot_paths'] = plot_paths
        
        return context


class ReportGenerationStage(PipelineStage):
    """
    Estágio para geração de relatório final consolidado.
    
    Agrega insights de todos os estágios anteriores e gera:
    1. Relatório textual com principais descobertas
    2. Tabela comparativa inter-tenant
    3. Identificação de "tenants barulhentos" baseado em critérios objetivos
    """
    
    def __init__(self):
        super().__init__("report_generation", "Geração de relatório final")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera relatório final com insights consolidados.
        
        Args:
            context: Contexto com resultados de todos os estágios anteriores.
            
        Returns:
            Contexto atualizado com caminho do relatório e tabela comparativa.
        """
        from src.report_generation import (
            generate_tenant_metrics,
            generate_markdown_report,
            generate_tenant_ranking_plot
        )
        
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame não disponível para geração de relatório.")
            return context
        
        # Configurar diretório de saída
        report_dir = os.path.join(context.get('output_dir', 'outputs'), 'reports')
        os.makedirs(report_dir, exist_ok=True)
        
        report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Recupera informações relevantes do contexto
        granger_matrices = context.get('granger_matrices', {})
        te_matrices = context.get('te_matrices', {})
        correlation_matrices = context.get('correlation_matrices', {})
        phase_comparison_results = context.get('phase_comparison_results', {})
        
        # Gerar métricas e ranking de tenants
        self.logger.info("Gerando métricas e ranking de tenants...")
        tenant_metrics = generate_tenant_metrics(
            granger_matrices,
            te_matrices, 
            correlation_matrices,
            phase_comparison_results
        )
        
        # Salvar a tabela de métricas
        metrics_table_path = os.path.join(report_dir, f"{report_filename}_tenant_metrics.csv")
        tenant_metrics.to_csv(metrics_table_path, index=False)
        
        # Criar visualização do ranking de tenants
        rank_plot_path = os.path.join(report_dir, f"{report_filename}_tenant_ranking.png")
        generate_tenant_ranking_plot(tenant_metrics, rank_plot_path)
        
        # Gerar relatório final em markdown
        self.logger.info("Gerando relatório markdown...")
        report_path = generate_markdown_report(
            tenant_metrics=tenant_metrics,
            context=context,
            rank_plot_path=rank_plot_path,
            metrics_table_path=metrics_table_path,
            out_dir=report_dir
        )
        
        # Atualizar contexto
        context['report_path'] = report_path
        context['tenant_metrics'] = tenant_metrics
        context['tenant_metrics_path'] = metrics_table_path
        context['tenant_ranking_path'] = rank_plot_path
        
        self.logger.info(f"Relatório completo gerado em: {report_path}")
        
        return context


class InsightAggregationStage(PipelineStage):
    """
    Estágio para agregação de insights e geração de tabela comparativa inter-tenant.
    Implementa as funcionalidades da Fase 3 do plano de trabalho.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        super().__init__(name="Insight Aggregation", description="Agregação de insights e comparativos inter-tenant")
        self.output_dir = output_dir
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Agrega insights de todos os tenants e gera tabela comparativa.
        
        Args:
            context: Contexto com resultados dos estágios anteriores
            
        Returns:
            Contexto atualizado com os resultados da agregação de insights
        """
        self.logger.info("Iniciando agregação de insights inter-tenant")
        
        if 'error' in context:
            self.logger.error(f"Erro em estágio anterior: {context['error']}")
            return context
            
        # Diretório de saída
        output_dir = self.output_dir or context.get('output_dir')
        if not output_dir:
            output_dir = os.path.join(os.getcwd(), 'outputs', 'insights')
        
        # Específico para insights    
        insights_dir = os.path.join(output_dir, 'insights')
        os.makedirs(insights_dir, exist_ok=True)
        
        # Verificar se temos os dados necessários
        if not all(key in context for key in ['tenant_metrics', 'phase_comparison_results']):
            self.logger.warning("Dados de métricas de tenant ou comparação de fases ausentes no contexto")
            # Tentar obter do relatório se disponível
            if 'report_context' not in context:
                self.logger.error("Não foi possível obter dados necessários para agregação de insights")
                context['error'] = "Dados necessários para agregação de insights não disponíveis"
                return context
            
            # Usar dados do relatório
            tenant_metrics = context['report_context'].get('tenant_metrics')
            phase_comparison_results = context['report_context'].get('phase_comparison_results')
        else:
            tenant_metrics = context.get('tenant_metrics')
            phase_comparison_results = context.get('phase_comparison_results')
            
        granger_matrices = context.get('granger_matrices', {})
        te_matrices = context.get('te_matrices', {})
        correlation_matrices = context.get('correlation_matrices', {})
        anomaly_metrics = context.get('anomaly_metrics', {})
        
        try:
            # Agregar insights para cada tenant
            self.logger.info("Agregando insights para cada tenant")
            tenant_insights = aggregate_tenant_insights(
                tenant_metrics=tenant_metrics,
                phase_comparison_results=phase_comparison_results,
                granger_matrices=granger_matrices,
                te_matrices=te_matrices,
                correlation_matrices=correlation_matrices,
                anomaly_metrics=anomaly_metrics
            )
            
            # Verificar se recebemos um erro da função de agregação
            if 'error_message' in tenant_insights:
                error_msg = tenant_insights['error_message']
                self.logger.error(f"Falha na agregação de insights: {error_msg}")
                context['error'] = error_msg
                return context
                
            context['tenant_insights'] = tenant_insights
            
            # Gerar tabela comparativa
            self.logger.info("Gerando tabela comparativa inter-tenant")
            comparative_table = generate_comparative_table(tenant_insights)
            context['comparative_table'] = comparative_table
            
            # Salvar tabela em CSV
            comparative_table_path = os.path.join(insights_dir, f"comparative_table_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            comparative_table.to_csv(comparative_table_path, index=False)
            context['comparative_table_path'] = comparative_table_path
            
            # Gerar visualização comparativa
            self.logger.info("Gerando visualização comparativa de métricas inter-tenant")
            comparative_metrics_path = plot_comparative_metrics(comparative_table, insights_dir)
            context['comparative_metrics_path'] = comparative_metrics_path
            
            # Salvar insights detalhados
            insights_json_file = os.path.join(insights_dir, f"tenant_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            import json
            
            # Converter dados para formato JSON serializável
            serializable_insights = {}
            for tenant, data in tenant_insights.items():
                serializable_insights[tenant] = {
                    k: (float(v) if isinstance(v, np.float32) or isinstance(v, np.float64) else v)
                    for k, v in data.items() 
                    if k != 'rank'  # Excluir rank por ser incompatível
                }
            
            with open(insights_json_file, 'w') as f:
                json.dump(serializable_insights, f, indent=2)
                
            context['insights_json_path'] = insights_json_file
            
            self.logger.info(f"Agregação de insights concluída. Resultados salvos em {insights_dir}")
            
        except Exception as e:
            self.logger.error(f"Erro durante agregação de insights: {str(e)}", exc_info=True)
            context['error'] = f"Erro na agregação de insights: {str(e)}"
        
        return context


class Pipeline:
    """Classe principal para orquestração do pipeline de análise."""

    def __init__(self, stages: List[PipelineStage], config_path: Optional[str] = None):
        """
        Inicializa o pipeline com uma lista de estágios.

        Args:
            stages: Lista de objetos PipelineStage a serem executados.
            config_path: Caminho opcional para o arquivo de configuração YAML.
        """
        self.stages = stages
        self.context: Dict[str, Any] = {}  # Contexto compartilhado entre estágios
        self.logger = logging.getLogger("pipeline.Pipeline")

        if config_path:
            self.configure_from_yaml(config_path)
        else:
            # Carregar configuração padrão se nenhum caminho for fornecido
            self._load_default_config()

    def configure_from_yaml(self, config_path: str) -> None:
        """
        Configura o pipeline a partir de um arquivo YAML.

        Args:
            config_path: Caminho para o arquivo de configuração YAML.
        """
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Atualiza o contexto com os dados do arquivo de configuração
            self.context['config'] = config_data
            self.logger.info(f"Configuração carregada de: {config_path}")

            # Aplicar configurações específicas, se necessário
            # Exemplo: self.context['data_root'] = config_data.get('data_root', config.DATA_ROOT)

        except FileNotFoundError:
            self.logger.error(f"Arquivo de configuração não encontrado: {config_path}")
            # Considerar levantar uma exceção ou usar configuração padrão
            self._load_default_config()
        except yaml.YAMLError as e:
            self.logger.error(f"Erro ao parsear arquivo YAML de configuração: {e}")
            # Considerar levantar uma exceção ou usar configuração padrão
            self._load_default_config()

    def _load_default_config(self) -> None:
        """Carrega uma configuração padrão para o pipeline."""
        self.logger.info("Carregando configuração padrão do pipeline.")
        self.context['config'] = {
            'data_root': config.DATA_ROOT,
            'processed_data_dir': config.PROCESSED_DATA_DIR,
            'output_dir': config.OUTPUT_DIR,
            'selected_metrics': None, # Ou valores padrão
            'selected_tenants': None, # Ou valores padrão
            'selected_rounds': None,  # Ou valores padrão
            'causality'