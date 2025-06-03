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
            self.logger.error("DataFrame não disponível para análise de correlação.")
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
        
        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                for metric in round_df['metric_name'].unique():
                    metric_key = f"{experiment_id}:{round_id}:{metric}"
                    correlation_matrices[metric_key] = {}
                    covariance_matrices[metric_key] = {}
                    
                    for phase in round_df['experimental_phase'].unique():
                        phase_key = f"{metric_key}:{phase}"
                        
                        # Calcular correlação
                        try:
                            corr = compute_correlation_matrix(round_df, metric, phase, round_id)
                            correlation_matrices[metric_key][phase] = corr
                            
                            if not corr.empty:
                                path = plot_correlation_heatmap(corr, metric, phase, round_id, out_dir)
                                plot_paths.append(path)
                        except Exception as e:
                            self.logger.error(f"Erro ao calcular correlação para {metric}, {phase}, {round_id}: {e}")
                        
                        # Calcular covariância
                        try:
                            cov = compute_covariance_matrix(round_df, metric, phase, round_id)
                            covariance_matrices[metric_key][phase] = cov
                            
                            if not cov.empty:
                                path = plot_covariance_heatmap(cov, metric, phase, round_id, out_dir)
                                plot_paths.append(path)
                        except Exception as e:
                            self.logger.error(f"Erro ao calcular covariância para {metric}, {phase}, {round_id}: {e}")
        
        # Atualizar contexto
        context['correlation_matrices'] = correlation_matrices
        context['covariance_matrices'] = covariance_matrices
        context['correlation_plot_paths'] = plot_paths
        
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


class Pipeline:
    """Classe principal para orquestrar a execução do pipeline completo."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa o pipeline.
        
        Args:
            config_path: Caminho para arquivo de configuração YAML. Se None, usa configuração padrão.
        """
        self.logger = logging.getLogger("pipeline.main")
        self.stages: List[PipelineStage] = []
        self.context: Dict[str, Any] = {'start_time': datetime.now()}
        
        # Carregar configuração
        self._load_configuration(config_path)
        
        # Configurar estágios padrão
        self._setup_default_stages()
    
    def _load_configuration(self, config_path: Optional[str]) -> None:
        """
        Carrega configuração do pipeline.
        
        Args:
            config_path: Caminho para arquivo de configuração.
        """
        # Valores padrão
        self.context['config'] = {}
        self.context['data_root'] = config.DATA_ROOT
        self.context['processed_data_dir'] = config.PROCESSED_DATA_DIR
        self.context['output_dir'] = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
        
        # Carregar de arquivo se especificado
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    pipeline_config = yaml.safe_load(f)
                    
                if pipeline_config:
                    self.context['config'] = pipeline_config
                    
                    # Atualizar caminhos se definidos
                    if 'data_root' in pipeline_config:
                        self.context['data_root'] = pipeline_config['data_root']
                    if 'processed_data_dir' in pipeline_config:
                        self.context['processed_data_dir'] = pipeline_config['processed_data_dir']
                    if 'output_dir' in pipeline_config:
                        self.context['output_dir'] = pipeline_config['output_dir']
                        
                self.logger.info(f"Configuração carregada de: {config_path}")
            except Exception as e:
                self.logger.error(f"Erro ao carregar configuração de {config_path}: {e}")
    
    def _setup_default_stages(self) -> None:
        """Configura os estágios padrão do pipeline."""
        self.stages = [
            DataIngestionStage(),
            DataExportStage(),
            DescriptiveAnalysisStage(),
            CorrelationAnalysisStage(),
            CausalityAnalysisStage(),
            PhaseComparisonStage(),
            ReportGenerationStage()
        ]
    
    def add_stage(self, stage: PipelineStage, position: Optional[int] = None) -> None:
        """
        Adiciona um estágio ao pipeline.
        
        Args:
            stage: Estágio a ser adicionado.
            position: Posição onde inserir. Se None, adiciona ao final.
        """
        if position is None:
            self.stages.append(stage)
        else:
            self.stages.insert(position, stage)
    
    def run(self) -> Dict[str, Any]:
        """
        Executa todos os estágios do pipeline em sequência.
        
        Returns:
            Contexto final com todos os resultados.
        """
        self.logger.info(f"Iniciando execução do pipeline com {len(self.stages)} estágios")
        start_time = time.time()
        
        try:
            # Executar cada estágio em sequência
            for stage in self.stages:
                self.context = stage.execute(self.context)
        except Exception as e:
            self.logger.error(f"Erro durante execução do pipeline: {str(e)}", exc_info=True)
            self.context['error'] = str(e)
        
        # Registrar tempo total
        elapsed_time = time.time() - start_time
        self.logger.info(f"Pipeline concluído em {elapsed_time:.2f} segundos")
        self.context['elapsed_time'] = elapsed_time
        self.context['end_time'] = datetime.now()
        
        return self.context


def parse_arguments():
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(description="Pipeline de análise de séries temporais multi-tenant")
    
    parser.add_argument(
        "--config", 
        type=str,
        help="Caminho para arquivo de configuração YAML"
    )
    parser.add_argument(
        "--data-root", 
        type=str,
        help="Diretório raiz dos dados brutos"
    )
    parser.add_argument(
        "--output-dir", 
        type=str,
        help="Diretório para outputs (plots, relatórios)"
    )
    parser.add_argument(
        "--selected-metrics", 
        nargs="+",
        help="Lista de métricas a serem processadas (ex: cpu_usage memory_usage)"
    )
    parser.add_argument(
        "--selected-tenants", 
        nargs="+",
        help="Lista de tenants a serem analisados (ex: tenant-a tenant-b)"
    )
    parser.add_argument(
        "--selected-rounds", 
        nargs="+",
        help="Lista de rounds a serem analisados (ex: round-1 round-2)"
    )
    
    return parser.parse_args()


def main():
    """Função principal para executar o pipeline."""
    # Parse argumentos da linha de comando
    args = parse_arguments()
    
    # Configurar arquivo de configuração
    config_path = None
    if args.config:
        config_path = args.config
    elif os.path.exists(os.path.join(config.CONFIG_DIR, 'pipeline_config.yaml')):
        config_path = os.path.join(config.CONFIG_DIR, 'pipeline_config.yaml')
    
    # Inicializar e executar pipeline
    pipeline = Pipeline(config_path)
    
    # Sobrescrever com argumentos da linha de comando
    if args.data_root:
        pipeline.context['data_root'] = args.data_root
    if args.output_dir:
        pipeline.context['output_dir'] = args.output_dir
    if args.selected_metrics:
        pipeline.context['config']['selected_metrics'] = args.selected_metrics
    if args.selected_tenants:
        pipeline.context['config']['selected_tenants'] = args.selected_tenants
    if args.selected_rounds:
        pipeline.context['config']['selected_rounds'] = args.selected_rounds
    
    # Executar pipeline
    result = pipeline.run()
    
    # Exibir resumo
    logger.info("=== Pipeline concluído ===")
    if 'error' in result:
        logger.error(f"Pipeline encontrou erro: {result['error']}")
        return 1
    
    logger.info(f"Tempo de execução: {result.get('elapsed_time', 0):.2f} segundos")
    
    if 'report_path' in result:
        logger.info(f"Relatório gerado: {result['report_path']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())