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
        force_reprocess = context.get('force_reprocess', False)
        
        processed_data_dir = context.get('processed_data_dir', config.PROCESSED_DATA_DIR)
        
        # Verificar se há um caminho de parquet de entrada especificado no config
        input_parquet_path = config_dict.get('input_parquet_path')
        
        # Determinar o nome do arquivo parquet de saída
        output_parquet_name = config_dict.get('output_parquet_name', 'consolidated_long.parquet')
        consolidated_long_path = os.path.join(processed_data_dir, output_parquet_name)
        
        # Caso 1: Arquivo de entrada parquet especificado - carrega diretamente
        if input_parquet_path and os.path.exists(input_parquet_path) and not force_reprocess:
            self.logger.info(f"Usando parquet específico de entrada: {input_parquet_path}")
            
            try:
                from src.data_ingestion import load_from_parquet
                df_long = load_from_parquet(input_parquet_path)
                self.logger.info(f"Dados carregados com sucesso. Total de registros: {len(df_long)}")
                
                # Adicionar ao contexto
                context['df_long'] = df_long
                context['consolidated_long_path'] = input_parquet_path
                
                return context
            except Exception as e:
                self.logger.error(f"Erro ao carregar arquivo parquet de entrada: {e}")
                self.logger.info("Continuando com a verificação de dados consolidados ou reprocessamento...")
        
        # Caso 2: Verificar se já existe um arquivo parquet consolidado
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
        
        # Caso 3: Processar dados brutos
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
        context['consolidated_long_path'] = consolidated_long_path
        
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
            
        # Obter configurações
        config_dict = context.get('config', {})
        processed_data_dir = context.get('processed_data_dir', config.PROCESSED_DATA_DIR)
        
        # Usar nome do arquivo de saída definido na configuração
        output_parquet_name = config_dict.get('output_parquet_name', 'consolidated_long.parquet')
        consolidated_long_path = os.path.join(processed_data_dir, output_parquet_name)
        
        # Verificar se estamos usando um arquivo de entrada específico
        input_parquet_path = config_dict.get('input_parquet_path')
        if input_parquet_path and os.path.exists(input_parquet_path) and context.get('consolidated_long_path') == input_parquet_path:
            self.logger.info(f"Usando arquivo parquet de entrada existente: {input_parquet_path}")
            # Não precisamos salvar novamente se estamos usando o arquivo de entrada diretamente
            return context
        
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
    """Estágio para análise de causalidade."""
    
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
        df_long = context.get('df_long')
        if df_long is None or df_long.empty:
            self.logger.error("DataFrame não disponível para análise de causalidade.")
            return context
            
        # Configurar diretório de saída
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'plots', 'causality')
        os.makedirs(out_dir, exist_ok=True)
        
        # Estruturas para armazenar resultados
        granger_matrices = {}
        te_matrices = {}
        plot_paths = []
        
        # Para simplificar o exemplo, vamos apenas gerar alguns grafos de causalidade
        # Na implementação real, você faria os cálculos de causalidade aqui
        
        # Exemplo: gerar grafos para métricas e fases
        experiments = df_long['experiment_id'].unique()
        
        for experiment_id in experiments:
            exp_df = df_long[df_long['experiment_id'] == experiment_id]
            for round_id in exp_df['round_id'].unique():
                round_df = exp_df[exp_df['round_id'] == round_id]
                for metric in round_df['metric_name'].unique():
                    for phase in round_df['experimental_phase'].unique():
                        # Aqui você calcularia a causalidade real
                        # Para o exemplo, vamos apenas gerar um grafo com dados simulados
                        try:
                            # Filtrar dados para essa fase/métrica/round
                            phase_df = filter_long_df(
                                round_df,
                                phase=phase,
                                metric=metric,
                                round_id=round_id
                            )
                            
                            # Transformar para formato wide
                            wide_df = get_wide_format_for_analysis(
                                phase_df,
                                metric=metric,
                                phase=phase,
                                round_id=round_id
                            )
                            
                            if not wide_df.empty and wide_df.shape[1] >= 2:
                                # Criar matriz de causalidade simulada para exemplificar
                                tenants = wide_df.columns
                                fake_matrix = pd.DataFrame(
                                    index=tenants,
                                    columns=tenants,
                                    data=0.3  # Valor de p-value simulado
                                )
                                np.fill_diagonal(fake_matrix.values, 1.0)  # Diagonal com 1 (sem causalidade)
                                
                                # Salvar matriz simulada para referência
                                granger_matrices[f"{experiment_id}:{round_id}:{phase}:{metric}"] = fake_matrix
                                
                                # Gerar grafo de causalidade
                                out_path = os.path.join(out_dir, f"causality_graph_{metric}_{phase}_{round_id}.png")
                                plot_causality_graph(fake_matrix, out_path, threshold=0.5, directed=True, metric=metric)
                                plot_paths.append(out_path)
                        except Exception as e:
                            self.logger.error(f"Erro ao processar causalidade para {metric}, {phase}, {round_id}: {e}")
        
        # Atualizar contexto
        context['granger_matrices'] = granger_matrices
        context['te_matrices'] = te_matrices
        context['causality_plot_paths'] = plot_paths
        
        return context


class ReportGenerationStage(PipelineStage):
    """Estágio para geração de relatório final."""
    
    def __init__(self):
        super().__init__("report_generation", "Geração de relatório final")
        
    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gera relatório final com resultados compilados.
        
        Args:
            context: Contexto com resultados de análises anteriores.
            
        Returns:
            Contexto atualizado com caminho do relatório.
        """
        # Configurar diretório de saída
        out_dir = os.path.join(context.get('output_dir', 'outputs'), 'reports')
        os.makedirs(out_dir, exist_ok=True)
        
        # Gerar relatório HTML ou Markdown
        report_path = os.path.join(out_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        # Exemplo simples de relatório
        with open(report_path, 'w') as f:
            f.write("<html><head><title>Relatório de Análise Multi-Tenant</title></head><body>\n")
            f.write("<h1>Relatório de Análise de Séries Temporais Multi-Tenant</h1>\n")
            f.write(f"<p>Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>\n")
            
            # Adicionar seções para cada tipo de análise
            f.write("<h2>1. Análise Descritiva</h2>\n")
            if 'descriptive_plot_paths' in context:
                f.write("<p>Plots gerados:</p><ul>\n")
                for path in context.get('descriptive_plot_paths', []):
                    rel_path = os.path.relpath(path, start=os.path.dirname(out_dir))
                    f.write(f"<li><a href='../{rel_path}'>{os.path.basename(path)}</a></li>\n")
                f.write("</ul>\n")
            
            f.write("<h2>2. Análise de Correlação</h2>\n")
            if 'correlation_plot_paths' in context:
                f.write("<p>Plots gerados:</p><ul>\n")
                for path in context.get('correlation_plot_paths', []):
                    rel_path = os.path.relpath(path, start=os.path.dirname(out_dir))
                    f.write(f"<li><a href='../{rel_path}'>{os.path.basename(path)}</a></li>\n")
                f.write("</ul>\n")
            
            f.write("<h2>3. Análise de Causalidade</h2>\n")
            if 'causality_plot_paths' in context:
                f.write("<p>Plots gerados:</p><ul>\n")
                for path in context.get('causality_plot_paths', []):
                    rel_path = os.path.relpath(path, start=os.path.dirname(out_dir))
                    f.write(f"<li><a href='../{rel_path}'>{os.path.basename(path)}</a></li>\n")
                f.write("</ul>\n")
            
            f.write("</body></html>\n")
        
        self.logger.info(f"Relatório gerado em: {report_path}")
        
        # Atualizar contexto
        context['report_path'] = report_path
        
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
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Força o reprocessamento dos dados brutos mesmo se existir arquivo parquet"
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
