# -*- coding: utf-8 -*-

"""
Módulo para a análise consolidada de múltiplas rodadas de execução.

Este módulo é responsável por agregar os resultados de várias execuções
do pipeline, permitindo uma análise estatística mais robusta e a
geração de visualizações que demonstram a consistência e a
variabilidade dos resultados.

Funcionalidades Principais:
- Carregar e consolidar os resultados de impacto de múltiplas rodadas.
- Calcular métricas estatísticas agregadas (média, desvio padrão, etc.).
- Preparar os dados para a geração de visualizações multi-round.

Classes:
- MultiRoundAnalysisStage: Orquestra a execução da análise multi-round.
"""

import pandas as pd
import os
from typing import Optional, Dict, List, Any
from .pipeline_stage import PipelineStage
from .config import PipelineConfig
from .visualization.multi_round_plots import (
    plot_aggregated_impact_boxplots,
    plot_aggregated_impact_bar_charts,
    plot_correlation_consistency_heatmap,
    plot_causality_consistency_matrix,
    plot_aggregated_causality_graph
)

class MultiRoundAnalysisStage(PipelineStage):
    """
    Representa a fase de análise consolidada de múltiplas rodadas no pipeline.

    Esta classe herda de `PipelineStage` e implementa a lógica para carregar,
    consolidar e analisar os resultados de todas as rodadas de execução
    definidas na configuração do experimento.
    """

    def __init__(self, config: PipelineConfig):
        """
        Inicializa a fase de análise multi-round.

        Args:
            config (Config): O objeto de configuração do pipeline.
        """
        super().__init__(
            "multi_round_analysis", 
            "Consolida e analisa os resultados de múltiplas rodadas."
        )
        self.config = config
        self.output_dir = self.config.get_output_dir()
        self.experiment_name = self.config.get_experiment_name()
        self.generated_plots: Dict[str, List[str]] = {}

    def _find_reports(self, analysis_type: str, filename: str) -> list[str]:
        """
        Encontra todos os relatórios de uma análise específica em todas as rodadas
        usando uma busca recursiva (glob).

        Args:
            analysis_type: O nome da subpasta da análise (ex: 'impact_analysis').
            filename: O nome do arquivo de relatório a ser encontrado.

        Returns:
            list[str]: Uma lista de caminhos para os arquivos CSV encontrados.
        """
        import glob
        
        base_path = os.path.join(self.output_dir, self.experiment_name)
        
        # Padrão de busca para encontrar os relatórios dentro das pastas de cada rodada
        search_pattern = os.path.join(base_path, 'round-*', analysis_type, filename)
        
        self.logger.info(f"Buscando por relatórios com o padrão: {search_pattern}")
        
        report_paths = glob.glob(search_pattern) # recursive=True não é mais necessário
        
        if report_paths:
            for path in report_paths:
                self.logger.info(f"Encontrado relatório '{filename}' em: {path}")
        else:
            self.logger.warning(f"Nenhum relatório '{filename}' encontrado para a análise '{analysis_type}' no caminho base '{base_path}'.")
            
        return report_paths

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a lógica de consolidação e análise dos resultados.

        Este método será o ponto de entrada para a execução da fase.
        Ele irá orquestrar a busca pelos arquivos de resultados,
        a consolidação dos dados e o cálculo das métricas agregadas.

        Args:
            context (Dict[str, Any]): O contexto do pipeline.

        Returns:
            Dict[str, Any]: O contexto do pipeline atualizado.
        """
        self.logger.info(f"Iniciando a fase: {self.name}")

        # --- Análise de Impacto Multi-Round ---
        self.logger.info("Iniciando análise de impacto multi-round...")
        impact_reports = self._find_reports('impact_analysis', 'impact_analysis_results.csv')
        if impact_reports:
            self._process_impact_analysis(impact_reports)
        else:
            self.logger.warning("Nenhum relatório de impacto encontrado. Pulando a análise de impacto multi-round.")

        # --- Análise de Correlação Multi-Round ---
        self.logger.info("Iniciando análise de correlação multi-round...")
        correlation_reports = self._find_reports('correlation_analysis', 'correlation_results.csv')
        if correlation_reports:
            self._process_correlation_analysis(correlation_reports)
        else:
            self.logger.warning("Nenhum relatório de correlação encontrado. Pulando a análise de correlação multi-round.")

        # --- Análise de Causalidade Multi-Round ---
        self.logger.info("Iniciando análise de causalidade multi-round...")
        causality_reports = self._find_reports('causality_analysis', 'causality_results.csv')
        if causality_reports:
            self._process_causality_analysis(causality_reports)
        else:
            self.logger.warning("Nenhum relatório de causalidade encontrado. Pulando a análise de causalidade multi-round.")

        self.logger.info(f"Fase {self.name} concluída.")
        
        # Retornar um DataFrame vazio ou um resumo, pois o principal são os artefatos salvos
        return context

    def _process_impact_analysis(self, impact_reports: List[str]):
        """
        Processa os relatórios de análise de impacto.
        """
        consolidated_df = self._consolidate_reports(impact_reports, 'round_id')
        if consolidated_df.empty:
            self.logger.error("A consolidação dos relatórios de impacto falhou.")
            return

        self.logger.info(f"Resultados de impacto de {len(impact_reports)} rodadas consolidados.")

        # Lógica para calcular estatísticas agregadas
        aggregated_stats = self._calculate_aggregated_stats(consolidated_df)

        # Salvar os resultados agregados em um novo CSV
        multi_round_output_dir = os.path.join(self.output_dir, self.experiment_name, self.name)
        os.makedirs(multi_round_output_dir, exist_ok=True)

        aggregated_csv_path = os.path.join(multi_round_output_dir, 'multi_round_aggregated_stats.csv')
        aggregated_stats.to_csv(aggregated_csv_path, index=False)
        self.logger.info(f"Estatísticas agregadas de impacto salvas em: {aggregated_csv_path}")

        # Gerar visualizações de impacto
        plots_output_dir = os.path.join(multi_round_output_dir, 'plots')
        os.makedirs(plots_output_dir, exist_ok=True)

        self.generated_plots['impact_boxplots'] = plot_aggregated_impact_boxplots(
            consolidated_df=consolidated_df, output_dir=plots_output_dir
        )
        self.generated_plots['impact_barcharts'] = plot_aggregated_impact_bar_charts(
            consolidated_df=consolidated_df, output_dir=plots_output_dir
        )

    def _consolidate_reports(self, report_paths: List[str], round_col_name: str) -> pd.DataFrame:
        """
        Consolida uma lista de relatórios CSV em um único DataFrame, adicionando uma coluna de rodada.

        Args:
            report_paths: Lista de caminhos para os arquivos CSV.
            round_col_name: Nome da coluna para armazenar o ID da rodada.

        Returns:
            Um DataFrame consolidado ou um DataFrame vazio se ocorrer um erro.
        """
        all_rounds_df = []
        for report_path in report_paths:
            try:
                # Extrai o nome da rodada (ex: 'round-1') do caminho do arquivo
                round_name = os.path.basename(os.path.dirname(os.path.dirname(report_path)))
                df = pd.read_csv(report_path)
                df[round_col_name] = round_name
                all_rounds_df.append(df)
            except Exception as e:
                self.logger.error(f"Falha ao ler ou processar o relatório {report_path}: {e}")
                continue
        
        if not all_rounds_df:
            self.logger.error("Nenhum relatório pôde ser lido com sucesso.")
            return pd.DataFrame()

        return pd.concat(all_rounds_df, ignore_index=True)

    def _process_correlation_analysis(self, correlation_reports: List[str]):
        """
        Processa os relatórios de análise de correlação de múltiplas rodadas.
        Consolida os dados, calcula estatísticas de consistência e gera visualizações.
        """
        # Usa o método _consolidate_reports para carregar e unir os relatórios
        consolidated_df = self._consolidate_reports(correlation_reports, round_col_name='round_id_from_path')
        if consolidated_df.empty:
            self.logger.error("A consolidação dos relatórios de correlação falhou.")
            return

        self.logger.info(f"Resultados de correlação de {len(correlation_reports)} rodadas consolidados.")

        # Colunas para agrupar e analisar a consistência
        group_by_cols = ['metric', 'phase', 'tenant1', 'tenant2']
        
        # Verificar se as colunas de agrupamento existem
        for col in group_by_cols:
            if col not in consolidated_df.columns:
                self.logger.error(f"Coluna de agrupamento '{col}' não encontrada no DataFrame de correlação. Colunas disponíveis: {consolidated_df.columns.tolist()}")
                return

        # Calcular estatísticas de consistência da correlação entre as rodadas
        correlation_consistency = consolidated_df.groupby(group_by_cols)['correlation'].agg(
            mean_correlation='mean',
            std_dev_correlation='std',
            min_correlation='min',
            max_correlation='max',
            rounds_count='count'
        ).reset_index()

        # Salvar os resultados de consistência em um novo CSV
        multi_round_output_dir = os.path.join(self.output_dir, self.experiment_name, self.name)
        os.makedirs(multi_round_output_dir, exist_ok=True)

        consistency_csv_path = os.path.join(multi_round_output_dir, 'multi_round_correlation_consistency.csv')
        correlation_consistency.to_csv(consistency_csv_path, index=False)
        self.logger.info(f"Estatísticas de consistência de correlação salvas em: {consistency_csv_path}")

        # Gerar visualizações (a função de plotagem pode precisar de adaptação)
        plots_output_dir = os.path.join(multi_round_output_dir, 'plots')
        os.makedirs(plots_output_dir, exist_ok=True)

        # A função plot_correlation_consistency_heatmap provavelmente espera um formato diferente.
        # A implementação da plotagem será ajustada a seguir.
        self.logger.info("Tentando gerar heatmap de consistência de correlação...")
        plot_path = plot_correlation_consistency_heatmap(
            correlation_consistency_df=correlation_consistency, # Passando o novo DataFrame
            output_dir=plots_output_dir
        )
        self.generated_plots['correlation_heatmap'] = [plot_path] if plot_path else []

    def _process_causality_analysis(self, causality_reports: List[str]):
        """
        Processa os relatórios de análise de causalidade de múltiplas rodadas.
        """
        all_causality_dfs = []
        for report_path in causality_reports:
            try:
                round_name = os.path.basename(os.path.dirname(os.path.dirname(report_path)))
                df = pd.read_csv(report_path)
                df['round'] = round_name
                all_causality_dfs.append(df)
            except Exception as e:
                self.logger.error(f"Falha ao ler o relatório de causalidade {report_path}: {e}")
                continue

        if not all_causality_dfs:
            self.logger.error("Nenhum relatório de causalidade pôde ser lido com sucesso.")
            return

        consolidated_df = pd.concat(all_causality_dfs, ignore_index=True)
        self.logger.info(f"Resultados de causalidade de {len(causality_reports)} rodadas consolidados.")

        # Filtrar apenas por links causais significativos
        causal_links = consolidated_df[consolidated_df['is_causal'] == True]

        # Calcular a frequência de cada link causal
        causality_frequency = causal_links.groupby(['source', 'target']).size().reset_index(name='frequency')
        causality_frequency['total_rounds'] = len(causality_reports)
        causality_frequency['consistency_rate'] = (causality_frequency['frequency'] / causality_frequency['total_rounds']) * 100

        # Salvar os resultados agregados
        multi_round_output_dir = os.path.join(self.output_dir, self.experiment_name, self.name)
        aggregated_csv_path = os.path.join(multi_round_output_dir, 'multi_round_causality_frequency.csv')
        causality_frequency.to_csv(aggregated_csv_path, index=False)
        self.logger.info(f"Frequência de causalidade agregada salva em: {aggregated_csv_path}")

        # Gerar visualizações de causalidade
        plots_output_dir = os.path.join(multi_round_output_dir, 'plots')
        os.makedirs(plots_output_dir, exist_ok=True)

        matrix_plot_path = plot_causality_consistency_matrix(
            causality_frequency, plots_output_dir
        )
        self.generated_plots['causality_matrix'] = [matrix_plot_path] if matrix_plot_path else []

        graph_plot_path = plot_aggregated_causality_graph(
            causality_frequency, plots_output_dir
        )
        self.generated_plots['causality_graph'] = [graph_plot_path] if graph_plot_path else []

    def _calculate_aggregated_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula estatísticas descritivas agregadas a partir do DataFrame consolidado.
        """
        # Colunas para agrupar os dados, alinhadas com os nomes no DataFrame
        group_by_cols = ['tenant_id', 'metric_name', 'experimental_phase']
        
        # Verificar se as colunas de agrupamento existem no DataFrame
        for col in group_by_cols:
            if col not in df.columns:
                self.logger.error(f"Coluna de agrupamento '{col}' não encontrada no DataFrame. Colunas disponíveis: {df.columns.tolist()}")
                # Levantar um erro ou retornar um DataFrame vazio para evitar falhas inesperadas
                raise KeyError(f"Coluna de agrupamento ausente: {col}")

        # Usar agregações nomeadas para maior clareza
        aggregated_df = df.groupby(group_by_cols).agg(
            mean_percentage_change=('percentage_change', 'mean'),
            median_percentage_change=('percentage_change', 'median'),
            std_percentage_change=('percentage_change', 'std'),
            min_percentage_change=('percentage_change', 'min'),
            max_percentage_change=('percentage_change', 'max'),
            rounds_count=('round_id', 'count'),  # Usar round_id para contagem
            significance_count=('is_significant', lambda x: x.sum()),
            significance_freq=('is_significant', lambda x: x.mean())
        ).reset_index()

        # Renomear colunas para o formato desejado, se necessário
        aggregated_df.rename(columns={
            'mean_percentage_change': 'mean_change',
            'median_percentage_change': 'median_change',
            'std_percentage_change': 'std_dev',
            'min_percentage_change': 'min_change',
            'max_percentage_change': 'max_change',
        }, inplace=True)

        # Calcular a frequência em porcentagem
        aggregated_df['significance_freq'] = aggregated_df['significance_freq'] * 100

        return aggregated_df

