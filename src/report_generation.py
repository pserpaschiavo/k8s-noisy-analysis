# -*- coding: utf-8 -*-
"""
Módulo para a geração de relatórios consolidados.
"""
import os
import logging
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime

from .pipeline_stage import PipelineStage
from .config import PipelineConfig
from .analysis_multi_round import MultiRoundAnalysisStage

logger = logging.getLogger(__name__)

class ReportGenerationStage(PipelineStage):
    """
    Gera o relatório final consolidando os resultados de todas as fases.
    """
    def __init__(self, config: PipelineConfig, 
                 descriptive_stats: pd.DataFrame, 
                 impact_results: pd.DataFrame,
                 correlation_results: pd.DataFrame,
                 causality_results: pd.DataFrame,
                 phase_comparison_results: pd.DataFrame,
                 multi_round_stage: MultiRoundAnalysisStage):
        """
        Inicializa a fase de geração de relatório.
        """
        super().__init__("report_generation", "Geração do Relatório Final")
        self.config = config
        self.descriptive_stats = descriptive_stats
        self.impact_results = impact_results
        self.correlation_results = correlation_results
        self.causality_results = causality_results
        self.phase_comparison_results = phase_comparison_results
        self.multi_round_stage = multi_round_stage
        self.output_dir = self.config.get_output_dir()
        self.experiment_name = self.config.get_experiment_name()

    def _execute_implementation(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa a geração do relatório.
        """
        self.logger.info(f"Iniciando a fase: {self.name}")
        
        report_dir = os.path.join(self.output_dir, self.experiment_name, self.name)
        os.makedirs(report_dir, exist_ok=True)
        
        self._generate_markdown_report(report_dir)
        
        self.logger.info(f"Fase {self.name} concluída. Relatório salvo em: {report_dir}")
        return context

    def _generate_markdown_report(self, report_dir: str):
        """
        Gera o conteúdo do relatório em formato Markdown.
        """
        report_filename = f"final_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path = os.path.join(report_dir, report_filename)

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"# Relatório de Análise de Impacto - {self.experiment_name}\n\n")
            f.write(f"*Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")
            
            f.write("## 1. Resumo da Análise\n")
            f.write("Este relatório apresenta os resultados consolidados da análise de impacto de vizinhos barulhentos em um ambiente multi-tenant. As seções a seguir detalham as análises de impacto, correlação, causalidade e os resultados agregados de múltiplas rodadas.\n\n")

            # --- Seção de Análise Multi-Round ---
            self._write_multi_round_section(f, report_dir)

            # --- Outras seções (resumidas ou a serem implementadas) ---
            f.write("## 2. Análise de Impacto por Rodada (Exemplo)\n")
            f.write("A análise de impacto individual por rodada avalia a mudança percentual nas métricas de desempenho em comparação com uma fase de linha de base. Abaixo, um resumo dos resultados de impacto.\n\n")
            if not self.impact_results.empty:
                f.write(self.impact_results.head().to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("Nenhum resultado de impacto disponível.\n\n")

            f.write("## 3. Análise de Correlação e Causalidade (Resumo)\n")
            f.write("As análises de correlação e causalidade exploram as interdependências entre as métricas dos diferentes tenants.\n\n")
            
            f.write("### Correlação\n")
            if not self.correlation_results.empty:
                f.write(self.correlation_results.head().to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("Nenhum resultado de correlação disponível.\n\n")

            f.write("### Causalidade\n")
            if not self.causality_results.empty:
                f.write(self.causality_results.head().to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write("Nenhum resultado de causalidade disponível.\n\n")

        self.logger.info(f"Relatório Markdown gerado com sucesso em: {report_path}")

    def _write_multi_round_section(self, f, report_dir: str):
        """
        Escreve a seção de análise multi-round no arquivo de relatório.
        """
        f.write("## Análise Consolidada Multi-Round\n\n")
        f.write("Esta seção agrega os resultados de todas as rodadas experimentais para fornecer uma visão estatisticamente mais robusta dos padrões de impacto, correlação e causalidade.\n\n")

        # Adicionar links para os CSVs agregados
        multi_round_output_dir = os.path.join(self.output_dir, self.experiment_name, "multi_round_analysis")
        
        aggregated_stats_csv = os.path.join(multi_round_output_dir, 'multi_round_aggregated_stats.csv')
        if os.path.exists(aggregated_stats_csv):
            f.write("### Estatísticas Agregadas de Impacto\n\n")
            f.write(f"A tabela completa com as estatísticas de impacto agregadas está disponível em: `../multi_round_analysis/multi_round_aggregated_stats.csv`\n\n")
            try:
                df_stats = pd.read_csv(aggregated_stats_csv)
                f.write("**Amostra das Estatísticas Agregadas:**\n\n")
                f.write(df_stats.head().to_markdown(index=False))
                f.write("\n\n")
            except Exception as e:
                self.logger.warning(f"Não foi possível ler o arquivo de estatísticas agregadas: {e}")

        causality_freq_csv = os.path.join(multi_round_output_dir, 'multi_round_causality_frequency.csv')
        if os.path.exists(causality_freq_csv):
            f.write("### Frequência de Causalidade\n\n")
            f.write(f"A tabela completa com a frequência de links causais está disponível em: `../multi_round_analysis/multi_round_causality_frequency.csv`\n\n")
            try:
                df_causality = pd.read_csv(causality_freq_csv)
                f.write("**Amostra da Frequência de Causalidade:**\n\n")
                f.write(df_causality.head().to_markdown(index=False))
                f.write("\n\n")
            except Exception as e:
                self.logger.warning(f"Não foi possível ler o arquivo de frequência de causalidade: {e}")

        # Adicionar plots gerados pela fase multi-round
        multi_round_plots = self.multi_round_stage.generated_plots
        if multi_round_plots:
            f.write("### Visualizações Consolidadas\n\n")
            for plot_type, plot_paths in multi_round_plots.items():
                # Garante que plot_paths seja sempre uma lista para iteração
                if not isinstance(plot_paths, list):
                    plot_paths = [plot_paths]

                if plot_paths:
                    title = plot_type.replace('_', ' ').title()
                    f.write(f"#### {title}\n\n")
                    # Itera sobre a lista de caminhos, tratando listas aninhadas
                    for path_item in plot_paths:
                        # Se o item for ele mesmo uma lista, itera sobre ela
                        if isinstance(path_item, list):
                            for path in path_item:
                                if path and os.path.exists(path):
                                    relative_path = os.path.relpath(path, report_dir)
                                    f.write(f"![{title} - {os.path.basename(path)}]({relative_path})\n\n")
                                elif path:
                                    self.logger.warning(f"Arquivo de plot (sublista) não encontrado: {path}")
                        # Se for um caminho de arquivo (string)
                        elif path_item and os.path.exists(path_item):
                            relative_path = os.path.relpath(path_item, report_dir)
                            f.write(f"![{title} - {os.path.basename(path_item)}]({relative_path})\n\n")
                        elif path_item:
                            self.logger.warning(f"Arquivo de plot não encontrado: {path_item}")
        else:
            f.write("Nenhuma visualização consolidada foi gerada.\n\n")
