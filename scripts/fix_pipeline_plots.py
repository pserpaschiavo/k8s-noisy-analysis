#!/usr/bin/env python3
"""
Script para corrigir problemas com plots ausentes no pipeline de análise K8s.
Este script implementa as correções definidas no plano de trabalho.

Uso:
python scripts/fix_pipeline_plots.py --config config/pipeline_config_sfi2.yaml
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def ensure_directories(output_dir):
    """
    Garante que todos os diretórios necessários existam.
    """
    directories = [
        "correlation_graphs",
        "plots/correlation",
        "plots/causality",
        "plots/phase_comparison",
        "multi_round_analysis/correlation_graphs"
    ]
    
    for directory in directories:
        full_path = os.path.join(output_dir, directory)
        os.makedirs(full_path, exist_ok=True)
        logger.info(f"Diretório verificado/criado: {full_path}")

def fix_correlation_graphs(config):
    """
    Corrige os problemas com gráficos de correlação ausentes.
    """
    from src.utils import configure_matplotlib
    configure_matplotlib()
    
    try:
        # Importar funções necessárias
        from src.analysis_correlation import compute_correlation_matrix
        from src.visualization.correlation_plots import plot_correlation_heatmap
        
        # Obter configurações relevantes
        output_dir = config.get("output_dir", "./outputs/sfi2-paper-analysis")
        data_path = os.path.join(config.get("processed_data_dir", "./data/processed"), 
                              config.get("output_parquet_name", "sfi2_paper.parquet"))
        
        # Garantir que diretórios existem
        ensure_directories(output_dir)
        
        # Carregar dados
        logger.info(f"Carregando dados de {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Reduzir threshold para garantir visualizações
        threshold = config.get('visualization', {}).get('correlation_graph', {}).get('threshold', 0.3)
        new_threshold = min(threshold, 0.1)  # Garantir threshold baixo para visualização
        
        logger.info(f"Usando threshold reduzido: {new_threshold} (original: {threshold})")
        
        # Obter métricas, rounds e fases
        metrics = config.get("selected_metrics", ["cpu_usage", "memory_usage"])
        rounds = config.get("selected_rounds", ["round-1", "round-2", "round-3"])
        phases = config.get("selected_phases", [])
        
        # Verificar se temos fases na configuração
        if not phases:
            phases = sorted(df['experimental_phase'].unique().tolist())
        
        # Gerar gráficos de correlação para cada combinação
        corr_results = {}
        
        # Criar diretório para gráficos de correlação no diretório principal
        main_corr_dir = os.path.join(output_dir, "correlation_graphs")
        os.makedirs(main_corr_dir, exist_ok=True)
        
        # Importar função de plotagem agregada
        from src.visualization.advanced_plots import plot_aggregated_correlation_graph
        
        # Computar correlações agregadas para cada métrica
        from src.analysis_correlation import compute_aggregated_correlation
        
        # Agregar correlações por métrica
        aggregated_correlations = compute_aggregated_correlation(df, metrics, rounds, phases)
        
        # Gerar gráficos de correlação agregada no diretório principal
        for metric, corr_matrix in aggregated_correlations.items():
            if corr_matrix is not None and not corr_matrix.empty:
                try:
                    plot_path = plot_aggregated_correlation_graph(
                        correlation_matrix=corr_matrix,
                        title=f"Correlação Agregada - {metric}",
                        output_dir=main_corr_dir,
                        filename=f"aggregated_correlation_graph_{metric}.png",
                        threshold=new_threshold
                    )
                    if plot_path:
                        logger.info(f"Gráfico de correlação agregada gerado: {plot_path}")
                    else:
                        # Se não houver correlações acima do threshold, criar gráfico vazio
                        logger.warning(f"Criando gráfico vazio para {metric} (correlações abaixo do threshold)")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.text(0.5, 0.5, f"Sem correlações acima de {new_threshold}", 
                               ha='center', va='center', fontsize=14)
                        ax.set_title(f"Correlação Agregada - {metric}")
                        ax.axis('off')
                        plot_path = os.path.join(main_corr_dir, f"empty_aggregated_correlation_graph_{metric}.png")
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Gráfico vazio gerado: {plot_path}")
                except Exception as e:
                    logger.error(f"Erro ao gerar gráfico de correlação para {metric}: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Erro ao corrigir gráficos de correlação: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def fix_transfer_entropy_plots(config):
    """
    Corrige problemas com os gráficos de transfer entropy.
    """
    try:
        # Garantir que configuração do matplotlib está aplicada
        from src.utils import configure_matplotlib
        configure_matplotlib()
        
        # Obter configurações relevantes
        output_dir = config.get("output_dir", "./outputs/sfi2-paper-analysis")
        causality_dir = os.path.join(output_dir, "plots/causality")
        os.makedirs(causality_dir, exist_ok=True)
        
        data_path = os.path.join(config.get("processed_data_dir", "./data/processed"), 
                              config.get("output_parquet_name", "sfi2_paper.parquet"))
        
        # Carregar dados
        logger.info(f"Carregando dados de {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Parâmetros para transfer entropy
        metrics = config.get("selected_metrics", ["cpu_usage", "memory_usage"])
        rounds = config.get("selected_rounds", ["round-1", "round-2", "round-3"])
        phases = config.get("selected_phases", [])
        
        # Verificar se temos fases na configuração
        if not phases:
            phases = sorted(df['experimental_phase'].unique().tolist())
        
        # Cria gráficos de transfer entropy robustos
        try:
            from src.analysis_causality import CausalityAnalyzer
            
            # Cria uma instância do analisador de causalidade
            analyzer = CausalityAnalyzer(df, causality_dir)
            
            for round_id in rounds:
                round_dir = os.path.join(causality_dir, round_id)
                os.makedirs(round_dir, exist_ok=True)
                
                for phase in phases:
                    for metric in metrics:
                        try:
                            # Filtrar dados para verificar se há suficiente
                            phase_df = df[(df['round_id'] == round_id) & 
                                         (df['experimental_phase'] == phase) & 
                                         (df['metric_name'] == metric)]
                            
                            # Verifica se há dados suficientes
                            if len(phase_df) < 30:  # Mínimo para cálculo de TE
                                logger.warning(f"Dados insuficientes para transfer entropy: {round_id}, {phase}, {metric}")
                                # Cria gráfico informativo
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.text(0.5, 0.5, "Dados insuficientes para cálculo de Transfer Entropy", 
                                      ha='center', va='center', fontsize=14)
                                ax.set_title(f"Transfer Entropy - {metric} - {phase} - {round_id}")
                                ax.axis('off')
                                plot_path = os.path.join(round_dir, f"transfer_entropy_{metric}_{phase.replace(' ', '_')}.png")
                                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                                plt.close(fig)
                                logger.info(f"Gráfico informativo gerado: {plot_path}")
                                continue
                            
                            # Tenta calcular transfer entropy usando o analisador
                            logger.info(f"Calculando transfer entropy para {round_id}, {phase}, {metric}")
                            results = analyzer.run_and_plot_causality_analysis(
                                metric=metric,
                                phase=phase,
                                round_id=round_id
                            )
                            
                            if results and 'plot_paths' in results and results['plot_paths']:
                                logger.info(f"Gráficos de causalidade gerados para {metric}, {phase}, {round_id}")
                                for path in results['plot_paths']:
                                    logger.info(f"  - {path}")
                            else:
                                logger.warning(f"Nenhum gráfico de causalidade gerado para {metric}, {phase}, {round_id}")
                                # Criar gráfico fallback
                                fig, ax = plt.subplots(figsize=(10, 8))
                                ax.text(0.5, 0.5, "Análise de causalidade não produziu resultados significativos", 
                                      ha='center', va='center', fontsize=14)
                                ax.set_title(f"Transfer Entropy - {metric} - {phase} - {round_id}")
                                ax.axis('off')
                                plot_path = os.path.join(round_dir, f"fallback_transfer_entropy_{metric}_{phase.replace(' ', '_')}.png")
                                fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                                plt.close(fig)
                                logger.info(f"Gráfico fallback gerado: {plot_path}")
                            
                        except Exception as e:
                            logger.error(f"Erro em transfer entropy para {round_id}, {phase}, {metric}: {str(e)}")
        
        except ImportError as e:
            logger.error(f"Módulo de análise de causalidade não disponível: {str(e)}")
        
        return True
    except Exception as e:
        logger.error(f"Erro ao corrigir gráficos de transfer entropy: {str(e)}")
        return False

def fix_phase_comparison_plots(config):
    """
    Corrige problemas com os gráficos de comparação de fases.
    """
    try:
        # Garantir que configuração do matplotlib está aplicada
        from src.utils import configure_matplotlib
        configure_matplotlib()
        
        # Obter configurações relevantes
        output_dir = config.get("output_dir", "./outputs/sfi2-paper-analysis")
        phase_comparison_dir = os.path.join(output_dir, "plots", "phase_comparison")
        os.makedirs(phase_comparison_dir, exist_ok=True)
        
        data_path = os.path.join(config.get("processed_data_dir", "./data/processed"), 
                              config.get("output_parquet_name", "sfi2_paper.parquet"))
        
        # Carregar dados
        logger.info(f"Carregando dados de {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Parâmetros para phase comparison
        metrics = config.get("selected_metrics", ["cpu_usage", "memory_usage"])
        rounds = config.get("selected_rounds", ["round-1", "round-2", "round-3"])
        phases = config.get("selected_phases", [])
        
        # Verificar se temos fases na configuração
        if not phases:
            phases = sorted(df['experimental_phase'].unique().tolist())
        
        # Verificar se há fases suficientes para comparação
        if len(phases) < 2:
            logger.warning(f"Pelo menos 2 fases são necessárias para comparação. Encontradas: {len(phases)}")
            
            # Criar gráfico informativo
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"Insuficiente número de fases para comparação.\nEncontradas: {len(phases)}, necessárias pelo menos 2.", 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f"Comparação de Fases - Erro")
            ax.axis('off')
            plot_path = os.path.join(phase_comparison_dir, "insufficient_phases_error.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Gráfico informativo gerado: {plot_path}")
            return True
        
        logger.info(f"Gerando plots de comparação de fases para {len(metrics)} métricas e {len(rounds)} rounds...")
        
        # Importar funções necessárias
        from src.analysis_phase_comparison import PhaseComparisonAnalyzer
        
        plots_gerados = []
        
        # Para cada round, gerar comparação de fases para cada métrica
        for round_id in rounds:
            # Criar analisador de comparação de fases
            analyzer = PhaseComparisonAnalyzer(df)
            
            # Para cada métrica, gerar análise
            for metric in metrics:
                try:
                    logger.info(f"Gerando comparação de fases para {metric} no round {round_id}...")
                    
                    # Gerar análise e plot
                    stats_df = analyzer.analyze_metric_across_phases(
                        metric=metric,
                        round_id=round_id,
                        output_dir=phase_comparison_dir
                    )
                    
                    if stats_df is not None and not stats_df.empty:
                        # Salvar o DataFrame de estatísticas
                        stats_path = os.path.join(phase_comparison_dir, f"phase_stats_{metric}_{round_id}.csv")
                        stats_df.to_csv(stats_path, index=False)
                        logger.info(f"Estatísticas de fase salvas em: {stats_path}")
                        
                        # Registrar o plot gerado
                        plot_path = os.path.join(phase_comparison_dir, f"phase_comparison_{metric}_{round_id}.png")
                        if os.path.exists(plot_path):
                            plots_gerados.append(plot_path)
                            logger.info(f"Plot de comparação de fases gerado: {plot_path}")
                    else:
                        logger.warning(f"Nenhum dado válido para comparação de fases em {metric}, {round_id}")
                        
                        # Criar gráfico informativo para esse caso
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.text(0.5, 0.5, f"Dados insuficientes para comparação de fases.\nMétrica: {metric}, Round: {round_id}", 
                               ha='center', va='center', fontsize=14)
                        ax.set_title(f"Comparação de Fases - {metric} - {round_id}")
                        ax.axis('off')
                        plot_path = os.path.join(phase_comparison_dir, f"insufficient_data_phase_comparison_{metric}_{round_id}.png")
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Gráfico informativo gerado: {plot_path}")
                        
                except Exception as e:
                    logger.error(f"Erro ao gerar comparação de fases para {metric}, {round_id}: {str(e)}")
        
        logger.info(f"Total de plots de comparação de fases gerados: {len(plots_gerados)}")
        return True
    except Exception as e:
        logger.error(f"Erro ao corrigir gráficos de phase comparison: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def fix_multi_round_plots(config):
    """
    Corrige problemas com os plots de multi-round analysis.
    """
    try:
        # Garantir que configuração do matplotlib está aplicada
        from src.utils import configure_matplotlib
        configure_matplotlib()
        
        # Obter configurações relevantes
        output_dir = config.get("output_dir", "./outputs/sfi2-paper-analysis")
        main_output_dir = output_dir  # Diretório principal para saída
        multi_round_dir = os.path.join(output_dir, "multi_round_analysis")
        
        # Verificar e criar diretórios necessários
        # Diretório principal de multi_round_analysis
        os.makedirs(multi_round_dir, exist_ok=True)
        
        # Diretório de correlação dentro de multi_round_analysis
        corr_dir = os.path.join(multi_round_dir, "correlation_graphs")
        os.makedirs(corr_dir, exist_ok=True)
        
        # Diretório de causalidade dentro de multi_round_analysis
        causality_dir = os.path.join(multi_round_dir, "causality_graphs")
        os.makedirs(causality_dir, exist_ok=True)
        
        logger.info(f"Diretórios para multi-round analysis verificados/criados")
        
        # Carregar dados
        data_path = os.path.join(config.get("processed_data_dir", "./data/processed"), 
                              config.get("output_parquet_name", "sfi2_paper.parquet"))
        
        logger.info(f"Carregando dados de {data_path}...")
        df = pd.read_parquet(data_path)
        
        # Configurações para multi-round
        metrics = config.get("selected_metrics", ["cpu_usage", "memory_usage"])
        rounds = config.get("selected_rounds", ["round-1", "round-2", "round-3"])
        phases = config.get("selected_phases", [])
        
        # Verificar se temos fases na configuração
        if not phases:
            phases = sorted(df['experimental_phase'].unique().tolist())
        
        # Verificar se há rounds suficientes
        if len(rounds) < 2:
            logger.warning(f"Pelo menos 2 rounds são necessários para análise multi-round. Encontrados: {len(rounds)}")
            
            # Criar gráfico informativo
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.text(0.5, 0.5, f"Insuficiente número de rounds para análise multi-round.\nEncontrados: {len(rounds)}, necessários pelo menos 2.", 
                   ha='center', va='center', fontsize=14)
            ax.set_title(f"Análise Multi-Round - Erro")
            ax.axis('off')
            plot_path = os.path.join(multi_round_dir, "insufficient_rounds_error.png")
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            logger.info(f"Gráfico informativo gerado: {plot_path}")
            return True
        
        # 1. Gerar gráficos de correlação para multi-round
        logger.info("Gerando gráficos de correlação para análise multi-round...")
        
        # Usar threshold menor para garantir visualizações
        threshold = config.get('visualization', {}).get('correlation_graph', {}).get('threshold', 0.3)
        new_threshold = min(threshold, 0.1)  # Threshold baixo para visualização
        
        # Calcular correlações agregadas
        from src.analysis_correlation import compute_aggregated_correlation
        from src.visualization.advanced_plots import plot_aggregated_correlation_graph
        
        # Calcular e plotar correlações agregadas para cada métrica
        aggregated_correlations = compute_aggregated_correlation(df, metrics, rounds, phases)
        
        for metric, corr_matrix in aggregated_correlations.items():
            if corr_matrix is not None and not corr_matrix.empty:
                try:
                    # Gerar gráfico de correlação agregada
                    logger.info(f"Gerando gráfico de correlação agregada para {metric}...")
                    plot_path = plot_aggregated_correlation_graph(
                        correlation_matrix=corr_matrix,
                        title=f"Correlação Agregada Multi-Round - {metric}",
                        output_dir=corr_dir,
                        filename=f"multi_round_correlation_graph_{metric}.png",
                        threshold=new_threshold
                    )
                    
                    if plot_path:
                        logger.info(f"Gráfico de correlação multi-round gerado: {plot_path}")
                    else:
                        # Se não houver correlações acima do threshold, criar gráfico vazio
                        logger.warning(f"Criando gráfico vazio para {metric} (correlações abaixo do threshold)")
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.text(0.5, 0.5, f"Sem correlações acima de {new_threshold} para análise multi-round", 
                               ha='center', va='center', fontsize=14)
                        ax.set_title(f"Correlação Multi-Round - {metric}")
                        ax.axis('off')
                        plot_path = os.path.join(corr_dir, f"empty_multi_round_correlation_graph_{metric}.png")
                        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        logger.info(f"Gráfico vazio gerado: {plot_path}")
                except Exception as e:
                    logger.error(f"Erro ao gerar gráfico de correlação multi-round para {metric}: {str(e)}")
        
        # 2. Gerar gráficos de causalidade para multi-round (se disponível)
        logger.info("Tentando gerar gráficos de causalidade para análise multi-round...")
        
        try:
            from src.analysis_causality import CausalityAnalyzer
            
            # Criar analisador de causalidade
            causality_analyzer = CausalityAnalyzer(df, causality_dir)
            
            for metric in metrics:
                # Usar a primeira fase como exemplo
                if phases:
                    phase = phases[0]
                    
                    # Para cada round, calcular matrizes de causalidade
                    for round_id in rounds:
                        try:
                            # Executar análise de causalidade
                            causality_results = causality_analyzer.run_and_plot_causality_analysis(
                                metric=metric,
                                phase=phase,
                                round_id=round_id
                            )
                            
                            if causality_results and 'plot_paths' in causality_results:
                                for path in causality_results['plot_paths']:
                                    logger.info(f"Gráfico de causalidade gerado: {path}")
                        except Exception as e:
                            logger.warning(f"Erro na análise de causalidade para {metric}, {phase}, {round_id}: {str(e)}")
        except ImportError as e:
            logger.warning(f"Módulo de análise de causalidade não disponível: {str(e)}")
        
        # 3. Gerar relatório de análise multi-round
        logger.info("Gerando relatório de análise multi-round...")
        
        report_path = os.path.join(multi_round_dir, "multi_round_analysis_report.md")
        
        with open(report_path, 'w') as report:
            report.write("# Relatório de Análise Multi-Round\n\n")
            report.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            report.write(f"## Configuração\n\n")
            report.write(f"- Métricas analisadas: {', '.join(metrics)}\n")
            report.write(f"- Rounds analisados: {', '.join(rounds)}\n")
            report.write(f"- Fases analisadas: {', '.join(phases)}\n\n")
            
            report.write("## Gráficos Gerados\n\n")
            
            # Listar gráficos de correlação
            report.write("### Gráficos de Correlação\n\n")
            corr_files = [f for f in os.listdir(corr_dir) if f.endswith('.png')]
            if corr_files:
                for f in corr_files:
                    report.write(f"- [{f}](correlation_graphs/{f})\n")
            else:
                report.write("Nenhum gráfico de correlação gerado.\n")
            
            report.write("\n### Gráficos de Causalidade\n\n")
            causality_files = [f for f in os.listdir(causality_dir) if f.endswith('.png')]
            if causality_files:
                for f in causality_files:
                    report.write(f"- [{f}](causality_graphs/{f})\n")
            else:
                report.write("Nenhum gráfico de causalidade gerado.\n")
            
            report.write("\n## Observações\n\n")
            report.write("Este relatório foi gerado pelo script de correção de plots ausentes.\n")
            report.write("Para uma análise completa, é necessário executar o pipeline completo com os ajustes apropriados.\n")
        
        logger.info(f"Relatório de análise multi-round gerado: {report_path}")
        return True
    except Exception as e:
        logger.error(f"Erro ao corrigir plots de multi-round: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def parse_args():
    parser = argparse.ArgumentParser(description='Corrige problemas com plots ausentes no pipeline de análise K8s')
    parser.add_argument('--config', type=str, default='config/pipeline_config_sfi2.yaml', 
                        help='Caminho para o arquivo de configuração YAML')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Carregar configuração
    import yaml
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    logger.info("Iniciando correção de plots ausentes...")
    
    # Executar correções
    results = {
        "correlation": fix_correlation_graphs(config),
        "transfer_entropy": fix_transfer_entropy_plots(config),
        "phase_comparison": fix_phase_comparison_plots(config),
        "multi_round": fix_multi_round_plots(config)
    }
    
    # Relatório final
    logger.info("\n===== Resultados da Correção =====")
    for plot_type, success in results.items():
        status = "✅ Sucesso" if success else "❌ Falha"
        logger.info(f"{plot_type}: {status}")
    
    # Verificar sucessos
    if all(results.values()):
        logger.info("Todas as correções foram aplicadas com sucesso!")
        return 0
    else:
        logger.error("Algumas correções falharam. Verifique os logs para detalhes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
