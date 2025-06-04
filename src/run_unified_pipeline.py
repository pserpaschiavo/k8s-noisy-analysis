#!/usr/bin/env python3
"""
Script: run_unified_pipeline.py
Descrição: Script unificado para execução do pipeline com todas as visualizações implementadas.

Este script serve como uma ponte temporária entre a implementação atual fragmentada e a
futura implementação baseada em plugins. Ele garante que todas as visualizações implementadas
sejam geradas corretamente, independente de qual versão do pipeline é executada.
"""

import os
import sys
import logging
import argparse
import warnings
from datetime import datetime
from typing import Dict, Any, Optional, List

# Importação para suprimir avisos
# Adiciona o diretório atual ao path de importação
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cria função para suprimir avisos do statsmodels
def suppress_statsmodels_warnings():
    """Context manager para suprimir avisos comuns do statsmodels."""
    import warnings
    import contextlib
    
    @contextlib.contextmanager
    def _suppress():
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'Maximum Likelihood optimization failed to converge')
            warnings.filterwarnings('ignore', 'Non-stationary starting autoregressive parameters')
            warnings.filterwarnings('ignore', 'Non-invertible starting MA parameters')
            warnings.filterwarnings('ignore', 'No frequency information was provided')
            yield
    
    return _suppress()

# Importação de diferentes versões do pipeline
from src.pipeline import Pipeline, parse_arguments
from src.pipeline_with_sliding_window import create_pipeline_with_sliding_window

# Configuração de logging
log_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'unified_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
print(f"Logging to {log_file}")

logger = logging.getLogger("unified_pipeline")

def verify_all_visualizations(base_output_dir: str) -> Dict[str, Dict[str, bool]]:
    """
    Verifica quais visualizações foram geradas e quais estão faltando.
    
    Args:
        base_output_dir: Diretório base de saída onde as visualizações são salvas.
        
    Returns:
        Dicionário com o status de cada visualização esperada.
    """
    expected_visualizations = {
        "descriptive": [
            "timeseries_multi_*.png",
            "barplot_*.png",
            "boxplot_*.png"
        ],
        "correlation": [
            "correlation_heatmap_*.png",  # Atualmente não gerado
            "covariance_heatmap_*.png"
        ],
        "causality": [
            "causality_graph_granger_*.png",
            "causality_graph_te_*.png"
        ],
        "sliding_window/correlation": [
            "sliding_corr_*.png",
            "sliding_corr_consolidated_*.png"
        ],
        "sliding_window/causality/granger": [
            "sliding_caus_granger_*.png"
        ],
        "sliding_window/causality/transfer_entropy": [
            "sliding_caus_transfer_entropy_*.png"
        ],
        "multi_round": [  # Diretório não existente
            "consistency_*.png",
            "robustness_*.png"
        ],
        "anomaly_detection": [  # Diretório não existente
            "anomaly_detection_*.png"
        ],
        "phase_comparison": [
            "phase_comparison_*.png"
        ]
    }
    
    # Resultado para armazenar o status de cada visualização
    visualization_status = {}
    
    # Verifica os diretórios e arquivos existentes
    for category, patterns in expected_visualizations.items():
        visualization_status[category] = {}
        category_dir = os.path.join(base_output_dir, "plots", category)
        
        # Verifica se o diretório existe
        if not os.path.exists(category_dir):
            logger.warning(f"Diretório não encontrado: {category_dir}")
            for pattern in patterns:
                visualization_status[category][pattern] = False
            continue
        
        # Verifica cada padrão de arquivo de forma mais robusta
        for pattern in patterns:
            # Extrai partes do padrão dividindo pelo asterisco
            pattern_parts = pattern.split('*')
            found = False
            
            for file in os.listdir(category_dir):
                # Todos os fragmentos do padrão devem estar presentes no arquivo
                match = True
                for part in pattern_parts:
                    if part and part not in file:
                        match = False
                        break
                
                if match:
                    found = True
                    break
            
            visualization_status[category][pattern] = found
            if not found:
                logger.warning(f"Visualização não encontrada: {pattern} em {category_dir}")
    
    return visualization_status

def create_missing_directories(base_output_dir: str) -> None:
    """
    Cria diretórios para categorias de visualização que possam estar faltando.
    
    Args:
        base_output_dir: Diretório base de saída onde as visualizações são salvas.
    """
    # Categorias principais e subcategorias
    categories = [
        "descriptive", "correlation", "causality", "multi_round", 
        "anomaly_detection", "phase_comparison"
    ]
    
    # Estrutura específica para sliding window com subcategorias
    sliding_window_categories = [
        "sliding_window/correlation",
        "sliding_window/causality/granger",
        "sliding_window/causality/transfer_entropy"
    ]
    
    plots_dir = os.path.join(base_output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        logger.info(f"Criado diretório base de plots: {plots_dir}")
    
    # Cria categorias principais
    for category in categories:
        category_dir = os.path.join(plots_dir, category)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
            logger.info(f"Criado diretório: {category_dir}")
    
    # Cria estrutura específica para sliding window
    for subcategory in sliding_window_categories:
        subcategory_dir = os.path.join(plots_dir, subcategory)
        if not os.path.exists(subcategory_dir):
            os.makedirs(subcategory_dir)
            logger.info(f"Criado diretório: {subcategory_dir}")

def run_unified_pipeline(config_path: Optional[str] = None, 
                        data_root: Optional[str] = None, 
                        output_dir: Optional[str] = None,
                        run_sliding_window: bool = True,
                        run_multi_round: bool = True,
                        force_reprocess: bool = False,
                        input_parquet_path: Optional[str] = None) -> None:
    """
    Executa a versão unificada do pipeline que garante que todas as visualizações sejam geradas.
    
    Args:
        config_path: Caminho para arquivo de configuração YAML.
        data_root: Caminho para diretório raiz de dados.
        output_dir: Caminho para diretório de saída.
        run_sliding_window: Se deve executar análise com janelas deslizantes.
        run_multi_round: Se deve executar análise multi-round.
        force_reprocess: Se deve forçar o reprocessamento dos dados brutos mesmo se existir o arquivo parquet.
        input_parquet_path: Caminho para um arquivo parquet existente a ser carregado diretamente.
    """
    logger.info("Iniciando execução unificada do pipeline")
    start_time = datetime.now()
    
    # Suprimir avisos repetitivos do statsmodels para melhorar legibilidade do log
    with suppress_statsmodels_warnings():
        # Configuração dos argumentos
        args = argparse.Namespace()
        args.config = config_path
        args.data_root = data_root
        args.output_dir = output_dir
        args.input_parquet_path = input_parquet_path
        
        # Cria diretórios necessários
        if output_dir:
            create_missing_directories(output_dir)
        
        try:
            # Primeiro executa o pipeline com janelas deslizantes (completo)
            if run_sliding_window:
                logger.info("Executando pipeline com janelas deslizantes")
                pipeline = create_pipeline_with_sliding_window()
                
                if args.config and hasattr(pipeline, 'configure_from_yaml'):
                    pipeline.configure_from_yaml(args.config)
                
                if args.data_root:
                    pipeline.context["config"]["data_root"] = args.data_root
                if args.output_dir:
                    pipeline.context["config"]["output_dir"] = args.output_dir
                
                # Adiciona flag de force_reprocess ao contexto
                pipeline.context["force_reprocess"] = force_reprocess
                    
                pipeline.run()
            
            # Executa o pipeline para análise multi-round se solicitado
            if run_multi_round and not run_sliding_window:
                # Se não executamos o pipeline com janelas deslizantes, precisamos executar o multi-round separadamente
                # Já que o pipeline com janelas deslizantes já inclui o estágio multi-round
                logger.info("Executando análise multi-round separadamente")
                
                # Importamos o estágio de análise multi-round
                from src.analysis_multi_round import MultiRoundAnalysisStage
                from src.pipeline import Pipeline
                
                # Criamos um pipeline específico só para multi-round
                from src.pipeline import DataIngestionStage, DataExportStage
                
                # Define diretórios de saída
                output_base = output_dir or "outputs"
                multi_round_output = os.path.join(output_base, "multi_round")
                os.makedirs(multi_round_output, exist_ok=True)
                
                # Cria e configura um pipeline com os estágios necessários
                pipeline_multi_round = Pipeline([
                    DataIngestionStage(),
                    DataExportStage(),
                    MultiRoundAnalysisStage(output_dir=output_base)
                ])
                
                # Configura o pipeline
                if config_path:
                    pipeline_multi_round.configure_from_yaml(config_path)
                
                # Sobrescreve as configurações se necessário
                if data_root:
                    pipeline_multi_round.context["config"]["data_root"] = data_root
                if output_dir:
                    pipeline_multi_round.context["config"]["output_dir"] = output_dir
                
                # Adiciona flag de force_reprocess ao contexto
                pipeline_multi_round.context["force_reprocess"] = force_reprocess
                
                # Executa o pipeline multi-round
                result_multi_round = pipeline_multi_round.run()
                
                if result_multi_round and "error" in result_multi_round:
                    logger.error(f"Erro na análise multi-round: {result_multi_round.get('error')}")
                else:
                    logger.info(f"Análise multi-round concluída. Resultados salvos em: {multi_round_output}")
            
            # Executa apenas a análise de janelas deslizantes, se não estiver executando o pipeline completo
            # mas queremos gerar as visualizações de janelas deslizantes
            if not run_sliding_window and "sliding_window" not in verify_all_visualizations(output_dir or "outputs"):
                logger.info("Executando análise de janelas deslizantes independentemente")
                
                # Define diretórios de saída
                output_base = output_dir or "outputs"
                sliding_window_output = os.path.join(output_base, "plots", "sliding_window")
                os.makedirs(sliding_window_output, exist_ok=True)
                
                # Importa e carrega os dados
                from src.data_ingestion import ingest_experiment_data
                from src.analysis_sliding_window import SlidingWindowAnalyzer
                
                # Carrega dados (similar ao que o DataIngestionStage faria)
                config_dict = {}
                if config_path:
                    import yaml
                    with open(config_path, 'r') as f:
                        config_dict = yaml.safe_load(f)
                
                # Define data_root garantindo que seja uma string
                data_root_path = ""
                if data_root is not None:
                    data_root_path = data_root
                elif config_dict and 'data_root' in config_dict:
                    data_root_path = config_dict['data_root']
                else:
                    from src import config
                    data_root_path = config.DATA_ROOT
                
                # Usa os mesmos parâmetros do pipeline principal
                df_long = ingest_experiment_data(
                    data_root=data_root_path,  # Agora garantidamente uma string
                    selected_metrics=config_dict.get('selected_metrics'),
                    selected_tenants=config_dict.get('selected_tenants'),
                    selected_rounds=config_dict.get('selected_rounds')
                )
                
                if df_long is None or df_long.empty:
                    logger.error("Falha ao carregar dados para análise de janelas deslizantes")
                else:
                    # Executa análise de janelas deslizantes diretamente
                    logger.info("Iniciando análise de janelas deslizantes")
                    
                    # Cria analisador e executa
                    analyzer = SlidingWindowAnalyzer(df_long)
                    
                    # Processa cada combinação
                    experiments = df_long['experiment_id'].unique()
                    for experiment_id in experiments:
                        exp_df = df_long[df_long['experiment_id'] == experiment_id]
                        for round_id in exp_df['round_id'].unique():
                            round_df = exp_df[exp_df['round_id'] == round_id]
                            for metric in round_df['metric_name'].unique():
                                for phase in round_df['experimental_phase'].unique():
                                    # Análise de correlação deslizante
                                    logger.info(f"Calculando correlação deslizante para {metric}, {phase}, {round_id}")
                                    
                                    try:
                                        # Processa correlação deslizante
                                        corr_results = analyzer.analyze_correlation_sliding_window(
                                            metric=metric,
                                            phase=phase,
                                            round_id=round_id,
                                            window_size='5min',
                                            step_size='1min'
                                        )
                                        
                                        if corr_results:
                                            # Gera visualizações
                                            corr_dir = os.path.join(sliding_window_output, "correlation")
                                            os.makedirs(corr_dir, exist_ok=True)
                                            
                                            analyzer.plot_sliding_window_correlation(
                                                corr_results,
                                                metric,
                                                phase,
                                                round_id,
                                                corr_dir
                                            )
                                            
                                        # Processa causalidade deslizante
                                        logger.info(f"Calculando causalidade deslizante para {metric}, {phase}, {round_id}")
                                        
                                        # Importa o analisador de causalidade
                                        from src.analysis_causality import CausalityAnalyzer
                                        
                                        causality_analyzer = CausalityAnalyzer(round_df)
                                        causality_results = analyzer.analyze_causality_sliding_window(
                                            metric=metric,
                                            phase=phase,
                                            round_id=round_id,
                                            window_size='5min',
                                            step_size='1min',
                                            method='granger'
                                        )
                                        
                                        if causality_results:
                                            # Gera visualizações
                                            caus_dir = os.path.join(sliding_window_output, "causality", "granger")
                                            os.makedirs(caus_dir, exist_ok=True)
                                            
                                            analyzer.plot_sliding_window_causality(
                                                causality_results,
                                                metric,
                                                phase,
                                                round_id,
                                                caus_dir,
                                                method='granger'
                                            )
                                    except Exception as e:
                                        logger.error(f"Erro ao processar janelas deslizantes para {metric}, {phase}, {round_id}: {e}", exc_info=True)
            
            # Verifica as visualizações geradas
            visualization_status = verify_all_visualizations(
                output_dir or "outputs"
            )
            
            # Exibe resumo
            logger.info("Resumo de visualizações geradas:")
            for category, patterns in visualization_status.items():
                status_symbols = []
                for pattern, status in patterns.items():
                    symbol = '✅' if status else '❌'
                    status_symbols.append(f"{pattern}: {symbol}")
                
                # Verifica se pelo menos uma visualização foi gerada nesta categoria
                any_generated = any(patterns.values())
                category_status = '✅' if any_generated else '❌'
                logger.info(f"  {category_status} {category}:")
                for status in status_symbols:
                    logger.info(f"    - {status}")
            
            # Sugere próximos passos se houver visualizações faltando
            missing_visualizations = []
            for category, patterns in visualization_status.items():
                for pattern, status in patterns.items():
                    if not status:
                        missing_visualizations.append(f"{category}/{pattern}")
            
            if missing_visualizations:
                logger.info("\nSugestões para gerar visualizações faltantes:")
                if 'sliding_window' in [vis.split('/')[0] for vis in missing_visualizations]:
                    logger.info("  - Para gerar visualizações de janelas deslizantes: Execute novamente com '--no-multi-round'")
                
                if 'multi_round' in [vis.split('/')[0] for vis in missing_visualizations]:
                    logger.info("  - Para gerar visualizações de análise multi-round: Execute novamente com '--no-sliding-window'")
                    
            logger.info("\nExecução do pipeline unificado concluída.")
            
        except Exception as e:
            logger.error(f"Erro durante a execução do pipeline: {e}", exc_info=True)
        
        # Calcula tempo total de execução
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Execução concluída em {duration:.2f} segundos")

def main():
    """Função principal para execução a partir da linha de comando."""
    parser = argparse.ArgumentParser(description="Pipeline unificado para análise de séries temporais multi-tenant")
    parser.add_argument("--config", help="Caminho para arquivo de configuração YAML")
    parser.add_argument("--data-root", help="Diretório raiz dos dados de experimento")
    parser.add_argument("--output-dir", help="Diretório para salvar resultados")
    parser.add_argument("--no-sliding-window", action="store_true", help="Desativa análise com janelas deslizantes")
    parser.add_argument("--no-multi-round", action="store_true", help="Desativa análise multi-round")
    parser.add_argument("--force-reprocess", action="store_true", help="Força o reprocessamento dos dados brutos, mesmo se existir arquivo parquet")
    parser.add_argument("--input-parquet-path", help="Caminho para um arquivo parquet existente para ser usado diretamente")
    
    args = parser.parse_args()
    
    run_unified_pipeline(
        config_path=args.config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        run_sliding_window=not args.no_sliding_window,
        run_multi_round=not args.no_multi_round,
        force_reprocess=args.force_reprocess,
        input_parquet_path=args.input_parquet_path
    )

if __name__ == "__main__":
    main()
