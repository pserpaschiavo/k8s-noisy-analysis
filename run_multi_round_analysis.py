import argparse
import logging
import pandas as pd
from src.analysis_multi_round import MultiRoundAnalysisStage
from src.parse_config import load_parse_config

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("run_multi_round_analysis")

def main():
    """
    Ponto de entrada principal para executar a análise de múltiplos rounds.
    """
    parser = argparse.ArgumentParser(description="Executa a análise de múltiplos rounds a partir de um arquivo de configuração.")
    parser.add_argument('--config', type=str, required=True, help='Caminho para o arquivo de configuração YAML do pipeline.')
    args = parser.parse_args()

    logger.info(f"Carregando configuração de {args.config}")
    config = load_parse_config(args.config)

    # Carregar dados processados
    processed_data_dir = config.get('processed_data_dir', './data/processed')
    output_parquet_name = config.get('output_parquet_name', 'sfi2_paper.parquet')
    
    # Construir o caminho completo para o arquivo parquet
    input_parquet_path = f"{processed_data_dir}/{output_parquet_name}"
    fallback_path = f"./data/processed/{output_parquet_name}"
    
    logger.info(f"Tentando carregar dados processados de {input_parquet_path}")
    try:
        df_long = pd.read_parquet(input_parquet_path)
        logger.info(f"Dados processados carregados com sucesso de {input_parquet_path}")
    except FileNotFoundError:
        logger.warning(f"Arquivo não encontrado em {input_parquet_path}. Tentando caminho alternativo: {fallback_path}")
        try:
            df_long = pd.read_parquet(fallback_path)
            logger.info(f"Dados processados carregados com sucesso de {fallback_path}")
        except FileNotFoundError:
            logger.error(f"Arquivo de dados processados não encontrado em nenhum dos caminhos. Execute o pipeline principal primeiro.")
            return

    # Diretório de saída para esta análise
    output_dir = config.get('output_dir', './outputs/sfi2-paper-analysis')

    # Preparar o contexto inicial para o estágio
    # Em um pipeline real, o contexto seria passado entre os estágios.
    # Aqui, simulamos o contexto necessário para a análise multi-round.
    # Precisamos das matrizes de causalidade de cada round, que não estão no df_long.
    # Por enquanto, vamos passar o df_long e o diretório de saída.
    # O estágio MultiRoundAnalysisStage é inteligente o suficiente para lidar com isso.
    initial_context = {
        'df_long': df_long,
        'output_dir': output_dir,
        'selected_metrics': config.get('selected_metrics', []),
        'selected_tenants': config.get('selected_tenants', []),
        'experiment_id': config.get('experiment_id', 'sfi2-paper')
    }

    # Instanciar e executar o estágio de análise multi-round
    logger.info("Iniciando o estágio de análise de múltiplos rounds.")
    multi_round_output_dir = f"{output_dir}/multi_round_analysis"
    multi_round_stage = MultiRoundAnalysisStage(output_dir=multi_round_output_dir)
    final_context = multi_round_stage.execute(initial_context)

    if 'error' in final_context:
        logger.error(f"A análise de múltiplos rounds falhou: {final_context['error']}")
    else:
        logger.info(f"Análise de múltiplos rounds concluída com sucesso. Resultados salvos em: {multi_round_output_dir}")

if __name__ == "__main__":
    main()
