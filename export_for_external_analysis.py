#!/usr/bin/env python3
"""
Script: export_for_external_analysis.py
Description: Utilitário para exportar análises específicas para ferramentas externas.

Este script permite exportar resultados de análises realizadas pelo pipeline para
formatos compatíveis com ferramentas de análise externa como Power BI, Tableau, etc.

Uso:
    ./export_for_external_analysis.py --config <arquivo_config> --output <diretorio_saida> [--round <id_rodada>] [--analyses <tipo_analise1,tipo_analise2>]

Exemplo:
    ./export_for_external_analysis.py --config config/pipeline_config_sfi2.yaml --output ./exports/powerbi --analyses impact,correlation
"""
import argparse
import json
import logging
import os
import pandas as pd
import numpy as np
import sys
import traceback
import yaml
from typing import List, Dict, Any, Optional

from src.config import PipelineConfig
from src.data_parquet_utils import ParquetDataManager, fix_parquet_generation
from src.data_export_utils import AnalysisDataExporter


def setup_logging(debug=False):
    """
    Configurar o sistema de logging.
    
    Args:
        debug: Se True, define o nível de log como DEBUG
    
    Returns:
        Logger configurado
    """
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )
    logger = logging.getLogger(__name__)
    
    if debug:
        logger.info("Modo de depuração ativado - mostrando logs detalhados")
        # Definir outros loggers específicos em DEBUG
        logging.getLogger('src.data_export_utils').setLevel(logging.DEBUG)
        logging.getLogger('src.data_parquet_utils').setLevel(logging.DEBUG)
    
    return logger


def load_existing_results(config: PipelineConfig, round_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Carrega os resultados existentes de análises prévias.
    
    Args:
        config: Configuração do pipeline
        round_id: ID da rodada (opcional)
        
    Returns:
        Dicionário com resultados ou None se não encontrado
    """
    logger = logging.getLogger(__name__)
    
    # Primeiro, tentamos carregar o arquivo parquet consolidado
    processed_data_path = config.get_processed_data_path()
    if processed_data_path and os.path.exists(processed_data_path):
        logger.info(f"Carregando dados processados de {processed_data_path}")
        try:
            df = pd.read_parquet(processed_data_path)
            if round_id:
                df = df[df['round_id'] == round_id]
                
            return {'consolidated_df': df}
        except Exception as e:
            logger.error(f"Erro ao carregar {processed_data_path}: {e}")
    
    logger.warning(f"Arquivo processado não encontrado em {processed_data_path}")
    return None


def export_analyses(config: PipelineConfig, output_dir: str, analyses: List[str], 
                   round_id: Optional[str] = None, results: Optional[Dict[str, Any]] = None,
                   output_format: str = 'parquet'):
    """
    Exporta análises específicas para o diretório de saída.
    
    Args:
        config: Configuração do pipeline
        output_dir: Diretório de saída
        analyses: Lista de tipos de análise a exportar
        round_id: ID da rodada (opcional)
        results: Resultados previamente carregados (opcional)
        output_format: Formato de saída (parquet, csv, excel, json, html)
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"export_analyses chamada com analyses={analyses}, round_id={round_id}, output_format={output_format}")
    
    # No modo de depuração, verificamos se estamos usando os dados de exemplo
    if results is not None:
        logger.debug(f"Usando dados pré-carregados. Chaves disponíveis: {list(results.keys())}")
        
        # Se forem dados de demonstração, adaptar ao formato esperado
        if 'descriptive' in results.keys() or 'correlation' in results.keys() or 'impact' in results.keys():
            logger.info("Detectados dados de demonstração - adaptando ao formato esperado pelo exportador")
            demo_data = results
            results = {}
            
            # Mapear dados de demonstração para formato de resultado
            if 'descriptive' in demo_data:
                results['descriptive_analysis'] = {'results': demo_data['descriptive']}
            if 'correlation' in demo_data:
                results['correlation_analysis'] = {'results': demo_data['correlation']}
            if 'impact' in demo_data:
                results['impact_analysis'] = {'results': demo_data['impact']}
            if 'multi_round' in demo_data:
                results['multi_round_analysis'] = {'results': demo_data['multi_round']}
            if 'phase' in demo_data:
                results['phase_comparison'] = {'results': demo_data['phase']}
            if 'causality' in demo_data:
                results['causality_analysis'] = {'results': demo_data['causality']}
                
            # No modo demo, criar estrutura per_round para compatibilidade
            if round_id:
                results['per_round'] = {
                    round_id: {
                        'descriptive_analysis': {'results': demo_data.get('descriptive', pd.DataFrame())},
                        'correlation_analysis': {'results': demo_data.get('correlation', pd.DataFrame())},
                        'impact_analysis': {'results': demo_data.get('impact', pd.DataFrame())},
                        'phase_comparison': {'results': demo_data.get('phase', pd.DataFrame())},
                        'causality_analysis': {'results': demo_data.get('causality', pd.DataFrame())}
                    }
                }
                
            logger.debug(f"Dados de demo reformatados. Novas chaves: {list(results.keys())}")
    
    # Se não temos resultados, tentar carregar
    if results is None:
        results = load_existing_results(config, round_id)
        if results is None:
            logger.error("Não foi possível carregar os resultados. Certifique-se de que o pipeline foi executado.")
            return False
    
    # Inicializar o exportador de dados
    data_exporter = AnalysisDataExporter(base_output_dir=output_dir, default_format=output_format)
    
    # Mapeamento de tipos de análise para nomes de diretório
    analysis_map = {
        'impact': 'impact_analysis',
        'correlation': 'correlation_analysis',
        'descriptive': 'descriptive_analysis',
        'causality': 'causality_analysis',
        'phase': 'phase_comparison',
        'fault': 'fault_tolerance_analysis',
        'multi': 'multi_round_analysis'
    }
    
    # Lista de análises exportadas com sucesso
    successful_exports = []
    
    # Exportar os dados consolidados primeiro
    if 'consolidated_df' in results and 'consolidated' in analyses:
        df = results['consolidated_df']
        
        # Depuração de tipos de dados
        if logger.level <= logging.DEBUG:
            debug_info = debug_dataframe_types(df)
            logger.debug(f"DataFrame para consolidated_data:\n{debug_info}")
        
        try:
            consolidated_path = data_exporter.save_analysis_result(
                df=df,
                analysis_type='consolidated_data',
                round_id=round_id,
                output_format=output_format
            )
            if consolidated_path:
                successful_exports.append(f"consolidated_data -> {consolidated_path}")
        except Exception as e:
            logger.error(f"Erro ao exportar consolidated_data: {str(e)}")
            # Tentar salvar em CSV se o formato original falhar
            if output_format != 'csv':
                logger.info(f"Tentando salvar consolidated_data em CSV como fallback")
                try:
                    fallback_path = data_exporter.save_analysis_result(
                        df=df,
                        analysis_type='consolidated_data',
                        round_id=round_id,
                        output_format='csv'
                    )
                    if fallback_path:
                        logger.info(f"Salvo com sucesso em CSV: {fallback_path}")
                        successful_exports.append(f"consolidated_data -> {fallback_path} (CSV fallback)")
                except Exception as csv_err:
                    logger.error(f"Também falhou ao salvar em CSV: {str(csv_err)}")
    
    # Verificar se temos resultados por rodada
    if 'per_round' in results and round_id:
        per_round_results = results['per_round']
        if round_id in per_round_results:
            round_results = per_round_results[round_id]
            
            # Exportar cada tipo de análise solicitada
            for analysis_short, analysis_full in analysis_map.items():
                if analysis_short in analyses and analysis_full in round_results:
                    # Exportar resultados desta análise
                    result_dfs = extract_dataframes_from_analysis(round_results[analysis_full])
                    
                    for result_name, df in result_dfs.items():
                        # Depuração de tipos de dados
                        if logger.level <= logging.DEBUG:
                            debug_info = debug_dataframe_types(df)
                            logger.debug(f"DataFrame para {analysis_full}/{result_name}:\n{debug_info}")
                            
                        # Tenta exportar, com tratamento de erros
                        try:
                            result_path = data_exporter.save_analysis_result(
                                df=df,
                                analysis_type=analysis_full,
                                round_id=round_id,
                                sub_type=result_name,
                                output_format=output_format
                            )
                            if result_path:
                                successful_exports.append(f"{analysis_full}:{result_name} -> {result_path}")
                        except Exception as e:
                            logger.error(f"Erro ao exportar {analysis_full}/{result_name}: {str(e)}")
                            # Tentar salvar em CSV se o formato original falhar
                            if output_format != 'csv':
                                logger.info(f"Tentando salvar {analysis_full}/{result_name} em CSV como fallback")
                                try:
                                    fallback_path = data_exporter.save_analysis_result(
                                        df=df,
                                        analysis_type=analysis_full,
                                        round_id=round_id,
                                        sub_type=result_name,
                                        output_format='csv'
                                    )
                                    if fallback_path:
                                        logger.info(f"Salvo com sucesso em CSV: {fallback_path}")
                                        successful_exports.append(f"{analysis_full}:{result_name} -> {fallback_path} (CSV fallback)")
                                except Exception as csv_err:
                                    logger.error(f"Também falhou ao salvar em CSV: {str(csv_err)}")
    
    # Processar todas as análises regulares que não estão em rodada específica
    for analysis_short, analysis_full in analysis_map.items():
        if analysis_short in analyses and analysis_full in results and analysis_short != 'multi':
            logger.debug(f"Processando análise global {analysis_short} ({analysis_full})")
            # Exportar resultados desta análise
            result_dfs = extract_dataframes_from_analysis(results[analysis_full])
            
            for result_name, df in result_dfs.items():
                # Depuração de tipos de dados
                if logger.level <= logging.DEBUG:
                    debug_info = debug_dataframe_types(df)
                    logger.debug(f"DataFrame para {analysis_full}/{result_name}:\n{debug_info}")
                    
                # Tenta exportar, com tratamento de erros
                try:
                    result_path = data_exporter.save_analysis_result(
                        df=df,
                        analysis_type=analysis_full,
                        sub_type=result_name,
                        output_format=output_format
                    )
                    if result_path:
                        successful_exports.append(f"{analysis_full}:{result_name} -> {result_path}")
                except Exception as e:
                    logger.error(f"Erro ao exportar {analysis_full}/{result_name}: {str(e)}")
                    # Tentar salvar em CSV se o formato original falhar
                    if output_format != 'csv':
                        logger.info(f"Tentando salvar {analysis_full}/{result_name} em CSV como fallback")
                        try:
                            fallback_path = data_exporter.save_analysis_result(
                                df=df,
                                analysis_type=analysis_full,
                                sub_type=result_name,
                                output_format='csv'
                            )
                            if fallback_path:
                                logger.info(f"Salvo com sucesso em CSV: {fallback_path}")
                                successful_exports.append(f"{analysis_full}:{result_name} -> {fallback_path} (CSV fallback)")
                        except Exception as csv_err:
                            logger.error(f"Também falhou ao salvar em CSV: {str(csv_err)}")

    # Se multi-round foi solicitado e está disponível
    if 'multi' in analyses and 'multi_round_analysis' in results:
        logger.debug(f"Processando análise multi-round")
        multi_results = results['multi_round_analysis']
        result_dfs = extract_dataframes_from_analysis(multi_results)
        
        for result_name, df in result_dfs.items():
            # Depuração de tipos de dados
            if logger.level <= logging.DEBUG:
                debug_info = debug_dataframe_types(df)
                logger.debug(f"DataFrame para multi_round_analysis/{result_name}:\n{debug_info}")
            
            # Tenta exportar, com tratamento de erros
            try:
                result_path = data_exporter.save_analysis_result(
                    df=df,
                    analysis_type='multi_round_analysis',
                    sub_type=result_name,
                    output_format=output_format
                )
                if result_path:
                    successful_exports.append(f"multi_round_analysis:{result_name} -> {result_path}")
            except Exception as e:
                logger.error(f"Erro ao exportar multi_round_analysis/{result_name}: {str(e)}")
                # Tentar salvar em CSV se o formato original falhar
                if output_format != 'csv':
                    logger.info(f"Tentando salvar multi_round_analysis/{result_name} em CSV como fallback")
                    try:
                        fallback_path = data_exporter.save_analysis_result(
                            df=df,
                            analysis_type='multi_round_analysis',
                            sub_type=result_name,
                            output_format='csv'
                        )
                        if fallback_path:
                            logger.info(f"Salvo com sucesso em CSV: {fallback_path}")
                            successful_exports.append(f"multi_round_analysis:{result_name} -> {fallback_path} (CSV fallback)")
                    except Exception as csv_err:
                        logger.error(f"Também falhou ao salvar em CSV: {str(csv_err)}")
    
    # Para debugging, vamos exibir mais informações
    logger.debug(f"Análises solicitadas: {analyses}")
    logger.debug(f"Análises disponíveis: {list(analysis_map.keys())}")
    logger.debug(f"Resultados disponíveis: {list(results.keys())}")
    if 'per_round' in results and round_id:
        logger.debug(f"Resultados por rodada disponíveis: {list(results['per_round'].get(round_id, {}).keys())}")

    # Relatório final de exportação
    if successful_exports:
        logger.info(f"Exportação concluída com sucesso. {len(successful_exports)} análises exportadas:")
        for export in successful_exports:
            logger.info(f"  - {export}")
        return True
    else:
        logger.warning("Nenhuma análise foi exportada. Verifique os tipos de análise solicitados.")
        # Sugerir possíveis soluções
        if not results:
            logger.warning("Não foram encontrados resultados para processar.")
        elif len(results) == 1 and 'consolidated_df' in results and 'consolidated' not in analyses:
            logger.warning("Apenas dados consolidados estão disponíveis. Adicione 'consolidated' à lista de análises.")
        return False


def extract_dataframes_from_analysis(analysis_results: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """
    Extrai DataFrames de resultados de análise.
    
    Args:
        analysis_results: Resultados de uma análise específica
        
    Returns:
        Dicionário com nome -> DataFrame
    """
    dataframes = {}
    logger = logging.getLogger(__name__)
    logger.debug(f"Extraindo DataFrames de análise com chaves: {list(analysis_results.keys())}")
    
    # Caso especial para dados de demonstração: pode ser diretamente um DataFrame
    if isinstance(analysis_results, pd.DataFrame):
        logger.debug("Resultado é diretamente um DataFrame")
        dataframes['results'] = analysis_results
        return dataframes
    
    # Caso especial: resultados é apenas {'results': DataFrame}
    if len(analysis_results) == 1 and 'results' in analysis_results and isinstance(analysis_results['results'], pd.DataFrame):
        logger.debug("Resultado contém apenas um DataFrame em 'results'")
        dataframes['results'] = analysis_results['results']
        return dataframes
        
    # Procurar por dataframes nos resultados (busca padrão)
    for key, value in analysis_results.items():
        if isinstance(value, pd.DataFrame):
            logger.debug(f"Encontrado DataFrame em chave '{key}'")
            dataframes[key] = value
        elif key.endswith('_df') and isinstance(value, pd.DataFrame):
            # Chaves que terminam com _df são provavelmente DataFrames
            logger.debug(f"Encontrado DataFrame em chave '{key}' (sufixo _df)")
            dataframes[key] = value
    
    # Se não encontramos dataframes diretamente, tentar processamento específico
    if not dataframes:
        # Tentar processar resultados de análise descritiva
        if 'summary_stats' in analysis_results:
            stats_dict = analysis_results['summary_stats']
            stats_rows = []
            
            for metric_name, metric_stats in stats_dict.items():
                for phase, phase_stats in metric_stats.items():
                    for tenant, tenant_stats in phase_stats.items():
                        for stat_name, stat_value in tenant_stats.items():
                            stats_rows.append({
                                'metric_name': metric_name,
                                'experimental_phase': phase,
                                'tenant_id': tenant,
                                'statistic': stat_name,
                                'value': stat_value
                            })
            
            if stats_rows:
                dataframes['summary_stats_df'] = pd.DataFrame(stats_rows)
        
        # Tentar processar resultados de causalidade
        if 'granger_results' in analysis_results:
            granger_dict = analysis_results['granger_results']
            granger_rows = []
            
            for phase, phase_results in granger_dict.items():
                for cause, effects in phase_results.items():
                    for effect, stats in effects.items():
                        granger_rows.append({
                            'experimental_phase': phase,
                            'cause': cause,
                            'effect': effect,
                            'p_value': stats.get('p_value'),
                            'f_value': stats.get('f_value'),
                            'lag': stats.get('lag')
                        })
            
            if granger_rows:
                dataframes['granger_results_df'] = pd.DataFrame(granger_rows)
    
    return dataframes


def generate_sample_data() -> dict:
    """
    Generate sample data for demo mode.
    
    Returns:
        Dictionary of DataFrames with sample data
    """
    # Create sample descriptive statistics
    descriptive_stats = pd.DataFrame({
        'metric': ['cpu_usage', 'memory_usage', 'network_latency', 'disk_io'],
        'mean': [45.3, 2048.6, 12.3, 150.8],
        'median': [42.1, 1986.3, 10.5, 142.1],
        'std_dev': [12.3, 512.4, 5.2, 45.6],
        'min': [10.2, 512.0, 2.1, 50.3],
        'max': [95.7, 4096.0, 45.8, 350.2]
    })
    
    # Create sample correlation data
    correlations = pd.DataFrame({
        'variable_x': ['cpu_usage', 'memory_usage', 'network_latency', 'disk_io'],
        'variable_y': ['response_time', 'response_time', 'response_time', 'response_time'],
        'pearson': [0.85, 0.72, 0.65, 0.58],
        'spearman': [0.82, 0.70, 0.67, 0.61],
        'p_value': [0.001, 0.005, 0.01, 0.02]
    })
    
    # Create sample impact analysis
    impact = pd.DataFrame({
        'component': ['api-server', 'etcd', 'controller', 'scheduler', 'proxy'],
        'impact_score': [0.92, 0.87, 0.75, 0.68, 0.45],
        'affected_services': [
            json.dumps(['frontend', 'auth', 'user-service']),
            json.dumps(['all']),
            json.dumps(['deployment', 'replicaset']),
            json.dumps(['pod-placement']),
            json.dumps(['service-discovery'])
        ],
        'mean_degradation': [42.5, 78.3, 35.2, 28.7, 15.3]
    })
    
    # Create sample multi-round analysis
    multi_round = pd.DataFrame({
        'round': [1, 2, 3, 4, 5],
        'success_rate': [0.95, 0.92, 0.85, 0.75, 0.65],
        'avg_response_time': [120, 145, 210, 350, 480],
        'error_count': [12, 24, 56, 98, 145],
        'affected_components': [
            json.dumps({'api-server': 0.2, 'controller': 0.1}),
            json.dumps({'api-server': 0.4, 'etcd': 0.3}),
            json.dumps({'api-server': 0.6, 'etcd': 0.5, 'controller': 0.3}),
            json.dumps({'api-server': 0.8, 'etcd': 0.7, 'controller': 0.6}),
            json.dumps({'api-server': 0.9, 'etcd': 0.8, 'controller': 0.7, 'scheduler': 0.5})
        ]
    })
    
    # Create sample phase comparison analysis
    phase_comparison = pd.DataFrame({
        'phase': ['1 - Baseline', '2 - Noise Injection', '3 - Recovery', '1 - Baseline', '2 - Noise Injection', '3 - Recovery'],
        'metric': ['cpu_usage', 'cpu_usage', 'cpu_usage', 'memory_usage', 'memory_usage', 'memory_usage'],
        'mean_value': [32.1, 78.5, 41.3, 1024.5, 2450.8, 1350.3],
        'percent_change': [0.0, 144.5, 28.7, 0.0, 139.2, 31.8],
        'p_value': [None, 0.001, 0.03, None, 0.001, 0.02],
        'is_significant': [False, True, True, False, True, True]
    })
    
    # Create sample causality analysis
    causality = pd.DataFrame({
        'cause': ['cpu_usage', 'memory_usage', 'disk_io', 'network_latency', 'api_server_errors'],
        'effect': ['response_time', 'response_time', 'throughput', 'error_rate', 'pod_failures'],
        'granger_p_value': [0.003, 0.012, 0.045, 0.008, 0.001],
        'lag': [2, 3, 1, 2, 1],
        'phase': ['2 - Noise Injection', '2 - Noise Injection', '3 - Recovery', '2 - Noise Injection', '3 - Recovery']
    })
    
    return {
        'descriptive': descriptive_stats,
        'correlation': correlations,
        'impact': impact,
        'multi_round': multi_round,
        'phase': phase_comparison,
        'causality': causality
    }

def export_for_external_validation(config: PipelineConfig, output_dir: str, round_id: Optional[str] = None):
    """
    Exporta dados específicos para validação externa em ferramentas como R, MATLAB, Python com outros pacotes, etc.
    Foca principalmente em dados para análise de Correlação, Causalidade de Granger e Transfer Entropy.
    
    Args:
        config: Configuração do pipeline
        output_dir: Diretório para os arquivos de saída
        round_id: ID da rodada específica (opcional)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Iniciando exportação para validação externa. Output dir: {output_dir}")
    
    # Cria estrutura de diretórios
    external_dirs = {
        'correlation': os.path.join(output_dir, 'correlation'),
        'granger': os.path.join(output_dir, 'granger'),
        'transfer_entropy': os.path.join(output_dir, 'transfer_entropy'),
        'timeseries': os.path.join(output_dir, 'timeseries')
    }
    
    for dir_name, dir_path in external_dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        logger.debug(f"Criado diretório: {dir_path}")
    
    # Carrega dados - obter caminhos para arquivos de análise
    output_base = config.get_base_output_dir()
    multi_round_dir = os.path.join(output_base, 'default_experiment', 'multi_round_analysis')
    correlation_path = os.path.join(multi_round_dir, 'multi_round_correlation_all.csv')
    causality_path = os.path.join(multi_round_dir, 'multi_round_causality_all.csv')
    
    # Obtém caminho para dados brutos (Parquet)
    processed_data_path = config.get_processed_data_path()
    
    # Verifica existência dos arquivos
    files_exist = {
        'correlation': os.path.exists(correlation_path),
        'causality': os.path.exists(causality_path),
        'processed_data': os.path.exists(processed_data_path) if processed_data_path else False
    }
    
    logger.info(f"Arquivos encontrados: {files_exist}")
    
    # 1. Exportar dados de Correlação
    if files_exist['correlation']:
        logger.info("Processando dados de correlação")
        try:
            corr_df = pd.read_csv(correlation_path)
            
            # Normaliza nomes de colunas
            if 'metric_name' in corr_df.columns and 'metric' not in corr_df.columns:
                corr_df = corr_df.rename(columns={'metric_name': 'metric'})
            
            # Agrupa por fase e métrica
            for (phase, metric), group in corr_df.groupby(['phase', 'metric']):
                # Cria matriz de correlação completa
                tenants = sorted(list(set(group['tenant1']) | set(group['tenant2'])))
                corr_matrix = pd.DataFrame(np.eye(len(tenants)), index=tenants, columns=tenants)
                
                for _, row in group.iterrows():
                    t1, t2, corr_value = row['tenant1'], row['tenant2'], row['mean_correlation']
                    corr_matrix.loc[t1, t2] = corr_matrix.loc[t2, t1] = corr_value
                    
                # Salva a matriz de correlação
                safe_metric = metric.replace('/', '_')
                safe_phase = phase.replace(' ', '_')
                filename = f"correlation_matrix_{safe_phase}_{safe_metric}.csv"
                corr_matrix.to_csv(os.path.join(external_dirs['correlation'], filename))
                logger.info(f"Salvo: {filename}")
            
            logger.info("Dados de correlação exportados com sucesso")
        except Exception as e:
            logger.error(f"Erro ao exportar dados de correlação: {e}")
    
    # 2. Exportar dados para Granger
    if files_exist['causality']:
        logger.info("Processando dados de causalidade")
        try:
            causality_df = pd.read_csv(causality_path)
            
            # Normaliza nomes de colunas
            if 'metric_name' in causality_df.columns and 'metric' not in causality_df.columns:
                causality_df = causality_df.rename(columns={'metric_name': 'metric'})
            
            # Seleciona colunas relevantes para Granger
            granger_columns = ['phase', 'metric', 'source', 'target', 'p-value', 'score']
            if all(col in causality_df.columns for col in granger_columns):
                granger_df = causality_df[granger_columns]
                
                # Agrupa por fase e métrica
                for (phase, metric), group in granger_df.groupby(['phase', 'metric']):
                    safe_metric = metric.replace('/', '_')
                    safe_phase = phase.replace(' ', '_')
                    filename = f"granger_results_{safe_phase}_{safe_metric}.csv"
                    group.to_csv(os.path.join(external_dirs['granger'], filename), index=False)
                    logger.info(f"Salvo: {filename}")
                
                logger.info("Dados de Granger exportados com sucesso")
            else:
                missing = [col for col in granger_columns if col not in causality_df.columns]
                logger.warning(f"Colunas necessárias não encontradas: {missing}")
        except Exception as e:
            logger.error(f"Erro ao exportar dados de Granger: {e}")
    
    # 3. Exportar dados para Transfer Entropy (TE)
    if files_exist['causality']:
        logger.info("Processando dados para Transfer Entropy")
        try:
            causality_df = pd.read_csv(causality_path)
            
            # Normaliza nomes de colunas
            if 'metric_name' in causality_df.columns and 'metric' not in causality_df.columns:
                causality_df = causality_df.rename(columns={'metric_name': 'metric'})
            
            # Para TE, precisamos dos pares fonte-destino
            te_columns = ['phase', 'metric', 'source', 'target']
            if all(col in causality_df.columns for col in te_columns):
                te_df = causality_df[te_columns].drop_duplicates()
                
                # Agrupa por fase e métrica
                for (phase, metric), group in te_df.groupby(['phase', 'metric']):
                    safe_metric = metric.replace('/', '_')
                    safe_phase = phase.replace(' ', '_')
                    filename = f"transfer_entropy_pairs_{safe_phase}_{safe_metric}.csv"
                    group.to_csv(os.path.join(external_dirs['transfer_entropy'], filename), index=False)
                    logger.info(f"Salvo: {filename}")
                
                logger.info("Dados para Transfer Entropy exportados com sucesso")
            else:
                missing = [col for col in te_columns if col not in causality_df.columns]
                logger.warning(f"Colunas necessárias não encontradas: {missing}")
        except Exception as e:
            logger.error(f"Erro ao exportar dados para Transfer Entropy: {e}")
    
    # 4. Exportar séries temporais brutas (opcional, mas útil para análises personalizadas)
    if files_exist['processed_data'] and processed_data_path:  # Garantir que não é None
        logger.info("Processando séries temporais brutas")
        try:
            # Carrega o arquivo Parquet de dados processados
            raw_df = pd.read_parquet(processed_data_path)
            
            if round_id:
                raw_df = raw_df[raw_df['round_id'] == round_id]
            
            # Exporta séries temporais por fase/métrica
            phases = raw_df['experimental_phase'].unique()
            metrics = raw_df['metric_name'].unique()
            
            for phase in phases:
                for metric in metrics:
                    subset = raw_df[(raw_df['experimental_phase'] == phase) & 
                                   (raw_df['metric_name'] == metric)]
                    
                    if not subset.empty:
                        # Pivot para formato mais adequado para ferramentas externas
                        pivot_df = subset.pivot_table(
                            index='timestamp', 
                            columns='tenant_id', 
                            values='metric_value'
                        )
                        
                        # Salva como CSV (formato mais universal)
                        safe_metric = metric.replace('/', '_')
                        safe_phase = phase.replace(' ', '_')
                        filename = f"timeseries_{safe_phase}_{safe_metric}.csv"
                        pivot_df.to_csv(os.path.join(external_dirs['timeseries'], filename))
                        logger.info(f"Salvo: {filename}")
            
            logger.info("Séries temporais exportadas com sucesso")
        except Exception as e:
            logger.error(f"Erro ao exportar séries temporais: {e}")
    
    # 5. Criar arquivo README com instruções de uso
    readme_content = """# Dados para Validação Externa de Correlação, Granger e Transfer Entropy

Este diretório contém dados exportados para validação em ferramentas externas como R, MATLAB, Python (com outros pacotes), etc.

## Estrutura de Diretórios

- `correlation/`: Matrizes de correlação por fase e métrica
- `granger/`: Resultados dos testes de causalidade de Granger
- `transfer_entropy/`: Pares para análise de Transfer Entropy
- `timeseries/`: Séries temporais brutas em formato pivot (timestamps × tenants)

## Exemplos de Uso

### R

```r
# Correlação
corr_matrix <- read.csv("correlation/correlation_matrix_Baseline_cpu_usage.csv", row.names=1)
library(corrplot)
corrplot(as.matrix(corr_matrix))

# Granger Causality
library(vars)
granger_pairs <- read.csv("transfer_entropy/transfer_entropy_pairs_Baseline_cpu_usage.csv")
timeseries_data <- read.csv("timeseries/timeseries_Baseline_cpu_usage.csv")

# Para cada par em granger_pairs, fazer teste:
source_col <- granger_pairs$source[1]
target_col <- granger_pairs$target[1]
var_data <- timeseries_data[, c(target_col, source_col)]
var_model <- VAR(var_data, p=3, type="const")
causality(var_model, cause=source_col)
```

### MATLAB

```matlab
% Correlação
corr_data = readtable("correlation/correlation_matrix_Baseline_cpu_usage.csv");
corr_matrix = table2array(corr_data(:, 2:end));
imagesc(corr_matrix);
colorbar;

% Transfer Entropy
te_pairs = readtable("transfer_entropy/transfer_entropy_pairs_Baseline_cpu_usage.csv");
ts_data = readtable("timeseries/timeseries_Baseline_cpu_usage.csv");

% Para cada par em te_pairs, calcular TE:
source = te_pairs.source{1};
target = te_pairs.target{1};
source_data = ts_data{:, source};
target_data = ts_data{:, target};
te_value = transferentropy(source_data, target_data);
```

### Python

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

# Correlação
corr_matrix = pd.read_csv("correlation/correlation_matrix_Baseline_cpu_usage.csv", index_col=0)
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Granger Causality
pairs = pd.read_csv("transfer_entropy/transfer_entropy_pairs_Baseline_cpu_usage.csv")
ts_data = pd.read_csv("timeseries/timeseries_Baseline_cpu_usage.csv", index_col=0)

# Para cada par, testar causalidade
for _, row in pairs.iterrows():
    source, target = row['source'], row['target']
    test_data = ts_data[[target, source]].dropna()
    result = grangercausalitytests(test_data, maxlag=5, verbose=False)
    p_values = [result[i+1][0]['ssr_ftest'][1] for i in range(5)]
    print(f"{source} -> {target}: min p-value = {min(p_values):.4f}")
```

## Validação Cruzada

Estes dados foram exportados para permitir a validação dos resultados obtidos no pipeline principal usando ferramentas estatísticas alternativas. Compare os resultados obtidos em ferramentas externas com os salvos em:

- Correlação: `multi_round_correlation_all.csv`
- Granger: `multi_round_causality_all.csv`

Gerado em: {datetime}
"""
    
    # Adiciona a data/hora atual
    from datetime import datetime
    readme_content = readme_content.replace('{datetime}', datetime.now().strftime('%Y-%m-%d %H:%M'))
    
    # Salva o README
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    logger.info(f"Arquivo README.md criado em {output_dir}")
    logger.info("Exportação para validação externa concluída com sucesso.")


def debug_dataframe_types(df: pd.DataFrame) -> str:
    """
    Generate a debug report of DataFrame column types.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        String with debug information
    """
    debug_info = []
    debug_info.append(f"DataFrame shape: {df.shape}")
    debug_info.append("Column types:")
    
    for col in df.columns:
        dtype = str(df[col].dtype)
        sample = df[col].iloc[0] if not df.empty else None
        sample_type = type(sample).__name__
        sample_repr = str(sample)[:50] + "..." if len(str(sample)) > 50 else str(sample)
        
        debug_info.append(f"  - {col}: dtype={dtype}, sample_type={sample_type}, sample={sample_repr}")
        
        # Check for mixed types
        if df[col].dtype == 'object':
            types_in_column = set()
            for val in df[col].dropna().head(10):
                types_in_column.add(type(val).__name__)
            if len(types_in_column) > 1:
                debug_info.append(f"    * Mixed types detected: {', '.join(types_in_column)}")
    
    return "\n".join(debug_info)


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(description="Exportar análises para ferramentas externas.")
    parser.add_argument('--config', type=str, required=True, help='Caminho para o arquivo de configuração YAML.')
    parser.add_argument('--output', type=str, required=True, help='Diretório para os arquivos de saída.')
    parser.add_argument('--round', type=str, help='ID da rodada específica (opcional).')
    parser.add_argument('--analyses', type=str, default='impact,correlation,descriptive',
                      help='Lista de tipos de análise a exportar, separados por vírgula. Valores possíveis: impact, correlation, descriptive, causality, phase, fault, multi, consolidated.')
    parser.add_argument('--format', type=str, default='parquet', choices=['parquet', 'csv', 'excel', 'json', 'html'],
                      help='Formato de saída dos arquivos. Valores possíveis: parquet, csv, excel, json, html. Padrão: parquet')
    parser.add_argument('--demo', action='store_true', help='Gerar dados de exemplo para demonstração/teste.')
    parser.add_argument('--external-validation', action='store_true', 
                      help='Exportar dados em formatos otimizados para validação externa de correlação, Granger e Transfer Entropy.')
    parser.add_argument('--debug', action='store_true', help='Ativar modo de depuração com logs detalhados.')
    args = parser.parse_args()

    logger = setup_logging(debug=args.debug)
    
    # Carregar configuração
    logger.info(f"Carregando configuração de {args.config}")
    try:
        # Passamos diretamente o caminho do arquivo para o PipelineConfig
        config = PipelineConfig(args.config)
    except Exception as e:
        logger.error(f"Erro ao carregar configuração: {e}")
        return 1
    
    # Criar diretório de saída
    os.makedirs(args.output, exist_ok=True)
    
    # Analisar a lista de análises
    analyses = [a.strip() for a in args.analyses.split(',')]
    
    # Se --demo foi especificado, usar dados de exemplo
    results = None
    if args.demo:
        logger.info("Gerando dados de exemplo para demonstração.")
        results = generate_sample_data()
        # Se estamos no modo demo e round_id não foi especificado, usar round-1
        if args.round is None:
            args.round = 'round-1'
            logger.info(f"Usando round_id padrão para demo: {args.round}")
    
    # Modo especial para validação externa
    if args.external_validation:
        logger.info("Modo de validação externa ativado - exportando dados otimizados para ferramentas externas")
        export_for_external_validation(config=config, output_dir=args.output, round_id=args.round)
        return 0
    
    # Exportar análises (modo padrão)
    success = export_analyses(
        config=config, 
        output_dir=args.output, 
        analyses=analyses, 
        round_id=args.round,
        output_format=args.format,
        results=results
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
