#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script: generate_advanced_analysis.py
Description:
    Gera o arquivo analise_avancada.csv a partir dos dados consolidados de:
    - Causalidade (p-value, score)
    - Correlação (mean_correlation)
    - Impacto (mean_percentage_change)
    
    O arquivo gerado é usado pelo script multi_round_full_analysis.py para
    visualizações de causalidade reprodutível.
"""

import os
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger("generate_advanced_analysis")

def generate_advanced_analysis(causality_df, correlation_df, impact_df, output_path=None):
    """
    Gera o arquivo analise_avancada.csv com dados consolidados para análise avançada.
    
    Args:
        causality_df: DataFrame com dados de causalidade (p-value, score)
        correlation_df: DataFrame com dados de correlação (mean_correlation)
        impact_df: DataFrame com dados de impacto (mean_percentage_change)
        output_path: Caminho para salvar o arquivo (default: './analise_avancada.csv')
        
    Returns:
        DataFrame com os dados consolidados
    """
    if causality_df.empty or correlation_df.empty or impact_df.empty:
        logger.warning("Pelo menos um dos DataFrames de entrada está vazio.")
        return None
    
    output_path = output_path or 'analise_avancada.csv'
    
    try:
        # Cópia dos dataframes para não modificar os originais
        causality_all_df = causality_df.copy()
        correlation_all_df = correlation_df.copy() 
        impact_stats_df = impact_df.copy()
        
        # 1. Prepara dados de causalidade
        # Determinar limiar para significância baseado no quantil 75% dos scores
        score_threshold = causality_all_df['score'].quantile(0.75) if 'score' in causality_all_df.columns else 0
        
        # Criar flag de significância baseada em p-value ou score
        if 'p-value' in causality_all_df.columns and 'score' in causality_all_df.columns:
            causality_all_df['is_significant'] = ((causality_all_df['p-value'] < 0.05) | 
                                               (causality_all_df['score'] > score_threshold)).fillna(False)
        elif 'p-value' in causality_all_df.columns:
            causality_all_df['is_significant'] = (causality_all_df['p-value'] < 0.05).fillna(False)
        elif 'score' in causality_all_df.columns:
            causality_all_df['is_significant'] = (causality_all_df['score'] > score_threshold).fillna(False)
        else:
            logger.error("Dados de causalidade não têm as colunas esperadas (p-value ou score)")
            return None
        
        # Garantir que temos as colunas esperadas
        required_causality = {'phase', 'metric', 'source', 'target'}
        missing = required_causality - set(causality_all_df.columns)
        if missing:
            logger.error(f"Colunas ausentes nos dados de causalidade: {missing}")
            return None
            
        # Agregação de dados de causalidade
        causality_agg = causality_all_df.groupby(['phase', 'metric', 'source', 'target']).agg(
            mean_p_value=('p-value', 'mean') if 'p-value' in causality_all_df.columns else ('is_significant', 'count'),
            mean_score=('score', 'mean') if 'score' in causality_all_df.columns else ('is_significant', 'count'),
            significant_frequency=('is_significant', 'sum')
        ).reset_index()

        # 2. Prepara dados de correlação
        # Garantir que temos as colunas esperadas
        required_correlation = {'tenant1', 'tenant2', 'phase'}
        
        # Verificar se temos a coluna de correlação com algum dos nomes possíveis
        correlation_column = None
        for col_name in ['correlation', 'mean_correlation']:
            if col_name in correlation_all_df.columns:
                correlation_column = col_name
                break
                
        if correlation_column is None:
            logger.error("Dados de correlação não têm coluna 'correlation' ou 'mean_correlation'")
            return None
            
        if 'metric_name' in correlation_all_df.columns:
            correlation_all_df.rename(columns={'metric_name': 'metric'}, inplace=True)
        elif 'metric' not in correlation_all_df.columns:
            logger.error("Dados de correlação não têm coluna 'metric' ou 'metric_name'")
            return None
            
        missing = required_correlation - set(correlation_all_df.columns)
        if missing:
            logger.error(f"Colunas ausentes nos dados de correlação: {missing}")
            return None
            
        correlation_all_df['tenant_pair'] = correlation_all_df.apply(
            lambda r: '-'.join(sorted([r['tenant1'], r['tenant2']])), axis=1
        )
        causality_agg['tenant_pair'] = causality_agg.apply(
            lambda r: '-'.join(sorted([r['source'], r['target']])), axis=1
        )
        
        # 3. Prepara dados de impacto
        # Garantir que temos as colunas esperadas
        required_impact = {'tenant_id', 'mean_percentage_change', 'experimental_phase'}
        if 'metric_name' in impact_stats_df.columns:
            impact_stats_df.rename(columns={'metric_name': 'metric'}, inplace=True)
        elif 'metric' not in impact_stats_df.columns:
            logger.error("Dados de impacto não têm coluna 'metric' ou 'metric_name'")
            return None
            
        if 'experimental_phase' in impact_stats_df.columns and 'phase' not in impact_stats_df.columns:
            impact_stats_df.rename(columns={'experimental_phase': 'phase'}, inplace=True)
            
        missing = (required_impact - {'experimental_phase'}) - set(impact_stats_df.columns)
        if missing:
            logger.error(f"Colunas ausentes nos dados de impacto: {missing}")
            return None
            
        impact_source_df = impact_stats_df.rename(
            columns={
                'tenant_id': 'source',
                'mean_cohen_d': 'source_mean_cohen_d' if 'mean_cohen_d' in impact_stats_df.columns else None,
                'mean_percentage_change': 'source_mean_percentage_change'
            }
        )
        
        impact_target_df = impact_stats_df.rename(
            columns={
                'tenant_id': 'target',
                'mean_cohen_d': 'target_mean_cohen_d' if 'mean_cohen_d' in impact_stats_df.columns else None,
                'mean_percentage_change': 'target_mean_percentage_change'
            }
        )

        # 4. Faz o merge de tudo
        # Merge correlação
        correlation_merge_cols = ['phase', 'metric', 'tenant_pair', correlation_column]
        
        df_adv = pd.merge(
            causality_agg, 
            correlation_all_df[correlation_merge_cols], 
            on=['phase', 'metric', 'tenant_pair'], 
            how='left'
        )
        
        # Renomear coluna de correlação para 'mean_correlation' se necessário
        if correlation_column != 'mean_correlation':
            df_adv.rename(columns={correlation_column: 'mean_correlation'}, inplace=True)
        
        # Merge impact source
        impact_source_cols = ['phase', 'metric', 'source', 'source_mean_percentage_change']
        if 'source_mean_cohen_d' in impact_source_df.columns:
            impact_source_cols.append('source_mean_cohen_d')
            
        df_adv = pd.merge(
            df_adv, 
            impact_source_df[impact_source_cols], 
            on=['phase', 'metric', 'source'], 
            how='left'
        )
        
        # Merge impact target
        impact_target_cols = ['phase', 'metric', 'target', 'target_mean_percentage_change']
        if 'target_mean_cohen_d' in impact_target_df.columns:
            impact_target_cols.append('target_mean_cohen_d')
            
        df_adv = pd.merge(
            df_adv, 
            impact_target_df[impact_target_cols], 
            on=['phase', 'metric', 'target'], 
            how='left'
        )
        
        # Elimina coluna temporária e preenche NaNs
        df_adv = df_adv.drop(columns=['tenant_pair']).fillna(0)
        
        # Adicionar índice de risco (opcional)
        if 'source_mean_percentage_change' in df_adv.columns and 'target_mean_percentage_change' in df_adv.columns:
            df_adv['risk_index'] = df_adv['significant_frequency'] * abs(df_adv['target_mean_percentage_change'])

        # 5. Salva o novo CSV
        df_adv.to_csv(output_path, index=False)
        logger.info(f"Dataset avançado '{output_path}' foi gerado com sucesso.")
        
        return df_adv
        
    except Exception as e:
        logger.error(f"Erro ao gerar análise avançada: {e}")
        return None


if __name__ == "__main__":
    # Exemplo de uso direto do script
    import argparse
    
    parser = argparse.ArgumentParser(description="Gera arquivo analise_avancada.csv")
    parser.add_argument("--causality", required=True, help="Caminho para arquivo de causalidade consolidado")
    parser.add_argument("--correlation", required=True, help="Caminho para arquivo de correlação consolidado")
    parser.add_argument("--impact", required=True, help="Caminho para arquivo de impacto consolidado")
    parser.add_argument("--output", default="analise_avancada.csv", help="Caminho para salvar o arquivo")
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    # Carregar dados
    try:
        causality_df = pd.read_csv(args.causality)
        correlation_df = pd.read_csv(args.correlation)
        impact_df = pd.read_csv(args.impact)
        
        # Gerar análise avançada
        result_df = generate_advanced_analysis(
            causality_df, correlation_df, impact_df, args.output
        )
        
        if result_df is not None:
            print("\nVisualização do novo dataset:")
            print(result_df.head())
            
    except Exception as e:
        logger.error(f"Erro ao executar o script: {e}")
