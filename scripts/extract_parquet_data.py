#!/usr/bin/env python3
"""
Script para extrair dados específicos do arquivo Parquet consolidado.
Permite exportar subconjuntos de dados por métrica, fase ou round.
"""

import os
import argparse
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Extrai dados específicos do arquivo Parquet consolidado')
    parser.add_argument('--input', '-i', default='data/processed/sfi2_paper_consolidated.parquet',
                       help='Caminho para o arquivo Parquet consolidado')
    parser.add_argument('--output', '-o', default='data/processed/extracted',
                       help='Diretório para salvar os arquivos de saída')
    parser.add_argument('--metric', '-m', default=None, help='Filtrar por métrica específica')
    parser.add_argument('--phase', '-p', default=None, help='Filtrar por fase específica')
    parser.add_argument('--round', '-r', default=None, help='Filtrar por round específico')
    parser.add_argument('--component', '-c', default=None, help='Filtrar por componente específico')
    parser.add_argument('--stats', '-s', action='store_true', 
                       help='Gerar arquivo com estatísticas dos dados')
    parser.add_argument('--format', '-f', choices=['csv', 'parquet'], default='parquet',
                       help='Formato de saída (csv ou parquet)')
    args = parser.parse_args()
    
    # Criar diretório de saída se não existir
    os.makedirs(args.output, exist_ok=True)
    
    # Carregar o arquivo Parquet
    print(f"Carregando dados de {args.input}...")
    df = pd.read_parquet(args.input)
    
    # Aplicar filtros
    filtered_df = df.copy()
    filter_desc = []
    
    if args.metric:
        filtered_df = filtered_df[filtered_df['metric'] == args.metric]
        filter_desc.append(f"metric-{args.metric}")
    
    if args.phase:
        filtered_df = filtered_df[filtered_df['phase'] == args.phase]
        filter_desc.append(f"phase-{args.phase}")
    
    if args.round:
        filtered_df = filtered_df[filtered_df['round_id'] == args.round]
        filter_desc.append(f"round-{args.round}")
    
    if args.component:
        filtered_df = filtered_df[filtered_df['component'] == args.component]
        filter_desc.append(f"component-{args.component}")
    
    # Verificar se há dados após os filtros
    if len(filtered_df) == 0:
        print("Nenhum dado encontrado com os filtros especificados.")
        return
    
    # Gerar nome do arquivo de saída
    filter_suffix = "_".join(filter_desc) if filter_desc else "all"
    output_file = os.path.join(args.output, f"sfi2_{filter_suffix}.{args.format}")
    
    # Exportar dados filtrados
    print(f"Exportando {len(filtered_df)} linhas para {output_file}...")
    
    if args.format == 'csv':
        filtered_df.to_csv(output_file, index=False)
    else:
        filtered_df.to_parquet(output_file, index=False)
    
    # Gerar estatísticas se solicitado
    if args.stats:
        # Remover valores infinitos para calcular estatísticas
        stats_df = filtered_df.copy()
        stats_df['value'] = stats_df['value'].replace([np.inf, -np.inf], np.nan)
        
        # Calcular estatísticas por grupo
        if any([args.metric, args.phase, args.round, args.component]):
            # Determinar colunas de agrupamento
            group_cols = []
            if not args.metric:
                group_cols.append('metric')
            if not args.phase:
                group_cols.append('phase')
            if not args.round:
                group_cols.append('round_id')
            if not args.component:
                group_cols.append('component')
                
            # Se houver colunas para agrupar, gerar estatísticas por grupo
            if group_cols:
                stats = stats_df.groupby(group_cols)['value'].agg([
                    'count', 'mean', 'std', 'min', 'max'
                ]).reset_index()
            else:
                # Estatísticas globais se todos os filtros foram aplicados
                stats = pd.DataFrame({
                    'count': [len(stats_df)],
                    'mean': [stats_df['value'].mean()],
                    'std': [stats_df['value'].std()],
                    'min': [stats_df['value'].min()],
                    'max': [stats_df['value'].max()]
                })
                
            # Exportar estatísticas
            stats_file = os.path.join(args.output, f"stats_{filter_suffix}.csv")
            stats.to_csv(stats_file, index=False)
            print(f"Estatísticas exportadas para {stats_file}")
    
    print("Concluído!")


if __name__ == "__main__":
    main()
