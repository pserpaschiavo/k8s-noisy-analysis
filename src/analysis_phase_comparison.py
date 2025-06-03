"""
Module: analysis_phase_comparison.py
Description: Implementa análises comparativas entre diferentes fases experimentais.

Este módulo oferece funcionalidade para comparar métricas e resultados entre as fases 
experimentais (baseline, ataque, recuperação), incluindo:

1. Comparação de estatísticas descritivas
2. Comparação de correlação/covariância
3. Comparação de causalidade
4. Visualizações comparativas
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union

# Configuração dos gráficos
plt.style.use('tableau-colorblind10')

def compute_phase_stats_comparison(df: pd.DataFrame, metric: str, round_id: str, tenants: List[str] = None) -> pd.DataFrame:
    """
    Calcula estatísticas comparativas de uma métrica entre diferentes fases para cada tenant.
    
    Args:
        df: DataFrame long com os dados
        metric: Nome da métrica para análise
        round_id: ID do round para análise
        tenants: Lista de tenants para analisar. Se None, usa todos.
        
    Returns:
        DataFrame com estatísticas comparativas por tenant e fase.
    """
    # Filtra os dados para o metric e round especificados
    subset = df[(df['metric_name'] == metric) & (df['round_id'] == round_id)]
    
    # Se tenants não for especificado, usa todos disponíveis
    if tenants is None:
        tenants = subset['tenant_id'].unique().tolist()
    
    # Lista de fases no formato esperado
    phases = ['1 - Baseline', '2 - Attack', '3 - Recovery']
    available_phases = [p for p in phases if p in subset['experimental_phase'].unique()]
    
    # Estrutura para armazenar resultados
    results = []
    
    # Para cada combinação de tenant e fase
    for tenant in tenants:
        tenant_data = {}
        tenant_data['tenant_id'] = tenant
        
        # Métricas base para o tenant (em baseline)
        baseline_data = subset[(subset['tenant_id'] == tenant) & 
                              (subset['experimental_phase'] == '1 - Baseline')]
        
        # Se não houver dados de baseline, usa médias gerais
        if len(baseline_data) == 0:
            baseline_stats = {
                'mean': subset['metric_value'].mean(),
                'std': subset['metric_value'].std(),
                'median': subset['metric_value'].median(),
                'min': subset['metric_value'].min(),
                'max': subset['metric_value'].max()
            }
        else:
            baseline_stats = {
                'mean': baseline_data['metric_value'].mean(),
                'std': baseline_data['metric_value'].std(),
                'median': baseline_data['metric_value'].median(),
                'min': baseline_data['metric_value'].min(),
                'max': baseline_data['metric_value'].max()
            }
        
        # Para cada fase, calcula estatísticas e variação relativa ao baseline
        for phase in available_phases:
            phase_data = subset[(subset['tenant_id'] == tenant) & 
                               (subset['experimental_phase'] == phase)]
            
            if len(phase_data) == 0:
                # Tenant não presente nesta fase
                tenant_data[f'{phase}_present'] = False
                continue
                
            tenant_data[f'{phase}_present'] = True
            tenant_data[f'{phase}_mean'] = phase_data['metric_value'].mean()
            tenant_data[f'{phase}_std'] = phase_data['metric_value'].std()
            tenant_data[f'{phase}_median'] = phase_data['metric_value'].median()
            tenant_data[f'{phase}_min'] = phase_data['metric_value'].min()
            tenant_data[f'{phase}_max'] = phase_data['metric_value'].max()
            
            # Calcula variação percentual em relação ao baseline (se baseline existir)
            if phase != '1 - Baseline' and '1 - Baseline' in available_phases:
                baseline_mean = baseline_stats['mean']
                if baseline_mean != 0:  # Evita divisão por zero
                    tenant_data[f'{phase}_vs_baseline_pct'] = ((tenant_data[f'{phase}_mean'] - baseline_mean) / 
                                                            abs(baseline_mean)) * 100
                else:
                    tenant_data[f'{phase}_vs_baseline_pct'] = np.nan
        
        results.append(tenant_data)
    
    # Converte para DataFrame
    result_df = pd.DataFrame(results)
    
    return result_df

def plot_phase_comparison(stats_df: pd.DataFrame, metric: str, out_path: str) -> None:
    """
    Gera visualização comparativa de métricas por fase e tenant.
    
    Args:
        stats_df: DataFrame com estatísticas por tenant e fase
        metric: Nome da métrica sendo visualizada
        out_path: Caminho para salvar a visualização
    """
    if stats_df.empty:
        logging.warning(f"DataFrame vazio para visualização de {metric}")
        return
    
    # Extrai columns relevantes
    phases = ['1 - Baseline', '2 - Attack', '3 - Recovery']
    tenants = stats_df['tenant_id'].tolist()
    
    # Prepara dados para plotagem
    plot_data = []
    for _, row in stats_df.iterrows():
        tenant = row['tenant_id']
        for phase in phases:
            if f'{phase}_present' in row and row[f'{phase}_present']:
                plot_data.append({
                    'tenant_id': tenant,
                    'phase': phase,
                    'mean': row[f'{phase}_mean'],
                    'std': row[f'{phase}_std'] if f'{phase}_std' in row else 0,
                    'vs_baseline': row[f'{phase}_vs_baseline_pct'] if f'{phase}_vs_baseline_pct' in row else 0
                })
    
    plot_df = pd.DataFrame(plot_data)
    if plot_df.empty:
        logging.warning(f"Dados insuficientes para visualização de {metric}")
        return
    
    # Cria subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
    
    # Plot 1: Médias por fase e tenant
    sns.barplot(x='tenant_id', y='mean', hue='phase', data=plot_df, ax=ax1)
    ax1.set_title(f'Média de {metric} por Tenant e Fase')
    ax1.set_xlabel('Tenant')
    ax1.set_ylabel(f'{metric} (média)')
    
    # Plot 2: Variação percentual em relação ao baseline
    baseline_plot_df = plot_df[plot_df['phase'] != '1 - Baseline']
    if not baseline_plot_df.empty:
        sns.barplot(x='tenant_id', y='vs_baseline', hue='phase', data=baseline_plot_df, ax=ax2)
        ax2.set_title(f'Variação % de {metric} em relação ao Baseline')
        ax2.set_xlabel('Tenant')
        ax2.set_ylabel('Variação % vs. Baseline')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    
    # Ajusta layout
    plt.tight_layout()
    
    # Salva figura
    plt.savefig(out_path)
    plt.close()
    
    logging.info(f"Visualização de comparação de fases salva em {out_path}")

def compare_correlation_matrices(corr_matrices: Dict[str, pd.DataFrame], metric: str, out_path: str) -> pd.DataFrame:
    """
    Compara matrizes de correlação entre diferentes fases experimentais.
    
    Args:
        corr_matrices: Dicionário de matrizes de correlação por fase
        metric: Nome da métrica analisada
        out_path: Caminho para salvar a visualização
        
    Returns:
        DataFrame mostrando as diferenças mais significativas entre fases
    """
    if not corr_matrices or len(corr_matrices) < 2:
        logging.warning("Insuficiente número de matrizes para comparação")
        return pd.DataFrame()
    
    # Garante que todas as matrizes têm os mesmos índices/colunas
    tenants = set()
    for mat in corr_matrices.values():
        tenants.update(mat.index)
    tenants = sorted(list(tenants))
    
    # Adapta todas as matrizes para ter os mesmos tenants
    for phase, mat in corr_matrices.items():
        for tenant in tenants:
            if tenant not in mat.index:
                mat.loc[tenant] = np.nan
                mat[tenant] = np.nan
    
    # Calcula diferenças entre fases
    phases = sorted(list(corr_matrices.keys()))
    diff_matrices = {}
    
    for i, phase1 in enumerate(phases):
        for j, phase2 in enumerate(phases):
            if i >= j:  # Evita duplicação e comparação de uma fase com ela mesma
                continue
                
            mat1 = corr_matrices[phase1]
            mat2 = corr_matrices[phase2]
            
            # Calcula matriz de diferenças
            diff_mat = mat2 - mat1
            diff_matrices[f"{phase1}_to_{phase2}"] = diff_mat
    
    # Prepara visualização de diferenças
    if diff_matrices:
        n_diffs = len(diff_matrices)
        fig, axes = plt.subplots(1, n_diffs, figsize=(7*n_diffs, 6))
        
        if n_diffs == 1:  # Garante que axes seja sempre um iterable
            axes = [axes]
            
        for ax, (diff_name, diff_mat) in zip(axes, diff_matrices.items()):
            # Plot heatmap
            im = sns.heatmap(diff_mat, vmin=-1, vmax=1, center=0, cmap='RdBu_r',
                      annot=True, fmt=".2f", ax=ax)
            ax.set_title(f'Diferença de Correlação: {diff_name}\n({metric})')
        
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        
        # Encontra as diferenças mais significativas
        results = []
        for diff_name, diff_mat in diff_matrices.items():
            # Transforma matriz em formato "long" para análise
            diff_long = diff_mat.stack().reset_index()
            diff_long.columns = ['tenant1', 'tenant2', 'diff']
            
            # Remove comparações na diagonal e duplicadas
            diff_long = diff_long[(diff_long['tenant1'] != diff_long['tenant2'])]
            
            # Seleciona top diferenças (positivas e negativas)
            top_pos = diff_long.nlargest(5, 'diff')
            top_neg = diff_long.nsmallest(5, 'diff')
            
            for _, row in pd.concat([top_pos, top_neg]).iterrows():
                results.append({
                    'phase_transition': diff_name,
                    'tenant1': row['tenant1'],
                    'tenant2': row['tenant2'],
                    'correlation_change': row['diff'],
                    'metric': metric
                })
        
        return pd.DataFrame(results)
    
    return pd.DataFrame()

class PhaseComparisonAnalyzer:
    """
    Classe para análises comparativas entre fases experimentais.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o analisador.
        
        Args:
            df: DataFrame long com dados de todas as fases
        """
        self.df = df
        self.logger = logging.getLogger("phase_comparison")
    
    def analyze_metric_across_phases(self, metric: str, round_id: str, output_dir: str) -> pd.DataFrame:
        """
        Realiza análise completa de uma métrica através das diferentes fases.
        
        Args:
            metric: Nome da métrica para análise
            round_id: ID do round para análise
            output_dir: Diretório onde salvar visualizações
            
        Returns:
            DataFrame com dados comparativos
        """
        # Assegura que o diretório existe
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Calcula estatísticas comparativas
        stats_df = compute_phase_stats_comparison(
            self.df, 
            metric=metric,
            round_id=round_id
        )
        
        # 2. Gera visualização comparativa
        out_path = os.path.join(output_dir, f'phase_comparison_{metric}_{round_id}.png')
        plot_phase_comparison(stats_df, metric, out_path)
        
        return stats_df
