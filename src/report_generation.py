"""
Module: report_generation.py
Description: Geração de relatórios e agregação de insights para análise multi-tenant.

Este módulo implementa funcionalidades para:
1. Consolidação de resultados de todas as análises
2. Identificação de "tenants barulhentos" com base em métricas objetivas
3. Geração de relatório final e tabela comparativa inter-tenant
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configuração dos gráficos
plt.style.use('tableau-colorblind10')
logger = logging.getLogger("report_generation")

def generate_tenant_metrics(
    granger_matrices: Dict[str, pd.DataFrame],
    te_matrices: Dict[str, pd.DataFrame],
    correlation_matrices: Dict[str, Dict[str, pd.DataFrame]],
    phase_comparison_results: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Gera métricas de impacto para cada tenant com base nas análises.
    
    Args:
        granger_matrices: Matrizes de causalidade de Granger
        te_matrices: Matrizes de Transfer Entropy
        correlation_matrices: Matrizes de correlação por métrica e fase
        phase_comparison_results: Resultados da comparação entre fases
        
    Returns:
        DataFrame com métricas e ranking de "barulho" de cada tenant
    """
    # Estrutura para armazenar métricas de cada tenant
    tenant_metrics = {}
    
    # Coleta todos os tenants presentes em qualquer matriz
    all_tenants = set()
    for matrix_dict in [granger_matrices, te_matrices]:
        for matrix in matrix_dict.values():
            if not matrix.empty:
                all_tenants.update(matrix.index)
    
    for corr_dict in correlation_matrices.values():
        for matrix in corr_dict.values():
            if not matrix.empty:
                all_tenants.update(matrix.index)
    
    # Inicializa métricas para cada tenant
    for tenant in all_tenants:
        tenant_metrics[tenant] = {
            'causality_impact_score': 0.0,  # Quanto este tenant causa impacto nos outros
            'causality_affected_score': 0.0,  # Quanto este tenant é afetado por outros
            'correlation_strength': 0.0,  # Força média das correlações com outros tenants
            'phase_variation': 0.0,  # Variação média entre fases (especialmente ataque)
            'noisy_score': 0.0,  # Score final para classificar tenants "barulhentos"
            'metrics_count': 0  # Contador para normalização
        }
    
    # 1. Análise de causalidade de Granger
    for key, matrix in granger_matrices.items():
        if matrix.empty:
            continue
            
        # Para cada tenant como fonte de causalidade
        for source in matrix.columns:
            if source not in tenant_metrics:
                continue
                
            # Valores baixos de p-valor = alta causalidade
            # Transformamos para 1 - p_value para que valores altos = mais causalidade
            causal_values = 1.0 - matrix[source].values
            # Remove diagonal e NaNs
            causal_values = [v for v in causal_values if not np.isnan(v) and v < 1.0]
            
            if causal_values:
                tenant_metrics[source]['causality_impact_score'] += np.mean(causal_values)
                tenant_metrics[source]['metrics_count'] += 1
    
    # 2. Análise de Transfer Entropy
    for key, matrix in te_matrices.items():
        if matrix.empty:
            continue
            
        for source in matrix.columns:
            if source not in tenant_metrics:
                continue
                
            # TE: valores mais altos = mais causalidade
            te_values = matrix[source].values
            # Remove diagonal e NaNs
            te_values = [v for v in te_values if not np.isnan(v) and v > 0]
            
            if te_values:
                tenant_metrics[source]['causality_impact_score'] += np.mean(te_values) * 5  # Peso maior para TE
                tenant_metrics[source]['metrics_count'] += 1
    
    # 3. Análise de correlação
    for metric_key, phase_dict in correlation_matrices.items():
        for phase, matrix in phase_dict.items():
            if matrix.empty:
                continue
                
            for tenant in matrix.index:
                if tenant not in tenant_metrics:
                    continue
                    
                # Valores absolutos de correlação (ignorando autocorrelação)
                corr_values = matrix.loc[tenant].abs().values
                corr_values = [v for v in corr_values if not np.isnan(v) and v < 1.0]
                
                if corr_values:
                    tenant_metrics[tenant]['correlation_strength'] += np.mean(corr_values)
                    tenant_metrics[tenant]['metrics_count'] += 1
    
    # 4. Análise de variação entre fases
    for key, stats_df in phase_comparison_results.items():
        if stats_df.empty:
            continue
            
        for _, row in stats_df.iterrows():
            tenant = row['tenant_id']
            if tenant not in tenant_metrics:
                continue
                
            # Variação percentual entre fases (especialmente ataque vs baseline)
            attack_vs_baseline_col = '2 - Attack_vs_baseline_pct'
            if attack_vs_baseline_col in row and not pd.isna(row[attack_vs_baseline_col]):
                variation = abs(row[attack_vs_baseline_col])
                tenant_metrics[tenant]['phase_variation'] += variation
                tenant_metrics[tenant]['metrics_count'] += 1
    
    # Normaliza e calcula score final
    for tenant, metrics in tenant_metrics.items():
        # Evita divisão por zero
        count = max(metrics['metrics_count'], 1)
        
        # Normalização
        metrics['causality_impact_score'] /= count
        metrics['correlation_strength'] /= count
        metrics['phase_variation'] /= count
        
        # Calcula o score final (ponderado)
        metrics['noisy_score'] = (
            metrics['causality_impact_score'] * 0.5 +  # 50% para causalidade
            metrics['correlation_strength'] * 0.3 +    # 30% para correlação
            metrics['phase_variation'] * 0.2           # 20% para variação de fase
        )
    
    # Converte para DataFrame, ordenado por score total
    df = pd.DataFrame.from_dict(tenant_metrics, orient='index')
    df.index.name = 'tenant_id'
    df = df.reset_index()
    df = df.sort_values(by='noisy_score', ascending=False)
    
    return df

def generate_markdown_report(
    tenant_metrics: pd.DataFrame,
    context: Dict[str, Any],
    rank_plot_path: str,
    metrics_table_path: str,
    out_dir: str
) -> str:
    """
    Gera relatório final em formato Markdown.
    
    Args:
        tenant_metrics: DataFrame com métricas de cada tenant
        context: Contexto com resultados de todas as análises
        rank_plot_path: Caminho para o plot de ranking de tenants
        metrics_table_path: Caminho para tabela completa de métricas
        out_dir: Diretório de saída para o relatório
        
    Returns:
        Caminho para o arquivo de relatório gerado
    """
    report_filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report_path = os.path.join(out_dir, f"{report_filename}.md")
    
    with open(report_path, 'w') as f:
        f.write("# Relatório de Análise Multi-Tenant\n\n")
        f.write(f"**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Seção de tenants barulhentos
        f.write("## Identificação de Tenants com Maior Impacto\n\n")
        
        # Identifica o tenant mais barulhento
        if not tenant_metrics.empty:
            top_tenant = tenant_metrics.iloc[0]['tenant_id']
            top_score = tenant_metrics.iloc[0]['noisy_score']
            f.write(f"**Tenant com maior impacto:** `{top_tenant}` (score: {top_score:.2f})\n\n")
            f.write(f"![Ranking de Tenants]({os.path.basename(rank_plot_path)})\n\n")
            
            # Tabela comparativa de tenants
            f.write("### Tabela Comparativa de Tenants\n\n")
            f.write("| Tenant | Score Total | Impacto Causal | Força de Correlação | Variação entre Fases |\n")
            f.write("|--------|------------|---------------|---------------------|----------------------|\n")
            
            for _, row in tenant_metrics.iterrows():
                f.write(f"| {row['tenant_id']} | {row['noisy_score']:.2f} | {row['causality_impact_score']:.2f} | ")
                f.write(f"{row['correlation_strength']:.2f} | {row['phase_variation']:.2f} |\n")
            
            f.write("\n*Tabela completa disponível em:* ")
            f.write(f"`{os.path.basename(metrics_table_path)}`\n\n")
        
        # Seção de visualizações
        f.write("## Visualizações Geradas\n\n")
        
        # Plots descritivos
        desc_plots = context.get('descriptive_plot_paths', [])
        if desc_plots:
            f.write(f"### Análise Descritiva ({len(desc_plots)} visualizações)\n\n")
            for i, path in enumerate(desc_plots[:3]):  # Limita a 3 exemplos
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(desc_plots) > 3:
                f.write(f"- *...e mais {len(desc_plots) - 3} visualizações*\n")
            f.write("\n")
                
        # Plots de correlação
        corr_plots = context.get('correlation_plot_paths', [])
        if corr_plots:
            f.write(f"### Análise de Correlação ({len(corr_plots)} visualizações)\n\n")
            for i, path in enumerate(corr_plots[:3]):
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(corr_plots) > 3:
                f.write(f"- *...e mais {len(corr_plots) - 3} visualizações*\n")
            f.write("\n")
                
        # Plots de causalidade
        causality_plots = context.get('causality_plot_paths', [])
        if causality_plots:
            f.write(f"### Análise de Causalidade ({len(causality_plots)} visualizações)\n\n")
            for i, path in enumerate(causality_plots[:3]):
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(causality_plots) > 3:
                f.write(f"- *...e mais {len(causality_plots) - 3} visualizações*\n")
            f.write("\n")
        
        # Plots de comparação entre fases
        phase_plots = context.get('phase_comparison_plot_paths', [])
        if phase_plots:
            f.write(f"### Comparação entre Fases ({len(phase_plots)} visualizações)\n\n")
            for i, path in enumerate(phase_plots[:3]):
                f.write(f"- [{os.path.basename(path)}]({os.path.relpath(path, out_dir)})\n")
            if len(phase_plots) > 3:
                f.write(f"- *...e mais {len(phase_plots) - 3} visualizações*\n")
            f.write("\n")
        
        # Metodologia
        f.write("## Metodologia\n\n")
        f.write("Este relatório utiliza uma metodologia de análise multi-dimensional para identificação de tenants com maior impacto:\n\n")
        f.write("1. **Análise de Causalidade**:\n")
        f.write("   - Causalidade de Granger: Testa se valores passados de um tenant ajudam a prever valores futuros de outro.\n")
        f.write("   - Transfer Entropy: Quantifica a transferência de informação direcional entre séries temporais.\n\n")
        f.write("2. **Análise de Correlação**:\n")
        f.write("   - Mede a força da relação linear entre métricas de diferentes tenants.\n")
        f.write("   - Valores mais altos indicam maior interdependência.\n\n")
        f.write("3. **Variação entre Fases**:\n")
        f.write("   - Quantifica a magnitude da alteração nas métricas durante fases de ataque vs. baseline.\n")
        f.write("   - Tenants com maior variação são mais sensíveis ao ambiente.\n\n")
        
        f.write("**O score final é calculado como média ponderada:**\n")
        f.write("- 50% Impacto Causal (maior peso para causalidade detectada via Transfer Entropy)\n")
        f.write("- 30% Força de Correlação\n")
        f.write("- 20% Variação entre Fases\n\n")
        
        f.write("### Limitações da Metodologia\n\n")
        f.write("- A causalidade estatística não implica necessariamente causalidade física direta.\n")
        f.write("- Correlação não implica causalidade; pode refletir fatores externos comuns.\n")
        f.write("- A análise presume séries temporais adequadamente amostradas e estacionárias.\n")
    
    return report_path

def generate_tenant_ranking_plot(tenant_metrics: pd.DataFrame, output_path: str) -> None:
    """
    Gera visualização do ranking de tenants por impacto.
    
    Args:
        tenant_metrics: DataFrame com métricas de tenants
        output_path: Caminho para salvar o plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot principal do score total
    ax = sns.barplot(
        x='tenant_id', 
        y='noisy_score', 
        data=tenant_metrics,
        palette="viridis"
    )
    
    # Adiciona valores sobre cada barra
    for i, v in enumerate(tenant_metrics['noisy_score']):
        ax.text(i, v + 0.05, f"{v:.2f}", ha='center', fontsize=9)
    
    plt.title('Ranking de "Tenants Barulhentos"\n(Maior Score = Maior Impacto no Ambiente)', fontsize=14)
    plt.xlabel('Tenant', fontsize=12)
    plt.ylabel('Score de Impacto', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Salva o plot
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    logger.info(f"Ranking de tenants salvo em: {output_path}")
