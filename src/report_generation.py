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
from typing import Dict, List, Any, Optional, Union, Tuple
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

def aggregate_tenant_insights(
    tenant_metrics: pd.DataFrame,
    phase_comparison_results: Dict[str, pd.DataFrame],
    granger_matrices: Dict[str, pd.DataFrame],
    te_matrices: Dict[str, pd.DataFrame],
    correlation_matrices: Dict[str, Dict[str, pd.DataFrame]],
    anomaly_metrics: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Agrega insights sobre cada tenant a partir de todas as análises disponíveis.
    
    Args:
        tenant_metrics: DataFrame com métricas de cada tenant
        phase_comparison_results: Resultados da comparação entre fases
        granger_matrices: Matrizes de causalidade de Granger
        te_matrices: Matrizes de Transfer Entropy
        correlation_matrices: Matrizes de correlação
        anomaly_metrics: Dicionário com informações sobre anomalias detectadas
        
    Returns:
        Dicionário com insights consolidados para cada tenant
    """
    insights = {}
    
    # Inicialização de insights para cada tenant
    for tenant in tenant_metrics['tenant_id'].unique():
        insights[tenant] = {
            # Informações gerais
            'name': tenant,
            'rank': None,
            'noisy_score': 0.0,
            
            # Insights específicos
            'is_noisy_tenant': False,
            'is_victim_tenant': False,
            'main_impacted_tenants': [],
            'main_impact_sources': [],
            
            # Métricas com comportamento anômalo
            'anomalous_metrics': [],
            
            # Padrões detectados
            'attack_phase_patterns': [],
            'correlation_patterns': [],
            
            # Recomendações geradas
            'recommendations': []
        }
    
    # Preencher informações básicas de ranking e score
    for idx, row in enumerate(tenant_metrics.itertuples()):
        tenant = row.tenant_id
        insights[tenant]['rank'] = idx + 1
        insights[tenant]['noisy_score'] = row.noisy_score
        
        # Determinar se é um "tenant barulhento" (top 25%)
        if idx < len(tenant_metrics) // 4:
            insights[tenant]['is_noisy_tenant'] = True
    
    # Analisar causalidade para determinar relações de impacto
    for metric, matrix in te_matrices.items():
        if matrix.empty:
            continue
            
        # Para cada tenant como fonte de causalidade
        for tenant in insights.keys():
            if tenant not in matrix.columns:
                continue
                
            # Identificar os principais tenants impactados por este tenant
            # (maiores valores de TE indicam mais influência)
            if tenant in matrix.columns:
                te_values = matrix[tenant].drop(tenant) if tenant in matrix.index else matrix[tenant]
                if not te_values.empty:
                    # Top 2 tenants mais influenciados, com TE > 0.05
                    impacted = te_values[te_values > 0.05].nlargest(2)
                    for impacted_tenant, te_val in impacted.items():
                        impact_info = {
                            'tenant': impacted_tenant,
                            'score': float(te_val),
                            'metric': metric
                        }
                        insights[tenant]['main_impacted_tenants'].append(impact_info)
                        
                        # Marcar o tenant impactado como potencial "vítima"
                        if impacted_tenant in insights:
                            insights[impacted_tenant]['is_victim_tenant'] = True
                            insights[impacted_tenant]['main_impact_sources'].append({
                                'tenant': tenant,
                                'score': float(te_val),
                                'metric': metric
                            })
    
    # Analisar comparação entre fases para identificar padrões durante ataque
    for metric, stats_df in phase_comparison_results.items():
        if stats_df.empty:
            continue
            
        for _, row in stats_df.iterrows():
            tenant = row['tenant_id']
            if tenant not in insights:
                continue
                
            # Verificar variação significativa durante ataque
            attack_vs_baseline_col = '2 - Attack_vs_baseline_pct'
            if attack_vs_baseline_col in row and not pd.isna(row[attack_vs_baseline_col]):
                variation = row[attack_vs_baseline_col]
                if abs(variation) > 30:  # Variação maior que 30%
                    pattern = {
                        'metric': metric,
                        'variation_pct': float(variation),
                        'direction': 'increase' if variation > 0 else 'decrease'
                    }
                    insights[tenant]['attack_phase_patterns'].append(pattern)
    
    # Analisar anomalias, se disponíveis
    if anomaly_metrics and isinstance(anomaly_metrics, dict):
        for metric, anomalies_df in anomaly_metrics.items():
            if anomalies_df.empty:
                continue
                
            for tenant in anomalies_df['tenant_id'].unique():
                if tenant not in insights:
                    continue
                    
                tenant_anomalies = anomalies_df[anomalies_df['tenant_id'] == tenant]
                if not tenant_anomalies.empty:
                    insights[tenant]['anomalous_metrics'].append({
                        'metric': metric,
                        'anomaly_count': len(tenant_anomalies),
                        'max_zscore': float(tenant_anomalies['z_score'].max())
                    })
    
    # Gerar recomendações específicas para cada tenant
    for tenant, tenant_insight in insights.items():
        recommendations = []
        
        # Para tenants barulhentos
        if tenant_insight['is_noisy_tenant']:
            recommendations.append("Considerar ajuste de limites de recursos para evitar impacto nos outros tenants.")
            
            # Se tiver anomalias, adicionar recomendação específica
            if tenant_insight['anomalous_metrics']:
                metrics_list = [m['metric'] for m in tenant_insight['anomalous_metrics']]
                recommendations.append(f"Investigar picos anômalos de utilização nas métricas: {', '.join(metrics_list)}.")
        
        # Para tenants vítimas
        if tenant_insight['is_victim_tenant'] and tenant_insight['main_impact_sources']:
            impact_sources = [s['tenant'] for s in tenant_insight['main_impact_sources']]
            recommendations.append(f"Considerar isolamento de {tenant} dos tenants que o impactam: {', '.join(impact_sources)}.")
        
        # Para tenants com padrões detectados durante ataque
        if tenant_insight['attack_phase_patterns']:
            metrics_with_patterns = [p['metric'] for p in tenant_insight['attack_phase_patterns']]
            recommendations.append(f"Monitorar em tempo real as métricas que mostraram maior sensibilidade durante ataque: {', '.join(metrics_with_patterns)}.")
        
        tenant_insight['recommendations'] = recommendations
    
    return insights

def generate_comparative_table(tenant_insights: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Gera tabela comparativa final para análise inter-tenant.
    
    Args:
        tenant_insights: Dicionário com insights agregados de cada tenant
        
    Returns:
        DataFrame com tabela comparativa detalhada
    """
    # Estrutura para armazenar dados da tabela
    table_data = []
    
    for tenant_id, insights in tenant_insights.items():
        # Dados básicos
        tenant_data = {
            'tenant_id': tenant_id,
            'rank': insights['rank'],
            'impact_score': insights['noisy_score'],
            'is_noisy': insights['is_noisy_tenant'],
            'is_victim': insights['is_victim_tenant'],
        }
        
        # Contagem de métricas anômalas
        tenant_data['anomaly_count'] = len(insights['anomalous_metrics'])
        
        # Principais tenants impactados por este tenant
        impacted_tenants = insights['main_impacted_tenants']
        tenant_data['impacts_others'] = len(impacted_tenants) > 0
        tenant_data['main_impacted'] = ', '.join([f"{item['tenant']} ({item['score']:.2f})" 
                                              for item in impacted_tenants[:2]])
        
        # Principais fontes de impacto neste tenant
        impact_sources = insights['main_impact_sources']
        tenant_data['impacted_by_others'] = len(impact_sources) > 0
        tenant_data['main_sources'] = ', '.join([f"{item['tenant']} ({item['score']:.2f})" 
                                              for item in impact_sources[:2]])
        
        # Comportamento durante fase de ataque
        attack_patterns = insights['attack_phase_patterns']
        tenant_data['attack_pattern_count'] = len(attack_patterns)
        tenant_data['attack_patterns'] = ', '.join([f"{p['metric']} ({p['direction']}: {p['variation_pct']:.1f}%)" 
                                                 for p in attack_patterns[:2]])
        
        # Recomendações principais
        recommendations = insights['recommendations']
        tenant_data['recommendation_count'] = len(recommendations)
        tenant_data['top_recommendation'] = recommendations[0] if recommendations else ""
        
        table_data.append(tenant_data)
    
    # Criar DataFrame e ordenar por rank
    if not table_data:
        return pd.DataFrame()
        
    df = pd.DataFrame(table_data)
    df = df.sort_values(by='rank')
    
    return df


def plot_comparative_metrics(comparative_table: pd.DataFrame, out_dir: str) -> Optional[str]:
    """
    Gera visualização comparativa das métricas de análise inter-tenant.
    
    Args:
        comparative_table: Tabela comparativa gerada
        out_dir: Diretório para salvar a visualização
        
    Returns:
        Caminho para o arquivo gerado ou None se não houver dados
    """
    if comparative_table.empty:
        return None
    
    # Extrair dados relevantes para visualização
    plot_data = comparative_table[['tenant_id', 'impact_score', 'is_noisy', 'is_victim', 
                                  'anomaly_count', 'attack_pattern_count']].copy()
    
    # Converter para formato mais adequado para visualização
    plot_data['tenant_type'] = 'Normal'
    plot_data.loc[plot_data['is_noisy'], 'tenant_type'] = 'Noisy'
    plot_data.loc[plot_data['is_victim'], 'tenant_type'] = 'Victim'
    plot_data.loc[plot_data['is_noisy'] & plot_data['is_victim'], 'tenant_type'] = 'Noisy+Victim'
    
    # Gerar visualização comparativa
    plt.figure(figsize=(12, 8))
    
    # Configurar cores por tipo de tenant
    colors = {
        'Normal': 'gray',
        'Noisy': 'red',
        'Victim': 'blue',
        'Noisy+Victim': 'purple'
    }
    
    # Barplot principal
    ax = sns.barplot(
        x='tenant_id', 
        y='impact_score', 
        data=plot_data,
        hue='tenant_type',
        palette=colors,
        dodge=False
    )
    
    # Adicionar pontos para anomalias e padrões de ataque
    for idx, row in enumerate(plot_data.itertuples()):
        if row.anomaly_count > 0:
            plt.scatter(idx, row.impact_score + 0.05, marker='*', s=200, color='yellow', 
                      edgecolor='black', label='_nolegend_')
        
        if row.attack_pattern_count > 0:
            plt.scatter(idx, row.impact_score + 0.1, marker='^', s=100, color='orange', 
                      edgecolor='black', label='_nolegend_')
    
    # Legendas customizadas para os marcadores
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=15,
               markeredgecolor='black', label='Anomalias Detectadas'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10,
               markeredgecolor='black', label='Padrões em Fase de Ataque')
    ]
    
    # Combinar legendas
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + legend_elements, loc='best')
    
    plt.title('Análise Comparativa Inter-Tenant', fontsize=14, fontweight='bold')
    plt.xlabel('Tenant', fontsize=12)
    plt.ylabel('Score de Impacto', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Salvar visualização
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"comparative_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    
    return out_path
