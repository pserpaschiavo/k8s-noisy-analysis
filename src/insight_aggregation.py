"""
Module: insight_aggregation.py
Description: Funções para agregação de insights inter-tenant e metodologias de comparação.

Este módulo complementa o report_generation.py fornecendo funcionalidades específicas para:
1. Agregação de insights sobre cada tenant
2. Tabelas comparativas inter-tenant
3. Visualizações comparativas para análise
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
import matplotlib.lines as mlines

# Configuração dos gráficos
plt.style.use('tableau-colorblind10')
logger = logging.getLogger("insight_aggregation")

def aggregate_tenant_insights(
    tenant_metrics: Optional[pd.DataFrame] = None,
    phase_comparison_results: Optional[Dict[str, pd.DataFrame]] = None,
    granger_matrices: Optional[Dict[str, pd.DataFrame]] = None,
    te_matrices: Optional[Dict[str, pd.DataFrame]] = None,
    correlation_matrices: Optional[Dict[str, Dict[str, pd.DataFrame]]] = None,
    anomaly_metrics: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict[str, Any]:
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
        Dicionário com insights consolidados para cada tenant ou mensagem de erro
    """
    # Dicionário para armazenar os insights
    insights: Dict[str, Dict[str, Any]] = {}
    
    # Verificar e registrar o status dos dados de entrada para debugging
    logger.info(f"Status dos dados para agregação de insights:")
    logger.info(f"- tenant_metrics tipo: {type(tenant_metrics)}")
    if tenant_metrics is not None:
        if hasattr(tenant_metrics, 'columns'):
            logger.info(f"  - Colunas disponíveis: {list(tenant_metrics.columns) if not tenant_metrics.empty else 'DataFrame vazio'}")
        elif isinstance(tenant_metrics, dict):
            logger.info(f"  - Chaves disponíveis: {list(tenant_metrics.keys()) if tenant_metrics else 'Dicionário vazio'}")
    logger.info(f"- phase_comparison_results presente: {phase_comparison_results is not None}")
    logger.info(f"- granger_matrices presente: {granger_matrices is not None}")
    logger.info(f"- te_matrices presente: {te_matrices is not None}")
    logger.info(f"- correlation_matrices presente: {correlation_matrices is not None}")
    logger.info(f"- anomaly_metrics presente: {anomaly_metrics is not None}")
    
    # Garantir que tenant_metrics é um DataFrame
    if tenant_metrics is None:
        logger.warning("tenant_metrics não disponível. Tentando criar um DataFrame básico a partir de outros dados.")
        
        # Tenta extrair uma lista de tenants de outras matrizes disponíveis
        tenants = set()
        
        # Tenta extrair de matrizes de correlação
        if correlation_matrices is not None:
            for method, matrices in correlation_matrices.items():
                for matrix_key, matrix in matrices.items():
                    if isinstance(matrix, pd.DataFrame):
                        tenants.update(matrix.index)
        
        # Tenta extrair de matrizes de causalidade
        if granger_matrices is not None:
            for key, matrix in granger_matrices.items():
                if isinstance(matrix, pd.DataFrame):
                    tenants.update(matrix.index)
                    
        # Tenta extrair de matrizes de TE
        if te_matrices is not None:
            for key, matrix in te_matrices.items():
                if isinstance(matrix, pd.DataFrame):
                    tenants.update(matrix.index)
        
        # Se encontramos algum tenant, cria um DataFrame básico
        if tenants:
            logger.info(f"Criando DataFrame básico com {len(tenants)} tenants: {tenants}")
            tenant_metrics = pd.DataFrame({
                'tenant_id': list(tenants),
                'noisy_score': [0.5] * len(tenants)  # Valor padrão neutro
            })
        else:
            logger.error("Não foi possível extrair lista de tenants de nenhum dos dados disponíveis")
            return {"error_message": "Dados necessários para agregação de insights não disponíveis"}
    
    # Converter para DataFrame se for um dicionário (compatibilidade)
    if isinstance(tenant_metrics, dict):
        try:
            # Primeiro verificar se é um dicionário que representa um DataFrame 
            # (chaves são índices e valores são dicionários de colunas)
            if tenant_metrics and isinstance(list(tenant_metrics.values())[0], dict):
                # Exemplo: {'tenant1': {'noisy_score': 0.8, ...}, 'tenant2': {...}}
                df_data = []
                for tenant_id, metrics in tenant_metrics.items():
                    tenant_data = metrics.copy()
                    tenant_data['tenant_id'] = tenant_id
                    df_data.append(tenant_data)
                tenant_metrics = pd.DataFrame(df_data)
            else:
                # Dicionário simples - converter diretamente
                tenant_metrics = pd.DataFrame(tenant_metrics)
        except Exception as e:
            logger.error(f"Erro ao converter tenant_metrics para DataFrame: {e}")
            return {"error_message": f"Erro ao processar métricas de tenant: {e}"}
    
    # Verificar se a coluna esperada está presente ou se precisamos criá-la
    if tenant_metrics.empty:
        logger.error("DataFrame de tenant_metrics está vazio")
        return {"error_message": "Dados de métricas de tenant vazios"}
    
    # Garantir que existe uma coluna tenant_id
    if 'tenant_id' not in tenant_metrics.columns:
        # Tentar encontrar ou criar coluna tenant_id
        if tenant_metrics.index.name == 'tenant_id':
            # Se o índice for tenant_id, reset para coluna
            tenant_metrics = tenant_metrics.reset_index()
        else:
            logger.error("Coluna 'tenant_id' ausente nas métricas de tenant e não pode ser derivada")
        logger.warning(f"Coluna 'tenant_id' não encontrada nas métricas de tenant. Colunas: {list(tenant_metrics.columns) if not tenant_metrics.empty else 'DataFrame vazio'}")
        return {"error_message": "Dados necessários para agregação de insights não disponíveis - coluna tenant_id ausente"}
    
    # Garantir que os outros parâmetros não são None
    phase_comparison_results = phase_comparison_results or {}
    granger_matrices = granger_matrices or {}
    te_matrices = te_matrices or {}
    correlation_matrices = correlation_matrices or {}
    anomaly_metrics = anomaly_metrics or {}
    
    # Inicialização de insights para cada tenant
    for tenant in tenant_metrics['tenant_id'].unique():
        insights[tenant] = {
            # Informações gerais
            'name': tenant,
            'rank': 0,
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
    sorted_metrics = tenant_metrics.sort_values(by='noisy_score', ascending=False).reset_index(drop=True)
    for idx, row in sorted_metrics.iterrows():
        tenant = row['tenant_id']
        insights[tenant]['rank'] = idx + 1
        insights[tenant]['noisy_score'] = row['noisy_score']
        
        # Determinar se é um "tenant barulhento" (top 25%)
        if idx < len(sorted_metrics) // 4:
            insights[tenant]['is_noisy_tenant'] = True
    
    # Analisar causalidade para determinar relações de impacto
    for metric_name, matrix in te_matrices.items():
        if matrix.empty:
            continue
            
        # Para cada tenant como fonte de causalidade
        for tenant in insights.keys():
            if tenant not in matrix.columns:
                continue
                
            # Identificar os principais tenants impactados por este tenant
            # (maiores valores de TE indicam mais influência)
            if tenant in matrix.columns:
                if tenant in matrix.index:
                    # Excluir diagonal (auto-influência)
                    te_values = matrix[tenant].drop(tenant)
                else:
                    te_values = matrix[tenant]
                    
                if not te_values.empty:
                    # Top 2 tenants mais influenciados, com TE > 0.05
                    significant_te = te_values[te_values > 0.05]
                    if not significant_te.empty:
                        top_impacted = significant_te.nlargest(2)
                        for impacted_tenant, te_val in top_impacted.items():
                            impact_info = {
                                'tenant': impacted_tenant,
                                'score': float(te_val),
                                'metric': metric_name
                            }
                            insights[tenant]['main_impacted_tenants'].append(impact_info)
                            
                            # Marcar o tenant impactado como potencial "vítima"
                            if impacted_tenant in insights:
                                insights[impacted_tenant]['is_victim_tenant'] = True
                                insights[impacted_tenant]['main_impact_sources'].append({
                                    'tenant': tenant,
                                    'score': float(te_val),
                                    'metric': metric_name
                                })
    
    # Analisar comparação entre fases para identificar padrões durante ataque
    for metric_name, stats_df in phase_comparison_results.items():
        if stats_df.empty:
            continue
            
        for _, row in stats_df.iterrows():
            tenant = row.get('tenant_id')
            if tenant not in insights:
                continue
                
            # Verificar variação significativa durante ataque
            attack_vs_baseline_col = '2 - Attack_vs_baseline_pct'
            if attack_vs_baseline_col in stats_df.columns and row.get(attack_vs_baseline_col) is not None:
                variation = row[attack_vs_baseline_col]
                if abs(variation) > 30:  # Variação maior que 30%
                    pattern = {
                        'metric': metric_name,
                        'variation_pct': float(variation),
                        'direction': 'increase' if variation > 0 else 'decrease'
                    }
                    insights[tenant]['attack_phase_patterns'].append(pattern)
    
    # Analisar anomalias, se disponíveis
    if anomaly_metrics and isinstance(anomaly_metrics, dict):
        for metric_name, anomalies_df in anomaly_metrics.items():
            if anomalies_df.empty:
                continue
                
            for tenant in anomalies_df['tenant_id'].unique():
                if tenant not in insights:
                    continue
                    
                tenant_anomalies = anomalies_df[anomalies_df['tenant_id'] == tenant]
                if not tenant_anomalies.empty:
                    insights[tenant]['anomalous_metrics'].append({
                        'metric': metric_name,
                        'anomaly_count': len(tenant_anomalies),
                        'max_zscore': float(tenant_anomalies['z_score'].max()) if 'z_score' in tenant_anomalies.columns else 0.0
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
    
    # Preparar arrays para anomalias e padrões de ataque
    anomaly_x = []
    anomaly_y = []
    attack_x = []
    attack_y = []
    
    # Coletar pontos para destacar
    for idx, (_, row) in enumerate(plot_data.iterrows()):
        if row['anomaly_count'] > 0:
            anomaly_x.append(idx)
            anomaly_y.append(float(row['impact_score']) + 0.05)
        
        if row['attack_pattern_count'] > 0:
            attack_x.append(idx)
            attack_y.append(float(row['impact_score']) + 0.1)
    
    # Plotar pontos destacados
    if anomaly_x:
        plt.scatter(anomaly_x, anomaly_y, marker='*', s=200, color='yellow', 
                   edgecolor='black', label='_nolegend_')
    
    if attack_x:
        plt.scatter(attack_x, attack_y, marker='^', s=100, color='orange', 
                   edgecolor='black', label='_nolegend_')
    
    # Legendas customizadas para os marcadores
    legend_elements = [
        mlines.Line2D([0], [0], marker='*', color='w', markerfacecolor='yellow', markersize=15,
               markeredgecolor='black', label='Anomalias Detectadas'),
        mlines.Line2D([0], [0], marker='^', color='w', markerfacecolor='orange', markersize=10,
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
