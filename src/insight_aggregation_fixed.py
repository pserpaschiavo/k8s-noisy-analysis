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
    insights = {}
    
    # Verificar e registrar o status dos dados de entrada para debugging
    logger.info("Status dos dados para agregação de insights:")
    logger.info(f"- tenant_metrics presente: {tenant_metrics is not None}")
    logger.info(f"- tenant_metrics tipo: {type(tenant_metrics)}")
    
    # Lista para coletar tenants de diferentes fontes
    all_tenants = set()
    
    # Tentar extrair tenant_metrics para um DataFrame adequado
    if tenant_metrics is None:
        logger.warning("tenant_metrics não disponível. Tentando criar a partir de outros dados.")
    elif isinstance(tenant_metrics, dict):
        logger.warning("tenant_metrics é um dicionário. Tentando converter para DataFrame.")
        try:
            # Se for dict[str, dict], converter para DataFrame
            tenant_data = []
            for tenant_id, metrics in tenant_metrics.items():
                if isinstance(metrics, dict):
                    row = metrics.copy()
                    row['tenant_id'] = tenant_id
                    tenant_data.append(row)
            
            if tenant_data:
                tenant_metrics = pd.DataFrame(tenant_data)
            else:
                # Formato diferente, tentar converter diretamente
                tenant_metrics = pd.DataFrame(tenant_metrics)
        except Exception as e:
            logger.error(f"Não foi possível converter tenant_metrics para DataFrame: {e}")
            tenant_metrics = None
    
    # Se temos tenant_metrics como DataFrame, extrair tenant_ids
    if isinstance(tenant_metrics, pd.DataFrame) and not tenant_metrics.empty:
        if 'tenant_id' in tenant_metrics.columns:
            all_tenants.update(tenant_metrics['tenant_id'].tolist())
            logger.info(f"Encontrados {len(all_tenants)} tenants em tenant_metrics")
        else:
            logger.warning("DataFrame de tenant_metrics não contém coluna 'tenant_id'")
    
    # Coletar tenants de outras fontes    
    # Extrair tenants de granger_matrices
    if granger_matrices:
        for metric, matrix in granger_matrices.items():
            if isinstance(matrix, pd.DataFrame):
                all_tenants.update(matrix.index)
                all_tenants.update(matrix.columns)
    
    # Extrair tenants de te_matrices
    if te_matrices:
        for metric, matrix in te_matrices.items():
            if isinstance(matrix, pd.DataFrame):
                all_tenants.update(matrix.index)
                all_tenants.update(matrix.columns)
                
    # Extrair tenants de correlation_matrices
    if correlation_matrices:
        for metric, phase_dict in correlation_matrices.items():
            for phase, matrix in phase_dict.items():
                if isinstance(matrix, pd.DataFrame):
                    all_tenants.update(matrix.index)
                    all_tenants.update(matrix.columns)
    
    # Se não temos tenant_metrics válido mas temos tenants, criar um DataFrame básico
    if (tenant_metrics is None or not isinstance(tenant_metrics, pd.DataFrame) or tenant_metrics.empty) and all_tenants:
        logger.info(f"Criando DataFrame básico para {len(all_tenants)} tenants")
        tenant_metrics = pd.DataFrame({
            'tenant_id': list(all_tenants),
            'noisy_score': [0.5] * len(all_tenants)  # Valor padrão neutro
        })
    
    # Verificação final de tenant_metrics
    if not isinstance(tenant_metrics, pd.DataFrame) or tenant_metrics.empty:
        logger.error("Não foi possível criar DataFrame válido de tenant_metrics")
        return {"error_message": "Dados necessários para agregação de insights não disponíveis"}
    
    # Verificar coluna tenant_id
    if 'tenant_id' not in tenant_metrics.columns:
        # Tentar usar o índice se for nomeado como tenant_id
        if tenant_metrics.index.name == 'tenant_id':
            tenant_metrics = tenant_metrics.reset_index()
        else:
            logger.error("DataFrame de tenant_metrics não contém coluna 'tenant_id' e não pode ser derivada")
            return {"error_message": "Dados necessários para agregação de insights não disponíveis - coluna tenant_id ausente"}
    
    # Inicializar valores padrão para parâmetros None
    phase_comparison_results = {} if phase_comparison_results is None else phase_comparison_results
    granger_matrices = {} if granger_matrices is None else granger_matrices
    te_matrices = {} if te_matrices is None else te_matrices
    correlation_matrices = {} if correlation_matrices is None else correlation_matrices
    anomaly_metrics = {} if anomaly_metrics is None else anomaly_metrics
    
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
    try:
        if 'noisy_score' in tenant_metrics.columns:
            sorted_metrics = tenant_metrics.sort_values(by='noisy_score', ascending=False).reset_index(drop=True)
            for idx, row in sorted_metrics.iterrows():
                tenant = row['tenant_id']
                if tenant in insights:
                    insights[tenant]['rank'] = idx + 1
                    insights[tenant]['noisy_score'] = float(row['noisy_score'])
                    
                    # Determinar se é um "tenant barulhento" (top 25%)
                    if idx < len(sorted_metrics) // 4:
                        insights[tenant]['is_noisy_tenant'] = True
    except Exception as e:
        logger.warning(f"Erro ao processar noisy_score: {e}")
    
    # Analisar causalidade para determinar relações de impacto
    try:
        for metric_name, matrix in te_matrices.items():
            if not isinstance(matrix, pd.DataFrame) or matrix.empty:
                continue
                
            # Para cada tenant como fonte de causalidade
            for tenant in insights.keys():
                if tenant not in matrix.columns:
                    continue
                    
                # Identificar os principais tenants impactados por este tenant
                if tenant in matrix.columns:
                    # Preparar série com valores de TE
                    te_values = None
                    if tenant in matrix.index:
                        # Excluir diagonal (auto-influência)
                        te_values = matrix[tenant].drop(tenant)
                    else:
                        te_values = matrix[tenant]
                        
                    if te_values is not None and not te_values.empty:
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
    except Exception as e:
        logger.warning(f"Erro ao processar matrizes de causalidade: {e}")
    
    # Analisar comparação entre fases para identificar padrões durante ataque
    try:
        for metric_name, stats_df in phase_comparison_results.items():
            if not isinstance(stats_df, pd.DataFrame) or stats_df.empty:
                continue
                
            for _, row in stats_df.iterrows():
                tenant = row.get('tenant_id')
                if tenant not in insights:
                    continue
                    
                # Verificar variação significativa durante ataque
                attack_vs_baseline_col = '2 - Attack_vs_baseline_pct'
                if attack_vs_baseline_col in stats_df.columns and row.get(attack_vs_baseline_col) is not None:
                    if pd.notna(row.get(attack_vs_baseline_col)):
                        variation = float(row[attack_vs_baseline_col])
                        if abs(variation) > 30:  # Variação maior que 30%
                            pattern = {
                                'metric': metric_name,
                                'variation_pct': float(variation),
                                'direction': 'increase' if variation > 0 else 'decrease'
                            }
                            insights[tenant]['attack_phase_patterns'].append(pattern)
    except Exception as e:
        logger.warning(f"Erro ao processar comparação de fases: {e}")
    
    # Analisar anomalias, se disponíveis
    try:
        if isinstance(anomaly_metrics, dict):
            for metric_name, anomalies_df in anomaly_metrics.items():
                if not isinstance(anomalies_df, pd.DataFrame) or anomalies_df.empty:
                    continue
                    
                if 'tenant_id' not in anomalies_df.columns:
                    continue
                    
                for tenant in anomalies_df['tenant_id'].unique():
                    if tenant not in insights:
                        continue
                        
                    tenant_anomalies = anomalies_df[anomalies_df['tenant_id'] == tenant]
                    if not tenant_anomalies.empty:
                        z_score_max = 0.0
                        if 'z_score' in tenant_anomalies.columns:
                            z_score_max = float(tenant_anomalies['z_score'].max())
                            
                        insights[tenant]['anomalous_metrics'].append({
                            'metric': metric_name,
                            'anomaly_count': len(tenant_anomalies),
                            'max_zscore': z_score_max
                        })
    except Exception as e:
        logger.warning(f"Erro ao processar anomalias: {e}")
    
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


# Outras funções do módulo - preservar a implementação existente
def generate_comparative_table(tenant_insights: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Gera tabela comparativa final para análise inter-tenant.
    
    Args:
        tenant_insights: Dicionário com insights agregados de cada tenant
        
    Returns:
        DataFrame com tabela comparativa detalhada
    """
    # Verificar se temos dados de insight válidos - não um dicionário de erro
    if not isinstance(tenant_insights, dict) or 'error_message' in tenant_insights:
        logger.error("Dados de insights inválidos para gerar tabela comparativa")
        return pd.DataFrame()
    
    # Estrutura para armazenar dados da tabela
    table_data = []
    
    for tenant_id, insights in tenant_insights.items():
        # Dados básicos
        tenant_data = {
            'tenant_id': tenant_id,
            'rank': insights.get('rank', 0),
            'impact_score': insights.get('noisy_score', 0.0),
            'is_noisy': insights.get('is_noisy_tenant', False),
            'is_victim': insights.get('is_victim_tenant', False),
        }
        
        # Contagem de métricas anômalas
        tenant_data['anomaly_count'] = len(insights.get('anomalous_metrics', []))
        
        # Principais tenants impactados por este tenant
        impacted_tenants = insights.get('main_impacted_tenants', [])
        tenant_data['impacts_others'] = len(impacted_tenants) > 0
        tenant_data['main_impacted'] = ', '.join([f"{item['tenant']} ({item['score']:.2f})" 
                                              for item in impacted_tenants[:2]])
        
        # Principais fontes de impacto neste tenant
        impact_sources = insights.get('main_impact_sources', [])
        tenant_data['impacted_by_others'] = len(impact_sources) > 0
        tenant_data['main_sources'] = ', '.join([f"{item['tenant']} ({item['score']:.2f})" 
                                              for item in impact_sources[:2]])
        
        # Comportamento durante fase de ataque
        attack_patterns = insights.get('attack_phase_patterns', [])
        tenant_data['attack_pattern_count'] = len(attack_patterns)
        tenant_data['attack_patterns'] = ', '.join([f"{p['metric']} ({p['direction']}: {p['variation_pct']:.1f}%)" 
                                                 for p in attack_patterns[:2]])
        
        # Recomendações principais
        recommendations = insights.get('recommendations', [])
        tenant_data['recommendation_count'] = len(recommendations)
        tenant_data['top_recommendation'] = recommendations[0] if recommendations else ""
        
        table_data.append(tenant_data)
    
    # Criar DataFrame e ordenar por rank
    if not table_data:
        logger.warning("Nenhum dado para tabela comparativa")
        return pd.DataFrame()
        
    df = pd.DataFrame(table_data)
    if 'rank' in df.columns:
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
    if comparative_table is None or comparative_table.empty:
        logger.warning("Tabela comparativa vazia, não é possível gerar visualização")
        return None
    
    # Verificar se temos todas as colunas necessárias
    required_columns = ['tenant_id', 'impact_score', 'is_noisy', 'is_victim', 
                         'anomaly_count', 'attack_pattern_count']
    missing_columns = [col for col in required_columns if col not in comparative_table.columns]
    
    if missing_columns:
        logger.warning(f"Colunas ausentes na tabela comparativa: {missing_columns}")
        # Adicionar colunas faltantes com valor padrão
        for col in missing_columns:
            if col == 'impact_score':
                comparative_table[col] = 0.0
            elif col in ['is_noisy', 'is_victim']:
                comparative_table[col] = False
            elif col in ['anomaly_count', 'attack_pattern_count']:
                comparative_table[col] = 0
            else:
                comparative_table[col] = "unknown"
    
    # Extrair dados relevantes para visualização
    plot_data = comparative_table[required_columns].copy()
    
    # Converter para formato mais adequado para visualização
    plot_data['tenant_type'] = 'Normal'
    plot_data.loc[plot_data['is_noisy'], 'tenant_type'] = 'Noisy'
    plot_data.loc[plot_data['is_victim'], 'tenant_type'] = 'Victim'
    plot_data.loc[plot_data['is_noisy'] & plot_data['is_victim'], 'tenant_type'] = 'Noisy+Victim'
    
    try:
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
    except Exception as e:
        logger.error(f"Erro ao gerar visualização comparativa: {e}", exc_info=True)
        return None
