"""
Module: insights_generation.py
Description: Módulo para geração automática de insights sobre os resultados da análise.

Este módulo implementa funções para extrair insights significativos dos resultados
da análise multi-round, detectar padrões relevantes e gerar resumos automáticos
em linguagem natural.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
import textwrap

logger = logging.getLogger(__name__)

def generate_automated_insights(
    aggregated_effects_df: pd.DataFrame,
    robustness_df: Optional[pd.DataFrame] = None,
    phase_correlations_df: Optional[pd.DataFrame] = None,
    correlation_stability_df: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Gera insights automáticos sobre os resultados da análise multi-round.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        robustness_df: DataFrame com resultados da análise de robustez
        phase_correlations_df: DataFrame com correlações intra-fase
        correlation_stability_df: DataFrame com estabilidade das correlações
        alpha: Nível de significância
        context: Contexto adicional para enriquecer os insights
        
    Returns:
        Dict[str, Any]: Dicionário com insights categorizados
    """
    insights = {
        "effect_size_insights": [],
        "robustness_insights": [],
        "correlation_insights": [],
        "anomalies": [],
        "recommendations": [],
        "summary": ""
    }
    
    # Verificar se os DataFrames estão vazios
    if aggregated_effects_df.empty:
        logger.warning("DataFrame de efeitos agregados vazio. Não é possível gerar insights.")
        return insights
    
    # Gerar insights para tamanhos de efeito
    insights["effect_size_insights"] = _generate_effect_size_insights(
        aggregated_effects_df, alpha
    )
    
    # Gerar insights de robustez se o DataFrame estiver disponível
    if robustness_df is not None and not robustness_df.empty:
        insights["robustness_insights"] = _generate_robustness_insights(
            robustness_df, aggregated_effects_df
        )
    
    # Gerar insights de correlação se os DataFrames estiverem disponíveis
    if phase_correlations_df is not None and not phase_correlations_df.empty:
        correlation_insights = _generate_correlation_insights(
            phase_correlations_df,
            correlation_stability_df if correlation_stability_df is not None else None
        )
        insights["correlation_insights"] = correlation_insights
    
    # Detectar anomalias nos dados
    if robustness_df is not None and not robustness_df.empty:
        anomalies = _detect_anomalies(
            aggregated_effects_df, robustness_df, phase_correlations_df
        )
        insights["anomalies"] = anomalies
    
    # Gerar recomendações baseadas em evidências
    recommendations = _generate_recommendations(
        insights["effect_size_insights"],
        insights["robustness_insights"],
        insights["correlation_insights"],
        insights["anomalies"]
    )
    insights["recommendations"] = recommendations
    
    # Criar resumo geral
    summary = _create_general_summary(
        aggregated_effects_df,
        robustness_df,
        phase_correlations_df,
        insights
    )
    insights["summary"] = summary
    
    return insights

def _generate_effect_size_insights(aggregated_effects_df: pd.DataFrame, alpha: float = 0.05) -> List[str]:
    """
    Gera insights sobre os tamanhos de efeito.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        alpha: Nível de significância
        
    Returns:
        List[str]: Lista de insights
    """
    insights = []
    
    # Contagem total de efeitos
    total_effects = aggregated_effects_df.shape[0]
    significant_effects = aggregated_effects_df[aggregated_effects_df['is_significant'] == True].shape[0]
    sig_percentage = (significant_effects / total_effects) * 100 if total_effects > 0 else 0
    
    insights.append(f"De {total_effects} combinações analisadas, {significant_effects} ({sig_percentage:.1f}%) apresentaram efeitos estatisticamente significativos (p < {alpha}).")
    
    # Análise por magnitude
    effect_magnitudes = aggregated_effects_df['effect_magnitude'].value_counts()
    if not effect_magnitudes.empty:
        for magnitude in ['large', 'medium', 'small', 'negligible']:
            if magnitude in effect_magnitudes.index:
                count = effect_magnitudes[magnitude]
                pct = (count / total_effects) * 100
                insights.append(f"Efeitos de magnitude '{magnitude}': {count} ({pct:.1f}%).")
    
    # Top efeitos significativos por magnitude
    sig_df = aggregated_effects_df[aggregated_effects_df['is_significant'] == True].copy()
    if not sig_df.empty:
        # Ordenar por magnitude absoluta do efeito
        sig_df['abs_effect'] = sig_df['mean_effect_size'].abs()
        top_effects = sig_df.sort_values('abs_effect', ascending=False).head(5)
        
        insights.append("\nPrincipais efeitos significativos:")
        for _, row in top_effects.iterrows():
            effect_dir = "aumento" if row['mean_effect_size'] > 0 else "redução"
            insights.append(f"- {row['experimental_phase']} causa {effect_dir} significativo em '{row['metric_name']}' para tenant {row['tenant_id']} " +
                          f"(Cohen's d = {row['mean_effect_size']:.2f}, p = {row['combined_p_value']:.4f}).")
    
    # Análise por fase experimental
    phase_analysis = sig_df.groupby('experimental_phase')['metric_name'].count()
    if not phase_analysis.empty:
        most_impactful_phase = phase_analysis.idxmax()
        most_impactful_count = phase_analysis.max()
        insights.append(f"\nA fase '{most_impactful_phase}' é a mais impactante, com {most_impactful_count} métricas significativamente afetadas.")
    
    # Análise por tenant
    tenant_analysis = sig_df.groupby('tenant_id')['metric_name'].count()
    if not tenant_analysis.empty:
        most_affected_tenant = tenant_analysis.idxmax()
        most_affected_count = tenant_analysis.max()
        insights.append(f"O tenant '{most_affected_tenant}' é o mais afetado, com {most_affected_count} métricas significativamente impactadas.")
    
    # Análise por métrica
    metric_analysis = sig_df.groupby('metric_name').size()
    if not metric_analysis.empty:
        most_affected_metric = metric_analysis.idxmax()
        most_affected_metric_count = metric_analysis.max()
        insights.append(f"A métrica '{most_affected_metric}' é a mais frequentemente impactada, com efeitos significativos em {most_affected_metric_count} combinações de fase/tenant.")
    
    # Análise de confiabilidade
    if 'reliability_category' in aggregated_effects_df.columns:
        reliability_counts = aggregated_effects_df['reliability_category'].value_counts()
        high_reliable = reliability_counts.get('high', 0)
        high_reliable_pct = (high_reliable / total_effects) * 100 if total_effects > 0 else 0
        insights.append(f"\n{high_reliable} ({high_reliable_pct:.1f}%) dos efeitos têm alta confiabilidade entre rounds.")
    
    return insights

def _generate_robustness_insights(robustness_df: pd.DataFrame, aggregated_effects_df: pd.DataFrame) -> List[str]:
    """
    Gera insights sobre a robustez dos achados estatísticos.
    
    Args:
        robustness_df: DataFrame com resultados da análise de robustez
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        
    Returns:
        List[str]: Lista de insights
    """
    insights = []
    
    # Contagem total de efeitos analisados
    total_analyzed = robustness_df.shape[0]
    
    # Distribuição de robustez
    robustness_counts = robustness_df['overall_robustness'].value_counts()
    high_robustness = robustness_counts.get('Alta', 0)
    medium_robustness = robustness_counts.get('Média', 0)
    low_robustness = robustness_counts.get('Baixa', 0)
    
    high_pct = (high_robustness / total_analyzed) * 100 if total_analyzed > 0 else 0
    medium_pct = (medium_robustness / total_analyzed) * 100 if total_analyzed > 0 else 0
    low_pct = (low_robustness / total_analyzed) * 100 if total_analyzed > 0 else 0
    
    insights.append(f"De {total_analyzed} efeitos analisados quanto à robustez, {high_robustness} ({high_pct:.1f}%) apresentam alta robustez, " +
                  f"{medium_robustness} ({medium_pct:.1f}%) média robustez, e {low_robustness} ({low_pct:.1f}%) baixa robustez.")
    
    # Efeitos com alta robustez e significância
    if 'is_robust_significance' in robustness_df.columns and 'is_robust_effect' in robustness_df.columns:
        robust_sig = robustness_df[(robustness_df['is_robust_significance'] == True) & 
                                  (robustness_df['is_robust_effect'] == True)]
        robust_sig_count = robust_sig.shape[0]
        robust_sig_pct = (robust_sig_count / total_analyzed) * 100 if total_analyzed > 0 else 0
        
        insights.append(f"{robust_sig_count} ({robust_sig_pct:.1f}%) dos efeitos são robustos tanto em magnitude quanto em significância estatística.")
    
    # Efeitos sensíveis a rounds específicos
    sensitive_effects = robustness_df[robustness_df['sensitive_rounds'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False)]
    if not sensitive_effects.empty:
        insights.append(f"\nForam identificados {sensitive_effects.shape[0]} efeitos cuja significância é sensível à remoção de rounds específicos:")
        
        # Listar os 3 principais efeitos sensíveis
        top_sensitive = sensitive_effects.head(3)
        for _, row in top_sensitive.iterrows():
            sensitive_rounds = ", ".join(row['sensitive_rounds']) if isinstance(row['sensitive_rounds'], list) else str(row['sensitive_rounds'])
            insights.append(f"- '{row['metric_name']}' em {row['experimental_phase']} (tenant {row['tenant_id']}) é sensível aos rounds: {sensitive_rounds}")
    
    # Frequência de rounds causando sensibilidade
    if sensitive_effects.shape[0] > 0:
        all_sensitive_rounds = []
        for rounds_list in sensitive_effects['sensitive_rounds']:
            if isinstance(rounds_list, list):
                all_sensitive_rounds.extend(rounds_list)
        
        if all_sensitive_rounds:
            from collections import Counter
            round_counts = Counter(all_sensitive_rounds)
            most_common_round = round_counts.most_common(1)
            if most_common_round:
                round_id, count = most_common_round[0]
                insights.append(f"O round '{round_id}' é o mais frequentemente responsável por alterações na significância ({count} efeitos).")
    
    return insights

def _generate_correlation_insights(
    phase_correlations_df: pd.DataFrame,
    correlation_stability_df: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Gera insights sobre padrões de correlação intra-fase.
    
    Args:
        phase_correlations_df: DataFrame com correlações intra-fase
        correlation_stability_df: DataFrame com estabilidade das correlações
        
    Returns:
        List[str]: Lista de insights
    """
    insights = []
    
    # Análise básica de correlações
    total_correlations = phase_correlations_df.shape[0]
    if 'correlation_strength' in phase_correlations_df.columns:
        strong_correlations = phase_correlations_df[phase_correlations_df['correlation_strength'] == 'strong'].shape[0]
        strong_pct = (strong_correlations / total_correlations) * 100 if total_correlations > 0 else 0
        
        insights.append(f"De {total_correlations} correlações analisadas, {strong_correlations} ({strong_pct:.1f}%) são fortes (|r| > 0.7).")
    
    # Correlações positivas vs negativas
    positive_corr = phase_correlations_df[phase_correlations_df['correlation'] > 0].shape[0]
    negative_corr = phase_correlations_df[phase_correlations_df['correlation'] < 0].shape[0]
    pos_pct = (positive_corr / total_correlations) * 100 if total_correlations > 0 else 0
    neg_pct = (negative_corr / total_correlations) * 100 if total_correlations > 0 else 0
    
    insights.append(f"{positive_corr} ({pos_pct:.1f}%) das correlações são positivas e {negative_corr} ({neg_pct:.1f}%) são negativas.")
    
    # Análise por fase
    phase_corr_counts = phase_correlations_df.groupby('experimental_phase').size()
    if not phase_corr_counts.empty:
        top_phase = phase_corr_counts.idxmax()
        top_phase_count = phase_corr_counts.max()
        
        # Força média de correlação por fase
        phase_corr_strength = phase_correlations_df.groupby('experimental_phase')['correlation'].apply(lambda x: x.abs().mean())
        if not phase_corr_strength.empty:
            strongest_phase = phase_corr_strength.idxmax()
            strongest_corr = phase_corr_strength.max()
            
            insights.append(f"\nA fase '{strongest_phase}' exibe as correlações mais fortes em média (|r| = {strongest_corr:.2f}).")
    
    # Análise por métrica
    metric_corr_counts = phase_correlations_df.groupby('metric_name').size()
    if not metric_corr_counts.empty:
        top_metric = metric_corr_counts.idxmax()
        top_metric_count = metric_corr_counts.max()
        
        # Força média de correlação por métrica
        metric_corr_strength = phase_correlations_df.groupby('metric_name')['correlation'].apply(lambda x: x.abs().mean())
        if not metric_corr_strength.empty:
            strongest_metric = metric_corr_strength.idxmax()
            strongest_metric_corr = metric_corr_strength.max()
            
            insights.append(f"A métrica '{strongest_metric}' exibe as correlações mais fortes em média (|r| = {strongest_metric_corr:.2f}).")
    
    # Análise de estabilidade de correlações
    if correlation_stability_df is not None and not correlation_stability_df.empty:
        avg_cv = correlation_stability_df['cv'].mean()
        
        if 'stability_category' in correlation_stability_df.columns:
            stability_counts = correlation_stability_df['stability_category'].value_counts()
            high_stability = stability_counts.get('high', 0)
            high_stability_pct = (high_stability / correlation_stability_df.shape[0]) * 100 if correlation_stability_df.shape[0] > 0 else 0
            
            insights.append(f"\n{high_stability} ({high_stability_pct:.1f}%) dos pares de correlação apresentam alta estabilidade entre rounds.")
        
        insights.append(f"A variabilidade média das correlações entre rounds (CV) é {avg_cv:.2f}.")
    
    # Top pares de correlação mais fortes e estáveis
    strong_correlations = phase_correlations_df.sort_values('correlation', key=abs, ascending=False).head(3)
    if not strong_correlations.empty:
        insights.append("\nPrincipais correlações intra-fase:")
        for _, row in strong_correlations.iterrows():
            corr_dir = "positivamente" if row['correlation'] > 0 else "negativamente"
            insights.append(f"- Em '{row['experimental_phase']}', tenant-pair {row['tenant_pair']} é {corr_dir} correlacionado " +
                          f"(r = {row['correlation']:.2f}) para a métrica '{row['metric_name']}'.")
    
    return insights

def _detect_anomalies(
    aggregated_effects_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
    phase_correlations_df: Optional[pd.DataFrame] = None
) -> List[str]:
    """
    Detecta anomalias nos dados e resultados.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        robustness_df: DataFrame com resultados da análise de robustez
        phase_correlations_df: DataFrame com correlações intra-fase
        
    Returns:
        List[str]: Lista de anomalias detectadas
    """
    anomalies = []
    
    # Detectar efeitos com baixa robustez mas alta significância
    if 'overall_robustness' in robustness_df.columns and 'combined_p_value' in aggregated_effects_df.columns:
        # Juntar os DataFrames para detecção de anomalias
        df_merged = pd.merge(
            aggregated_effects_df,
            robustness_df[['metric_name', 'experimental_phase', 'tenant_id', 'overall_robustness']],
            on=['metric_name', 'experimental_phase', 'tenant_id'],
            how='inner'
        )
        
        # Efeitos com baixa robustez mas alta significância
        anomalous_effects = df_merged[(df_merged['overall_robustness'] == 'Baixa') & 
                                     (df_merged['is_significant'] == True) &
                                     (df_merged['combined_p_value'] < 0.01)]
        
        if not anomalous_effects.empty:
            anomalies.append(f"Detectados {anomalous_effects.shape[0]} efeitos com baixa robustez mas alta significância estatística (p < 0.01):")
            
            for _, row in anomalous_effects.head(3).iterrows():
                anomalies.append(f"- '{row['metric_name']}' em {row['experimental_phase']} (tenant {row['tenant_id']}) " +
                               f"com p={row['combined_p_value']:.4f} mas robustez {row['overall_robustness']}.")
    
    # Detectar efeitos com alto CV
    if 'coefficient_of_variation' in aggregated_effects_df.columns:
        high_cv_effects = aggregated_effects_df[aggregated_effects_df['coefficient_of_variation'] > 0.5]
        if not high_cv_effects.empty:
            anomalies.append(f"\nDetectados {high_cv_effects.shape[0]} efeitos com alta variabilidade entre rounds (CV > 0.5):")
            
            for _, row in high_cv_effects.head(3).iterrows():
                anomalies.append(f"- '{row['metric_name']}' em {row['experimental_phase']} (tenant {row['tenant_id']}), " +
                               f"CV={row['coefficient_of_variation']:.2f}")
    
    # Detectar correlações inconsistentes entre rounds
    if phase_correlations_df is not None and 'round_id' in phase_correlations_df.columns:
        # Identificar pares métrica-fase-tenant_pair com correlações que mudam de sinal entre rounds
        if 'correlation' in phase_correlations_df.columns:
            # Agrupar por métrica, fase e par de tenants
            grouped = phase_correlations_df.groupby(['metric_name', 'experimental_phase', 'tenant_pair'])
            
            inconsistent_pairs = []
            for name, group in grouped:
                # Verificar se há correlações de sinais diferentes
                pos_corrs = sum(group['correlation'] > 0)
                neg_corrs = sum(group['correlation'] < 0)
                
                if pos_corrs > 0 and neg_corrs > 0:
                    inconsistent_pairs.append((name, pos_corrs, neg_corrs))
            
            if inconsistent_pairs:
                anomalies.append(f"\nDetectados {len(inconsistent_pairs)} pares de correlação que mudam de sinal entre rounds:")
                
                for (metric, phase, tenant_pair), pos_count, neg_count in inconsistent_pairs[:3]:
                    anomalies.append(f"- '{metric}' em {phase} para {tenant_pair}: {pos_count} correlações positivas e {neg_count} negativas entre rounds.")
    
    # Se não encontrou anomalias
    if not anomalies:
        anomalies.append("Nenhuma anomalia significativa detectada nos dados.")
    
    return anomalies

def _generate_recommendations(
    effect_insights: List[str],
    robustness_insights: List[str],
    correlation_insights: List[str],
    anomalies: List[str]
) -> List[str]:
    """
    Gera recomendações baseadas nos insights e anomalias detectadas.
    
    Args:
        effect_insights: Lista de insights sobre tamanhos de efeito
        robustness_insights: Lista de insights sobre robustez
        correlation_insights: Lista de insights sobre correlações
        anomalies: Lista de anomalias detectadas
        
    Returns:
        List[str]: Lista de recomendações
    """
    recommendations = []
    
    # Recomendações baseadas em efeitos significativos
    if any("significativos" in insight for insight in effect_insights):
        recommendations.append("Concentre a atenção nas métricas com efeitos significativos e de alta magnitude, " +
                             "especialmente aqueles com alta confiabilidade entre rounds.")
    
    # Recomendações baseadas em robustez
    if any("baixa robustez" in insight for insight in robustness_insights) or \
       any("sensível" in insight for insight in robustness_insights):
        recommendations.append("Para resultados com baixa robustez, considere realizar rounds adicionais para aumentar a confiabilidade " +
                             "ou investigar fatores específicos que podem estar causando instabilidade nos resultados.")
    
    # Recomendações baseadas em correlações
    if any("correlações mais fortes" in insight for insight in correlation_insights):
        recommendations.append("Explore mais profundamente as fases com correlações fortes entre tenants, " +
                             "pois podem indicar comportamentos sistêmicos importantes ou oportunidades de otimização conjunta.")
    
    # Recomendações baseadas em anomalias
    if any("baixa robustez mas alta significância" in anomaly for anomaly in anomalies):
        recommendations.append("Investigue com cautela os efeitos que apresentam alta significância estatística mas baixa robustez, " +
                             "pois podem representar falsos positivos ou condições específicas que não generalizam.")
    
    if any("mudam de sinal entre rounds" in anomaly for anomaly in anomalies):
        recommendations.append("Revisite as correlações que mudam de sinal entre rounds para identificar possíveis fatores externos " +
                             "ou condições específicas que alteram fundamentalmente as relações entre tenants.")
    
    # Recomendações gerais
    recommendations.append("Implemente monitoramento contínuo para as métricas e tenants identificados como mais sensíveis às fases experimentais.")
    recommendations.append("Estabeleça limiares de alerta baseados nos tamanhos de efeito observados para detectar anomalias em tempo real.")
    recommendations.append("Consolide os descobrimentos em um modelo preditivo para antecipar o comportamento do sistema em diferentes cenários.")
    
    return recommendations

def _create_general_summary(
    aggregated_effects_df: pd.DataFrame,
    robustness_df: Optional[pd.DataFrame],
    phase_correlations_df: Optional[pd.DataFrame],
    insights: Dict[str, Any]
) -> str:
    """
    Cria um resumo geral de todos os insights e recomendações.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        robustness_df: DataFrame com resultados da análise de robustez
        phase_correlations_df: DataFrame com correlações intra-fase
        insights: Dicionário com insights categorizados
        
    Returns:
        str: Resumo geral em formato de texto
    """
    total_effects = aggregated_effects_df.shape[0]
    significant_effects = aggregated_effects_df[aggregated_effects_df['is_significant'] == True].shape[0]
    sig_pct = (significant_effects / total_effects) * 100 if total_effects > 0 else 0
    
    # Obter principais métricas afetadas
    sig_df = aggregated_effects_df[aggregated_effects_df['is_significant'] == True]
    top_metrics = sig_df.groupby('metric_name').size().sort_values(ascending=False).head(3)
    top_metrics_str = ", ".join([f"'{m}'" for m in top_metrics.index])
    
    # Obter principais fases de impacto
    top_phases = sig_df.groupby('experimental_phase').size().sort_values(ascending=False).head(2)
    top_phases_str = ", ".join([f"'{p}'" for p in top_phases.index])
    
    # Robustez geral
    robustness_summary = ""
    if robustness_df is not None and not robustness_df.empty and 'overall_robustness' in robustness_df.columns:
        high_robustness = robustness_df[robustness_df['overall_robustness'] == 'Alta'].shape[0]
        high_rob_pct = (high_robustness / robustness_df.shape[0]) * 100 if robustness_df.shape[0] > 0 else 0
        robustness_summary = f" {high_robustness} ({high_rob_pct:.1f}%) demonstram alta robustez;"
    
    # Correlações gerais
    correlation_summary = ""
    if phase_correlations_df is not None and not phase_correlations_df.empty:
        strong_correlations = phase_correlations_df[phase_correlations_df['correlation'].abs() > 0.7].shape[0]
        strong_corr_pct = (strong_correlations / phase_correlations_df.shape[0]) * 100 if phase_correlations_df.shape[0] > 0 else 0
        correlation_summary = f" {strong_correlations} correlações fortes (|r| > 0.7) identificadas;"
    
    # Construir resumo
    summary = textwrap.dedent(f"""
        # Resumo Executivo: Análise Multi-Round

        A análise consolidou dados de múltiplos rounds experimentais para identificar efeitos consistentes e robustos.
        
        ## Principais Descobertas:
        - De {total_effects} combinações analisadas, {significant_effects} ({sig_pct:.1f}%) mostram efeitos estatisticamente significativos.
        -{robustness_summary}
        -{correlation_summary}
        - As métricas mais impactadas são {top_metrics_str}.
        - As fases com maior impacto são {top_phases_str}.
        
        ## Recomendações Principais:
        - {insights['recommendations'][0] if insights['recommendations'] else 'Não há recomendações específicas.'}
        - {insights['recommendations'][1] if len(insights['recommendations']) > 1 else ''}
        
        ## Próximos Passos Sugeridos:
        1. Investigar anomalias detectadas, especialmente efeitos com alta significância mas baixa robustez.
        2. Aprofundar análise nas métricas e fases de maior impacto.
        3. Implementar monitoramento contínuo baseado nos padrões identificados.
    """)
    
    return summary.strip()

def generate_markdown_report(
    insights: Dict[str, Any],
    output_path: Optional[str] = None
) -> str:
    """
    Gera um relatório markdown completo a partir dos insights.
    
    Args:
        insights: Dicionário com insights categorizados
        output_path: Caminho para salvar o relatório (opcional)
        
    Returns:
        str: Relatório completo em formato markdown
    """
    report = insights["summary"] + "\n\n"
    
    # Adicionar seção de tamanhos de efeito
    report += "## Detalhamento dos Tamanhos de Efeito\n\n"
    for insight in insights["effect_size_insights"]:
        report += insight + "\n"
    
    # Adicionar seção de robustez
    if insights["robustness_insights"]:
        report += "\n## Análise de Robustez\n\n"
        for insight in insights["robustness_insights"]:
            report += insight + "\n"
    
    # Adicionar seção de correlações
    if insights["correlation_insights"]:
        report += "\n## Correlações Intra-fase\n\n"
        for insight in insights["correlation_insights"]:
            report += insight + "\n"
    
    # Adicionar seção de anomalias
    if insights["anomalies"]:
        report += "\n## Anomalias Detectadas\n\n"
        for anomaly in insights["anomalies"]:
            report += anomaly + "\n"
    
    # Adicionar seção de recomendações
    report += "\n## Recomendações Detalhadas\n\n"
    for i, recommendation in enumerate(insights["recommendations"], 1):
        report += f"{i}. {recommendation}\n"
    
    # Salvar relatório se caminho foi fornecido
    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Relatório markdown salvo em: {output_path}")
    
    return report
