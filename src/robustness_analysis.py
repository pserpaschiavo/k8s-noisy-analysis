"""
Module: robustness_analysis.py
Description: Módulo para análises de robustez de resultados estatísticos.

Este módulo implementa funções para analisar a robustez dos tamanhos de efeito,
validar a estabilidade dos resultados entre rounds e realizar testes de sensibilidade
para avaliar a confiabilidade das conclusões estatísticas.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import os

logger = logging.getLogger(__name__)

def perform_robustness_analysis(
    effect_sizes_df: pd.DataFrame, 
    output_dir: Optional[str] = None,
    alpha: float = 0.05,
    metrics: Optional[List[str]] = None,
    phases: Optional[List[str]] = None,
    tenants: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Realiza análise de robustez dos tamanhos de efeito através de técnicas leave-one-out
    e análise de sensibilidade com diferentes limiares de significância.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito por round
        output_dir: Diretório para salvar os gráficos (opcional)
        alpha: Nível de significância (padrão: 0.05)
        metrics: Lista de métricas para analisar (se None, usa todas)
        phases: Lista de fases para analisar (se None, usa todas)
        tenants: Lista de tenants para analisar (se None, usa todos)
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]: DataFrame com resultados da análise e dicionário de caminhos para gráficos
    """
    if effect_sizes_df.empty:
        logger.warning("DataFrame de tamanhos de efeito vazio.")
        return pd.DataFrame(), {}
    
    # Verificar colunas necessárias
    required_cols = ['round_id', 'metric_name', 'experimental_phase', 'tenant_id', 'effect_size', 'p_value']
    missing_cols = [col for col in required_cols if col not in effect_sizes_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return pd.DataFrame(), {}
    
    # Filtrar dados se especificado
    filtered_df = effect_sizes_df.copy()
    
    if metrics:
        filtered_df = filtered_df[filtered_df['metric_name'].isin(metrics)]
    
    if phases:
        filtered_df = filtered_df[filtered_df['experimental_phase'].isin(phases)]
    
    if tenants:
        filtered_df = filtered_df[filtered_df['tenant_id'].isin(tenants)]
    
    if filtered_df.empty:
        logger.warning("DataFrame filtrado está vazio.")
        return pd.DataFrame(), {}
    
    # Agrupar por métrica, fase e tenant
    grouped = filtered_df.groupby(['metric_name', 'experimental_phase', 'tenant_id', 'baseline_phase'])
    
    # Inicializar DataFrames e dicionário de resultados
    robustness_results = []
    sensitivity_results = []
    output_paths = {}
    
    # Para cada grupo, realizar análise leave-one-out
    for (metric, phase, tenant, baseline), group in grouped:
        if len(group) <= 1:
            # Não é possível fazer leave-one-out com apenas um round
            logger.warning(f"Insuficientes rounds para {metric}/{phase}/{tenant}. Pelo menos 2 são necessários.")
            continue
        
        rounds = group['round_id'].unique()
        original_effect = group['effect_size'].mean()
        original_p = _combine_p_values(group['p_value'].values)
        significant_original = original_p < alpha
        
        # Análise leave-one-out
        loo_effects = []
        loo_p_values = []
        loo_significant = []
        sensitive_rounds = []
        
        for round_id in rounds:
            # Remover um round e calcular métricas
            loo_group = group[group['round_id'] != round_id]
            loo_effect = loo_group['effect_size'].mean()
            loo_p = _combine_p_values(loo_group['p_value'].values)
            loo_sig = loo_p < alpha
            
            loo_effects.append(loo_effect)
            loo_p_values.append(loo_p)
            loo_significant.append(loo_sig)
            
            # Verificar se a remoção do round alterou a significância
            if loo_sig != significant_original:
                sensitive_rounds.append(round_id)
        
        # Calcular estatísticas de robustez
        effect_std = np.std(loo_effects)
        effect_cv = effect_std / abs(original_effect) if original_effect != 0 else float('inf')
        effect_range = max(loo_effects) - min(loo_effects)
        effect_rel_range = effect_range / abs(original_effect) if original_effect != 0 else float('inf')
        
        # Determinar robustez
        is_robust_effect = effect_cv < 0.3
        is_robust_significance = len(sensitive_rounds) == 0
        
        robustness_results.append({
            'metric_name': metric,
            'experimental_phase': phase,
            'tenant_id': tenant,
            'baseline_phase': baseline,
            'original_effect': original_effect,
            'original_p_value': original_p,
            'effect_std_loo': effect_std,
            'effect_cv_loo': effect_cv,
            'effect_range_loo': effect_range,
            'effect_rel_range_loo': effect_rel_range,
            'sensitive_rounds': sensitive_rounds,
            'is_robust_effect': is_robust_effect,
            'is_robust_significance': is_robust_significance,
            'overall_robustness': 'Alta' if (is_robust_effect and is_robust_significance) else
                                  'Média' if (is_robust_effect or is_robust_significance) else
                                  'Baixa'
        })
        
        # Análise de sensibilidade com diferentes valores de alpha
        alpha_values = [0.001, 0.01, 0.05, 0.1]
        alpha_significant = []
        
        for a in alpha_values:
            sig = original_p < a
            alpha_significant.append(sig)
        
        # Encontrar o alpha limiar onde a significância muda
        if True in alpha_significant and False in alpha_significant:
            change_points = []
            for i in range(1, len(alpha_values)):
                if alpha_significant[i] != alpha_significant[i-1]:
                    change_points.append((alpha_values[i-1] + alpha_values[i]) / 2)
            
            threshold_alpha = change_points[0] if change_points else None
        else:
            threshold_alpha = None if all(alpha_significant) else float('inf')
        
        sensitivity_results.append({
            'metric_name': metric,
            'experimental_phase': phase,
            'tenant_id': tenant,
            'baseline_phase': baseline,
            'p_value': original_p,
            'significant_at_0.001': alpha_significant[0],
            'significant_at_0.01': alpha_significant[1],
            'significant_at_0.05': alpha_significant[2],
            'significant_at_0.1': alpha_significant[3],
            'threshold_alpha': threshold_alpha,
            'sensitivity': 'Alta' if threshold_alpha and threshold_alpha > 0.05 else
                          'Média' if threshold_alpha and threshold_alpha > 0.01 else
                          'Baixa'
        })
        
        # Gerar gráficos se diretório de saída for fornecido
        if output_dir:
            # Criar diretório se não existir
            os.makedirs(output_dir, exist_ok=True)
            
            # Gráfico de análise leave-one-out
            plt.figure(figsize=(10, 6))
            
            # Plot de barras para efeitos leave-one-out
            bar_positions = np.arange(len(rounds))
            bars = plt.bar(
                bar_positions, 
                loo_effects, 
                color=['red' if r in sensitive_rounds else 'blue' for r in rounds],
                alpha=0.7
            )
            
            # Adicionar linha horizontal para efeito original
            plt.axhline(y=original_effect, color='black', linestyle='--', label='Efeito Original')
            
            # Rótulos e título
            plt.xlabel('Round Omitido')
            plt.ylabel("Tamanho de Efeito (Cohen's d)")
            plt.title(f'Análise Leave-One-Out - {metric}\nFase: {phase}, Tenant: {tenant}')
            plt.xticks(bar_positions, rounds, rotation=45)
            
            # Adicionar p-values como texto acima das barras
            for i, (p, sig) in enumerate(zip(loo_p_values, loo_significant)):
                color = 'green' if sig else 'red'
                plt.text(
                    i, 
                    loo_effects[i] * (1.1 if loo_effects[i] > 0 else 0.9),
                    f'p={p:.3f}',
                    ha='center',
                    color=color,
                    fontsize=8
                )
            
            # Legenda para barras vermelhas (sensíveis)
            if sensitive_rounds:
                plt.figtext(0.01, 0.01, "Barras vermelhas: rounds cuja remoção altera a significância", ha="left", fontsize=8)
            
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Salvar gráfico
            safe_metric = metric.replace(' ', '_')
            safe_phase = phase.replace(' ', '_')
            safe_tenant = tenant.replace(' ', '_')
            filename = f"robustness_loo_{safe_metric}_{safe_phase}_{safe_tenant}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths[f"{metric}_{phase}_{tenant}_loo"] = output_path
            
            # Gráfico de análise de sensibilidade
            plt.figure(figsize=(8, 6))
            
            # Plot de linha para p-valor vs. diferentes alphas
            plt.axhline(y=original_p, color='blue', linewidth=2, label='p-valor')
            
            for a, style, label in zip(alpha_values, ['-', '--', '-.', ':'], 
                                     ['α=0.001', 'α=0.01', 'α=0.05', 'α=0.1']):
                plt.axhline(y=a, color='red', linestyle=style, alpha=0.7, label=label)
            
            # Estilo do gráfico
            plt.yscale('log')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.xlim(-0.5, 0.5)  # Apenas para visualização
            
            # Rótulos e título
            plt.xlabel('Análise de Sensibilidade')
            plt.ylabel('p-valor (escala log)')
            plt.title(f'Análise de Sensibilidade - {metric}\nFase: {phase}, Tenant: {tenant}')
            plt.legend()
            
            # Salvar gráfico
            filename = f"sensitivity_{safe_metric}_{safe_phase}_{safe_tenant}.png"
            output_path = os.path.join(output_dir, filename)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            output_paths[f"{metric}_{phase}_{tenant}_sensitivity"] = output_path
    
    # Criar DataFrames de resultado
    robustness_df = pd.DataFrame(robustness_results) if robustness_results else pd.DataFrame()
    sensitivity_df = pd.DataFrame(sensitivity_results) if sensitivity_results else pd.DataFrame()
    
    # Combinar resultados
    if not robustness_df.empty and not sensitivity_df.empty:
        result_df = pd.merge(
            robustness_df, 
            sensitivity_df[['metric_name', 'experimental_phase', 'tenant_id', 'sensitivity', 'threshold_alpha']], 
            on=['metric_name', 'experimental_phase', 'tenant_id']
        )
    else:
        result_df = robustness_df
    
    return result_df, output_paths

def _combine_p_values(p_values: np.ndarray, method: str = 'fisher') -> float:
    """
    Combina múltiplos p-valores usando o método de Fisher ou Stouffer.
    
    Args:
        p_values: Array de p-valores a combinar
        method: Método de combinação ('fisher' ou 'stouffer')
        
    Returns:
        float: p-valor combinado
    """
    if len(p_values) == 0:
        return 1.0
    
    # Sanitizar p-valores (remover valores extremos)
    p_values = np.clip(p_values, 1e-10, 1.0)
    
    if method.lower() == 'fisher':
        # Método de Fisher: χ² = -2 * Σ log(p_i)
        chi_square = -2 * np.sum(np.log(p_values))
        combined_p = 1 - stats.chi2.cdf(chi_square, 2 * len(p_values))
    elif method.lower() == 'stouffer':
        # Método de Stouffer: Z = Σz_i / √n onde z_i = Φ⁻¹(1-p_i)
        z_scores = stats.norm.ppf(1 - p_values)
        z = np.sum(z_scores) / np.sqrt(len(p_values))
        combined_p = 1 - stats.norm.cdf(z)
    else:
        logger.warning(f"Método de combinação '{method}' não reconhecido. Usando Fisher.")
        chi_square = -2 * np.sum(np.log(p_values))
        combined_p = 1 - stats.chi2.cdf(chi_square, 2 * len(p_values))
    
    return combined_p

def generate_robustness_summary(robustness_df: pd.DataFrame) -> str:
    """
    Gera um resumo textual dos resultados da análise de robustez.
    
    Args:
        robustness_df: DataFrame com resultados da análise de robustez
        
    Returns:
        str: Resumo em formato de texto
    """
    if robustness_df.empty:
        return "Não há dados suficientes para análise de robustez."
    
    # Estatísticas gerais
    total_effects = len(robustness_df)
    high_robustness = sum(robustness_df['overall_robustness'] == 'Alta')
    medium_robustness = sum(robustness_df['overall_robustness'] == 'Média')
    low_robustness = sum(robustness_df['overall_robustness'] == 'Baixa')
    
    high_sensitivity = sum(robustness_df['sensitivity'] == 'Alta')
    medium_sensitivity = sum(robustness_df['sensitivity'] == 'Média')
    low_sensitivity = sum(robustness_df['sensitivity'] == 'Baixa')
    
    # Contagens por métrica
    metric_summary = robustness_df.groupby('metric_name')['overall_robustness'].value_counts().unstack().fillna(0)
    
    # Contagens por fase
    phase_summary = robustness_df.groupby('experimental_phase')['overall_robustness'].value_counts().unstack().fillna(0)
    
    # Resumo textual
    summary = "## Resumo da Análise de Robustez\n\n"
    
    summary += f"### Estatísticas Gerais\n"
    summary += f"- Total de efeitos analisados: {total_effects}\n"
    summary += f"- Robustez Alta: {high_robustness} ({high_robustness/total_effects*100:.1f}%)\n"
    summary += f"- Robustez Média: {medium_robustness} ({medium_robustness/total_effects*100:.1f}%)\n"
    summary += f"- Robustez Baixa: {low_robustness} ({low_robustness/total_effects*100:.1f}%)\n\n"
    
    summary += f"### Análise de Sensibilidade\n"
    summary += f"- Sensibilidade Alta: {high_sensitivity} ({high_sensitivity/total_effects*100:.1f}%)\n"
    summary += f"- Sensibilidade Média: {medium_sensitivity} ({medium_sensitivity/total_effects*100:.1f}%)\n"
    summary += f"- Sensibilidade Baixa: {low_sensitivity} ({low_sensitivity/total_effects*100:.1f}%)\n\n"
    
    summary += "### Robustez por Métrica\n"
    for metric in metric_summary.index:
        high = metric_summary.loc[metric, 'Alta'] if 'Alta' in metric_summary.columns else 0
        medium = metric_summary.loc[metric, 'Média'] if 'Média' in metric_summary.columns else 0
        low = metric_summary.loc[metric, 'Baixa'] if 'Baixa' in metric_summary.columns else 0
        total = high + medium + low
        
        summary += f"- {metric}: "
        summary += f"Alta {high}/{total} ({high/total*100:.1f}%), "
        summary += f"Média {medium}/{total} ({medium/total*100:.1f}%), "
        summary += f"Baixa {low}/{total} ({low/total*100:.1f}%)\n"
    
    summary += "\n### Robustez por Fase\n"
    for phase in phase_summary.index:
        high = phase_summary.loc[phase, 'Alta'] if 'Alta' in phase_summary.columns else 0
        medium = phase_summary.loc[phase, 'Média'] if 'Média' in phase_summary.columns else 0
        low = phase_summary.loc[phase, 'Baixa'] if 'Baixa' in phase_summary.columns else 0
        total = high + medium + low
        
        summary += f"- {phase}: "
        summary += f"Alta {high}/{total} ({high/total*100:.1f}%), "
        summary += f"Média {medium}/{total} ({medium/total*100:.1f}%), "
        summary += f"Baixa {low}/{total} ({low/total*100:.1f}%)\n"
    
    # Alertas para efeitos com baixa robustez
    if low_robustness > 0:
        summary += "\n### Alertas para Baixa Robustez\n"
        low_robust_effects = robustness_df[robustness_df['overall_robustness'] == 'Baixa']
        
        for _, row in low_robust_effects.iterrows():
            metric = row['metric_name']
            phase = row['experimental_phase']
            tenant = row['tenant_id']
            
            summary += f"- **{metric}** (Fase: {phase}, Tenant: {tenant}):"
            if not row['is_robust_effect']:
                summary += " Instabilidade no tamanho de efeito."
            if not row['is_robust_significance']:
                summary += f" Sensível a rounds: {', '.join(row['sensitive_rounds'])}."
            summary += "\n"
    
    return summary
