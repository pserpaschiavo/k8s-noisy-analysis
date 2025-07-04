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
import seaborn as sns
from pathlib import Path
import os
from datetime import datetime

from src.effect_aggregation import aggregate_effect_sizes

logger = logging.getLogger(__name__)

def perform_robustness_analysis(
    effect_sizes_df: pd.DataFrame, 
    aggregated_effects_df: pd.DataFrame,
    alpha: float = 0.05,
    leave_one_out: bool = True,
    sensitivity_test: bool = True,
    reliability_threshold: dict = {"high": 0.7, "medium": 0.5, "low": 0.3}
) -> Dict[str, Any]:
    """
    Realiza análise de robustez usando leave-one-out e variando o limiar de significância.
    
    Args:
        effect_sizes_df: DataFrame original com tamanhos de efeito
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        alpha: Nível de significância padrão
        leave_one_out: Se True, realiza análise leave-one-out
        sensitivity_test: Se True, realiza teste de sensibilidade variando alpha
        reliability_threshold: Dicionário com limiares para categorias de confiabilidade
        
    Returns:
        Dict: Resultados da análise de robustez, incluindo:
          - leave_one_out: Resultados da análise leave-one-out
          - sensitivity: Resultados da análise de sensibilidade
          - robustness_score: Pontuações de robustez/confiabilidade
          - summary: Resumo da análise
    """
    if effect_sizes_df.empty or aggregated_effects_df.empty:
        logging.warning("DataFrames vazios. Não é possível realizar análise de robustez.")
        return {"error": "DataFrames vazios"}
    
    results = {}
    
    # 1. Análise Leave-One-Out
    loo_results = {}
    if leave_one_out:
        loo_results = _perform_leave_one_out_analysis(
            effect_sizes_df, 
            aggregated_effects_df
        )
        results["leave_one_out"] = loo_results
    
    # 2. Teste de Sensibilidade
    sensitivity_results = {}
    if sensitivity_test:
        sensitivity_results = _perform_sensitivity_analysis(
            effect_sizes_df,
            aggregated_effects_df,
            alpha_range=[0.01, 0.05, 0.1]
        )
        results["sensitivity"] = sensitivity_results
    
    # 3. Calcular pontuações de robustez
    robustness_scores = calculate_robustness_scores(
        leave_one_out_results=loo_results,
        sensitivity_results=sensitivity_results
    )
    results["robustness_score"] = robustness_scores
    
    # 4. Adicionar categorias de confiabilidade ao DataFrame agregado
    if isinstance(robustness_scores, pd.DataFrame) and not robustness_scores.empty:
        # Criar um DataFrame com as pontuações de robustez
        enhanced_df = aggregated_effects_df.copy()
        
        # Mesclar com as pontuações de robustez
        key_cols = ['metric_name', 'experimental_phase', 'tenant_id']
        enhanced_df = enhanced_df.merge(
            robustness_scores[key_cols + ['reliability']], 
            on=key_cols,
            how='left'
        )
        
        # Categorizar a confiabilidade
        enhanced_df['reliability_category'] = enhanced_df['reliability'].apply(
            lambda x: 'high' if x >= reliability_threshold['high'] else 
                     ('medium' if x >= reliability_threshold['medium'] else 
                      ('low' if x >= reliability_threshold['low'] else 'very_low'))
        )
        
        results["enhanced_aggregated_effects"] = enhanced_df
    
    # 5. Resumo da análise
    results["summary"] = {
        "total_combinations": len(aggregated_effects_df),
        "leave_one_out_performed": leave_one_out,
        "sensitivity_test_performed": sensitivity_test,
        "high_reliability": len(robustness_scores[robustness_scores['reliability'] >= reliability_threshold['high']]) if isinstance(robustness_scores, pd.DataFrame) else 0,
        "medium_reliability": len(robustness_scores[(robustness_scores['reliability'] < reliability_threshold['high']) & 
                                                  (robustness_scores['reliability'] >= reliability_threshold['medium'])]) if isinstance(robustness_scores, pd.DataFrame) else 0,
        "low_reliability": len(robustness_scores[(robustness_scores['reliability'] < reliability_threshold['medium']) & 
                                               (robustness_scores['reliability'] >= reliability_threshold['low'])]) if isinstance(robustness_scores, pd.DataFrame) else 0,
        "very_low_reliability": len(robustness_scores[robustness_scores['reliability'] < reliability_threshold['low']]) if isinstance(robustness_scores, pd.DataFrame) else 0
    }
    
    return results

def _perform_leave_one_out_analysis(
    effect_sizes_df: pd.DataFrame,
    aggregated_effects_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Realiza análise leave-one-out para avaliar a estabilidade dos resultados.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        
    Returns:
        Dict: Resultados da análise leave-one-out
    """
    logging.info("Realizando análise leave-one-out...")
    
    # Inicializar dicionário para resultados
    results = {
        "effect_changes": [],
        "significance_changes": [],
        "variability": pd.DataFrame()
    }
    
    # Agrupar por combinação métrica × fase × tenant
    key_cols = ['metric_name', 'experimental_phase', 'tenant_id']
    
    # Analisar cada combinação
    combinations = effect_sizes_df.groupby(key_cols).size().reset_index().drop(0, axis=1)
    
    loo_data = []
    
    for _, row in combinations.iterrows():
        metric, phase, tenant = row['metric_name'], row['experimental_phase'], row['tenant_id']
        
        # Filtrar dados para esta combinação
        combo_data = effect_sizes_df[
            (effect_sizes_df['metric_name'] == metric) &
            (effect_sizes_df['experimental_phase'] == phase) &
            (effect_sizes_df['tenant_id'] == tenant)
        ]
        
        if combo_data.shape[0] < 2:
            # Precisamos de pelo menos 2 rounds para fazer leave-one-out
            continue
        
        rounds = combo_data['round_id'].unique()
        original_effect = aggregated_effects_df[
            (aggregated_effects_df['metric_name'] == metric) &
            (aggregated_effects_df['experimental_phase'] == phase) &
            (aggregated_effects_df['tenant_id'] == tenant)
        ]['mean_effect_size'].values[0]
        
        original_significance = aggregated_effects_df[
            (aggregated_effects_df['metric_name'] == metric) &
            (aggregated_effects_df['experimental_phase'] == phase) &
            (aggregated_effects_df['tenant_id'] == tenant)
        ]['combined_p_value'].values[0] < 0.05
        
        # Para cada round, remover e recalcular
        loo_effects = []
        loo_significance = []
        
        for round_id in rounds:
            # Remover este round
            reduced_data = combo_data[combo_data['round_id'] != round_id]
            
            # Recalcular agregação
            try:
                loo_agg = aggregate_effect_sizes(reduced_data)
                
                # Extrair efeito e significância
                loo_effect = loo_agg[
                    (loo_agg['metric_name'] == metric) &
                    (loo_agg['experimental_phase'] == phase) &
                    (loo_agg['tenant_id'] == tenant)
                ]['mean_effect_size'].values[0]
                
                loo_pvalue = loo_agg[
                    (loo_agg['metric_name'] == metric) &
                    (loo_agg['experimental_phase'] == phase) &
                    (loo_agg['tenant_id'] == tenant)
                ]['combined_p_value'].values[0]
                
                loo_sig = loo_pvalue < 0.05
                
                loo_effects.append(loo_effect)
                loo_significance.append(loo_sig)
            except Exception as e:
                logging.warning(f"Erro na análise LOO para {metric}/{phase}/{tenant} sem o round {round_id}: {str(e)}")
                continue
        
        # Calcular métricas de variabilidade
        if len(loo_effects) > 0:
            effect_std = np.std(loo_effects)
            effect_range = np.max(loo_effects) - np.min(loo_effects)
            effect_cv = effect_std / abs(np.mean(loo_effects)) if np.mean(loo_effects) != 0 else float('inf')
            
            # Calcular mudanças de significância
            sig_changes = np.sum(np.array(loo_significance) != original_significance)
            sig_change_prop = sig_changes / len(loo_significance)
            
            loo_data.append({
                'metric_name': metric,
                'experimental_phase': phase,
                'tenant_id': tenant,
                'original_effect': original_effect,
                'loo_effect_std': effect_std,
                'loo_effect_range': effect_range,
                'loo_effect_cv': effect_cv,
                'sig_changes': sig_changes,
                'sig_change_proportion': sig_change_prop,
                'original_significant': original_significance,
                'n_rounds': len(rounds)
            })
    
    # Converter para DataFrame
    if loo_data:
        results["variability"] = pd.DataFrame(loo_data)
    
    return results

def _perform_sensitivity_analysis(
    effect_sizes_df: pd.DataFrame,
    aggregated_effects_df: pd.DataFrame,
    alpha_range: List[float] = [0.01, 0.05, 0.1]
) -> Dict[str, Any]:
    """
    Realiza análise de sensibilidade variando o limiar de significância.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        alpha_range: Lista de valores de alpha para teste
        
    Returns:
        Dict: Resultados da análise de sensibilidade
    """
    logging.info("Realizando análise de sensibilidade...")
    
    results = {
        "alpha_sensitivity": [],
        "sensitivity_curves": pd.DataFrame()
    }
    
    # Agrupar por combinação métrica × fase × tenant
    key_cols = ['metric_name', 'experimental_phase', 'tenant_id']
    
    # Analisar cada combinação
    combinations = aggregated_effects_df[key_cols].drop_duplicates()
    
    sensitivity_data = []
    
    for _, row in combinations.iterrows():
        metric, phase, tenant = row['metric_name'], row['experimental_phase'], row['tenant_id']
        
        # Filtrar dados para esta combinação
        combo_data = effect_sizes_df[
            (effect_sizes_df['metric_name'] == metric) &
            (effect_sizes_df['experimental_phase'] == phase) &
            (effect_sizes_df['tenant_id'] == tenant)
        ]
        
        if combo_data.shape[0] < 2:
            # Precisamos de pelo menos 2 rounds para análise significativa
            continue
        
        # Obter p-valor combinado original
        original_p = aggregated_effects_df[
            (aggregated_effects_df['metric_name'] == metric) &
            (aggregated_effects_df['experimental_phase'] == phase) &
            (aggregated_effects_df['tenant_id'] == tenant)
        ]['combined_p_value'].values[0]
        
        # Para cada alpha no range, verificar significância
        significance_by_alpha = {}
        for alpha in alpha_range:
            significance_by_alpha[alpha] = original_p < alpha
        
        # Calcular curva de sensibilidade
        # Use mais pontos para traçar uma curva suave
        detailed_alphas = np.logspace(-4, -1, 20)  # 20 pontos de 0.0001 a 0.1
        significance_curve = []
        
        for alpha in detailed_alphas:
            significance_curve.append({
                'metric_name': metric,
                'experimental_phase': phase,
                'tenant_id': tenant,
                'alpha': alpha,
                'is_significant': original_p < alpha
            })
        
        # Adicionar dados de sensibilidade
        sensitivity_data.append({
            'metric_name': metric,
            'experimental_phase': phase,
            'tenant_id': tenant,
            'original_p_value': original_p,
            'sig_at_001': original_p < 0.01,
            'sig_at_005': original_p < 0.05,
            'sig_at_01': original_p < 0.1,
            'p_stability': original_p / min(alpha_range),
            'smallest_alpha_significant': min([a for a in detailed_alphas if original_p < a], 
                                           default=float('inf'))
        })
    
    # Converter para DataFrames
    if sensitivity_data:
        results["alpha_sensitivity"] = pd.DataFrame(sensitivity_data)
        
    if significance_curve:
        results["sensitivity_curves"] = pd.DataFrame(significance_curve)
    
    return results

def calculate_robustness_scores(
    leave_one_out_results: Optional[Dict[str, Any]] = None,
    sensitivity_results: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Calcula pontuações de robustez com base nas análises leave-one-out e de sensibilidade.
    
    Args:
        leave_one_out_results: Resultados da análise leave-one-out
        sensitivity_results: Resultados da análise de sensibilidade
        
    Returns:
        DataFrame: Pontuações de robustez para cada combinação métrica × fase × tenant
    """
    logging.info("Calculando pontuações de robustez...")
    
    robustness_data = []
    
    # Verificar se temos dados de LOO
    loo_df = None
    if leave_one_out_results and 'variability' in leave_one_out_results:
        loo_df = leave_one_out_results['variability']
    
    # Verificar se temos dados de sensibilidade
    sensitivity_df = None
    if sensitivity_results and 'alpha_sensitivity' in sensitivity_results:
        sensitivity_df = sensitivity_results['alpha_sensitivity']
    
    # Se não temos nenhum dos dois, não podemos calcular robustez
    if loo_df is None and sensitivity_df is None:
        logging.warning("Dados insuficientes para calcular robustez.")
        return pd.DataFrame()
    
    # Mesclar dados se ambos estiverem disponíveis
    key_cols = ['metric_name', 'experimental_phase', 'tenant_id']
    
    if loo_df is not None and sensitivity_df is not None:
        combined_df = pd.merge(loo_df, sensitivity_df, on=key_cols, how='outer')
    elif loo_df is not None:
        combined_df = loo_df
    else:
        combined_df = sensitivity_df
    
    # Calcular pontuação de robustez para cada linha
    for _, row in combined_df.iterrows():
        robustness_score = 1.0  # Começar com pontuação perfeita
        
        # Penalidades baseadas em leave-one-out
        if 'loo_effect_cv' in row and not pd.isna(row['loo_effect_cv']):
            # Penalidade baseada no coeficiente de variação: quanto maior, menos robusto
            # Limitar CV a 2.0 para evitar penalidades extremas
            cv_penalty = min(row['loo_effect_cv'], 2.0) / 2.0
            robustness_score -= cv_penalty * 0.3  # CV contribui 30% para a pontuação
        
        if 'sig_change_proportion' in row and not pd.isna(row['sig_change_proportion']):
            # Penalidade baseada na proporção de mudanças de significância
            robustness_score -= row['sig_change_proportion'] * 0.4  # Mudanças de sig contribuem 40%
        
        # Penalidades baseadas na sensibilidade ao alpha
        if 'p_stability' in row and not pd.isna(row['p_stability']):
            # Penalidade baseada na estabilidade do p-valor
            # p_stability é a razão entre p-valor e o alpha mínimo testado
            p_stability = min(row['p_stability'], 10.0)  # Limitar para evitar penalidades extremas
            p_penalty = p_stability / 10.0
            robustness_score -= p_penalty * 0.2  # P-valor contribui 20%
        
        # Ajuste baseado no número de rounds (mais rounds = mais confiança)
        if 'n_rounds' in row and not pd.isna(row['n_rounds']):
            n_rounds = row['n_rounds']
            # Ajuste de 0 a 0.1 baseado em rounds
            round_bonus = min(0.1, (n_rounds - 2) * 0.02)  # +0.02 para cada round além de 2
            robustness_score += round_bonus
        
        # Garantir que a pontuação fique entre 0 e 1
        robustness_score = max(0.0, min(1.0, robustness_score))
        
        robustness_data.append({
            'metric_name': row['metric_name'],
            'experimental_phase': row['experimental_phase'],
            'tenant_id': row['tenant_id'],
            'reliability': robustness_score
        })
    
    return pd.DataFrame(robustness_data)

def generate_robustness_summary(
    robustness_results: Dict[str, Any], 
    output_dir: str
) -> Dict[str, str]:
    """
    Gera visualizações e relatório de resumo para os resultados da análise de robustez.
    
    Args:
        robustness_results: Resultados da análise de robustez
        output_dir: Diretório para salvar os relatórios e visualizações
        
    Returns:
        Dict[str, str]: Caminhos para os arquivos gerados
    """
    outputs = {}
    
    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Visualização da estabilidade leave-one-out
    if 'leave_one_out' in robustness_results and 'variability' in robustness_results['leave_one_out']:
        loo_df = robustness_results['leave_one_out']['variability']
        
        if not loo_df.empty:
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            sns.histplot(loo_df['loo_effect_cv'], bins=20, kde=True)
            plt.title('Distribuição do Coef. de Variação')
            plt.xlabel('Coeficiente de Variação')
            
            plt.subplot(2, 2, 2)
            sns.histplot(loo_df['sig_change_proportion'], bins=20, kde=True)
            plt.title('Proporção de Mudanças de Significância')
            plt.xlabel('Proporção')
            
            plt.subplot(2, 2, 3)
            sns.scatterplot(data=loo_df, x='loo_effect_cv', y='sig_change_proportion', 
                            hue='original_significant', size='n_rounds',
                            palette=['gray', 'blue'])
            plt.title('CV vs. Mudanças de Significância')
            plt.xlabel('Coeficiente de Variação')
            plt.ylabel('Proporção de Mudanças de Significância')
            
            plt.subplot(2, 2, 4)
            sns.boxplot(data=loo_df, x='original_significant', y='loo_effect_cv')
            plt.title('CV por Status de Significância Original')
            plt.xlabel('Significativo Originalmente')
            plt.ylabel('Coeficiente de Variação')
            
            plt.tight_layout()
            loo_path = os.path.join(output_dir, 'leave_one_out_stability.png')
            plt.savefig(loo_path, dpi=300)
            plt.close()
            
            outputs['leave_one_out_plot'] = loo_path
    
    # 2. Visualização da sensibilidade ao alpha
    if 'sensitivity' in robustness_results and 'alpha_sensitivity' in robustness_results['sensitivity']:
        sens_df = robustness_results['sensitivity']['alpha_sensitivity']
        
        if not sens_df.empty:
            plt.figure(figsize=(12, 6))
            
            plt.subplot(1, 2, 1)
            sns.histplot(sens_df['original_p_value'], bins=30, kde=True)
            plt.title('Distribuição de p-valores')
            plt.xlabel('p-valor')
            plt.xscale('log')
            
            plt.subplot(1, 2, 2)
            sig_counts = [
                sens_df['sig_at_001'].sum(),
                sens_df['sig_at_005'].sum() - sens_df['sig_at_001'].sum(),
                sens_df['sig_at_01'].sum() - sens_df['sig_at_005'].sum(),
                len(sens_df) - sens_df['sig_at_01'].sum()
            ]
            labels = ['p<0.01', '0.01≤p<0.05', '0.05≤p<0.1', 'p≥0.1']
            plt.pie(sig_counts, labels=labels, autopct='%1.1f%%', startangle=90,
                   colors=['darkgreen', 'green', 'lightgreen', 'gray'])
            plt.title('Distribuição de Significância por Alpha')
            
            plt.tight_layout()
            sens_path = os.path.join(output_dir, 'alpha_sensitivity.png')
            plt.savefig(sens_path, dpi=300)
            plt.close()
            
            outputs['sensitivity_plot'] = sens_path
    
    # 3. Visualização das pontuações de robustez
    if 'robustness_score' in robustness_results:
        robustness_df = robustness_results['robustness_score']
        
        if not robustness_df.empty:
            plt.figure(figsize=(10, 6))
            
            sns.histplot(robustness_df['reliability'], bins=20, kde=True)
            plt.title('Distribuição de Pontuações de Confiabilidade')
            plt.xlabel('Pontuação de Confiabilidade')
            plt.axvline(x=0.7, color='green', linestyle='--', label='Alto (>0.7)')
            plt.axvline(x=0.5, color='blue', linestyle='--', label='Médio (>0.5)')
            plt.axvline(x=0.3, color='red', linestyle='--', label='Baixo (>0.3)')
            plt.legend()
            
            plt.tight_layout()
            robustness_path = os.path.join(output_dir, 'robustness_scores.png')
            plt.savefig(robustness_path, dpi=300)
            plt.close()
            
            outputs['robustness_plot'] = robustness_path
    
    # 4. Gerar relatório de resumo em Markdown
    if 'summary' in robustness_results:
        summary = robustness_results['summary']
        
        summary_md = [
            "# Relatório de Análise de Robustez\n",
            f"Data de geração: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n",
            "## Resumo da Análise\n",
            f"- Total de combinações analisadas: {summary.get('total_combinations', 'N/A')}\n",
            f"- Análise Leave-One-Out: {'Realizada' if summary.get('leave_one_out_performed', False) else 'Não realizada'}\n",
            f"- Teste de Sensibilidade: {'Realizado' if summary.get('sensitivity_test_performed', False) else 'Não realizado'}\n\n",
            "## Distribuição de Confiabilidade\n",
            f"- Alta confiabilidade: {summary.get('high_reliability', 0)} combinações\n",
            f"- Média confiabilidade: {summary.get('medium_reliability', 0)} combinações\n",
            f"- Baixa confiabilidade: {summary.get('low_reliability', 0)} combinações\n",
            f"- Muito baixa confiabilidade: {summary.get('very_low_reliability', 0)} combinações\n\n"
        ]
        
        # Adicionar recomendações
        summary_md.extend([
            "## Recomendações\n",
            "- Considerar como confiáveis apenas combinações com pontuação alta (>0.7)\n",
            "- Para combinações com confiabilidade média (>0.5), interpretar com cautela\n",
            "- Combinações com confiabilidade baixa ou muito baixa podem ser instáveis e devem ser interpretadas com ressalvas\n",
            "- Considerar realizar mais rounds experimentais para combinações importantes com baixa confiabilidade\n\n"
        ])
        
        # Adicionar informações sobre visualizações
        if outputs:
            summary_md.extend(["## Visualizações geradas\n"])
            for key, path in outputs.items():
                filename = os.path.basename(path)
                summary_md.append(f"- {key.replace('_', ' ').title()}: [{filename}]({filename})\n")
        
        summary_path = os.path.join(output_dir, 'robustness_summary.md')
        with open(summary_path, 'w') as f:
            f.writelines(summary_md)
        
        outputs['summary_report'] = summary_path
    
    return outputs
