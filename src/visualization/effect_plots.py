import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

def generate_effect_size_heatmap(
    aggregated_effects_df: pd.DataFrame,
    output_dir: str,
    metric: Optional[str] = None,
    tenant: Optional[str] = None,
    cmap: str = 'RdBu_r',
    show_significance_markers: bool = True,
    alpha: float = 0.05,
    filename_prefix: str = ''
) -> str:
    """
    Gera um heatmap dos tamanhos de efeito médios.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        output_dir: Diretório para salvar o gráfico
        metric: Métrica específica a visualizar (se None, gera para todas)
        tenant: Tenant específico a visualizar (se None, gera para todos)
        cmap: Colormap para os valores de efeito
        show_significance_markers: Se True, adiciona marcadores para valores significativos
        alpha: Nível de significância
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo gerado
    """
    if aggregated_effects_df.empty:
        logger.warning("DataFrame de efeitos agregados vazio. Não é possível gerar heatmap.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['metric_name', 'experimental_phase', 'tenant_id', 'mean_effect_size', 'is_significant']
    missing_cols = [col for col in required_cols if col not in aggregated_effects_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica e tenant se especificados
    filtered_df = aggregated_effects_df
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{metric}'.")
            return ""
    
    if tenant:
        filtered_df = filtered_df[filtered_df['tenant_id'] == tenant]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para o tenant '{tenant}'.")
            return ""
    
    # Determinar métricas exclusivas
    metrics_to_plot = [metric] if metric else filtered_df['metric_name'].unique()
    
    output_paths = []
    
    for current_metric in metrics_to_plot:
        # Filtrar dados para a métrica atual
        metric_df = filtered_df[filtered_df['metric_name'] == current_metric]
        
        if metric_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{current_metric}'.")
            continue
        
        # Determinar tenants a serem plotados
        tenants_to_plot = [tenant] if tenant else metric_df['tenant_id'].unique()
        
        for current_tenant in tenants_to_plot:
            # Filtrar dados para o tenant atual
            tenant_df = metric_df[metric_df['tenant_id'] == current_tenant]
            
            if tenant_df.empty:
                logger.warning(f"Nenhum dado para o tenant '{current_tenant}' na métrica '{current_metric}'.")
                continue
            
            # Formatar os dados para o heatmap
            # Pivot: linhas são tenants, colunas são fases, valores são mean_effect_size
            phases = sorted(tenant_df['experimental_phase'].unique())
            
            # Filtrar a fase de baseline, que é tipicamente usada como referência
            baseline_phase = tenant_df['baseline_phase'].iloc[0]  # Assume mesmo baseline para todas as entradas
            phases = [phase for phase in phases if phase != baseline_phase]
            
            if not phases:
                logger.warning(f"Sem fases não-baseline para o tenant '{current_tenant}' na métrica '{current_metric}'.")
                continue
            
            # Criar figura
            plt.figure(figsize=(10, 6))
            
            # Preparar dados para o heatmap
            # Usamos um DataFrame auxiliar para ter controle total sobre a apresentação
            effect_values = []
            significance_markers = []
            
            for phase in phases:
                phase_data = tenant_df[tenant_df['experimental_phase'] == phase]
                if phase_data.empty:
                    effect_values.append(np.nan)
                    significance_markers.append('')
                else:
                    effect_values.append(phase_data['mean_effect_size'].iloc[0])
                    if show_significance_markers and phase_data['is_significant'].iloc[0]:
                        significance_markers.append('*')
                    else:
                        significance_markers.append('')
            
            # Criar matriz para o heatmap
            effect_matrix = np.array(effect_values).reshape(1, len(phases))
            
            # Definir limites de colorbar baseados nos valores
            vmax = max(1.0, np.nanmax(np.abs(effect_matrix)))
            vmin = -vmax  # Simétrico para melhor visualização
            
            # Gerar heatmap
            ax = plt.gca()
            heatmap = sns.heatmap(
                effect_matrix,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={'label': "Cohen's d"},
                linewidths=0.5,
                ax=ax
            )
            
            # Adicionar marcadores de significância
            if show_significance_markers:
                for i, marker in enumerate(significance_markers):
                    if marker:
                        ax.text(
                            i + 0.5,  # x-position (cell center)
                            0.5,     # y-position (cell center)
                            marker,
                            horizontalalignment='left',
                            verticalalignment='top',
                            color='black',
                            fontsize=14
                        )
            
            # Rótulos e título
            ax.set_title(f"Tamanho de Efeito por Fase Experimental\n{current_metric.replace('_', ' ').title()} - {current_tenant}")
            ax.set_yticklabels(['Efeito'], rotation=0)
            ax.set_xticklabels(phases, rotation=45, ha='right')
            
            # Adicionar nota sobre significância
            plt.figtext(0.01, 0.01, "* p < {:.2f}".format(alpha), ha="left", fontsize=8)
            
            # Salvar figura
            safe_metric = current_metric.replace(' ', '_')
            safe_tenant = current_tenant.replace(' ', '_')
            filename = f"{filename_prefix}effect_size_heatmap_{safe_metric}_{safe_tenant}.png"
            output_path = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Heatmap salvo em: {output_path}")
            output_paths.append(output_path)
    
    # Se não foi gerado nenhum heatmap, retorna string vazia
    if not output_paths:
        logger.warning("Nenhum heatmap foi gerado.")
        return ""
    
    # Retorna o caminho do último arquivo salvo se vários foram gerados
    return output_paths[-1]

def plot_effect_error_bars(
    aggregated_effects_df: pd.DataFrame,
    output_dir: str,
    metric: Optional[str] = None,
    phases: Optional[List[str]] = None,
    sort_by_magnitude: bool = True,
    show_significance_markers: bool = True,
    alpha: float = 0.05,
    filename_prefix: str = ''
) -> str:
    """
    Cria gráficos de barras de erro para os tamanhos de efeito com intervalos de confiança.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        output_dir: Diretório para salvar o gráfico
        metric: Métrica específica a visualizar (se None, gera para todas)
        phases: Lista de fases a visualizar (se None, usa todas)
        sort_by_magnitude: Se True, ordena por magnitude do efeito
        show_significance_markers: Se True, destaca valores significativos
        alpha: Nível de significância
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo gerado
    """
    if aggregated_effects_df.empty:
        logger.warning("DataFrame de efeitos agregados vazio. Não é possível gerar gráfico de barras de erro.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['metric_name', 'experimental_phase', 'tenant_id', 'mean_effect_size', 
                    'ci_lower', 'ci_upper', 'is_significant']
    missing_cols = [col for col in required_cols if col not in aggregated_effects_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica se especificada
    filtered_df = aggregated_effects_df
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{metric}'.")
            return ""
    
    # Filtrar por fases se especificadas
    if phases:
        filtered_df = filtered_df[filtered_df['experimental_phase'].isin(phases)]
        if filtered_df.empty:
            logger.warning("Nenhum dado para as fases especificadas.")
            return ""
    
    # Determinar métricas exclusivas
    metrics_to_plot = [metric] if metric else filtered_df['metric_name'].unique()
    
    output_paths = []
    
    for current_metric in metrics_to_plot:
        # Filtrar dados para a métrica atual
        metric_df = filtered_df[filtered_df['metric_name'] == current_metric]
        
        if metric_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{current_metric}'.")
            continue
        
        # Excluir a fase de baseline se estiver presente
        baseline_phases = metric_df['baseline_phase'].unique()
        metric_df = metric_df[~metric_df['experimental_phase'].isin(baseline_phases)]
        
        if metric_df.empty:
            logger.warning(f"Sem dados não-baseline para a métrica '{current_metric}'.")
            continue
        
        # Ordenar por magnitude do efeito se especificado
        if sort_by_magnitude:
            metric_df = metric_df.sort_values('mean_effect_size', key=abs, ascending=False)
        
        # Criar um identificador único para cada combinação de fase e tenant
        metric_df['phase_tenant'] = metric_df['experimental_phase'] + ' - ' + metric_df['tenant_id']
        
        # Criar figura
        plt.figure(figsize=(12, max(6, len(metric_df) * 0.3)))  # Ajusta altura com base no número de itens
        
        # Extrair dados para o gráfico
        phase_tenants = metric_df['phase_tenant'].values
        mean_effects = metric_df['mean_effect_size'].to_numpy()
        ci_lowers = metric_df['ci_lower'].to_numpy()
        ci_uppers = metric_df['ci_upper'].to_numpy()
        is_significant = metric_df['is_significant'].values
        
        # Calcular barras de erro (diferença entre média e limites de IC)
        yerr = np.vstack((mean_effects - ci_lowers, ci_uppers - mean_effects))
        
        # Criar cores com base na significância
        colors = ['#1f77b4' if sig else '#aaaaaa' for sig in is_significant]
        
        # Criar o gráfico de barras de erro
        ax = plt.gca()
        bars = ax.barh(range(len(phase_tenants)), mean_effects, xerr=yerr, color=colors, 
                      alpha=0.7, capsize=5, height=0.6)
        
        # Adicionar marcadores de significância
        if show_significance_markers:
            for i, sig in enumerate(is_significant):
                if sig:
                    ax.text(
                        mean_effects[i] + 0.05 if mean_effects[i] >= 0 else mean_effects[i] - 0.05,
                        i,
                        '*',
                        horizontalalignment='left' if mean_effects[i] >= 0 else 'right',
                        verticalalignment='center',
                        color='black',
                        fontsize=12
                    )
        
        # Adicionar uma linha vertical em x=0
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Rótulos e título
        ax.set_yticks(range(len(phase_tenants)))
        ax.set_yticklabels(phase_tenants)
        ax.set_xlabel("Tamanho de Efeito (Cohen's d)")
        ax.set_title(f"Tamanhos de Efeito com IC95%\n{current_metric.replace('_', ' ').title()}")
        
        # Adicionar nota sobre significância
        plt.figtext(0.01, 0.01, "* p < {:.2f}".format(alpha), ha="left", fontsize=8)
        
        # Adicionar áreas sombreadas para interpretar magnitudes
        ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible')
        ax.axvspan(0.2, 0.5, alpha=0.1, color='blue', label='Small')
        ax.axvspan(0.5, 0.8, alpha=0.1, color='green', label='Medium')
        ax.axvspan(0.8, ax.get_xlim()[1], alpha=0.1, color='red', label='Large')
        ax.axvspan(ax.get_xlim()[0], -0.8, alpha=0.1, color='red')
        ax.axvspan(-0.8, -0.5, alpha=0.1, color='green')
        ax.axvspan(-0.5, -0.2, alpha=0.1, color='blue')
        
        # Adicionar legenda
        ax.legend(loc='upper right')
        
        # Salvar figura
        safe_metric = current_metric.replace(' ', '_')
        filename = f"{filename_prefix}effect_errorbar_{safe_metric}.png"
        output_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de barras de erro salvo em: {output_path}")
        output_paths.append(output_path)
    
    # Se não foi gerado nenhum gráfico, retorna string vazia
    if not output_paths:
        logger.warning("Nenhum gráfico de barras de erro foi gerado.")
        return ""
    
    # Retorna o caminho do último arquivo salvo se vários foram gerados
    return output_paths[-1]

def plot_effect_scatter(
    aggregated_effects_df: pd.DataFrame,
    output_dir: str,
    metric: Optional[str] = None,
    filename_prefix: str = ''
) -> str:
    """
    Gera gráficos de dispersão multidimensionais relacionando tamanho de efeito, p-valor e variabilidade.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        output_dir: Diretório para salvar o gráfico
        metric: Métrica específica a visualizar (se None, gera para todas)
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo gerado
    """
    if aggregated_effects_df.empty:
        logger.warning("DataFrame de efeitos agregados vazio. Não é possível gerar gráfico de dispersão.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['metric_name', 'experimental_phase', 'tenant_id', 'mean_effect_size', 
                     'combined_p_value', 'coefficient_of_variation']
    missing_cols = [col for col in required_cols if col not in aggregated_effects_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica se especificada
    filtered_df = aggregated_effects_df
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{metric}'.")
            return ""
    
    # Determinar métricas exclusivas
    metrics_to_plot = [metric] if metric else filtered_df['metric_name'].unique()
    
    output_paths = []
    
    for current_metric in metrics_to_plot:
        # Filtrar dados para a métrica atual
        metric_df = filtered_df[filtered_df['metric_name'] == current_metric]
        
        if metric_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{current_metric}'.")
            continue
        
        # Excluir a fase de baseline se estiver presente
        baseline_phases = metric_df['baseline_phase'].unique()
        metric_df = metric_df[~metric_df['experimental_phase'].isin(baseline_phases)]
        
        if metric_df.empty:
            logger.warning(f"Sem dados não-baseline para a métrica '{current_metric}'.")
            continue
        
        # Preparar dados para o gráfico
        effect_sizes = metric_df['mean_effect_size'].to_numpy()
        log_p_values = -np.log10(metric_df['combined_p_value'].to_numpy())  # Transformação -log10 para p-valores
        cv_values = np.abs(metric_df['coefficient_of_variation'].to_numpy())  # Valores absolutos do CV
        
        # Criar figura
        plt.figure(figsize=(10, 8))
        ax = plt.gca()
        
        # Criar scatter plot com tamanhos variáveis
        scatter = ax.scatter(
            effect_sizes,
            log_p_values,
            s=100 * (1 / (cv_values + 0.1)),  # Tamanho inversamente proporcional ao CV
            c=np.abs(effect_sizes),  # Cor baseada na magnitude do efeito
            cmap='viridis',
            alpha=0.7
        )
        
        # Adicionar linhas de referência
        ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax.axhline(y=-np.log10(0.05), color='red', linestyle='--', alpha=0.5)
        
        # Adicionar texto para cada ponto
        for i, row in metric_df.iterrows():
            phase = row['experimental_phase']
            tenant = row['tenant_id']
            ax.annotate(
                f"{phase}\n{tenant}",
                (row['mean_effect_size'], -np.log10(row['combined_p_value'])),
                fontsize=8,
                alpha=0.7,
                ha='center',
                va='bottom',
                xytext=(0, 5),
                textcoords='offset points'
            )
        
        # Rótulos e título
        ax.set_xlabel("Tamanho de Efeito (Cohen's d)")
        ax.set_ylabel("-log10(p-valor)")
        ax.set_title(f"Relação entre Tamanho de Efeito, Significância e Variabilidade\n{current_metric.replace('_', ' ').title()}")
        
        # Adicionar barra de cores
        colorbar = plt.colorbar(scatter)
        colorbar.set_label('|Cohen\'s d|')
        
        # Adicionar texto explicativo
        plt.figtext(0.01, 0.01, "Tamanho do círculo inversamente proporcional ao coeficiente de variação", ha="left", fontsize=8)
        plt.figtext(0.01, 0.03, "Linha vermelha: limite de significância (p=0.05)", ha="left", fontsize=8)
        
        # Salvar figura
        safe_metric = current_metric.replace(' ', '_')
        filename = f"{filename_prefix}effect_scatter_{safe_metric}.png"
        output_path = os.path.join(output_dir, filename)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de dispersão salvo em: {output_path}")
        output_paths.append(output_path)
    
    # Se não foi gerado nenhum gráfico, retorna string vazia
    if not output_paths:
        logger.warning("Nenhum gráfico de dispersão foi gerado.")
        return ""
    
    # Retorna o caminho do último arquivo salvo se vários foram gerados
    return output_paths[-1]

def generate_effect_forest_plot(
    effect_sizes_df: pd.DataFrame,
    aggregated_effects_df: pd.DataFrame,
    output_dir: str,
    metric: str,
    phase: str,
    tenant: str,
    filename_prefix: str = ''
) -> str:
    """
    Gera um gráfico de floresta (forest plot) para visualizar os tamanhos de efeito
    em diferentes rounds, juntamente com o efeito agregado.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito individuais por round
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        output_dir: Diretório para salvar o gráfico
        metric: Métrica para visualizar
        phase: Fase experimental para visualizar
        tenant: Tenant para visualizar
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo gerado
    """
    if effect_sizes_df.empty or aggregated_effects_df.empty:
        logger.warning("DataFrames de efeitos vazios. Não é possível gerar forest plot.")
        return ""
    
    # Filtrar dados para a combinação específica
    individual_df = effect_sizes_df[
        (effect_sizes_df['metric_name'] == metric) & 
        (effect_sizes_df['experimental_phase'] == phase) & 
        (effect_sizes_df['tenant_id'] == tenant)
    ]
    
    aggregated_row = aggregated_effects_df[
        (aggregated_effects_df['metric_name'] == metric) & 
        (aggregated_effects_df['experimental_phase'] == phase) & 
        (aggregated_effects_df['tenant_id'] == tenant)
    ]
    
    if individual_df.empty or aggregated_row.empty:
        logger.warning(f"Dados insuficientes para forest plot: {metric}, {phase}, {tenant}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Preparar dados para o forest plot
    rounds = individual_df['round_id'].values
    effects = individual_df['effect_size'].to_numpy()
    
    # Estimar intervalos de confiança para rounds individuais
    # Usamos uma aproximação simples baseada no p-valor
    p_values = individual_df['p_value'].to_numpy()
    z_scores = stats.norm.ppf(1 - p_values/2)  # Two-tailed z-score
    std_errs = np.abs(effects / z_scores)
    ci_lowers = effects - 1.96 * std_errs
    ci_uppers = effects + 1.96 * std_errs
    
    # Obter o efeito agregado e seu IC
    mean_effect = aggregated_row['mean_effect_size'].values[0]
    ci_lower = aggregated_row['ci_lower'].values[0]
    ci_upper = aggregated_row['ci_upper'].values[0]
    
    # Criar figura
    plt.figure(figsize=(10, max(6, len(rounds) * 0.5)))
    ax = plt.gca()
    
    # Plotar efeitos individuais
    for i, (effect, lower, upper, round_id) in enumerate(zip(effects, ci_lowers, ci_uppers, rounds)):
        # Plotar o ponto
        ax.scatter(effect, i, s=100, color='blue', zorder=2)
        
        # Plotar a linha do intervalo de confiança
        ax.plot([lower, upper], [i, i], color='blue', linewidth=2, zorder=1)
        
        # Adicionar rótulo do round
        ax.text(-0.1, i, round_id, ha='right', va='center', fontsize=10)
    
    # Adicionar linha separadora
    ax.axhline(y=len(rounds) - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Plotar efeito agregado (abaixo dos efeitos individuais)
    ax.scatter(mean_effect, len(rounds) + 1, s=200, color='red', marker='D', zorder=3)
    ax.plot([ci_lower, ci_upper], [len(rounds) + 1, len(rounds) + 1], color='red', linewidth=3, zorder=2)
    ax.text(-0.1, len(rounds) + 1, "Agregado", ha='right', va='center', fontsize=12, fontweight='bold')
    
    # Adicionar linha vertical em x=0
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
    
    # Configurar eixos
    ax.set_yticks(list(range(len(rounds))) + [len(rounds) + 1])
    ax.set_yticklabels(list(rounds) + ["Agregado"])
    ax.set_xlabel("Tamanho de Efeito (Cohen's d)")
    ax.set_title(f"Forest Plot: {metric.replace('_', ' ').title()}\n{phase} - {tenant}")
    
    # Adicionar áreas sombreadas para interpretar magnitudes
    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligible')
    ax.axvspan(0.2, 0.5, alpha=0.1, color='blue', label='Small')
    ax.axvspan(0.5, 0.8, alpha=0.1, color='green', label='Medium')
    ax.axvspan(0.8, ax.get_xlim()[1], alpha=0.1, color='red', label='Large')
    ax.axvspan(ax.get_xlim()[0], -0.8, alpha=0.1, color='red')
    ax.axvspan(-0.8, -0.5, alpha=0.1, color='green')
    ax.axvspan(-0.5, -0.2, alpha=0.1, color='blue')
    
    # Adicionar legenda
    ax.legend(loc='lower right')
    
    # Salvar figura
    safe_metric = metric.replace(' ', '_')
    safe_phase = phase.replace(' ', '_')
    safe_tenant = tenant.replace(' ', '_')
    filename = f"{filename_prefix}forest_plot_{safe_metric}_{safe_phase}_{safe_tenant}.png"
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Forest plot salvo em: {output_path}")
    return output_path
