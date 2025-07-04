import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
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
    show_reliability_markers: bool = True,
    alpha: float = 0.05,
    filename_prefix: str = '',
    group_by: str = 'tenant'  # 'tenant' ou 'metric'
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
        show_reliability_markers: Se True, adiciona marcadores para indicar confiabilidade
        alpha: Nível de significância
        filename_prefix: Prefixo para o nome do arquivo
        group_by: Como agrupar os dados ('tenant' ou 'metric')
        
    Returns:
        str: Caminho para o arquivo gerado
    """
    if aggregated_effects_df.empty:
        logger.warning("DataFrame de efeitos agregados vazio. Não é possível gerar heatmap.")
        return ""
    
    # Verificar colunas necessárias
    required_cols = ['metric_name', 'experimental_phase', 'tenant_id', 'mean_effect_size', 'is_significant']
    if show_reliability_markers and 'reliability_category' not in aggregated_effects_df.columns:
        show_reliability_markers = False
        logger.warning("Coluna 'reliability_category' não encontrada. Marcadores de confiabilidade desativados.")
    
    missing_cols = [col for col in required_cols if col not in aggregated_effects_df.columns]
    if missing_cols:
        logger.error(f"Colunas necessárias ausentes: {missing_cols}")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica e tenant se especificados
    filtered_df = aggregated_effects_df.copy()
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
    
    # Remover fase de baseline dos dados
    baseline_phases = filtered_df['baseline_phase'].unique()
    filtered_df = filtered_df[~filtered_df['experimental_phase'].isin(baseline_phases)]
    
    if filtered_df.empty:
        logger.warning("Sem dados não-baseline para análise.")
        return ""
    
    output_paths = []
    
    # Dependendo de group_by, organizamos o heatmap de forma diferente
    if group_by == 'tenant':
        # Para cada métrica, criar um heatmap com tenants nas linhas e fases nas colunas
        metrics_to_plot = [metric] if metric else filtered_df['metric_name'].unique()
        
        for current_metric in metrics_to_plot:
            metric_df = filtered_df[filtered_df['metric_name'] == current_metric]
            if metric_df.empty:
                continue
                
            # Listar fases e tenants exclusivos
            phases = sorted(metric_df['experimental_phase'].unique())
            tenants = sorted(metric_df['tenant_id'].unique())
            
            if not phases or not tenants:
                continue
                
            # Preparar matriz para o heatmap
            heatmap_data = np.zeros((len(tenants), len(phases)))
            significance_markers = np.empty((len(tenants), len(phases)), dtype=object)
            reliability_markers = np.empty((len(tenants), len(phases)), dtype=object)
            
            # Preencher a matriz
            for i, t in enumerate(tenants):
                for j, p in enumerate(phases):
                    data = metric_df[(metric_df['tenant_id'] == t) & (metric_df['experimental_phase'] == p)]
                    if not data.empty:
                        heatmap_data[i, j] = data['mean_effect_size'].values[0]
                        
                        if show_significance_markers:
                            significance_markers[i, j] = '*' if data['is_significant'].values[0] else ''
                        
                        if show_reliability_markers and 'reliability_category' in data.columns:
                            if data['reliability_category'].values[0] == 'high':
                                reliability_markers[i, j] = '✓'
                            elif data['reliability_category'].values[0] == 'low':
                                reliability_markers[i, j] = '!'
                            else:
                                reliability_markers[i, j] = ''
                    else:
                        heatmap_data[i, j] = np.nan
                        significance_markers[i, j] = ''
                        reliability_markers[i, j] = ''
            
            # Criar figura com tamanho adaptativo baseado no número de tenants
            plt.figure(figsize=(max(8, len(phases) * 1.2), max(6, len(tenants) * 0.5)))
            
            # Gerar heatmap
            vmax = max(1.0, np.nanmax(np.abs(heatmap_data)))
            vmin = -vmax
            
            ax = plt.gca()
            heatmap = sns.heatmap(
                heatmap_data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={'label': "Cohen's d"},
                linewidths=0.5,
                yticklabels=tenants,
                xticklabels=phases,
                ax=ax
            )
            
            # Adicionar marcadores de significância e confiabilidade
            for i in range(len(tenants)):
                for j in range(len(phases)):
                    annotation = ''
                    
                    if significance_markers[i, j]:
                        annotation += significance_markers[i, j]
                    
                    if reliability_markers[i, j]:
                        annotation += reliability_markers[i, j]
                        
                    if annotation:
                        ax.text(
                            j + 0.85,  # x-position (right side of cell)
                            i + 0.2,   # y-position (top of cell)
                            annotation,
                            fontsize=10,
                            fontweight='bold',
                            color='black'
                        )
            
            # Rótulos e título
            ax.set_title(f"Tamanho de Efeito por Tenant e Fase\n{current_metric.replace('_', ' ').title()}")
            ax.set_ylabel('Tenant')
            ax.set_xlabel('Fase Experimental')
            plt.xticks(rotation=45, ha='right')
            
            # Adicionar legenda
            legend_text = ""
            if show_significance_markers:
                legend_text += "* = p < {:.2f}".format(alpha)
            if show_reliability_markers:
                if legend_text:
                    legend_text += "   "
                legend_text += "✓ = Alta Confiabilidade   ! = Baixa Confiabilidade"
            
            if legend_text:
                plt.figtext(0.01, 0.01, legend_text, ha="left", fontsize=8)
            
            # Salvar figura
            safe_metric = current_metric.replace(' ', '_')
            filename = f"{filename_prefix}effect_size_heatmap_{safe_metric}_by_tenant.png"
            output_path = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Heatmap salvo em: {output_path}")
            output_paths.append(output_path)
            
    else:  # group_by == 'metric'
        # Para cada tenant, criar um heatmap com métricas nas linhas e fases nas colunas
        tenants_to_plot = [tenant] if tenant else filtered_df['tenant_id'].unique()
        
        for current_tenant in tenants_to_plot:
            tenant_df = filtered_df[filtered_df['tenant_id'] == current_tenant]
            if tenant_df.empty:
                continue
                
            # Listar fases e métricas exclusivas
            phases = sorted(tenant_df['experimental_phase'].unique())
            metrics_list = sorted(tenant_df['metric_name'].unique())
            
            if not phases or not metrics_list:
                continue
                
            # Preparar matriz para o heatmap
            heatmap_data = np.zeros((len(metrics_list), len(phases)))
            significance_markers = np.empty((len(metrics_list), len(phases)), dtype=object)
            reliability_markers = np.empty((len(metrics_list), len(phases)), dtype=object)
            
            # Preencher a matriz
            for i, m in enumerate(metrics_list):
                for j, p in enumerate(phases):
                    data = tenant_df[(tenant_df['metric_name'] == m) & (tenant_df['experimental_phase'] == p)]
                    if not data.empty:
                        heatmap_data[i, j] = data['mean_effect_size'].values[0]
                        
                        if show_significance_markers:
                            significance_markers[i, j] = '*' if data['is_significant'].values[0] else ''
                        
                        if show_reliability_markers and 'reliability_category' in data.columns:
                            if data['reliability_category'].values[0] == 'high':
                                reliability_markers[i, j] = '✓'
                            elif data['reliability_category'].values[0] == 'low':
                                reliability_markers[i, j] = '!'
                            else:
                                reliability_markers[i, j] = ''
                    else:
                        heatmap_data[i, j] = np.nan
                        significance_markers[i, j] = ''
                        reliability_markers[i, j] = ''
            
            # Criar figura com tamanho adaptativo baseado no número de métricas
            plt.figure(figsize=(max(8, len(phases) * 1.2), max(6, len(metrics_list) * 0.5)))
            
            # Gerar heatmap
            vmax = max(1.0, np.nanmax(np.abs(heatmap_data)))
            vmin = -vmax
            
            ax = plt.gca()
            heatmap = sns.heatmap(
                heatmap_data,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=0,
                annot=True,
                fmt=".2f",
                cbar_kws={'label': "Cohen's d"},
                linewidths=0.5,
                yticklabels=[m.replace('_', ' ').title() for m in metrics_list],
                xticklabels=phases,
                ax=ax
            )
            
            # Adicionar marcadores de significância e confiabilidade
            for i in range(len(metrics_list)):
                for j in range(len(phases)):
                    annotation = ''
                    
                    if significance_markers[i, j]:
                        annotation += significance_markers[i, j]
                    
                    if reliability_markers[i, j]:
                        annotation += reliability_markers[i, j]
                        
                    if annotation:
                        ax.text(
                            j + 0.85,  # x-position (right side of cell)
                            i + 0.2,   # y-position (top of cell)
                            annotation,
                            fontsize=10,
                            fontweight='bold',
                            color='black'
                        )
            
            # Rótulos e título
            ax.set_title(f"Tamanho de Efeito por Métrica e Fase\nTenant: {current_tenant}")
            ax.set_ylabel('Métrica')
            ax.set_xlabel('Fase Experimental')
            plt.xticks(rotation=45, ha='right')
            
            # Adicionar legenda
            legend_text = ""
            if show_significance_markers:
                legend_text += "* = p < {:.2f}".format(alpha)
            if show_reliability_markers:
                if legend_text:
                    legend_text += "   "
                legend_text += "✓ = Alta Confiabilidade   ! = Baixa Confiabilidade"
            
            if legend_text:
                plt.figtext(0.01, 0.01, legend_text, ha="left", fontsize=8)
            
            # Salvar figura
            safe_tenant = current_tenant.replace(' ', '_')
            filename = f"{filename_prefix}effect_size_heatmap_{safe_tenant}_by_metric.png"
            output_path = os.path.join(output_dir, filename)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Heatmap salvo em: {output_path}")
            output_paths.append(output_path)
    
    # Gerar um heatmap consolidado de todas as métricas por tenant (visão geral)
    if group_by == 'tenant' and not metric and len(filtered_df['metric_name'].unique()) > 1:
        try:
            # Calcular média dos efeitos por tenant (independente da métrica)
            pivot_df = filtered_df.pivot_table(
                index='tenant_id', 
                columns='experimental_phase', 
                values='mean_effect_size', 
                aggfunc='mean'
            )
            
            if not pivot_df.empty:
                # Ordenar tenants por média de efeito absoluto (maior impacto primeiro)
                pivot_df['avg_abs_effect'] = pivot_df.abs().mean(axis=1)
                pivot_df = pivot_df.sort_values('avg_abs_effect', ascending=False)
                pivot_df = pivot_df.drop(columns=['avg_abs_effect'])
                
                plt.figure(figsize=(max(8, len(pivot_df.columns) * 1.2), max(6, len(pivot_df) * 0.5)))
                
                # Gerar heatmap
                vmax = max(1.0, np.abs(pivot_df).max().max())
                vmin = -vmax
                
                ax = plt.gca()
                heatmap = sns.heatmap(
                    pivot_df,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    center=0,
                    annot=True,
                    fmt=".2f",
                    cbar_kws={'label': "Cohen's d Médio"},
                    linewidths=0.5,
                    ax=ax
                )
                
                # Rótulos e título
                ax.set_title("Visão Geral: Tamanho de Efeito Médio por Tenant e Fase\n(Todas as Métricas)")
                ax.set_ylabel('Tenant')
                ax.set_xlabel('Fase Experimental')
                plt.xticks(rotation=45, ha='right')
                
                # Salvar figura
                filename = f"{filename_prefix}effect_size_heatmap_overview_all_metrics.png"
                output_path = os.path.join(output_dir, filename)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Heatmap consolidado salvo em: {output_path}")
                output_paths.append(output_path)
        except Exception as e:
            logger.error(f"Erro ao gerar heatmap consolidado: {e}")
    
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
    show_reliability_indicators: bool = True,
    alpha: float = 0.05,
    filename_prefix: str = '',
    group_by: str = 'tenant'  # 'tenant' ou 'phase'
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
        show_reliability_indicators: Se True, varia a opacidade com base na confiabilidade
        alpha: Nível de significância
        filename_prefix: Prefixo para o nome do arquivo
        group_by: Como agrupar os dados ('tenant' ou 'phase')
        
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
    
    # Verificar coluna de confiabilidade
    has_reliability = 'reliability_category' in aggregated_effects_df.columns
    if not has_reliability:
        show_reliability_indicators = False
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar por métrica se especificada
    filtered_df = aggregated_effects_df.copy()
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
    
    # Remover fase de baseline dos dados
    baseline_phases = filtered_df['baseline_phase'].unique()
    filtered_df = filtered_df[~filtered_df['experimental_phase'].isin(baseline_phases)]
    
    if filtered_df.empty:
        logger.warning("Sem dados não-baseline para análise.")
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
        
        # Agrupar os dados conforme especificado
        if group_by == 'tenant':
            # Gerar um gráfico por fase, com tenants no eixo y
            phases_to_plot = metric_df['experimental_phase'].unique()
            
            for phase in phases_to_plot:
                phase_df = metric_df[metric_df['experimental_phase'] == phase].copy()
                if phase_df.empty:
                    continue
                
                # Ordenar por magnitude do efeito se especificado
                if sort_by_magnitude:
                    phase_df['abs_effect'] = phase_df['mean_effect_size'].abs()
                    phase_df = phase_df.sort_values('abs_effect', ascending=False)
                    phase_df = phase_df.drop(columns=['abs_effect'])
                
                # Criar figura com altura adaptável
                plt.figure(figsize=(10, max(5, len(phase_df) * 0.4)))
                ax = plt.gca()
                
                # Extrair dados para o gráfico
                tenants = phase_df['tenant_id'].values
                mean_effects = phase_df['mean_effect_size'].values
                ci_lowers = phase_df['ci_lower'].values
                ci_uppers = phase_df['ci_upper'].values
                is_significant = phase_df['is_significant'].values
                
                # Calcular barras de erro (diferença entre média e limites de IC)
                yerr = np.vstack((mean_effects - ci_lowers, ci_uppers - mean_effects))
                
                # Determinar cores e opacidades
                colors = ['#1f77b4' if sig else '#aaaaaa' for sig in is_significant]
                
                if show_reliability_indicators and has_reliability:
                    alphas = []
                    for rel in phase_df['reliability_category'].values:
                        if rel == 'high':
                            alphas.append(0.9)
                        elif rel == 'medium':
                            alphas.append(0.7)
                        else:
                            alphas.append(0.4)
                else:
                    alphas = [0.7] * len(tenants)
                
                # Criar o gráfico de barras de erro horizontais
                y_pos = np.arange(len(tenants))
                
                for i, (pos, mean, yerr_i, color, alpha_val) in enumerate(zip(y_pos, mean_effects, yerr.T, colors, alphas)):
                    ax.barh(pos, mean, xerr=[[yerr_i[0]], [yerr_i[1]]], color=color, alpha=alpha_val, 
                           capsize=5, height=0.6, error_kw={'elinewidth': 1.5})
                
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
                                fontsize=12,
                                fontweight='bold'
                            )
                
                # Adicionar linha vertical em x=0
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Adicionar áreas sombreadas para interpretar magnitudes
                xlim = ax.get_xlim()
                xmin = min(xlim[0], -1.0)
                xmax = max(xlim[1], 1.0)
                ax.set_xlim(xmin, xmax)
                
                ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligível')
                ax.axvspan(0.2, 0.5, alpha=0.1, color='blue', label='Pequeno')
                ax.axvspan(0.5, 0.8, alpha=0.1, color='green', label='Médio')
                ax.axvspan(0.8, xmax, alpha=0.1, color='red', label='Grande')
                ax.axvspan(xmin, -0.8, alpha=0.1, color='red')
                ax.axvspan(-0.8, -0.5, alpha=0.1, color='green')
                ax.axvspan(-0.5, -0.2, alpha=0.1, color='blue')
                
                # Rótulos e título
                ax.set_yticks(y_pos)
                ax.set_yticklabels(tenants)
                ax.set_xlabel("Tamanho de Efeito (Cohen's d)")
                ax.set_title(f"Tamanhos de Efeito com IC95%\n{current_metric.replace('_', ' ').title()} - Fase: {phase}")
                
                # Adicionar legendas
                legend_items = []
                if show_significance_markers:
                    legend_items.append(f"* p < {alpha:.3f}")
                
                if show_reliability_indicators and has_reliability:
                    legend_items.append("Opacidade indica confiabilidade (maior = mais confiável)")
                
                if legend_items:
                    plt.figtext(0.01, 0.01, "  |  ".join(legend_items), ha="left", fontsize=8)
                
                # Salvar figura
                safe_metric = current_metric.replace(' ', '_')
                safe_phase = phase.replace(' ', '_')
                filename = f"{filename_prefix}effect_errorbar_{safe_metric}_{safe_phase}.png"
                output_path = os.path.join(output_dir, filename)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Gráfico de barras de erro salvo em: {output_path}")
                output_paths.append(output_path)
                
        else:  # group_by == 'phase'
            # Gerar um gráfico por tenant, com fases no eixo y
            tenants_to_plot = metric_df['tenant_id'].unique()
            
            for tenant_id in tenants_to_plot:
                tenant_df = metric_df[metric_df['tenant_id'] == tenant_id].copy()
                if tenant_df.empty:
                    continue
                
                # Ordenar por magnitude do efeito se especificado
                if sort_by_magnitude:
                    tenant_df['abs_effect'] = tenant_df['mean_effect_size'].abs()
                    tenant_df = tenant_df.sort_values('abs_effect', ascending=False)
                    tenant_df = tenant_df.drop(columns=['abs_effect'])
                else:
                    # Ordenar por fase se não ordenar por magnitude
                    tenant_df = tenant_df.sort_values('experimental_phase')
                
                # Criar figura
                plt.figure(figsize=(10, max(5, len(tenant_df) * 0.4)))
                ax = plt.gca()
                
                # Extrair dados para o gráfico
                phases = tenant_df['experimental_phase'].values
                mean_effects = tenant_df['mean_effect_size'].values
                ci_lowers = tenant_df['ci_lower'].values
                ci_uppers = tenant_df['ci_upper'].values
                is_significant = tenant_df['is_significant'].values
                
                # Calcular barras de erro (diferença entre média e limites de IC)
                mean_effects_np = np.array(mean_effects)
                ci_lowers_np = np.array(ci_lowers)
                ci_uppers_np = np.array(ci_uppers)
                yerr = np.vstack((mean_effects_np - ci_lowers_np, ci_uppers_np - mean_effects_np))
                
                # Determinar cores e opacidades
                colors = ['#1f77b4' if sig else '#aaaaaa' for sig in is_significant]
                
                if show_reliability_indicators and has_reliability:
                    alphas = []
                    for rel in tenant_df['reliability_category'].values:
                        if rel == 'high':
                            alphas.append(0.9)
                        elif rel == 'medium':
                            alphas.append(0.7)
                        else:
                            alphas.append(0.4)
                else:
                    phase_count = len(phases) if phases is not None else 0
                    alphas = [0.7] * phase_count
                
                # Criar o gráfico de barras de erro horizontais
                y_pos = np.arange(len(tenant_df))
                
                for i, (pos, mean, yerr_i, color, alpha_val) in enumerate(zip(y_pos, mean_effects, yerr.T, colors, alphas)):
                    ax.barh(pos, mean, xerr=[[yerr_i[0]], [yerr_i[1]]], color=color, alpha=alpha_val, 
                           capsize=5, height=0.6, error_kw={'elinewidth': 1.5})
                
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
                                fontsize=12,
                                fontweight='bold'
                            )
                
                # Adicionar linha vertical em x=0
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Adicionar áreas sombreadas para interpretar magnitudes
                xlim = ax.get_xlim()
                xmin = min(xlim[0], -1.0)
                xmax = max(xlim[1], 1.0)
                ax.set_xlim(xmin, xmax)
                
                ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligível')
                ax.axvspan(0.2, 0.5, alpha=0.1, color='blue', label='Pequeno')
                ax.axvspan(0.5, 0.8, alpha=0.1, color='green', label='Médio')
                ax.axvspan(0.8, xmax, alpha=0.1, color='red', label='Grande')
                ax.axvspan(xmin, -0.8, alpha=0.1, color='red')
                ax.axvspan(-0.8, -0.5, alpha=0.1, color='green')
                ax.axvspan(-0.5, -0.2, alpha=0.1, color='blue')
                
                # Rótulos e título
                ax.set_yticks(y_pos)
                ax.set_yticklabels(tenant_df['experimental_phase'].tolist())
                ax.set_xlabel("Tamanho de Efeito (Cohen's d)")
                ax.set_title(f"Tamanhos de Efeito por Fase com IC95%\n{current_metric.replace('_', ' ').title()} - Tenant: {tenant_id}")
                
                # Adicionar legendas
                legend_items = []
                if show_significance_markers:
                    legend_items.append(f"* p < {alpha:.3f}")
                
                if show_reliability_indicators and has_reliability:
                    legend_items.append("Opacidade indica confiabilidade (maior = mais confiável)")
                
                if legend_items:
                    plt.figtext(0.01, 0.01, "  |  ".join(legend_items), ha="left", fontsize=8)
                
                # Salvar figura
                safe_metric = current_metric.replace(' ', '_')
                safe_tenant = tenant_id.replace(' ', '_')
                filename = f"{filename_prefix}effect_errorbar_{safe_metric}_{safe_tenant}.png"
                output_path = os.path.join(output_dir, filename)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Gráfico de barras de erro salvo em: {output_path}")
                output_paths.append(output_path)
        
        # Criar visualização consolidada para esta métrica (todas as fases e tenants)
        if len(metric_df['tenant_id'].unique()) > 1 and len(metric_df['experimental_phase'].unique()) > 1:
            try:
                # Criar figura grande
                plt.figure(figsize=(12, 8))
                ax = plt.gca()
                
                # Usar cores diferentes para cada fase
                phases_unique = sorted(metric_df['experimental_phase'].unique())
                tenant_unique = sorted(metric_df['tenant_id'].unique())
                
                # Criar paleta de cores
                import matplotlib.cm as cm
                phase_colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(phases_unique)))
                
                # Criar rótulos combinados para cada ponto de dados
                metric_df['combined_label'] = metric_df['tenant_id'] + ' - ' + metric_df['experimental_phase']
                
                # Ordenar por combinação de tenant e fase
                if sort_by_magnitude:
                    metric_df['abs_effect'] = metric_df['mean_effect_size'].abs()
                    metric_df = metric_df.sort_values(['tenant_id', 'abs_effect'], ascending=[True, False])
                    metric_df = metric_df.drop(columns=['abs_effect'])
                else:
                    metric_df = metric_df.sort_values(['tenant_id', 'experimental_phase'])
                
                # Extrair dados para o gráfico
                combined_labels = metric_df['combined_label'].values
                mean_effects = metric_df['mean_effect_size'].values
                ci_lowers = metric_df['ci_lower'].values
                ci_uppers = metric_df['ci_upper'].values
                is_significant = metric_df['is_significant'].values
                
                # Calcular barras de erro (diferença entre média e limites de IC)
                mean_effects_np = np.array(mean_effects)
                ci_lowers_np = np.array(ci_lowers)
                ci_uppers_np = np.array(ci_uppers)
                yerr = np.vstack((mean_effects_np - ci_lowers_np, ci_uppers_np - mean_effects_np))
                
                # Criar o gráfico de barras de erro horizontais
                y_pos = np.arange(len(combined_labels))
                
                # Determinar cores por fase
                colors = []
                alphas = []
                
                for _, row in metric_df.iterrows():
                    phase_idx = phases_unique.index(row['experimental_phase'])
                    colors.append(phase_colors[phase_idx])
                    
                    if show_reliability_indicators and has_reliability:
                        if row['reliability_category'] == 'high':
                            alphas.append(0.9)
                        elif row['reliability_category'] == 'medium':
                            alphas.append(0.7)
                        else:
                            alphas.append(0.4)
                    else:
                        alphas.append(0.7)
                
                # Plotar cada barra
                for i, (pos, mean, yerr_i, color, alpha_val) in enumerate(zip(y_pos, mean_effects, yerr.T, colors, alphas)):
                    ax.barh(pos, mean, xerr=[[yerr_i[0]], [yerr_i[1]]], color=color, alpha=alpha_val, 
                           capsize=4, height=0.5, error_kw={'elinewidth': 1.2})
                
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
                                fontsize=12,
                                fontweight='bold'
                            )
                
                # Adicionar linha vertical em x=0
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
                
                # Adicionar linhas horizontais de separação entre tenants
                current_tenant = None
                for i, tenant in enumerate(metric_df['tenant_id'].values):
                    if tenant != current_tenant and i > 0:
                        ax.axhline(y=i-0.5, color='gray', linestyle='--', alpha=0.3)
                        current_tenant = tenant
                    elif i == 0:
                        current_tenant = tenant
                
                # Adicionar áreas sombreadas para interpretar magnitudes
                xlim = ax.get_xlim()
                xmin = min(xlim[0], -1.0)
                xmax = max(xlim[1], 1.0)
                ax.set_xlim(xmin, xmax)
                
                # Rótulos e título
                ax.set_yticks(y_pos)
                ax.set_yticklabels(combined_labels)
                ax.set_xlabel("Tamanho de Efeito (Cohen's d)")
                ax.set_title(f"Visão Consolidada: Tamanhos de Efeito com IC95%\n{current_metric.replace('_', ' ').title()}")
                
                # Criar legenda personalizada para fases
                from matplotlib.lines import Line2D
                phase_handles = [Line2D([0], [0], color=phase_colors[i], lw=4, alpha=0.7) 
                               for i in range(len(phases_unique))]
                ax.legend(phase_handles, phases_unique, title="Fase", loc='lower right')
                
                # Adicionar legendas adicionais
                legend_items = []
                if show_significance_markers:
                    legend_items.append(f"* p < {alpha:.3f}")
                
                if show_reliability_indicators and has_reliability:
                    legend_items.append("Opacidade indica confiabilidade (maior = mais confiável)")
                
                if legend_items:
                    plt.figtext(0.01, 0.01, "  |  ".join(legend_items), ha="left", fontsize=8)
                
                # Salvar figura
                safe_metric = current_metric.replace(' ', '_')
                filename = f"{filename_prefix}effect_errorbar_{safe_metric}_consolidated.png"
                output_path = os.path.join(output_dir, filename)
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Gráfico de barras de erro consolidado salvo em: {output_path}")
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Erro ao gerar gráfico consolidado: {e}")
    
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
    phase: Optional[str] = None,
    tenant: Optional[str] = None,
    highlight_significant: bool = True,
    show_labels: bool = True,
    min_effect_size: Optional[float] = None,
    color_by: str = 'effect_size',  # 'effect_size', 'phase', 'tenant' ou 'reliability'
    filename_prefix: str = ''
) -> str:
    """
    Gera gráficos de dispersão multidimensionais relacionando tamanho de efeito, p-valor e variabilidade.
    
    Args:
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        output_dir: Diretório para salvar o gráfico
        metric: Métrica específica a visualizar (se None, gera para todas as métricas)
        phase: Fase específica a visualizar (se None, gera para todas as fases)
        tenant: Tenant específico a visualizar (se None, gera para todos os tenants)
        highlight_significant: Se True, destaca valores significativos
        show_labels: Se True, adiciona rótulos aos pontos
        min_effect_size: Tamanho de efeito mínimo para exibir (em valor absoluto)
        color_by: Variável para colorir os pontos ('effect_size', 'phase', 'tenant' ou 'reliability')
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
    
    # Verificar coluna de confiabilidade
    has_reliability = 'reliability_category' in aggregated_effects_df.columns
    if color_by == 'reliability' and not has_reliability:
        logger.warning("Coluna 'reliability_category' não encontrada. Usando 'effect_size' para colorir.")
        color_by = 'effect_size'
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Filtrar dados
    filtered_df = aggregated_effects_df.copy()
    
    if metric:
        filtered_df = filtered_df[filtered_df['metric_name'] == metric]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{metric}'.")
            return ""
    
    if phase:
        filtered_df = filtered_df[filtered_df['experimental_phase'] == phase]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para a fase '{phase}'.")
            return ""
            
    if tenant:
        filtered_df = filtered_df[filtered_df['tenant_id'] == tenant]
        if filtered_df.empty:
            logger.warning(f"Nenhum dado para o tenant '{tenant}'.")
            return ""
    
    # Remover fase de baseline dos dados
    baseline_phases = filtered_df['baseline_phase'].unique()
    filtered_df = filtered_df[~filtered_df['experimental_phase'].isin(baseline_phases)]
    
    if filtered_df.empty:
        logger.warning("Sem dados não-baseline para análise.")
        return ""
    
    # Filtrar por tamanho de efeito mínimo, se especificado
    if min_effect_size is not None:
        filtered_df = filtered_df[filtered_df['mean_effect_size'].abs() >= min_effect_size]
        if filtered_df.empty:
            logger.warning(f"Nenhum efeito com magnitude >= {min_effect_size}.")
            return ""
    
    # Determinar métricas exclusivas
    if metric:
        metrics_to_plot = [metric]
    else:
        # Se não tiver uma métrica específica, gerar um único gráfico consolidado
        metrics_to_plot = ['all']
    
    output_paths = []
    
    for current_metric in metrics_to_plot:
        if current_metric == 'all':
            metric_df = filtered_df
            metric_title = "Todas as Métricas"
        else:
            metric_df = filtered_df[filtered_df['metric_name'] == current_metric]
            metric_title = current_metric.replace('_', ' ').title()
            
        if metric_df.empty:
            logger.warning(f"Nenhum dado para a métrica '{current_metric}'.")
            continue
        
        # Preparar dados para o gráfico
        effect_sizes = np.array(metric_df['mean_effect_size'].values)
        
        # Tratar p-valores para evitar valores infinitos ao aplicar log10
        p_values = np.array(metric_df['combined_p_value'].values)
        # Substituir zeros por o menor valor positivo representável
        p_values = np.maximum(p_values, np.finfo(float).eps)
        log_p_values = -np.log10(p_values)
        
        # Limitar valores extremos para evitar problemas de visualização
        MAX_LOG_P = 16  # Equivale a p-valor de aproximadamente 10^-16
        log_p_values = np.minimum(log_p_values, MAX_LOG_P)
        
        cv_values = np.array(metric_df['coefficient_of_variation'].abs().values)
        
        # Calcular tamanhos de ponto inversos ao CV (mais estável = maior)
        # Limitar range para melhor visualização
        point_sizes = 100 / (cv_values + 0.1)
        point_sizes = np.clip(point_sizes, 20, 200)  # Min/max tamanho
        
        # Definir cores com base na variável selecionada
        import matplotlib.cm as cm
        from matplotlib.colors import Normalize
        
        if color_by == 'effect_size':
            # Cores baseadas na magnitude do efeito
            colors = np.abs(effect_sizes)
            cmap = cm.get_cmap('viridis')
            norm = Normalize(vmin=0, vmax=max(1.0, np.max(colors)))
            scatter_colors = cmap(norm(colors))
            cbar_label = '|Cohen\'s d|'
            
        elif color_by == 'phase':
            # Cores baseadas na fase
            phases = metric_df['experimental_phase'].unique()
            phase_map = {phase: i for i, phase in enumerate(phases)}
            color_indices = np.array([phase_map[p] for p in metric_df['experimental_phase']])
            cmap = cm.get_cmap('tab10')
            scatter_colors = cmap(color_indices / max(1, len(phases) - 1))
            cbar_label = None  # Usaremos legenda em vez de colorbar
            
        elif color_by == 'tenant':
            # Cores baseadas no tenant
            tenants = metric_df['tenant_id'].unique()
            tenant_map = {t: i for i, t in enumerate(tenants)}
            color_indices = np.array([tenant_map[t] for t in metric_df['tenant_id']])
            cmap = cm.get_cmap('tab20' if len(tenants) > 10 else 'tab10')
            scatter_colors = cmap(color_indices / max(1, len(tenants) - 1))
            cbar_label = None  # Usaremos legenda em vez de colorbar
            
        elif color_by == 'reliability' and has_reliability:
            # Cores baseadas na confiabilidade
            reliability_mapping = {'high': 0.9, 'medium': 0.5, 'low': 0.1}
            reliability_values = np.array([reliability_mapping.get(r, 0.5) for r in metric_df['reliability_category']])
            cmap = cm.get_cmap('RdYlGn')  # Vermelho (baixo) para Verde (alto)
            scatter_colors = cmap(reliability_values)
            cbar_label = 'Confiabilidade'
        else:
            # Fallback para tamanho de efeito
            colors = np.abs(effect_sizes)
            cmap = cm.get_cmap('viridis')
            norm = Normalize(vmin=0, vmax=max(1.0, np.max(colors)))
            scatter_colors = cmap(norm(colors))
            cbar_label = '|Cohen\'s d|'
        
        # Configurar visualização com base nos dados disponíveis
        if current_metric == 'all' and len(metric_df) > 50:
            # Muitos pontos, reduzir rótulos
            show_labels = False
        
        # Criar figura principal com estilo aprimorado
        plt.style.use('seaborn-v0_8-whitegrid')  # Usar estilo moderno do Seaborn para melhor legibilidade
        fig = plt.figure(figsize=(14, 12))  # Figura maior para melhor visualização
        
        # Criar três subplots para diferentes perspectivas
        # 1. Tamanho de efeito vs. p-valor
        ax1 = fig.add_subplot(2, 2, 1)
        scatter1 = ax1.scatter(
            effect_sizes, 
            log_p_values,
            s=point_sizes,
            c=scatter_colors,
            alpha=0.85,  # Maior opacidade para melhor visibilidade
            edgecolors='white',  # Bordas brancas para melhor contraste
            linewidth=0.8,  # Espessura da borda
            zorder=10  # Garantir que os pontos estejam acima da grade
        )
        
        # Adicionar linhas de referência com melhor visualização
        # Linha para p=0.05 (limite de significância)
        sig_line = -np.log10(0.05)
        ax1.axhline(y=sig_line, color='red', linestyle='--', alpha=0.7, label='p=0.05', linewidth=1.5, zorder=5)
        
        # Adicionar área sombreada para região não significativa
        ax1.axhspan(0, sig_line, color='red', alpha=0.1, zorder=1, label='Não significativo')
        
        # Linha vertical em x=0 (sem efeito)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=5)
        
        # Linhas para magnitudes de efeito (pequeno, médio, grande)
        ax1.axvline(x=0.2, color='blue', linestyle=':', alpha=0.4, linewidth=1.2, zorder=4, label='Efeito pequeno')
        ax1.axvline(x=0.5, color='green', linestyle=':', alpha=0.4, linewidth=1.2, zorder=4, label='Efeito médio')
        ax1.axvline(x=0.8, color='purple', linestyle=':', alpha=0.4, linewidth=1.2, zorder=4, label='Efeito grande')
        ax1.axvline(x=-0.2, color='blue', linestyle=':', alpha=0.4, linewidth=1.2, zorder=4)
        ax1.axvline(x=-0.5, color='green', linestyle=':', alpha=0.4, linewidth=1.2, zorder=4)
        ax1.axvline(x=-0.8, color='purple', linestyle=':', alpha=0.4, linewidth=1.2, zorder=4)
        
        # Rótulos aprimorados
        ax1.set_xlabel("Tamanho de Efeito (Cohen's d)", fontsize=11, fontweight='bold')
        ax1.set_ylabel("-log10(p-valor)", fontsize=11, fontweight='bold')
        ax1.set_title("Efeito vs. Significância", fontsize=13, fontweight='bold')
        
        # Adicionar linha de grade sutil
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Adicionar uma legenda contextual
        ax1.legend(loc='upper right', framealpha=0.9, fontsize=9)
        
        # 2. Tamanho de efeito vs. coeficiente de variação (com design aprimorado)
        ax2 = fig.add_subplot(2, 2, 2)
        scatter2 = ax2.scatter(
            effect_sizes, 
            cv_values,
            s=point_sizes,
            c=scatter_colors,
            alpha=0.85,
            edgecolors='white',
            linewidth=0.8,
            zorder=10
        )
        
        # Adicionar linhas de referência e áreas sombreadas para interpretação
        ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1.5, zorder=5)
        
        # Áreas sombreadas para diferentes níveis de variabilidade
        ylim = ax2.get_ylim()
        max_y = max(ylim[1], 1.0)
        
        # Adicionar áreas sombreadas para interpretação da variabilidade
        ax2.axhspan(0, 0.1, color='green', alpha=0.1, zorder=1, label='Baixa variabilidade (CV < 0.1)')
        ax2.axhspan(0.1, 0.3, color='orange', alpha=0.1, zorder=1, label='Média variabilidade (0.1 ≤ CV < 0.3)')
        ax2.axhspan(0.3, max_y, color='red', alpha=0.1, zorder=1, label='Alta variabilidade (CV ≥ 0.3)')
        
        # Linhas de referência para diferentes níveis de CV
        ax2.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, linewidth=1.2, zorder=5, label='CV=0.1')
        ax2.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, linewidth=1.2, zorder=5, label='CV=0.3')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, linewidth=1.2, zorder=5, label='CV=0.5')
        
        # Rótulos aprimorados
        ax2.set_xlabel("Tamanho de Efeito (Cohen's d)", fontsize=11, fontweight='bold')
        ax2.set_ylabel("Coeficiente de Variação", fontsize=11, fontweight='bold')
        ax2.set_title("Efeito vs. Variabilidade", fontsize=13, fontweight='bold')
        ax2.set_ylim(bottom=0, top=min(max_y, 1.0))  # Limitar para evitar valores extremos
        
        # Adicionar grade sutil
        ax2.grid(True, alpha=0.2, linestyle='-', linewidth=0.5)
        
        # Adicionar legenda contextual
        ax2.legend(loc='upper right', framealpha=0.9, fontsize=9)
        
        # 3. Visualização 3D aprimorada: Efeito, p-valor, CV
        from mpl_toolkits.mplot3d import Axes3D  # Importação necessária para projeção 3D
        ax3 = fig.add_subplot(2, 2, (3, 4), projection='3d')
        
        # Criar um scatter plot 3D com melhor contraste e visibilidade
        scatter3 = ax3.scatter(
            effect_sizes,
            log_p_values,
            cv_values,
            s=point_sizes * 0.9,  # Tamanho ligeiramente maior para melhor visibilidade
            c=scatter_colors,
            alpha=0.9,  # Maior opacidade para visualização 3D
            edgecolors='white',
            linewidth=0.8,
            depthshade=True  # Adicionar efeito de profundidade
        )
        
        # Melhorar a representação de planos de referência
        x_min, x_max = min(effect_sizes) - 0.2, max(effect_sizes) + 0.2
        y_min, y_max = min(log_p_values) - 0.5, max(log_p_values) + 0.5
        z_min, z_max = 0, min(max(cv_values) * 1.1, 1.0)
        
        # Definir limites de visualização para melhor enquadramento
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_zlim(z_min, z_max)
        
        # Criar uma malha mais densa para planos de referência mais suaves
        xx, zz = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(z_min, z_max, 10))
        
        # 1. Plano para p=0.05 (significância estatística)
        sig_level = -np.log10(0.05)
        yy_sig = np.ones_like(xx) * sig_level
        ax3.plot_surface(xx, yy_sig, zz, color='red', alpha=0.1, label='p=0.05')
        
        # 2. Plano para CV=0.3 (limite de variabilidade)
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
        zz_cv = np.ones_like(xx) * 0.3
        ax3.plot_surface(xx, yy, zz_cv, color='orange', alpha=0.1, label='CV=0.3')
        
        # 3. Plano para efeito=0 (sem efeito)
        yy, zz = np.meshgrid(np.linspace(y_min, y_max, 10), np.linspace(z_min, z_max, 10))
        xx_zero = np.zeros_like(yy)
        ax3.plot_surface(xx_zero, yy, zz, color='gray', alpha=0.1, label='Efeito=0')
        
        # Adicionar linhas de referência nas intersecções dos planos (para maior clareza visual)
        ax3.plot([x_min, x_max], [sig_level, sig_level], [z_min, z_min], 'r--', alpha=0.5, lw=1.5)
        ax3.plot([0, 0], [y_min, y_max], [z_min, z_min], 'k--', alpha=0.5, lw=1.5)
        ax3.plot([x_min, x_min], [y_min, y_max], [0.3, 0.3], 'orange', linestyle='--', alpha=0.5, lw=1.5)
        
        # Rótulos aprimorados
        ax3.set_xlabel("Tamanho de Efeito (Cohen's d)", fontsize=11, fontweight='bold')
        ax3.set_ylabel("-log10(p-valor)", fontsize=11, fontweight='bold')
        ax3.set_zlabel("Coeficiente de Variação", fontsize=11, fontweight='bold')
        ax3.set_title("Visualização 3D: Efeito, Significância e Variabilidade", fontsize=13, fontweight='bold')
        
        # Ajuste de perspectiva para visão ideal
        ax3.view_init(elev=25, azim=40)
        
        # Adicionar uma anotação para ajudar na interpretação
        ax3.text(x_max*0.8, y_min*1.2, z_min, 
                "Quadrante ideal:\nAlto efeito, Alta significância, Baixa variabilidade", 
                color='darkgreen', fontsize=9)
        
        # Adicionar legenda de cores, se aplicável
        if cbar_label:
            cbar = fig.colorbar(scatter1, ax=[ax1, ax2, ax3], label=cbar_label, shrink=0.6)
        elif color_by == 'phase':
            # Legenda para fases
            from matplotlib.lines import Line2D
            phases = metric_df['experimental_phase'].unique()
            phase_colors = [cmap(i / max(1, len(phases) - 1)) for i in range(len(phases))]
            legend_elements = [Line2D([0], [0], marker='o', color='w', 
                              label=phase, markerfacecolor=color, markersize=10)
                              for phase, color in zip(phases, phase_colors)]
            fig.legend(handles=legend_elements, loc='upper right', title='Fase')
            
        elif color_by == 'tenant':
            # Legenda para tenants
            from matplotlib.lines import Line2D
            tenants = metric_df['tenant_id'].unique()
            tenant_colors = [cmap(i / max(1, len(tenants) - 1)) for i in range(len(tenants))]
            
            # Se houver muitos tenants, limitar a legenda aos principais
            if len(tenants) > 10:
                # Selecionar os tenants com maior efeito médio absoluto
                tenant_effects = {}
                for t in tenants:
                    tenant_effects[t] = metric_df[metric_df['tenant_id'] == t]['mean_effect_size'].abs().mean()
                
                # Ordenar tenants por efeito e selecionar os 10 principais
                top_tenants = sorted(tenant_effects.keys(), key=lambda t: tenant_effects[t], reverse=True)[:10]
                legend_elements = [Line2D([0], [0], marker='o', color='w',
                                  label=t, markerfacecolor=tenant_colors[list(tenants).index(t)], markersize=10)
                                  for t in top_tenants]
                fig.legend(handles=legend_elements, loc='upper right', title='Top Tenants')
            else:
                legend_elements = [Line2D([0], [0], marker='o', color='w',
                                  label=t, markerfacecolor=color, markersize=10)
                                  for t, color in zip(tenants, tenant_colors)]
                fig.legend(handles=legend_elements, loc='upper right', title='Tenant')
                
        elif color_by == 'reliability' and has_reliability:
            # Legenda para categorias de confiabilidade
            from matplotlib.lines import Line2D
            rel_categories = ['high', 'medium', 'low']
            # Definir mapping novamente para esta seção
            rel_mapping = {'high': 0.9, 'medium': 0.5, 'low': 0.1}
            rel_colors = [cmap(rel_mapping[r]) for r in rel_categories]
            rel_labels = ['Alta', 'Média', 'Baixa']
            
            legend_elements = [Line2D([0], [0], marker='o', color='w',
                              label=label, markerfacecolor=color, markersize=10)
                              for label, color in zip(rel_labels, rel_colors)]
            fig.legend(handles=legend_elements, loc='upper right', title='Confiabilidade')
            
        # Adicionar rótulos aos pontos, se solicitado
        if show_labels:
            annotate_data = []
            
            # Determinar os rótulos com base nos dados disponíveis
            if current_metric == 'all':
                # Para gráficos consolidados, mostrar métrica + fase ou métrica + tenant
                if phase:
                    # Se já temos uma fase fixa, rotulamos por métrica + tenant
                    for i, (m, t) in enumerate(zip(metric_df['metric_name'], metric_df['tenant_id'])):
                        # Abreviar nomes longos
                        m_short = m[:10] + '...' if len(m) > 12 else m
                        t_short = t[:10] + '...' if len(t) > 12 else t
                        annotate_data.append((effect_sizes[i], log_p_values[i], cv_values[i], f"{m_short}/{t_short}"))
                elif tenant:
                    # Se já temos um tenant fixo, rotulamos por métrica + fase
                    for i, (m, p) in enumerate(zip(metric_df['metric_name'], metric_df['experimental_phase'])):
                        m_short = m[:10] + '...' if len(m) > 12 else m
                        annotate_data.append((effect_sizes[i], log_p_values[i], cv_values[i], f"{m_short}/{p}"))
                else:
                    # Se não temos fase nem tenant fixos, priorizar pontos significativos ou grandes efeitos
                    sig_threshold = -np.log10(0.05)
                    for i, (m, t, p) in enumerate(zip(metric_df['metric_name'], 
                                                    metric_df['tenant_id'], 
                                                    metric_df['experimental_phase'])):
                        # Só anotar pontos significativos ou com grande efeito
                        if log_p_values[i] > sig_threshold or abs(effect_sizes[i]) > 0.8:
                            m_short = m[:8] + '...' if len(m) > 10 else m
                            annotate_data.append((effect_sizes[i], log_p_values[i], cv_values[i], f"{m_short}"))
            else:
                # Para gráficos de uma métrica específica, rotular por fase + tenant
                for i, (p, t) in enumerate(zip(metric_df['experimental_phase'], metric_df['tenant_id'])):
                    annotate_data.append((effect_sizes[i], log_p_values[i], cv_values[i], f"{p}/{t}"))
            
            # Adicionar anotações aos subplots
            for x, y, z, text in annotate_data:
                # Gráfico 1: Efeito vs p-valor
                ax1.annotate(
                    text,
                    (x, y),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="gray")
                )
                
                # Gráfico 2: Efeito vs CV
                ax2.annotate(
                    text,
                    (x, z),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="gray")
                )
                
                # Para o gráfico 3D, limitamos as anotações para não sobrecarregar
                # Verificar se o ponto é significativo ou tem grande efeito
                if y > -np.log10(0.01) or abs(x) > 0.8:
                    ax3.text(
                        x, y, z,
                        text,
                        fontsize=8,
                        alpha=0.7
                    )
        
        # Adicionar título global com estilo aprimorado
        fig.suptitle(f"Análise Multivariada de Tamanhos de Efeito\n{metric_title}", 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Adicionar uma descrição explicativa sobre como interpretar os gráficos
        description = (
            "Guia de Interpretação:\n"
            "• Pontos acima da linha vermelha horizontal são estatisticamente significativos (p < 0.05)\n"
            "• Pontos mais à esquerda/direita de zero têm maior magnitude de efeito (Cohen's d)\n"
            "• Pontos maiores têm menor variabilidade entre rounds (mais confiáveis)\n"
            "• As cores indicam " + ("intensidade do efeito" if color_by == 'effect_size' else 
                                 "fase experimental" if color_by == 'phase' else
                                 "tenant" if color_by == 'tenant' else
                                 "nível de confiabilidade")
        )
        
        # Adicionar a descrição na parte inferior da figura
        fig.text(0.5, 0.01, description, ha='center', va='bottom', 
                fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        
        # Ajustar layout com mais espaço para acomodar a descrição
        plt.tight_layout(rect=(0.02, 0.06, 0.98, 0.94))  # Deixar espaço para o título e descrição
        
        # Salvar figura
        if current_metric == 'all':
            filename = f"{filename_prefix}effect_scatter_all_metrics.png"
        else:
            safe_metric = current_metric.replace(' ', '_')
            filename = f"{filename_prefix}effect_scatter_{safe_metric}.png"
            
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de dispersão multivariado salvo em: {output_path}")
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
    metric: str = None,
    phase: str = None,
    tenant: str = None,
    show_reliability_indicator: bool = True,
    confidence_level: float = 0.95,
    sort_by: str = 'effect_size',  # 'effect_size', 'p_value', 'round_id'
    custom_title: str = None,
    effect_measure: str = 'effect_size',  # or 'eta_squared'
    cmap: str = 'viridis',
    filename_prefix: str = ''
) -> str:
    """
    Gera um gráfico de floresta (forest plot) para visualizar os tamanhos de efeito
    em diferentes rounds, juntamente com o efeito agregado.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito individuais por round
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        output_dir: Diretório para salvar o gráfico
        metric: Métrica para visualizar (None para gerar para todas as métricas)
        phase: Fase experimental para visualizar (None para gerar para todas as fases)
        tenant: Tenant para visualizar (None para gerar para todos os tenants)
        show_reliability_indicator: Se True, adiciona indicadores de confiabilidade
        confidence_level: Nível de confiança para os intervalos
        sort_by: Como ordenar os rounds ('effect_size', 'p_value', 'round_id')
        custom_title: Título personalizado para o gráfico (None para título automático)
        effect_measure: Medida de efeito a ser usada ('effect_size' para Cohen's d, 'eta_squared' para η²)
        cmap: Colormap para pontos coloridos por significância
        filename_prefix: Prefixo para o nome do arquivo
        
    Returns:
        str: Caminho para o arquivo gerado ou lista de caminhos se múltiplos gráficos forem gerados
    """
    if effect_sizes_df.empty or aggregated_effects_df.empty:
        logger.warning("DataFrames de efeitos vazios. Não é possível gerar forest plot.")
        return ""
    
    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)
    
    # Determinar todas as combinações para visualização
    if metric is None:
        metrics = sorted(aggregated_effects_df['metric_name'].unique())
    else:
        metrics = [metric]
        
    if phase is None:
        phases = sorted(aggregated_effects_df['experimental_phase'].unique())
    else:
        phases = [phase]
        
    if tenant is None:
        tenants = sorted(aggregated_effects_df['tenant_id'].unique())
    else:
        tenants = [tenant]
    
    output_paths = []
    
    # Gerar forest plots para cada combinação
    for m in metrics:
        for p in phases:
            for t in tenants:
                # Filtrar dados para a combinação específica
                individual_df = effect_sizes_df[
                    (effect_sizes_df['metric_name'] == m) & 
                    (effect_sizes_df['experimental_phase'] == p) & 
                    (effect_sizes_df['tenant_id'] == t)
                ]
                
                aggregated_row = aggregated_effects_df[
                    (aggregated_effects_df['metric_name'] == m) & 
                    (aggregated_effects_df['experimental_phase'] == p) & 
                    (aggregated_effects_df['tenant_id'] == t)
                ]
                
                if individual_df.empty or aggregated_row.empty:
                    logger.warning(f"Dados insuficientes para forest plot: {m}, {p}, {t}")
                    continue
                
                # Ordenar os rounds conforme especificado
                if sort_by == 'effect_size':
                    individual_df = individual_df.sort_values(by=effect_measure, ascending=False)
                elif sort_by == 'p_value':
                    individual_df = individual_df.sort_values(by='p_value')
                elif sort_by == 'round_id':
                    individual_df = individual_df.sort_values(by='round_id')
                
                # Preparar dados para o forest plot
                rounds = individual_df['round_id'].values
                effects = individual_df[effect_measure].to_numpy()
                
                # Usar 95% CI já calculados se disponíveis, caso contrário estimar baseado em p-valores
                if 'ci_lower' in individual_df.columns and 'ci_upper' in individual_df.columns:
                    ci_lowers = individual_df['ci_lower'].to_numpy()
                    ci_uppers = individual_df['ci_upper'].to_numpy()
                else:
                    # Estimar intervalos de confiança para rounds individuais baseados no p-valor
                    p_values = individual_df['p_value'].to_numpy()
                    
                    # Calcula o z-score correspondente ao nível de confiança
                    z_value = stats.norm.ppf(1 - (1 - confidence_level) / 2)
                    
                    # Substituir p-valores muito pequenos para evitar problemas
                    p_values_adj = np.clip(p_values, 1e-10, 1-1e-10)
                    z_scores = np.abs(stats.norm.ppf(p_values_adj/2))  # Two-tailed z-score
                    
                    # Calcular erro padrão e intervalos de confiança
                    # Usar uma abordagem mais robusta para casos onde z_scores pode ser zero
                    std_errs = np.zeros_like(effects)
                    nonzero_mask = (z_scores > 0)
                    std_errs[nonzero_mask] = np.abs(effects[nonzero_mask] / z_scores[nonzero_mask])
                    std_errs[~nonzero_mask] = np.std(effects) if len(effects) > 1 else 0.2 * np.abs(effects[~nonzero_mask] + 1e-10)
                    
                    ci_lowers = effects - z_value * std_errs
                    ci_uppers = effects + z_value * std_errs
                
                # Obter o efeito agregado e seu IC
                mean_effect = aggregated_row[f'mean_{effect_measure}'].values[0]
                ci_lower = aggregated_row['ci_lower'].values[0]
                ci_upper = aggregated_row['ci_upper'].values[0]
                
                # Determine se temos informações de robustez/confiabilidade
                has_reliability = 'reliability' in aggregated_row.columns
                reliability_score = aggregated_row['reliability'].values[0] if has_reliability else None
                
                # Criar figura com tamanho apropriado
                plt.figure(figsize=(12, max(6, len(rounds) * 0.5 + 2)))
                ax = plt.gca()
                
                # Determinar cores para os pontos baseadas na significância dos efeitos
                p_values = individual_df['p_value'].values
                significant = p_values < 0.05
                colors = np.array(['#1f77b4' if sig else '#d3d3d3' for sig in significant])
                
                # Se temos uma coluna de confiabilidade para rounds individuais, usar para tamanhos dos pontos
                if 'reliability' in individual_df.columns and show_reliability_indicator:
                    reliabilities = individual_df['reliability'].values
                    sizes = 50 + 100 * reliabilities
                else:
                    sizes = np.ones_like(effects) * 100
                
                # Plotar efeitos individuais
                for i, (effect, lower, upper, round_id, color, size) in enumerate(
                    zip(effects, ci_lowers, ci_uppers, rounds, colors, sizes)):
                    # Plotar o ponto
                    ax.scatter(effect, i, s=size, color=color, zorder=2, alpha=0.8)
                    
                    # Plotar a linha do intervalo de confiança
                    ax.plot([lower, upper], [i, i], color=color, linewidth=2, zorder=1)
                    
                    # Adicionar rótulo do round
                    ax.text(-0.01, i, round_id, ha='right', va='center', fontsize=10, transform=ax.get_yaxis_transform())
                    
                    # Adicionar p-valor
                    p_val = individual_df['p_value'].iloc[i]
                    p_text = f"p={p_val:.4f}" if p_val >= 0.0001 else "p<0.0001"
                    ax.text(1.01, i, p_text, ha='left', va='center', fontsize=9, transform=ax.get_yaxis_transform())
                
                # Adicionar linha separadora
                ax.axhline(y=len(rounds) - 0.5, color='gray', linestyle='--', alpha=0.5)
                
                # Plotar efeito agregado (abaixo dos efeitos individuais)
                agg_color = 'darkred'
                agg_marker = 'D'
                
                # Tamanho baseado em confiabilidade
                agg_size = 200
                if reliability_score is not None and show_reliability_indicator:
                    agg_size = 100 + 200 * reliability_score
                
                ax.scatter(mean_effect, len(rounds) + 1, s=agg_size, color=agg_color, marker=agg_marker, zorder=3)
                ax.plot([ci_lower, ci_upper], [len(rounds) + 1, len(rounds) + 1], color=agg_color, linewidth=3, zorder=2)
                ax.text(-0.01, len(rounds) + 1, "Agregado", ha='right', va='center', 
                        fontsize=12, fontweight='bold', transform=ax.get_yaxis_transform())
                
                # Adicionar p-valor agregado
                if 'combined_p_value' in aggregated_row.columns:
                    combined_p = aggregated_row['combined_p_value'].values[0]
                    p_text = f"p={combined_p:.4f}" if combined_p >= 0.0001 else "p<0.0001"
                    ax.text(1.01, len(rounds) + 1, p_text, ha='left', va='center', 
                          fontsize=10, fontweight='bold', transform=ax.get_yaxis_transform())
                
                # Adicionar linha vertical em x=0
                ax.axvline(x=0, color='black', linestyle='-', alpha=0.2)
                
                # Configurar eixos
                ax.set_yticks([])  # Remover ticks do eixo y (usamos textos em vez disso)
                
                # Adicionar título e rótulos
                if effect_measure == 'effect_size':
                    xlabel = "Tamanho de Efeito (Cohen's d)"
                    interpret_areas = True
                else:
                    xlabel = "Tamanho de Efeito (η²)"
                    interpret_areas = False
                    
                ax.set_xlabel(xlabel)
                
                if custom_title:
                    title = custom_title
                else:
                    title = f"Forest Plot: {m.replace('_', ' ').title()}\n{p} - {t}"
                ax.set_title(title)
                
                # Adicionar áreas sombreadas para interpretar magnitudes (apenas para Cohen's d)
                if interpret_areas:
                    x_min, x_max = ax.get_xlim()
                    # Ajustar limites para garantir visualização adequada
                    x_min = min(x_min, -1.0)
                    x_max = max(x_max, 1.0)
                    ax.set_xlim(x_min, x_max)
                    
                    # Áreas sombreadas para Cohen's d
                    ax.axvspan(-0.2, 0.2, alpha=0.1, color='gray', label='Negligível')
                    ax.axvspan(0.2, 0.5, alpha=0.1, color='blue', label='Pequeno')
                    ax.axvspan(0.5, 0.8, alpha=0.1, color='green', label='Médio')
                    ax.axvspan(0.8, x_max, alpha=0.1, color='red', label='Grande')
                    ax.axvspan(x_min, -0.8, alpha=0.1, color='red')
                    ax.axvspan(-0.8, -0.5, alpha=0.1, color='green')
                    ax.axvspan(-0.5, -0.2, alpha=0.1, color='blue')
                
                # Adicionar legenda
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=10, label='Significativo (p<0.05)'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#d3d3d3', markersize=10, label='Não significativo'),
                    Line2D([0], [0], marker=agg_marker, color='w', markerfacecolor=agg_color, markersize=10, label='Efeito Agregado')
                ]
                
                if show_reliability_indicator and (has_reliability or 'reliability' in individual_df.columns):
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=7, label='Baixa confiabilidade'),
                    )
                    legend_elements.append(
                        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12, label='Alta confiabilidade')
                    )
                
                ax.legend(handles=legend_elements, loc='lower right')
                
                # Adicionar informações de robustez se disponíveis
                if show_reliability_indicator and reliability_score is not None:
                    robustness_text = f"Confiabilidade: {reliability_score:.2f}"
                    plt.figtext(0.02, 0.02, robustness_text, fontsize=10, ha='left')
                
                # Salvar figura
                safe_metric = m.replace(' ', '_').replace('/', '_')
                safe_phase = p.replace(' ', '_').replace('/', '_')
                safe_tenant = t.replace(' ', '_').replace('/', '_')
                
                filename = f"{filename_prefix}forest_plot_{safe_metric}_{safe_phase}_{safe_tenant}.png"
                output_path = os.path.join(output_dir, filename)
                
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Forest plot salvo em: {output_path}")
                output_paths.append(output_path)
    
    # Retornar uma lista de caminhos ou um único caminho dependendo do caso
    if len(output_paths) == 1:
        return output_paths[0]
    elif len(output_paths) > 1:
        return output_paths
    else:
        return ""
