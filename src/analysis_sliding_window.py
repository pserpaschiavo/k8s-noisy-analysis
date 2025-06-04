"""
Module: analysis_sliding_window.py
Description: Implementa análises de séries temporais com janelas deslizantes.

Este módulo oferece ferramentas para realizar análises de correlação e causalidade
em janelas temporais deslizantes, permitindo visualizar a evolução desses indicadores
ao longo do tempo, ao invés de apenas valores agregados para todo o período.
"""
import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
from datetime import datetime, timedelta

# Importações de módulos do projeto
from src.analysis_correlation import compute_correlation_matrix
from src.analysis_causality import CausalityAnalyzer

# Configuração dos gráficos
plt.style.use('tableau-colorblind10')
logger = logging.getLogger("sliding_window")


class SlidingWindowAnalyzer:
    """
    Classe para análises em janelas deslizantes de séries temporais.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o analisador de janelas deslizantes.
        
        Args:
            df: DataFrame no formato long com os dados das séries temporais
        """
        self.df = df
        self.logger = logging.getLogger("sliding_window.analyzer")
    
    def analyze_correlation_sliding_window(
        self,
        metric: str,
        phase: str,
        round_id: str,
        window_size: str = '5min',
        step_size: str = '1min',
        method: str = 'pearson',
        min_periods: int = 3,
        tenant_pairs: List[Tuple[str, str]] = None
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Analisa a correlação entre tenants em janelas deslizantes.
        
        Args:
            metric: Nome da métrica para análise
            phase: Fase experimental para análise
            round_id: ID do round para análise
            window_size: Tamanho da janela (formato pandas offset string, ex: '5min', '1h')
            step_size: Tamanho do passo para deslizar a janela (mesmo formato)
            method: Método de correlação ('pearson', 'spearman', 'kendall')
            min_periods: Número mínimo de períodos para calcular correlação
            tenant_pairs: Lista de tuplas (tenant1, tenant2) para analisar.
                          Se None, usa todas as combinações possíveis.
                          
        Returns:
            Dicionário com pares de tenants como chaves e DataFrames de correlação por janela como valores.
            Cada DataFrame tem índice 'window_start' e colunas 'window_end' e 'correlation'.
        """
        subset = self.df[(self.df['metric_name'] == metric) & 
                         (self.df['experimental_phase'] == phase) & 
                         (self.df['round_id'] == round_id)]
        
        if subset.empty:
            self.logger.warning(f"Sem dados para {metric}, {phase}, {round_id}")
            return {}
        
        # Converte timestamp para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
            subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
        
        # Obtém lista de tenants disponíveis
        tenants = subset['tenant_id'].unique()
        
        # Se tenant_pairs não for fornecido, gera todas as combinações possíveis
        if tenant_pairs is None:
            tenant_pairs = []
            for i, t1 in enumerate(tenants):
                for j, t2 in enumerate(tenants):
                    if i < j:  # Evita repetição e auto-correlação
                        tenant_pairs.append((t1, t2))
        
        # Prepara formato wide para facilitar a análise
        wide_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Determina timestamps de início e fim
        start_time = wide_df.index.min()
        end_time = wide_df.index.max()
        
        # Converte window_size e step_size para timedelta
        window_delta = pd.Timedelta(window_size)
        step_delta = pd.Timedelta(step_size)
        
        # Dicionário para armazenar resultados
        results = {}
        
        # Para cada par de tenants
        for tenant1, tenant2 in tenant_pairs:
            if tenant1 not in wide_df.columns or tenant2 not in wide_df.columns:
                self.logger.warning(f"Tenant {tenant1} ou {tenant2} não encontrado nos dados")
                continue
                
            # Lista para armazenar correlações de cada janela
            window_results = []
            
            # Desliza a janela
            current_start = start_time
            while current_start + window_delta <= end_time:
                current_end = current_start + window_delta
                
                # Filtra dados dentro da janela
                window_data = wide_df.loc[(wide_df.index >= current_start) & 
                                         (wide_df.index < current_end)]
                
                # Verifica se há dados suficientes
                if len(window_data) >= min_periods:
                    # Calcula correlação
                    corr = window_data[tenant1].corr(window_data[tenant2], method=method)
                    window_results.append({
                        'window_start': current_start,
                        'window_end': current_end,
                        'correlation': corr
                    })
                
                # Avança a janela
                current_start += step_delta
            
            # Converte para DataFrame
            if window_results:
                results[(tenant1, tenant2)] = pd.DataFrame(window_results)
            
        return results
    
    def analyze_causality_sliding_window(
        self,
        metric: str,
        phase: str,
        round_id: str,
        window_size: str = '8min',  # Aumentado para 8min para capturar mais pontos
        step_size: str = '2min',    # Aumentado para 2min para melhorar performance
        method: str = 'granger',
        max_lag: int = 3,
        min_periods: int = 5,       # Reduzido para permitir mais janelas válidas
        tenant_pairs: List[Tuple[str, str]] = None,
        bins: int = 5               # Número de bins para transfer entropy
    ) -> Dict[Tuple[str, str], pd.DataFrame]:
        """
        Analisa a causalidade entre tenants em janelas deslizantes.
        
        Args:
            metric: Nome da métrica para análise
            phase: Fase experimental para análise
            round_id: ID do round para análise
            window_size: Tamanho da janela (formato pandas offset string, ex: '5min', '1h')
            step_size: Tamanho do passo para deslizar a janela
            method: Método de causalidade ('granger' ou 'transfer_entropy')
            max_lag: Atraso máximo para testes de causalidade
            min_periods: Número mínimo de períodos para calcular causalidade
            tenant_pairs: Lista de tuplas (tenant1, tenant2) para analisar,
                         onde tenant1 é potencial causa de tenant2
                          
        Returns:
            Dicionário com pares de tenants (causa, efeito) como chaves e 
            DataFrames com scores/p-valores por janela como valores.
        """
        subset = self.df[(self.df['metric_name'] == metric) & 
                         (self.df['experimental_phase'] == phase) & 
                         (self.df['round_id'] == round_id)]
        
        if subset.empty:
            self.logger.warning(f"Sem dados para {metric}, {phase}, {round_id}")
            return {}
        
        # Converte timestamp para datetime se necessário
        if not pd.api.types.is_datetime64_any_dtype(subset['timestamp']):
            subset['timestamp'] = pd.to_datetime(subset['timestamp'], errors='coerce')
        
        # Obtém lista de tenants disponíveis
        tenants = subset['tenant_id'].unique()
        
        # Se tenant_pairs não for fornecido, gera todas as combinações possíveis
        if tenant_pairs is None:
            tenant_pairs = []
            for t1 in tenants:
                for t2 in tenants:
                    if t1 != t2:  # Evita auto-causalidade
                        tenant_pairs.append((t1, t2))
        
        # Prepara formato wide para facilitar a análise
        wide_df = subset.pivot_table(index='timestamp', columns='tenant_id', values='metric_value')
        
        # Determina timestamps de início e fim
        start_time = wide_df.index.min()
        end_time = wide_df.index.max()
        
        # Converte window_size e step_size para timedelta
        window_delta = pd.Timedelta(window_size)
        step_delta = pd.Timedelta(step_size)
        
        # Dicionário para armazenar resultados
        results = {}
        
        # Inicializa o analisador de causalidade
        causality_analyzer = CausalityAnalyzer(self.df)
        
        # Para cada par de tenants
        for source, target in tenant_pairs:
            if source not in wide_df.columns or target not in wide_df.columns:
                self.logger.warning(f"Tenant {source} ou {target} não encontrado nos dados")
                continue
                
            # Lista para armazenar resultados de causalidade de cada janela
            window_results = []
            
            # Desliza a janela
            current_start = start_time
            while current_start + window_delta <= end_time:
                current_end = current_start + window_delta
                
                # Cria um subset temporário do DataFrame com apenas os dados na janela atual
                window_subset = subset[(subset['timestamp'] >= current_start) & 
                                     (subset['timestamp'] < current_end)]
                
                # Verifica se há dados suficientes
                tenant_data = window_subset.groupby('tenant_id').size()
                if len(window_subset) >= min_periods and source in tenant_data and target in tenant_data:
                    # Calcula causalidade
                    try:
                        if method == 'granger':
                            # Formato específico para teste de Granger
                            window_wide = window_subset.pivot_table(
                                index='timestamp', columns='tenant_id', values='metric_value'
                            )
                            # Garantir ordenação temporal
                            window_wide = window_wide.sort_index()
                            # Preencher valores faltantes por interpolação
                            window_wide = window_wide.interpolate()
                            
                            source_series = window_wide[source].values
                            target_series = window_wide[target].values
                            
                            # Calcula teste de causalidade de Granger
                            try:
                                result = causality_analyzer._granger_causality_test(
                                    source_series, target_series, max_lag=max_lag
                                )
                                # Extrai p-valor para o lag com menor p-valor
                                min_p_value = min([v[0] for v in result.values()])
                                window_results.append({
                                    'window_start': current_start,
                                    'window_end': current_end,
                                    'p_value': min_p_value,
                                    'causality_score': 1.0 - min_p_value  # Converte p-value para score (maior = mais causalidade)
                                })
                            except:
                                self.logger.warning(f"Erro no teste de Granger para janela {current_start} a {current_end}")
                        
                        elif method == 'transfer_entropy':
                            # Formato específico para cálculo de Transfer Entropy
                            window_wide = window_subset.pivot_table(
                                index='timestamp', columns='tenant_id', values='metric_value'
                            )
                            # Garantir ordenação temporal
                            window_wide = window_wide.sort_index()
                            
                            source_series = window_wide[source].values
                            target_series = window_wide[target].values
                            
                            # Calcula Transfer Entropy
                            from src.analysis_causality import _transfer_entropy
                            te_value = _transfer_entropy(target_series, source_series, bins=8, k=1)
                            window_results.append({
                                'window_start': current_start,
                                'window_end': current_end,
                                'transfer_entropy': te_value
                            })
                            
                    except Exception as e:
                        self.logger.warning(f"Erro na análise de causalidade para janela {current_start} a {current_end}: {e}")
                
                # Avança a janela
                current_start += step_delta
            
            # Converte para DataFrame
            if window_results:
                results[(source, target)] = pd.DataFrame(window_results)
            
        return results

    def plot_sliding_window_correlation(
        self,
        results: Dict[Tuple[str, str], pd.DataFrame],
        metric: str,
        phase: str,
        round_id: str,
        out_dir: str,
        top_n: int = None
    ) -> List[str]:
        """
        Plota a evolução da correlação em janelas deslizantes para pares de tenants.
        
        Args:
            results: Resultado da função analyze_correlation_sliding_window
            metric: Nome da métrica analisada
            phase: Fase experimental analisada
            round_id: ID do round analisado
            out_dir: Diretório para salvar os plots
            top_n: Se fornecido, plota apenas os top_n pares com maior correlação média
            
        Returns:
            Lista de caminhos para os arquivos de imagem gerados
        """
        if not results:
            return []
            
        # Paleta de cores aprimorada para melhor visualização
        from matplotlib import cm
        color_palette = cm.viridis(np.linspace(0, 1, max(len(results), 10)))
        
        # Filtra top_n pares se solicitado
        if top_n is not None and len(results) > top_n:
            # Calcula correlação média para cada par
            avg_corrs = {pair: df['correlation'].abs().mean() for pair, df in results.items()}
            # Ordena pares por correlação média decrescente
            sorted_pairs = sorted(avg_corrs.items(), key=lambda x: x[1], reverse=True)
            # Seleciona top_n pares
            top_pairs = [pair for pair, _ in sorted_pairs[:top_n]]
            results = {pair: df for pair, df in results.items() if pair in top_pairs}
        
        output_paths = []  
        
        # Configura estilo do matplotlib para maior legibilidade
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.titlesize': 16,
            'figure.figsize': (12, 7),
            'figure.dpi': 120
        })
        
        # Gera um plot para cada par
        for idx, ((tenant1, tenant2), df) in enumerate(results.items()):
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Formata o timestamp para exibição mais legível
            df['formatted_time'] = df['window_start'].dt.strftime('%H:%M:%S')
            
            # Plota correlação ao longo do tempo com estilo aprimorado
            ax.plot(df['window_start'], df['correlation'], 'o-', 
                   markersize=5, linewidth=2.5, 
                   color=color_palette[idx % len(color_palette)],
                   markeredgecolor='black', markeredgewidth=0.5)
            
            # Adiciona linha horizontal em y=0 para referência
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, zorder=1)
            
            # Adiciona áreas sombreadas para destacar níveis de correlação
            ax.axhspan(0.7, 1.0, alpha=0.15, color='green', label='Correlação forte positiva')
            ax.axhspan(-1.0, -0.7, alpha=0.15, color='red', label='Correlação forte negativa')
            ax.axhspan(0.3, 0.7, alpha=0.1, color='lightgreen', label='Correlação moderada positiva')
            ax.axhspan(-0.7, -0.3, alpha=0.1, color='salmon', label='Correlação moderada negativa')
            
            # Melhora a formatação do gráfico com título mais informativo
            title = f'Correlação Deslizante entre {tenant1} e {tenant2}'
            subtitle = f'Métrica: {metric} | Fase: {phase} | Round: {round_id}'
            ax.set_title(title, fontweight='bold')
            fig.suptitle(subtitle, fontsize=12, y=0.97)
            
            # Formata eixos para melhor legibilidade
            ax.set_xlabel('Tempo (início da janela)', fontsize=12)
            ax.set_ylabel('Coeficiente de Correlação', fontsize=12)
            
            # Melhora formatação do eixo de tempo
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()  # Rotaciona labels para evitar sobreposição
            
            # Adiciona grid aprimorada para melhor legibilidade
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, color='#CCCCCC', alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.3, color='#DDDDDD', alpha=0.4)
            ax.minorticks_on()
            
            # Configura os limites do eixo Y para garantir que -1 a 1 estejam sempre visíveis
            ax.set_ylim(min(-1.1, df['correlation'].min() - 0.05), max(1.1, df['correlation'].max() + 0.05))
            
            # Adiciona estatísticas importantes no plot
            mean_corr = df['correlation'].mean()
            std_corr = df['correlation'].std()
            min_corr = df['correlation'].min()
            max_corr = df['correlation'].max()
            stats_text = (f'Estatísticas:\n'
                          f'Média: {mean_corr:.3f}\n'
                          f'Desvio: {std_corr:.3f}\n'
                          f'Min: {min_corr:.3f}\n'
                          f'Max: {max_corr:.3f}')
            
            # Adiciona caixa de texto com as estatísticas
            ax.text(0.02, 0.02, stats_text, transform=ax.transAxes,
                   bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.5', alpha=0.8),
                   fontsize=10, verticalalignment='bottom')
            
            # Adiciona legenda melhorada com posicionamento otimizado
            handles, labels = ax.get_legend_handles_labels()
            unique_labels = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            if unique_labels:
                ax.legend(*zip(*unique_labels), loc='upper right', framealpha=0.9, 
                        fancybox=True, shadow=True, borderpad=1)
            
            # Salva o plot com qualidade aprimorada
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, 
                f"sliding_corr_{tenant1}_vs_{tenant2}_{metric}_{phase}_{round_id}.png"
            )
            fig.tight_layout()
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            output_paths.append(out_path)
            
        # Se tiver muitos pares, gera um plot consolidado com os principais
        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for idx, ((tenant1, tenant2), df) in enumerate(results.items()):
                label = f"{tenant1} vs {tenant2}"
                color = color_palette[idx % len(color_palette)]
                ax.plot(df['window_start'], df['correlation'], 'o-', 
                       markersize=4, linewidth=2, alpha=0.8,
                       color=color, markeredgecolor='black', markeredgewidth=0.5,
                       label=label)
            
            # Adiciona faixas de correlação
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=1)
            ax.axhspan(0.7, 1.0, alpha=0.15, color='green', label='Corr. forte +')
            ax.axhspan(-1.0, -0.7, alpha=0.15, color='red', label='Corr. forte -')
            
            # Títulos e formatação de eixos
            title = f'Comparação de Correlações Deslizantes - Top Pares'
            subtitle = f'Métrica: {metric} | Fase: {phase} | Round: {round_id}'
            ax.set_title(title, fontweight='bold')
            fig.suptitle(subtitle, fontsize=12, y=0.97)
            
            ax.set_xlabel('Tempo (início da janela)', fontsize=12)
            ax.set_ylabel('Coeficiente de Correlação', fontsize=12)
            
            # Formatação de data/hora
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
            fig.autofmt_xdate()
            
            # Grid e estilo
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.set_ylim(-1.1, 1.1)
            
            # Legenda em posição otimizada com duas colunas para muitos itens
            ncol = 1 if len(results) < 6 else 2
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', ncol=ncol, 
                    frameon=True, fancybox=True, shadow=True)
            
            # Salva o plot consolidado
            out_path = os.path.join(
                out_dir, 
                f"sliding_corr_consolidated_{metric}_{phase}_{round_id}.png"
            )
            fig.tight_layout(rect=[0, 0, 0.85, 0.95])
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            plt.close()
            
            output_paths.append(out_path)
            
        return output_paths
        
    def plot_sliding_window_causality(
        self,
        results: Dict[Tuple[str, str], pd.DataFrame],
        metric: str,
        phase: str,
        round_id: str,
        out_dir: str,
        method: str = 'granger',
        top_n: int = None
    ) -> List[str]:
        """
        Plota a evolução da causalidade em janelas deslizantes para pares de tenants.
        
        Args:
            results: Resultado da função analyze_causality_sliding_window
            metric: Nome da métrica analisada
            phase: Fase experimental analisada
            round_id: ID do round analisado
            out_dir: Diretório para salvar os plots
            method: Método de causalidade ('granger' ou 'transfer_entropy')
            top_n: Se fornecido, plota apenas os top_n pares com maior causalidade média
            
        Returns:
            Lista de caminhos para os arquivos de imagem gerados
        """
        if not results:
            return []
            
        # Define qual coluna usar baseado no método
        if method == 'granger':
            value_col = 'causality_score'
            y_label = 'Escore de Causalidade (1-p_valor)'
            threshold = 0.95  # Equivalente a p < 0.05
            threshold_label = 'Limiar de significância (p < 0.05)'
        else:
            value_col = 'transfer_entropy'
            y_label = 'Transfer Entropy'
            threshold = 0.05  # Valor arbitrário para TE
            threshold_label = 'Limiar de referência'
            
        # Filtra top_n pares se solicitado
        if top_n is not None and len(results) > top_n:
            # Calcula causalidade média para cada par
            if method == 'granger':
                avg_values = {pair: df[value_col].mean() for pair, df in results.items() 
                             if value_col in df.columns}
            else:
                avg_values = {pair: df[value_col].mean() for pair, df in results.items() 
                             if value_col in df.columns}
                
            # Ordena pares por causalidade média decrescente
            sorted_pairs = sorted(avg_values.items(), key=lambda x: x[1], reverse=True)
            # Seleciona top_n pares
            top_pairs = [pair for pair, _ in sorted_pairs[:top_n]]
            results = {pair: df for pair, df in results.items() if pair in top_pairs}
        
        output_paths = []
        
        # Define cores para diferentes pares
        pair_colors = plt.cm.tab20(np.linspace(0, 1, max(10, len(results))))
        
        # Gera um plot para cada par
        for i, ((source, target), df) in enumerate(results.items()):
            if value_col not in df.columns:
                self.logger.warning(f"Coluna {value_col} não encontrada para par {source}->{target}")
                continue
                
            # Cria figura e eixos
            fig, ax = plt.subplots(figsize=(12, 6))
            color = pair_colors[i % len(pair_colors)]
            
            # Plota linha principal de causalidade
            ax.plot(df['window_start'], df[value_col], 'o-', 
                   markersize=5, linewidth=2.5, color=color,
                   markeredgecolor='black', markeredgewidth=0.5)
            
            # Destaca pontos de causalidade significativa
            mask = df[value_col] > threshold
            if mask.any():
                # Destaca pontos significativos
                ax.plot(df.loc[mask, 'window_start'], df.loc[mask, value_col], 
                       'ro', markersize=8, alpha=0.7, 
                       label='Causalidade significativa')
            
            # Adiciona linha horizontal para o limiar de significância
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=threshold_label)
            
            # Formata o eixo de tempo
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()  # Rotaciona labels de data
            
            # Títulos e formatação do gráfico
            title = f'Causalidade Deslizante: {source} → {target}'
            subtitle = f'Métrica: {metric} | Fase: {phase} | Round: {round_id}'
            ax.set_title(title, fontweight='bold')
            plt.suptitle(subtitle, fontsize=12)
            
            # Configurações dos eixos
            ax.set_xlabel('Tempo (início da janela)', fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            
            # Adiciona grid 
            ax.grid(True, which='major', linestyle='-', linewidth=0.5, alpha=0.7)
            ax.grid(True, which='minor', linestyle=':', linewidth=0.3, alpha=0.3)
            
            # Adiciona estatísticas no gráfico
            mean_val = df[value_col].mean()
            std_val = df[value_col].std()
            min_val = df[value_col].min()
            max_val = df[value_col].max()
            sig_pct = (df[value_col] > threshold).mean() * 100
            
            stats = f'Média: {mean_val:.3f}\nDesvio: {std_val:.3f}\n'
            stats += f'Min/Max: {min_val:.3f}/{max_val:.3f}\n'
            stats += f'% Significativo: {sig_pct:.1f}%'
            
            box_props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
            ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=10,
                   bbox=box_props, verticalalignment='bottom')
            
            # Legenda
            ax.legend(loc='best')
            
            # Salva o plot
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(
                out_dir, 
                f"sliding_caus_{method}_{source}_to_{target}_{metric}_{phase}_{round_id}.png"
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=120)
            plt.close(fig)
            
            output_paths.append(out_path)
            
        # Se tiver muitos pares, gera um plot consolidado com os principais
        if len(results) > 1:
            fig, ax = plt.subplots(figsize=(14, 8))
            
            for i, ((source, target), df) in enumerate(results.items()):
                if value_col not in df.columns:
                    continue
                label = f"{source} → {target}"
                ax.plot(df['window_start'], df[value_col], '-o', 
                       markersize=4, linewidth=2, alpha=0.8,
                       color=pair_colors[i % len(pair_colors)],
                       label=label)
            
            # Adiciona linha horizontal para o limiar de significância
            ax.axhline(y=threshold, color='red', linestyle='--', 
                      linewidth=1.5, alpha=0.7, label=threshold_label)
            
            # Formata o eixo de tempo
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
            plt.gcf().autofmt_xdate()
            
            # Títulos e formatação
            title = f'Comparativo de Causalidade Deslizante - Top Pares'
            subtitle = f'Métrica: {metric} | Fase: {phase} | Round: {round_id}'
            ax.set_title(title, fontweight='bold')
            plt.suptitle(subtitle, fontsize=12)
            
            # Configurações dos eixos
            ax.set_xlabel('Tempo (início da janela)', fontsize=12) 
            ax.set_ylabel(y_label, fontsize=12)
            ax.grid(True, alpha=0.3)
            
            # Legenda em coluna ao lado do gráfico
            ncol = 1 if len(results) < 6 else 2
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', 
                     ncol=ncol, frameon=True, framealpha=0.8)
            
            # Salva o plot consolidado
            out_path = os.path.join(
                out_dir, 
                f"sliding_caus_{method}_consolidated_{metric}_{phase}_{round_id}.png"
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=120, bbox_inches='tight')
            plt.close(fig)
            
            output_paths.append(out_path)
            
        return output_paths


class SlidingWindowStage:
    """
    Estágio do pipeline para análises de janelas deslizantes.
    """
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa análises em janelas deslizantes e gera visualizações.
        
        Args:
            context: Contexto do pipeline com os dados e configurações
            
        Returns:
            Contexto atualizado com os resultados das análises
        """
        logger = logging.getLogger("pipeline.sliding_window")
        logger.info("Iniciando análises com janelas deslizantes")
        
        # Extrai configurações do contexto
        df_long = context.get("df_long")
        if df_long is None or df_long.empty:
            logger.error("DataFrame não encontrado ou vazio no contexto")
            return context
            
        config = context.get("config", {})
        output_dir = config.get("output_dir", "outputs")
        metrics = config.get("selected_metrics")
        # Usa as métricas do DataFrame se não foram especificadas no config
        if not metrics:
            logger.info("Nenhuma métrica selecionada no config, usando todas as métricas disponíveis no DataFrame")
            metrics = df_long['metric_name'].unique().tolist()
        rounds = config.get("selected_rounds")
        # Usa os rounds do DataFrame se não foram especificados no config
        if not rounds:
            logger.info("Nenhum round selecionado no config, usando todos os rounds disponíveis no DataFrame")
            rounds = df_long['round_id'].unique().tolist()
        
        # Inicializa analisador
        analyzer = SlidingWindowAnalyzer(df_long)
        
        # Resultados para armazenar no contexto
        correlation_results = {}
        causality_results = {}
        
        # Para cada métrica e round
        for metric in metrics:
            for round_id in rounds:
                # Obtém todas as fases disponíveis para este round
                phases = df_long[(df_long['metric_name'] == metric) & 
                               (df_long['round_id'] == round_id)]['experimental_phase'].unique()
                
                for phase in phases:
                    logger.info(f"Analisando janelas deslizantes para {metric}, {phase}, {round_id}")
                    
                    # Análise de correlação deslizante
                    corr_results = analyzer.analyze_correlation_sliding_window(
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        window_size='5min',  # Parâmetros configuráveis
                        step_size='1min',
                        method='pearson',
                        min_periods=3
                    )
                    
                    if corr_results:
                        # Gera visualizações
                        out_dir = os.path.join(output_dir, "plots", "sliding_window", "correlation")
                        paths = analyzer.plot_sliding_window_correlation(
                            results=corr_results,
                            metric=metric,
                            phase=phase,
                            round_id=round_id,
                            out_dir=out_dir,
                            top_n=5  # Limita para 5 pares mais correlacionados
                        )
                        logger.info(f"Gerados {len(paths)} plots de correlação deslizante")
                        
                        # Salva resultados no dicionário
                        key = (metric, phase, round_id)
                        correlation_results[key] = {
                            'data': corr_results,
                            'plots': paths
                        }
                    
                    # Análise de causalidade deslizante (Granger)
                    causality_results_granger = analyzer.analyze_causality_sliding_window(
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        window_size='8min',  # Aumentado para capturar mais pontos
                        step_size='2min',    # Aumentado para melhorar a performance
                        method='granger',
                        max_lag=2,           # Reduzido para minimizar erros em séries curtas
                        min_periods=5        # Reduzido para permitir mais janelas válidas
                    )
                    
                    if causality_results_granger:
                        # Gera visualizações
                        out_dir = os.path.join(output_dir, "plots", "sliding_window", "causality", "granger")
                        paths = analyzer.plot_sliding_window_causality(
                            results=causality_results_granger,
                            metric=metric,
                            phase=phase,
                            round_id=round_id,
                            out_dir=out_dir,
                            method='granger',
                            top_n=5  # Limita para 5 pares mais causais
                        )
                        logger.info(f"Gerados {len(paths)} plots de causalidade Granger deslizante")
                        
                        # Salva resultados no dicionário
                        key = (metric, phase, round_id)
                        causality_results[key] = {
                            'method': 'granger',
                            'data': causality_results_granger,
                            'plots': paths
                        }
                    
                    # Análise de causalidade deslizante (Transfer Entropy)
                    causality_results_te = analyzer.analyze_causality_sliding_window(
                        metric=metric,
                        phase=phase,
                        round_id=round_id,
                        window_size='12min',  # Aumentado ainda mais para garantir pontos suficientes
                        step_size='4min',     # Passos maiores para melhor performance
                        method='transfer_entropy',
                        min_periods=8,        # Mantido em 8 pontos mínimos
                        bins=5                # Reduzido o número de bins para melhor estimação com menos pontos
                    )
                    
                    if causality_results_te:
                        # Gera visualizações
                        out_dir = os.path.join(output_dir, "plots", "sliding_window", "causality", "transfer_entropy")
                        paths = analyzer.plot_sliding_window_causality(
                            results=causality_results_te,
                            metric=metric,
                            phase=phase,
                            round_id=round_id,
                            out_dir=out_dir,
                            method='transfer_entropy',
                            top_n=5  # Limita para 5 pares mais causais
                        )
                        logger.info(f"Gerados {len(paths)} plots de Transfer Entropy deslizante")
                        
                        # Salva resultados no dicionário
                        key = (metric, phase, round_id, 'te')
                        causality_results[key] = {
                            'method': 'transfer_entropy',
                            'data': causality_results_te,
                            'plots': paths
                        }
        
        # Atualiza o contexto com os resultados
        context["sliding_window_correlation"] = correlation_results
        context["sliding_window_causality"] = causality_results
        
        logger.info("Análises com janelas deslizantes concluídas")
        return context
