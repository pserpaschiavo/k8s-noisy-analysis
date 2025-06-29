# Plano de Implementação do Análise Multi-Round

Este documento detalha a implementação técnica das funções necessárias para completar o plano de trabalho definido em `work-plan-20250629.md`.

## 1. Extração de Tamanhos de Efeito (`extract_effect_sizes`)

### Assinatura da função
```python
def extract_effect_sizes(
    df_long: pd.DataFrame,
    rounds: List[str],
    metrics: List[str],
    phases: List[str],
    tenants: List[str],
    baseline_phase: str = "1 - Baseline",
    use_cache: bool = True,
    parallel: bool = False
) -> pd.DataFrame:
    """
    Extrai estatísticas de tamanho de efeito (Cohen's d) e p-valores para
    comparações de fase vs. baseline para cada métrica, tenant e round.
    
    Args:
        df_long: DataFrame em formato longo com todos os dados
        rounds: Lista de rounds para análise
        metrics: Lista de métricas para análise
        phases: Lista de fases para análise
        tenants: Lista de tenants para análise
        baseline_phase: Nome da fase de baseline (default: "1 - Baseline")
        use_cache: Se True, usa cache para evitar recálculos
        parallel: Se True, paraleliza o processamento
        
    Returns:
        DataFrame com colunas: round_id, metric_name, experimental_phase, 
        tenant_id, baseline_phase, effect_size, p_value, eta_squared
    """
```

### Algoritmo
1. Inicializar uma lista vazia para armazenar resultados
2. Para cada combinação de round × métrica × fase × tenant:
   - Filtrar os dados da fase experimental e da fase baseline
   - Verificar se há dados suficientes para cálculos estatísticos
   - Calcular Cohen's d e p-valor usando t-test independente
   - Calcular Eta-squared se solicitado
   - Adicionar resultado à lista com todas as informações necessárias
3. Converter a lista de resultados em um DataFrame
4. Adicionar informações de qualidade (pontuação de confiança)
5. Retornar o DataFrame de resultados

### Implementação de Cohen's d
Para calcular o Cohen's d, usaremos a seguinte fórmula:
```python
def cohens_d(group1, group2):
    """
    Calcula o tamanho de efeito (Cohen's d) entre dois grupos.
    
    Args:
        group1: Array-like com valores do grupo 1
        group2: Array-like com valores do grupo 2
        
    Returns:
        float: Tamanho de efeito (Cohen's d)
    """
    # Calcula médias e desvios padrão para ambos os grupos
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    # Pooled standard deviation
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    # Cohen's d
    d = (mean2 - mean1) / pooled_std
    return d
```

### Implementação de Eta-squared
```python
def eta_squared(group1, group2):
    """
    Calcula o tamanho de efeito Eta-squared entre dois grupos.
    
    Args:
        group1: Array-like com valores do grupo 1
        group2: Array-like com valores do grupo 2
        
    Returns:
        float: Tamanho de efeito (Eta-squared)
    """
    # Combina os grupos em um único array
    all_data = np.concatenate([group1, group2])
    labels = np.concatenate([np.zeros(len(group1)), np.ones(len(group2))])
    
    # Calcula a soma dos quadrados total
    ss_total = np.sum((all_data - np.mean(all_data))**2)
    
    # Calcula a soma dos quadrados entre grupos
    mean1, mean2 = np.mean(group1), np.mean(group2)
    ss_between = len(group1) * (mean1 - np.mean(all_data))**2 + len(group2) * (mean2 - np.mean(all_data))**2
    
    # Eta-squared
    eta_sq = ss_between / ss_total
    return eta_sq
```

## 2. Extração de Correlações Intra-Fase (`extract_phase_correlations`)

### Assinatura da função
```python
def extract_phase_correlations(
    df_long: pd.DataFrame,
    rounds: List[str],
    metrics: List[str],
    phases: List[str],
    tenants: Optional[List[str]] = None,
    method: str = 'pearson',
    min_periods: int = 3
) -> pd.DataFrame:
    """
    Extrai as correlações intra-fase entre tenants para cada métrica × fase × round.
    
    Args:
        df_long: DataFrame em formato longo com todos os dados
        rounds: Lista de rounds para análise
        metrics: Lista de métricas para análise
        phases: Lista de fases para análise
        tenants: Lista de tenants para análise (opcional)
        method: Método de correlação ('pearson', 'spearman', 'kendall')
        min_periods: Número mínimo de períodos para calcular correlação
        
    Returns:
        DataFrame com colunas: round_id, metric_name, experimental_phase, 
        tenant_pair, correlation
    """
```

## 3. Agregação de Tamanhos de Efeito (`aggregate_effect_sizes`)

### Assinatura da função
```python
def aggregate_effect_sizes(
    effect_sizes_df: pd.DataFrame,
    alpha: float = 0.05,
    p_value_method: str = 'fisher',
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Agrega os tamanhos de efeito através dos rounds, calculando médias,
    desvios padrão, intervalos de confiança e p-valores combinados.
    
    Args:
        effect_sizes_df: DataFrame com tamanhos de efeito por round × métrica × fase × tenant
        alpha: Nível de significância (default: 0.05)
        p_value_method: Método para combinar p-valores ('fisher', 'stouffer')
        confidence_level: Nível de confiança para IC (default: 0.95)
        
    Returns:
        DataFrame com estatísticas agregadas
    """
```

### Algoritmo
1. Agrupar o DataFrame por métrica × fase × tenant
2. Para cada grupo:
   - Calcular média e desvio padrão do tamanho de efeito
   - Calcular IC95% usando bootstrapping ou estatística t
   - Combinar p-valores usando o método especificado
   - Calcular coeficiente de variação e outras métricas de estabilidade
3. Adicionar métricas de qualidade e confiança
4. Retornar o DataFrame agregado

### Implementação do método de Fisher para combinar p-valores
```python
def combine_pvalues_fisher(p_values):
    """
    Combina múltiplos p-valores usando o método de Fisher.
    
    Args:
        p_values: Lista de p-valores a combinar
        
    Returns:
        float: p-valor combinado
    """
    # Substituir valores zero ou nan para evitar problemas
    p_values = np.array(p_values)
    p_values = np.clip(p_values, 1e-10, 1.0)
    
    # Estatística de teste de Fisher
    chi_square = -2 * np.sum(np.log(p_values))
    
    # Graus de liberdade = 2 * número de p-valores
    df = 2 * len(p_values)
    
    # p-valor combinado
    combined_p = 1.0 - stats.chi2.cdf(chi_square, df)
    return combined_p
```

### Implementação do método de Stouffer para combinar p-valores
```python
def combine_pvalues_stouffer(p_values, weights=None):
    """
    Combina múltiplos p-valores usando o método de Stouffer.
    
    Args:
        p_values: Lista de p-valores a combinar
        weights: Pesos para cada p-valor (opcional)
        
    Returns:
        float: p-valor combinado
    """
    # Converter p-valores para z-scores
    z_scores = stats.norm.ppf(1 - np.array(p_values))
    
    if weights is None:
        weights = np.ones_like(z_scores)
    weights = np.array(weights)
    
    # Z-score combinado
    z_combined = np.sum(weights * z_scores) / np.sqrt(np.sum(weights**2))
    
    # p-valor combinado
    combined_p = 1 - stats.norm.cdf(z_combined)
    return combined_p
```

## 4. Visualização de Tamanhos de Efeito (`generate_effect_size_heatmap`)

### Assinatura da função
```python
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
```

## 5. Análise de Robustez (`perform_robustness_analysis`)

### Assinatura da função
```python
def perform_robustness_analysis(
    effect_sizes_df: pd.DataFrame,
    aggregated_effects_df: pd.DataFrame,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Realiza análise de robustez usando leave-one-out e variando o limiar de significância.
    
    Args:
        effect_sizes_df: DataFrame original com tamanhos de efeito
        aggregated_effects_df: DataFrame com tamanhos de efeito agregados
        alpha: Nível de significância padrão
        
    Returns:
        Dict: Resultados da análise de robustez
    """
```

## 6. Atualização do Arquivo de Configuração YAML

Adicionar ao arquivo `config/pipeline_config_sfi2.yaml`:

```yaml
# Configuração para análise multi-round
multi_round_analysis:
  effect_size:
    methods: ["cohen_d", "eta_squared"]
    baseline_phase: "1 - Baseline"
  meta_analysis:
    p_value_combination: "fisher"  # ou "stouffer"
    confidence_level: 0.95
    alpha: 0.05
  performance:
    use_cache: true
    parallel_processing: false
  visualization:
    heatmap_colormap: "RdBu_r"
    show_significance_markers: true
    effect_size_thresholds:
      small: 0.2
      medium: 0.5
      large: 0.8
```
