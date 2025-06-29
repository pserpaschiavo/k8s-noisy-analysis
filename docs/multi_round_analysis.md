# Análise Multi-Round

Este README documenta as funcionalidades e a implementação da análise de múltiplos rounds para o projeto k8s-noisy-analysis.

## Visão Geral

A análise multi-round consolida os resultados estatísticos obtidos de múltiplos rounds em uma análise robusta por métrica × fase × tenant, permitindo generalização e visualização dos efeitos com confiança.

## Funcionalidades Principais

### 1. Extração de Tamanhos de Efeito

- Calcula o Cohen's d e Eta-squared para comparações entre fases experimentais e baseline
- Realiza testes estatísticos (t-test) e retorna p-valores
- Suporta cache para evitar recálculos
- Paralelização opcional para maior desempenho

### 2. Agregação Estatística

- Calcula média, desvio padrão e intervalo de confiança (IC95%) dos tamanhos de efeito
- Combina p-valores de múltiplos rounds usando os métodos de Fisher ou Stouffer
- Calcula métricas de estabilidade (coeficiente de variação)

### 3. Visualização Consolidada

- Gera heatmaps dos tamanhos de efeito médios
- Cria boxplots de variabilidade por round
- Produz gráficos de error bars com IC95%
- Disponibiliza scatter plots para relacionar efeito × p-valor × variabilidade

### 4. Análise de Robustez

- Implementa análise leave-one-out para detectar dependência de rounds específicos
- Calcula a estabilidade das conclusões com diferentes limiares de significância
- Quantifica a incerteza usando métodos bayesianos ou bootstrap

## Uso

Para executar a análise multi-round:

```bash
python run_multi_round_analysis.py --config config/pipeline_config_sfi2.yaml
```

## Configuração

A análise multi-round pode ser configurada no arquivo YAML de configuração:

```yaml
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

## Interpretação dos Resultados

### Tamanho de Efeito (Cohen's d)

- 0.2 - 0.5: Efeito pequeno
- 0.5 - 0.8: Efeito médio
- > 0.8: Efeito grande

### Eta-squared

- 0.01 - 0.06: Efeito pequeno
- 0.06 - 0.14: Efeito médio
- > 0.14: Efeito grande

### P-valor Combinado

Um p-valor combinado < 0.05 indica que o efeito observado é estatisticamente significativo considerando todos os rounds analisados.

## Saídas

A análise multi-round gera os seguintes artefatos:

- Arquivo CSV com todos os tamanhos de efeito calculados
- Gráficos de heatmap para visualização dos efeitos
- Gráficos de boxplot e error bars
- Relatório consolidado em formato Markdown
