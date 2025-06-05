# Fórmulas e Algoritmos de Referência para Cálculo de Impacto

Este documento serve como uma referência técnica com as principais fórmulas e procedimentos utilizados para calcular os diferentes componentes do score de impacto em ambientes Kubernetes multi-tenant.

## Causalidade de Granger

### Procedimento Passo a Passo

1. **Teste de Estacionariedade (ADF)**:
   ```
   H0: A série temporal contém uma raiz unitária (não estacionária)
   H1: A série temporal é estacionária
   ```
   Se p-valor > 0.05, aplicamos diferenciação para tornar a série estacionária.

2. **Teste de Causalidade**:
   
   Comparação entre:
   
   **Modelo Restrito**: 
   ```
   Y(t) = α₀ + α₁Y(t-1) + ... + α₌Y(t-p) + ε(t)
   ```
   
   **Modelo Irrestrito**:
   ```
   Y(t) = α₀ + α₁Y(t-1) + ... + α₌Y(t-p) + β₁X(t-1) + ... + β₌X(t-p) + ε(t)
   ```

3. **Teste F**:
   ```
   F = ((RSS_r - RSS_ur)/m) / (RSS_ur/(n-k))
   ```
   onde:
   - RSS_r: Soma de quadrados residuais do modelo restrito
   - RSS_ur: Soma de quadrados residuais do modelo irrestrito
   - m: Número de restrições (parâmetros adicionais no modelo irrestrito)
   - n: Número de observações
   - k: Número total de parâmetros no modelo irrestrito

4. **Conversão para Score de Causalidade**:
   ```
   score_granger = 1 - p_valor
   ```

5. **Implementação com Statsmodels**:
   ```python
   from statsmodels.tsa.stattools import grangercausalitytests
   result = grangercausalitytests(data, maxlag=5, verbose=False)
   p_values = [result[lag][0]['ssr_chi2test'][1] for lag in range(1, 6)]
   min_p_value = min(p_values) if p_values else np.nan
   score_granger = 1 - min_p_value
   ```

## Transfer Entropy

### Fórmula Matemática

A Transfer Entropy de X para Y é definida como:

```
TE(X→Y) = H(Y₍₊₁|Y₍) - H(Y₍₊₁|Y₍,X₍)
```

Onde:
- H(Y₍₊₁|Y₍) é a entropia condicional de Y₍₊₁ dado Y₍
- H(Y₍₊₁|Y₍,X₍) é a entropia condicional de Y₍₊₁ dado Y₍ e X₍

### Cálculo da Entropia Condicional

```
H(Y|X) = -∑∑ p(x,y) log(p(y|x))
```

### Procedimento para Cálculo da TE

1. **Discretização**:
   ```python
   def discretize(series, bins=8):
       return np.digitize(series, 
                          np.linspace(np.min(series), 
                                      np.max(series), bins))
   ```

2. **Cálculo da TE**:
   ```python
   def transfer_entropy(source, target, k=1):
       from pyinform.transferentropy import transfer_entropy
       source_disc = discretize(source)
       target_disc = discretize(target)
       te = transfer_entropy(source_disc, target_disc, k=k)
       return te
   ```

3. **Conversão para Score**:
   ```python
   # Aplicamos peso maior à TE por capturar relações não-lineares
   score_te = te_value * 5
   ```

## Correlação

### Correlação de Pearson

```
r = Σ[(X_i - X̄)(Y_i - Ȳ)] / √[Σ(X_i - X̄)² Σ(Y_i - Ȳ)²]
```

Onde:
- X̄ é a média de X
- Ȳ é a média de Y

### Cross-Correlation Function (CCF)

Para cada lag k:

```
CCF(X,Y,k) = Σ[(X_{t} - X̄)(Y_{t+k} - Ȳ)] / √[Σ(X_t - X̄)² Σ(Y_t - Ȳ)²]
```

Para k > 0, correlação entre X e valores futuros de Y
Para k < 0, correlação entre Y e valores futuros de X

### Normalização para CCF

```python
# Normalizar séries (Z-score)
x_norm = (x - np.mean(x)) / np.std(x)
y_norm = (y - np.mean(y)) / np.std(y)

# Calcular CCF para lags positivos (X → Y)
ccf_pos = []
for lag in range(max_lag+1):
    if lag == 0:
        # Para lag=0, correlação contemporânea
        corr = np.corrcoef(x_norm, y_norm)[0, 1]
    else:
        # Para lag > 0, correlação com deslocamento
        corr = np.corrcoef(x_norm[:-lag], y_norm[lag:])[0, 1]
    ccf_pos.append(corr)
```

## Variação entre Fases

### Variação Percentual

```
variação_pct = ((valor_ataque - valor_baseline) / valor_baseline) * 100
```

### Tratamento de Valores Extremos

```python
# Limitação para evitar valores extremos devido a divisão por valores muito pequenos
if abs(baseline_value) < 0.001:
    baseline_value = 0.001 * np.sign(baseline_value)
    
# Para baseline zero, usamos diferença absoluta
if baseline_value == 0:
    variation = attack_value
else:
    variation = ((attack_value - baseline_value) / baseline_value) * 100
```

## Score Final Composto

### Normalização

```python
# Normalizar cada métrica pela quantidade de ocorrências
for tenant, metrics in tenant_metrics.items():
    count = max(metrics['metrics_count'], 1)  # Evitar divisão por zero
    
    metrics['causality_impact_score'] /= count
    metrics['correlation_strength'] /= count
    metrics['phase_variation'] /= count
```

### Cálculo Ponderado

```python
noisy_score = (
    causality_impact_score * 0.5 +  # 50% para causalidade
    correlation_strength * 0.3 +    # 30% para correlação
    phase_variation * 0.2           # 20% para variação entre fases
)
```

## Métricas de Robustez

### Robustez de Causalidade

```
R = N_significativo / N_total
```

Onde:
- N_significativo: Número de rounds onde a relação causal foi estatisticamente significativa
- N_total: Número total de rounds analisados

### Coeficiente de Variação

```
CV = (desvio_padrão / média) * 100
```

### Registro de Causalidade

Formato: "(robustez) (contagem/total) [média TE]"

Exemplo: "0.67 (2/3) [0.123]" indica que a relação causal foi significativa em 2 de 3 rounds, com robustez 0.67 e valor médio de TE de 0.123.

## Classificação de Tenant Barulhento

```python
# Ordenar tenants por score de impacto
sorted_metrics = tenant_metrics.sort_values(by='noisy_score', ascending=False)

# Classificar top 25% como "barulhento"
for idx, row in sorted_metrics.iterrows():
    tenant = row['tenant_id']
    if idx < len(sorted_metrics) // 4:
        insights[tenant]['is_noisy_tenant'] = True
```
