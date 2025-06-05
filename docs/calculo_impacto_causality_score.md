# Cálculos de Impacto em Ambiente Kubernetes Multi-Tenant

Este documento detalha os algoritmos e fórmulas utilizados para calcular os diferentes scores de impacto no sistema de análise de séries temporais multi-tenant. Essas métricas são fundamentais para a identificação e classificação de "tenants barulhentos" e para entender as relações de influência entre diferentes workloads em um ambiente Kubernetes.

## 1. Visão Geral dos Scores Compostos

O sistema utiliza uma combinação ponderada de três dimensões principais para calcular o impacto total de um tenant no ambiente:

```
Noisy Score = (Impacto Causal × 0.5) + (Força de Correlação × 0.3) + (Variação entre Fases × 0.2)
```

Onde:
- **Impacto Causal (50%)**: Avalia o quanto um tenant causa alterações em outros
- **Força de Correlação (30%)**: Mede a força das relações lineares entre tenants
- **Variação entre Fases (20%)**: Quantifica mudanças de comportamento durante fases experimentais

## 2. Cálculo do Impacto Causal

O impacto causal é calculado combinando duas métricas complementares de causalidade: Causalidade de Granger e Transfer Entropy.

### 2.1 Causalidade de Granger

A causalidade de Granger testa se os valores passados de uma série temporal X ajudam a prever os valores futuros de uma série Y, além do que se pode prever usando apenas os valores passados de Y.

**Procedimento de cálculo:**

1. Para cada par de tenants (A, B) e cada métrica relevante:
   
   a. Extrai-se as séries temporais dos dois tenants
   b. Verifica-se estacionariedade (teste ADF) e aplica-se diferenciação se necessário
   c. Compara-se dois modelos:
      - Modelo restrito: Y(t) = f(Y(t-1), Y(t-2), ..., Y(t-p))
      - Modelo irrestrito: Y(t) = f(Y(t-1), ..., Y(t-p), X(t-1), ..., X(t-p))
   d. Calcula-se o p-valor do teste F para comparação dos modelos
   e. Converte-se para score de impacto: `impacto = 1 - p_valor`
   f. Armazena-se o menor p-valor encontrado considerando diferentes lags

**Implementação no código:**
```python
# Testa causalidade de Granger para cada lag
from statsmodels.tsa.stattools import grangercausalitytests
test_results = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                
# Obtém o menor p-valor entre todos os lags testados
p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
min_p_value = min(p_values) if p_values else np.nan
```

### 2.2 Transfer Entropy (TE)

A Transfer Entropy mede a quantidade de informação que flui de uma série temporal para outra, quantificando a redução da incerteza sobre os valores futuros de uma série quando conhecemos valores passados de outra série.

**Procedimento de cálculo:**

1. Para cada par de tenants (A, B) e cada métrica:
   
   a. Extrai-se as séries temporais alinhadas temporalmente
   b. Discretiza-se os valores das séries contínuas em bins (padrão: 8 bins)
   c. Calcula-se a Transfer Entropy:
      ```
      TE(X→Y) = H(Yt+1|Yt) - H(Yt+1|Yt,Xt)
      ```
      onde H representa a entropia condicional
   d. Valores mais altos de TE indicam maior transferência de informação (maior causalidade)

**Implementação no código:**
```python
def _transfer_entropy(target_series, source_series, bins=8, k=1):
    # Discretização dos dados
    source_disc = np.digitize(source_series, 
                            np.linspace(np.min(source_series), 
                                        np.max(source_series), bins))
    target_disc = np.digitize(target_series, 
                            np.linspace(np.min(target_series), 
                                        np.max(target_series), bins))
    
    # Calcula TE usando pyinform
    from pyinform.transferentropy import transfer_entropy
    te_value = transfer_entropy(source_disc, target_disc, k=k)
    return te_value
```

### 2.3 Combinação das Métricas de Causalidade

O score de impacto causal final é calculado como uma combinação ponderada:

```python
# No cálculo das métricas finais
tenant_metrics[source]['causality_impact_score'] += np.mean(causal_values)  # Para Granger
tenant_metrics[source]['causality_impact_score'] += np.mean(te_values) * 5  # Peso maior para TE
```

A Transfer Entropy recebe um peso maior (5x) por ser uma medida mais robusta que captura relações não-lineares.

## 3. Cálculo da Força de Correlação

A força de correlação mede a intensidade das relações lineares entre pares de tenants, independentemente da direção de influência.

**Procedimento de cálculo:**

1. Para cada par de tenants (A, B) e cada métrica:
   
   a. Calcula-se a correlação de Pearson entre as séries temporais
   b. Utiliza-se o valor absoluto da correlação (ignorando o sinal)
   c. Considera-se apenas correlações significativas (geralmente |corr| > 0.2)

**Implementação no código:**
```python
# Valores absolutos de correlação (ignorando autocorrelação)
corr_values = matrix.loc[tenant].abs().values
corr_values = [v for v in corr_values if not np.isnan(v) and v < 1.0]

if corr_values:
    tenant_metrics[tenant]['correlation_strength'] += np.mean(corr_values)
    tenant_metrics[tenant]['metrics_count'] += 1
```

### 3.1 Cross-Correlation Function (CCF)

A CCF estende a análise de correlação para incluir defasagens temporais, permitindo identificar padrões onde um tenant influencia outro com atraso.

**Procedimento de cálculo:**

1. Para cada par de tenants (A, B):
   
   a. Normaliza-se as séries temporais (z-score)
   b. Para cada lag de -max_lag a +max_lag:
      - Desloca-se uma série em relação à outra
      - Calcula-se a correlação entre as partes sobrepostas
   c. Identifica-se o lag com maior correlação absoluta
   d. Determina-se a direção de influência com base no sinal do lag

**Implementação no código:**
```python
# Normalizar dados (importante para CCF)
ts1_norm = (ts1 - ts1.mean()) / ts1.std()
ts2_norm = (ts2 - ts2.mean()) / ts2.std()

# Calcular CCF para lags positivos (tenant1 -> tenant2)
for lag in range(max_lag+1):
    if lag == 0:
        # Para lag=0, correlação é simétrica
        corr = np.corrcoef(ts1_norm, ts2_norm)[0, 1]
    else:
        # Para lag > 0, calcular correlação com deslocamento
        corr = np.corrcoef(ts1_norm[:-lag], ts2_norm[lag:])[0, 1]
```

## 4. Cálculo da Variação entre Fases

A variação entre fases quantifica o quanto o comportamento de um tenant se altera entre diferentes fases experimentais (baseline, ataque, recuperação).

**Procedimento de cálculo:**

1. Para cada tenant e cada métrica:
   
   a. Calcula-se as estatísticas descritivas para cada fase
   b. Determina-se a variação percentual entre fases:
      ```
      variação_pct = ((valor_ataque - valor_baseline) / valor_baseline) * 100
      ```
   c. Valores absolutos maiores indicam tenants mais sensíveis a mudanças no ambiente

**Implementação no código:**
```python
# Analisar variações entre fases
for key, phase_df in phase_comparison_results.items():
    if phase_df.empty:
        continue
        
    for _, row in phase_df.iterrows():
        tenant = row['tenant_id']
        if tenant not in tenant_metrics:
            continue
            
        # Extrair variação percentual entre fases (em valor absoluto)
        if 'variation_pct' in row and not pd.isna(row['variation_pct']):
            tenant_metrics[tenant]['phase_variation'] += abs(row['variation_pct'])
            tenant_metrics[tenant]['metrics_count'] += 1
```

## 5. Normalização e Cálculo do Score Final

Após calcular as métricas individuais, o sistema normaliza os valores e aplica a ponderação para obter o score final.

**Procedimento de cálculo:**

1. Para cada tenant:
   
   a. Normaliza-se cada métrica pelo número de ocorrências
   b. Aplica-se os pesos para cada dimensão
   c. Calcula-se o score final

**Implementação no código:**
```python
# Normaliza e calcula score final
for tenant, metrics in tenant_metrics.items():
    # Evita divisão por zero
    count = max(metrics['metrics_count'], 1)
    
    # Normalização
    metrics['causality_impact_score'] /= count
    metrics['correlation_strength'] /= count
    metrics['phase_variation'] /= count
    
    # Calcula o score final (ponderado)
    metrics['noisy_score'] = (
        metrics['causality_impact_score'] * 0.5 +  # 50% para causalidade
        metrics['correlation_strength'] * 0.3 +    # 30% para correlação
        metrics['phase_variation'] * 0.2           # 20% para variação entre fases
    )
```

## 6. Classificação e Identificação de Tenant Barulhento

Após calcular os scores finais para todos os tenants, o sistema:

1. Ordena os tenants por score de impacto (do maior para o menor)
2. Classifica como "tenant barulhento" aqueles no quartil superior (top 25%)
3. Gera insights sobre relações de impacto, incluindo os principais tenants afetados

**Classificação no código:**
```python
# Preencher informações básicas de ranking e score
sorted_metrics = tenant_metrics.sort_values(by='noisy_score', ascending=False).reset_index(drop=True)
for idx, row in sorted_metrics.iterrows():
    tenant = row['tenant_id']
    insights[tenant]['rank'] = idx + 1
    insights[tenant]['noisy_score'] = row['noisy_score']
    
    # Determinar se é um "tenant barulhento" (top 25%)
    if idx < len(sorted_metrics) // 4:
        insights[tenant]['is_noisy_tenant'] = True
```

## 7. Análise de Casos Específicos em Multi-Round

Para experimentos com múltiplos rounds, o sistema implementa métodos adicionais para garantir consistência e robustez nas conclusões:

### 7.1 Métrica de Robustez para Causalidade

```
R = N_significativo / N_total
```

Onde:
- R > 0.75: Relação causal altamente robusta
- 0.5 ≤ R ≤ 0.75: Relação causal moderadamente robusta
- 0.25 ≤ R < 0.5: Relação causal fracamente robusta
- R < 0.25: Relação causal não robusta (possivelmente espúria)

### 7.2 Coeficiente de Variação (CV)

```
CV = (desvio_padrão / média) * 100
```

Permite classificar a consistência de métricas entre rounds:
- CV < 15%: Alta consistência
- CV entre 15% e 30%: Consistência média
- CV > 30%: Baixa consistência

## 8. Interpretação dos Resultados

Os scores gerados permitem diferentes níveis de interpretação:

### 8.1 Nível Individual

- **Score de Impacto Alto**: Indica tenant com forte influência sobre outros tenants
- **Score de Correlação Alto**: Sugere forte acoplamento com outros tenants
- **Variação entre Fases Alta**: Indica sensibilidade a mudanças no ambiente

### 8.2 Nível de Pares

- **TE(A→B) > TE(B→A)**: Sugere que A tem maior influência causal sobre B
- **CCF com Lag Positivo**: Indica que mudanças em A precedem mudanças em B
- **Correlação Alta com Variação de Fase Baixa**: Sugere acoplamento estrutural não circunstancial

### 8.3 Nível Sistêmico

- **Tenant Barulhento**: Aquele com maior score de impacto, tipicamente no quartil superior
- **Vítimas Potenciais**: Tenants com alta variação entre fases e que são influenciados pelos "barulhentos"
- **Clusters de Influência**: Grupos de tenants com forte interdependência causal e correlacional

## 9. Limitações e Considerações

É importante reconhecer algumas limitações dessas métricas:

1. **Causalidade Estatística vs. Física**: A causalidade inferida é estatística, não necessariamente causal no sentido físico.

2. **Limitações do Teste de Granger**:
   - Assume relações lineares
   - Sensível à estacionariedade das séries
   - Pode falhar ao detectar algumas relações causais complexas

3. **Desafios da Transfer Entropy**:
   - Requer séries temporais relativamente longas
   - A discretização pode influenciar os resultados
   - Computacionalmente mais intensiva

4. **Contextualização Necessária**:
   - Os scores devem ser interpretados no contexto do experimento específico
   - A identificação de "tenant barulhento" depende dos critérios e limiares escolhidos
   - Recomenda-se validação cruzada com conhecimento do domínio

## 10. Conclusão

O sistema de análise multi-tenant implementa um conjunto abrangente de métricas para identificação de relações de influência entre tenants em um ambiente Kubernetes. A combinação de diferentes perspectivas (causalidade, correlação e variação entre fases) permite uma compreensão robusta e multidimensional das dinâmicas entre workloads, fundamental para estratégias eficazes de isolamento de recursos e mitigação de problemas de vizinhança barulhenta.

Estas métricas constituem uma base quantitativa sólida para decisões de engenharia relacionadas ao isolamento de tenants, alocação de recursos e otimização de desempenho em ambientes multi-tenant.
