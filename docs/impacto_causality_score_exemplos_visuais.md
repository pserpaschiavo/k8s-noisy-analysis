# Visualização do Impacto em Ambiente Kubernetes Multi-Tenant
*Material Complementar para Seminário*

Este documento apresenta visualizações e exemplos para ilustrar como os diferentes componentes do score de impacto funcionam na prática, complementando o material didático principal.

## Exemplos Visuais de Causalidade

### Exemplo 1: Detecção de Causalidade via Teste de Granger

Considerando duas séries temporais de utilização de CPU por diferentes tenants:

```
Tenant A (possível causador): [60%, 75%, 85%, 78%, 65%, 55%, 70%, 80%]
                               ↓    ↓    ↓    ↓    ↓    ↓    ↓
Tenant B (possível afetado):   [40%, 50%, 65%, 70%, 72%, 60%, 55%, 65%]
```

Ao aplicar o teste de causalidade de Granger:

1. **Modelo Restrito**: Previsão de B baseada apenas em valores anteriores de B
   - Erro médio quadrático: 85.2

2. **Modelo Irrestrito**: Previsão de B baseada em valores anteriores de B e A
   - Erro médio quadrático: 45.7

3. **Resultado do Teste F**: p-valor = 0.03 (significativo em nível 5%)

4. **Interpretação**: Com 97% de confiança (1 - p-valor), podemos dizer que a atividade do Tenant A influencia causalmente o Tenant B.

### Exemplo 2: Transfer Entropy em Ação

Consideremos dois tenants com padrões de uso de memória ao longo do tempo:

```
Tenant C (discretizado): [1, 2, 4, 6, 7, 6, 5, 4, 3, 5, 7]
Tenant D (discretizado): [2, 2, 3, 5, 6, 7, 6, 5, 4, 4, 6]
```

A Transfer Entropy calcula:

1. **Incerteza sobre D usando apenas D**: H(D_future | D_past) = 1.2 bits
2. **Incerteza sobre D usando C e D**: H(D_future | D_past, C_past) = 0.7 bits
3. **TE(C→D)** = 1.2 - 0.7 = 0.5 bits

4. **TE(D→C)** = 0.2 bits (calculado inversamente)

5. **Interpretação**: Como TE(C→D) > TE(D→C), há maior fluxo de informação de C para D do que vice-versa, sugerindo que C tem maior influência causal sobre D.

## Visualização de Correlação

### Exemplo: Heatmap de Correlação entre Tenants

Matriz de correlação para utilização de CPU entre 4 tenants:

```
       | Tenant A | Tenant B | Tenant C | Tenant D |
-------|----------|----------|----------|----------|
Tenant A|   1.00   |   0.32   |   0.78   |  -0.15   |
Tenant B|   0.32   |   1.00   |   0.23   |   0.65   |
Tenant C|   0.78   |   0.23   |   1.00   |  -0.08   |
Tenant D|  -0.15   |   0.65   |  -0.08   |   1.00   |
```

**Interpretação do heatmap:**
- **Correlação forte positiva** (A-C: 0.78): Quando A aumenta, C tende a aumentar
- **Correlação moderada positiva** (B-D: 0.65): Quando B aumenta, D tende a aumentar
- **Correlação fraca negativa** (A-D: -0.15): Leve tendência oposta
- **Correlação diagonal = 1.00**: Auto-correlação (sempre 1)

### Cross-Correlation Function (CCF) Visual

Correlação cruzada entre Tenants E e F ao longo do tempo:

```
Lag   CCF Valor
-5     0.12
-4     0.18
-3     0.25
-2     0.42
-1     0.56
 0     0.65
+1     0.82  ← Máximo (Tenant E influencia Tenant F com lag +1)
+2     0.67
+3     0.40
+4     0.22
+5     0.10
```

**Interpretação:**
- O maior valor (0.82) ocorre no lag +1
- Isso sugere que mudanças no Tenant E precedem mudanças similares no Tenant F por 1 período
- A correlação contemporânea (lag 0) também é forte (0.65)

## Exemplo de Variação entre Fases

### Comportamento de um Tenant ao Longo de Três Fases

Dados de latência média (ms) para o Tenant G:

```
Fase 1 (Baseline): Média = 20ms, Desvio padrão = 2.5ms
Fase 2 (Ataque):   Média = 52ms, Desvio padrão = 12.8ms
Fase 3 (Recuperação): Média = 28ms, Desvio padrão = 4.2ms
```

**Cálculo da variação percentual:**
```
Variação (Baseline → Ataque) = ((52 - 20) / 20) * 100 = 160%
Variação (Ataque → Recuperação) = ((28 - 52) / 52) * 100 = -46.15%
```

**Interpretação:**
- O Tenant G sofreu um aumento dramático de 160% na latência durante a fase de ataque
- A recuperação foi parcial, com latência ainda 40% acima da linha de base
- Este perfil é típico de um tenant vítima que foi significativamente impactado por outro tenant barulhento

## Exemplo Completo de Cálculo do Score

Vamos calcular o score completo para o Tenant X:

**Dados coletados:**
1. **Impacto Causal**:
   - Granger: 1 - p-valor médio = 1 - 0.15 = 0.85
   - TE média = 0.42 (multiplicado por 5 = 2.10)
   - Média combinada = (0.85 + 2.10) / 2 = 1.475

2. **Força de Correlação**:
   - Correlação média absoluta = 0.65

3. **Variação entre Fases**:
   - Variação percentual média = 120%

**Aplicando a fórmula ponderada:**
```
Noisy Score = (1.475 × 0.5) + (0.65 × 0.3) + (120 × 0.2)
            = 0.7375 + 0.195 + 24
            = 24.9325
```

**Interpretação:**
- O Tenant X tem um score muito alto (24.9325)
- A maior contribuição vem da variação entre fases (120% × 0.2 = 24)
- O tenant demonstra um padrão claro de "tenant barulhento"

## Verificação de Robustez em Multi-Round

### Exemplo de Consistência Causal

Considerando a relação causal entre Tenants Y e Z em três rounds experimentais:

```
Round 1: TE(Y→Z) = 0.41, p-valor = 0.03 (significativo)
Round 2: TE(Y→Z) = 0.38, p-valor = 0.04 (significativo)
Round 3: TE(Y→Z) = 0.12, p-valor = 0.22 (não significativo)
```

**Cálculo da métrica de robustez:**
```
R = N_significativo / N_total = 2 / 3 = 0.67
```

**Registro de Causalidade:** "0.67 (2/3) [0.30]"

**Interpretação:**
- A relação causal Y→Z é moderadamente robusta (0.5 ≤ R ≤ 0.75)
- Significativa em 2 de 3 rounds
- O valor médio de TE nos casos significativos é 0.30

### Coeficiente de Variação (CV)

Para o Noisy Score do Tenant Y em três rounds:

```
Round 1: Score = 18.6
Round 2: Score = 15.9
Round 3: Score = 21.2
```

**Cálculo do Coeficiente de Variação:**
```
Média = (18.6 + 15.9 + 21.2) / 3 = 18.57
Desvio Padrão = 2.65
CV = (2.65 / 18.57) * 100 = 14.27%
```

**Interpretação:**
- CV < 15% indica alta consistência no score do Tenant Y entre rounds
- O tenant mostra um comportamento barulhento consistente nos três experimentos

## Recomendações Baseadas na Análise

### Exemplo de Tenant Barulhento
Para um tenant identificado como barulhento (noisy_score > 10):

1. **Isolamento de recursos:**
   ```
   Aplicar limites de recursos (CPU, memória) mais rigorosos:
   
   resources:
     limits:
       cpu: 1000m
       memory: 1Gi
     requests:
       cpu: 500m
       memory: 512Mi
   ```

2. **Segregação de nó:**
   ```
   Aplicar taint no nó:
   kubectl taint nodes node1 noisy-tenant=true:NoSchedule
   
   Aplicar toleration no tenant barulhento:
   tolerations:
   - key: "noisy-tenant"
     operator: "Equal"
     value: "true"
     effect: "NoSchedule"
   ```

### Exemplo de Tenant Vítima
Para um tenant identificado como vítima (high-correlation, high-variation):

1. **Garantias de QoS:**
   ```
   Definir PriorityClass alta para o tenant:
   
   apiVersion: scheduling.k8s.io/v1
   kind: PriorityClass
   metadata:
     name: high-priority-tenant
   value: 1000000
   ```

2. **Definição de Pod Anti-Affinity:**
   ```
   affinity:
     podAntiAffinity:
       requiredDuringSchedulingIgnoredDuringExecution:
       - labelSelector:
           matchExpressions:
           - key: tenant
             operator: In
             values:
             - barulhento
         topologyKey: kubernetes.io/hostname
   ```

## Conclusão

Este material visual complementa a documentação principal, ilustrando através de exemplos concretos como o sistema calcula e interpreta os scores de impacto em ambiente Kubernetes multi-tenant.

Utilizando uma abordagem multidimensional que considera causalidade, correlação e variação entre fases, o sistema consegue identificar com precisão os tenants "barulhentos" e seus impactos, fornecendo insights valiosos para engenheiros e administradores de plataforma.

## Justificativas Metodológicas e Definições de Variáveis

### Por que usamos cada ferramenta?

- **Causalidade de Granger:**
  - Permite identificar relações de causa e efeito temporais entre tenants, mesmo em sistemas complexos.
  - É amplamente reconhecida em análise de séries temporais e fácil de interpretar.
  - Limitação: só detecta relações lineares e exige estacionariedade.

- **Transfer Entropy (TE):**
  - Captura relações não-lineares e direcionalidade na transferência de informação entre tenants.
  - Não assume modelo linear, sendo mais robusta em ambientes ruidosos.
  - Complementa o teste de Granger, cobrindo casos onde Granger pode falhar.

- **Correlação (Pearson e CCF):**
  - Mede o grau de associação linear entre tenants, útil para identificar acoplamentos e padrões de co-variação.
  - A CCF permite detectar influências com atraso (lags), enriquecendo a análise de dependências temporais.
  - Limitação: não distingue causa de efeito, apenas associação.

- **Variação entre Fases:**
  - Quantifica a sensibilidade de cada tenant a mudanças no ambiente (ex: ataque, recuperação).
  - Ajuda a identificar vítimas e agressores em cenários de contenção de recursos.

### Definições das Variáveis Principais

- **Tenant:** Um grupo de workloads (pods, aplicações) que compartilham recursos em um mesmo cluster Kubernetes.
- **Lag:** Defasagem temporal entre duas séries, usada para identificar influência com atraso.
- **p-valor:** Probabilidade de obter um resultado tão extremo quanto o observado, assumindo que não há relação causal. Usado para avaliar significância estatística.
- **TE(X→Y):** Transfer Entropy da série X para Y, quantificando o quanto o passado de X reduz a incerteza sobre o futuro de Y.
- **Correlação de Pearson (r):** Mede a força e direção da relação linear entre duas séries.
- **CCF:** Cross-Correlation Function, mede correlação entre séries em diferentes lags.
- **Variação percentual:** Mudança relativa entre valores de uma métrica em diferentes fases experimentais.
- **Noisy Score:** Score composto que classifica o impacto de cada tenant, combinando causalidade, correlação e variação entre fases.
- **Robustez (R):** Proporção de rounds em que uma relação causal foi estatisticamente significativa.
- **CV:** Coeficiente de Variação, mede a consistência de uma métrica entre rounds.

Essas escolhas metodológicas garantem uma análise robusta, multidimensional e interpretável do impacto de tenants em ambientes Kubernetes multi-tenant.
