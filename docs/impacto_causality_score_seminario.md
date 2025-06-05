# Detecção de Impacto em Ambientes Multi-Tenant Kubernetes
*Material Didático para Seminário*

## Objetivos da Apresentação

Este material explica didaticamente como nosso sistema calcula o impacto entre diferentes tenants (inquilinos) em um ambiente Kubernetes compartilhado. Usamos três componentes principais para construir um score que identifica tenants "barulhentos" - aqueles que mais afetam o desempenho dos outros.

## A Fórmula do Score Composto

Nosso sistema utiliza uma equação ponderada para combinar diferentes indicadores de impacto:

```
Noisy Score = (Impacto Causal × 0.5) + (Força de Correlação × 0.3) + (Variação entre Fases × 0.2)
```

## 1. Impacto Causal (50% do score)

O impacto causal analisa relações de causa e efeito entre os tenants, respondendo à pergunta: "O comportamento do tenant A **causa** mudanças no comportamento do tenant B?"

### Causalidade de Granger

Este teste verifica se o histórico do tenant A ajuda a prever o comportamento futuro do tenant B, além do que B consegue prever baseado apenas em seu próprio histórico.

**Ilustração visual do conceito:**
```
Modelo 1 (Restrito):   B(t) = f(B(t-1), B(t-2), ...) + erro1
Modelo 2 (Irrestrito): B(t) = f(B(t-1), B(t-2), ..., A(t-1), A(t-2), ...) + erro2
```

Se erro2 for significativamente menor que erro1, então A "causa" B no sentido de Granger.

**Como calculamos:**
1. Transformamos os dados para estacionariedade (teste ADF)
2. Testamos diferentes defasagens (lags)
3. Calculamos o p-valor do teste F
4. Convertemos para score: `impacto = 1 - p_valor`

### Transfer Entropy (TE)

Enquanto Granger assume relações lineares, a TE pode capturar relações não-lineares entre tenants.

**O que significa:** Quantifica a informação transferida do tenant A para o tenant B ao longo do tempo.

**Formula conceitual:**
```
TE(A→B) = Incerteza_sobre_B_conhecendo_apenas_B - Incerteza_sobre_B_conhecendo_A_e_B
```

**Por que damos mais peso ao TE:** 
- Captura relações não-lineares
- Mais robusta a ruídos
- No código, multiplicamos por 5 para dar maior peso aos resultados de TE

## 2. Força de Correlação (30% do score)

A correlação mede o grau de associação linear entre tenants, respondendo à pergunta: "Quando o tenant A muda, o tenant B também muda de forma similar?"

**O que medimos:**
- Correlação tradicional (em um mesmo momento)
- Correlação cruzada (com defasagens temporais)

### Correlação Tradicional

**Como calculamos:**
1. Para cada par de tenants, calculamos o coeficiente de correlação de Pearson
2. Usamos o valor absoluto (ignorando o sinal)
3. Armazenamos os valores significativos (geralmente |r| > 0.2)

### Cross-Correlation Function (CCF)

A CCF estende a análise para incluir defasagens temporais, permitindo detectar padrões onde um tenant influencia outro com atraso.

**Exemplo visual:**

```
Correlação Máxima em Lag +3:
Tenant A: [a1, a2, a3, a4, a5, a6, a7, ...]
                     ↓   ↓   ↓   ↓
Tenant B: [b1, b2, b3, b4, b5, b6, b7, ...]
                 ↑   ↑   ↑   ↑
```

Este padrão sugere que mudanças no Tenant A precedem mudanças similares no Tenant B por 3 períodos.

## 3. Variação entre Fases (20% do score)

Este componente mede a sensibilidade de um tenant às mudanças no ambiente entre diferentes fases experimentais (baseline, ataque, recuperação).

**O que calculamos:**
```
Variação(%) = ((valor_ataque - valor_baseline) / valor_baseline) * 100
```

**Interpretação:**
- Valores altos (positivos ou negativos): Tenant muito sensível a mudanças no ambiente
- Valores próximos a zero: Tenant estável, pouco afetado por mudanças no ambiente

**Por que é importante:** Tenants com alta variação frequentemente são vítimas de "noisy neighbors" ou são os próprios causadores do problema.

## Exemplo Prático: Descobrindo o Tenant Barulhento

**Situação:** Em um cluster Kubernetes com tenants A, B, C e D:
1. O sistema coleta dados de desempenho ao longo das três fases experimentais
2. Calcula cada componente do score para cada tenant
3. Aplica a fórmula ponderada para obter o score final
4. Classifica os tenants por score (do maior para o menor)

**Resultado exemplo (dados reais de experimento):**

| Tenant | Score Total | Impacto Causal | Força de Correlação | Variação entre Fases |
|--------|------------|---------------|---------------------|----------------------|
| B | 9.03 | 0.25 | 0.10 | 44.39 |
| D | 1.43 | 0.20 | 0.14 | 6.45 |
| C | 1.40 | 0.38 | 0.17 | 5.81 |
| A | 1.04 | 0.30 | 0.12 | 4.25 |

**Interpretação:** Tenant B é claramente identificado como "barulhento" devido ao seu alto score total (9.03), impulsionado principalmente por sua enorme variação entre fases (44.39).

## Validação Multi-Round

Para garantir que nossas conclusões são consistentes, implementamos métricas de robustez:

1. **Métrica de Robustez para Causalidade:** 
   ```
   R = N_significativo / N_total
   ```
   Classificação:
   - R > 0.75: Relação causal altamente robusta
   - 0.5 ≤ R ≤ 0.75: Moderadamente robusta
   - 0.25 ≤ R < 0.5: Fracamente robusta
   - R < 0.25: Não robusta (possivelmente espúria)

2. **Coeficiente de Variação (CV):**
   ```
   CV = (desvio_padrão / média) * 100
   ```
   Classificação:
   - CV < 15%: Alta consistência
   - CV entre 15% e 30%: Consistência média
   - CV > 30%: Baixa consistência

## Aplicações Práticas

### Identificação de Problemas
- Detectar automaticamente tenants problemáticos antes que afetem todo o cluster
- Isolar a causa raiz de problemas de desempenho em ambientes compartilhados

### Otimização de Recursos
- Orientar estratégias de isolamento de recursos (CPU, memória, I/O)
- Informar decisões sobre alocação e limitação de recursos

### Planejamento de Capacidade
- Prever o impacto de adicionar novos tenants ao sistema
- Estimar a "compatibilidade" entre diferentes tipos de workloads

## Limitações e Considerações

É importante compreender as limitações destas métricas:

1. **Causalidade Estatística ≠ Causalidade Física**
   - As relações identificadas são estatísticas e podem não refletir mecanismos físicos de causa e efeito

2. **Limitações das Técnicas**
   - Granger: Assume relações lineares
   - TE: Requer séries temporais relativamente longas
   - Correlação: Pode detectar associações espúrias

3. **Contextualização**
   - Os scores devem ser interpretados no contexto específico do experimento
   - Validação cruzada com conhecimento do domínio é recomendada

## Conclusão

O sistema de análise multi-tenant implementa um conjunto abrangente de métricas para identificação das dinâmicas entre workloads em um ambiente Kubernetes compartilhado. A abordagem multidimensional (causalidade, correlação e variação) fornece uma base sólida para decisões de engenharia relacionadas ao isolamento de recursos e mitigação de problemas de "noisy neighbors".
