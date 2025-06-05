# Resumo Executivo: Score de Impacto em Kubernetes Multi-Tenant

## O Problema

Em ambientes Kubernetes multi-tenant, um tenant "barulhento" pode degradar significativamente o desempenho de outros tenants que compartilham os mesmos recursos. Detectar e quantificar esse impacto é fundamental.

## Nossa Solução

Desenvolvemos um sistema de análise que calcula um **Noisy Score** para identificar tenants barulhentos baseado em três dimensões complementares:

```
Noisy Score = (Impacto Causal × 0.5) + (Força de Correlação × 0.3) + (Variação entre Fases × 0.2)
```

## Componentes do Score

### 1. Impacto Causal (50%)

**Pergunta:** O tenant A **causa** mudanças no tenant B?

**Métodos:**
- **Causalidade de Granger**: Testa se valores passados de A melhoram a previsão de B
- **Transfer Entropy**: Mede o fluxo de informação de A para B (captura relações não-lineares)

### 2. Força de Correlação (30%)

**Pergunta:** Quando A muda, B também muda de forma similar?

**Métodos:**
- **Correlação de Pearson**: Mede associação linear instantânea
- **Cross-Correlation (CCF)**: Identifica correlações com defasagem temporal

### 3. Variação entre Fases (20%)

**Pergunta:** O quanto o comportamento do tenant muda entre fases do experimento?

**Método:**
- Variação percentual: `((valor_ataque - valor_baseline) / valor_baseline) * 100`

## Resultados Principais

| Tenant | Score Total | Impacto Causal | Força de Correlação | Variação entre Fases |
|--------|------------|---------------|---------------------|----------------------|
| B | 9.03 | 0.25 | 0.10 | 44.39 |
| D | 1.43 | 0.20 | 0.14 | 6.45 |
| C | 1.40 | 0.38 | 0.17 | 5.81 |
| A | 1.04 | 0.30 | 0.12 | 4.25 |

## Principais Conclusões

1. **Identificação Precisa**: O sistema identifica com precisão quais tenants causam maior impacto no ambiente compartilhado.

2. **Multidimensionalidade**: A combinação de diferentes perspectivas (causalidade, correlação, variação) oferece uma visão holística.

3. **Validação Robusta**: Métricas de robustez em análises multi-round garantem consistência dos resultados.

4. **Aplicabilidade Prática**: Os insights orientam decisões de engenharia como isolamento de recursos e priorização de workloads.

## Próximos Passos

1. Implementar detecção e mitigação automatizada para tenants barulhentos
2. Estender para análises em janelas deslizantes para detecção em tempo real
3. Desenvolver recomendações específicas de configuração baseadas nos padrões de impacto

## Para Saber Mais

- Documentação completa: `/docs/calculo_impacto_causality_score.md`
- Material didático para seminário: `/docs/impacto_causality_score_seminario.md`
- Exemplos visuais: `/docs/impacto_causality_score_exemplos_visuais.md`
