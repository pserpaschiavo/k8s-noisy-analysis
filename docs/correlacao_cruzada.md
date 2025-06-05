# Correlação Cruzada na Análise Multi-Tenant

## Visão Geral

A correlação cruzada (Cross-Correlation Function - CCF) é uma poderosa técnica para analisar relações temporais entre séries, identificando não apenas se duas séries estão correlacionadas, mas também se existe uma defasagem (lag) nessa relação. Este documento explica como usar e interpretar os gráficos de correlação cruzada gerados pelo pipeline.

## Funcionalidade Implementada

A funcionalidade de correlação cruzada foi implementada no módulo `analysis_correlation.py` e está agora integrada ao pipeline principal através do estágio `CorrelationAnalysisStage`. Os plots gerados são armazenados no diretório:

```
outputs/demo-experiment-1-round/plots/correlation/cross_correlation/
```

## Como Executar

Para gerar plots de correlação cruzada, execute o pipeline unificado:

```bash
python -m src.run_unified_pipeline --config config/pipeline_config.yaml
```

## Interpretando os Plots de Correlação Cruzada

Os plots de correlação cruzada mostram:

1. **Eixo X**: Defasagem (lag) entre as séries temporais, de -N a +N
   - Lag negativo: o segundo tenant influencia o primeiro
   - Lag zero: correlação contemporânea (sem defasagem)
   - Lag positivo: o primeiro tenant influencia o segundo

2. **Eixo Y**: Valor da correlação entre as séries para cada defasagem
   - Valores próximos a +1 indicam forte correlação positiva
   - Valores próximos a -1 indicam forte correlação negativa
   - Valores próximos a 0 indicam fraca ou nenhuma correlação

3. **Marcação Vermelha**: Indica o ponto de maior correlação em valor absoluto
   - A anotação indica o valor da correlação e o lag onde ocorre
   - Sugere a defasagem ótima entre as séries temporais

4. **Faixa Cinza**: Representa um intervalo de confiança aproximado
   - Valores dentro desta faixa podem não ser estatisticamente significativos

5. **Título**: Inclui informação direcional baseada no lag de maior correlação
   - "A → B" indica que A tende a influenciar B
   - "B → A" indica que B tende a influenciar A
   - "Contemporânea" indica que a maior correlação ocorre sem defasagem

## Exemplo de Análise

Um exemplo de interpretação:

- Se o plot mostra correlação máxima em lag +3 para tenant-a e tenant-b:
  - Isso sugere que mudanças na atividade do tenant-a precedem mudanças similares no tenant-b por 3 períodos
  - Pode indicar uma relação causal ou influência de tenant-a sobre tenant-b

- Se o valor máximo ocorre em lag negativo:
  - Sugere que o segundo tenant influencia o primeiro
  - Um lag -2 entre tenant-c e tenant-d indicaria que tenant-d influencia tenant-c com 2 períodos de antecedência

## Importância para a Análise Multi-Tenant

A correlação cruzada é particularmente valiosa para:

1. **Identificação do tenant barulhento**: Tenants que sistematicamente influenciam outros com lags positivos são potenciais candidatos a "tenants barulhentos"

2. **Propagação de efeitos**: Entender quanto tempo leva para o comportamento de um tenant afetar outros

3. **Planejamento de mitigação**: Conhecer a defasagem temporal permite ações preventivas mais eficazes

## Limitações

- A CCF identifica apenas relações lineares entre séries temporais
- Não prova causalidade, apenas sugere relações temporais
- Defasagens muito longas podem ser espúrias ou coincidências
