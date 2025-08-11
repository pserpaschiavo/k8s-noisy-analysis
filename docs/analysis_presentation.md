# Análise de Noisy Neighbor — Guia e Discussão

## Novidades desta execução

- Exportação de CSVs para todos os gráficos:
  - ECDF por métrica e fase: `figs/csv/ecdf_{Métrica}.csv`.
  - Estatísticas de boxplot por fase: `figs/csv/box_stats_{Métrica}.csv`.
  - Impacto agregado (base dos gráficos): `figs/csv/impact_aggregated_stats.csv`.
  - Correlação (matriz simétrica) por fase e métrica: `figs/csv/correlation_matrix_{metricaSlug}_{fase}.csv`.
  - Causalidade (matriz alvo×fonte) por fase e métrica: `figs/csv/causality_matrix_{Métrica}_{fase}.csv`.
  - Top-15 links por consistência: `figs/csv/causality_consistency_top15.csv`.
- Heatmaps de correlação agora incluem a métrica no título e no nome do arquivo (quando disponível).
- Heatmaps de causalidade são gerados para todas as métricas por padrão; nomes exibem a métrica em formato legível.
- Nomes “bonitos” de métricas aplicados de forma consistente em todos os gráficos.

## Guia de interpretação dos gráficos

- ECDF por Fase (distribuição acumulada)
  - O que mostra: a distribuição acumulada dos valores da métrica por fase do experimento.
  - Como ler: curvas mais íngremes indicam menor variabilidade; deslocamentos para a direita sugerem valores maiores. Compare fases para ver mudanças sob ruído.
  - Observação: as fases estão na ordem numérica (1-Baseline → 7-Recovery).

- Boxplots por Fase
  - O que mostra: mediana, IQR e outliers por fase.
  - Como ler: medianas/IQR mais altos em fases com ruído indicam aumento de nível/variabilidade da métrica frente ao baseline.

- Impacto Agregado por Fase (um gráfico por métrica)
  - O que mostra: impacto percentual médio (± desvio padrão) por tenant em cada fase.
  - Como ler: barras acima de zero significam aumento da métrica sob ruído; abaixo de zero, redução. As barras de erro ilustram a variabilidade entre rounds. Compare tenants para ver quem é mais afetado.
  - Observação: o eixo x segue a ordem numérica das fases.

- Heatmaps de Correlação (por fase)
  - O que mostra: correlação média entre tenants (matriz simétrica, valores em [-1, 1]).
  - Como ler: cores fortes indicam co-movimento mais intenso (positivo ou negativo). A diagonal é 1 por definição. Heurística (valor absoluto): ~0,1 fraco, ~0,3 moderado, ~0,5+ forte (dependente do contexto).

- Heatmaps de Causalidade (por fase e métrica)
  - O que mostra: influência direcional média (ex.: Granger/TE) de uma fonte → alvo (matriz assimétrica).
  - Como ler: cores mais quentes indicam influência maior. Foque em setas partindo do tenant ruidoso (tenant-nsy) para as vítimas. Repetição ao longo dos rounds reforça a evidência.

- Consistência de Causalidade (Top-15)
  - O que mostra: os links causais mais frequentes ao longo dos rounds.
  - Como ler: a taxa de consistência (%) indica com que frequência o link reaparece; quanto maior, mais estável o efeito.

## Discussão dos resultados (base atual)

- Impacto
  - Observa-se impacto percentual diferente de zero para tenants vítimas em várias fases com ruído. Em geral, CPU-Noise e Combined-Noise exibem os efeitos mais pronunciados para a métrica CPU Usage. Para rede/ disco/ memória, os efeitos aparecem alinhados à métrica, porém normalmente menores que os de CPU nesta base.
  - As barras de erro (DP) mostram variabilidade entre rounds, mas vários casos permanecem consistentemente acima (ou abaixo) de zero, indicando padrões reprodutíveis de impacto.

- Correlação
  - As magnitudes médias de |correlação| por fase são modestas e não aumentam de forma sistemática sob ruído em relação ao baseline (médias típicas ~0,15–0,18 em valor absoluto). Isso é esperado, pois correlação captura co-movimento e não direção/causa.
  - Ainda assim, os heatmaps ajudam a localizar tenants potencialmente acoplados e a priorizar onde investigar causalidade.

- Causalidade
  - Os links causais mais frequentes apresentam consistência alta entre rounds (tipicamente na faixa de ~60–70% nesta base), o que sugere uma estrutura direcional estável sob injeção de ruído.
  - Essa repetição de setas na direção esperada (do tenant ruidoso para vítimas) reforça a validade dos métodos Granger/TE usados.

## As observações validam as ferramentas?

- Impacto percentual e agregação por round
  - Métrica diretamente conectada a SLO/SLA; fácil de comunicar. A agregação entre rounds reduz ruído amostral e destaca efeitos persistentes.

- Correlação
  - Boa ferramenta de triagem para identificar co-movimentos entre tenants, reduzindo o espaço de busca para causalidade, sem assumir direção.

- Granger/Transfer Entropy (direcionalidade)
  - Fornecem sentido (source → target). A consistência multi-round aumenta a confiança e filtra detecções espúrias.

Em conjunto, os resultados observados (impactos consistentes, padrões direcionais recorrentes e suporte de correlação) corroboram o uso dessas ferramentas neste cenário.

## O fenômeno de noisy neighbor foi constatado?

Sim. Há evidências convergentes de que a atividade de um tenant afeta o comportamento de outros:
- Impactos percentuais alinhados às fases de ruído (especialmente CPU-Noise e Combined-Noise em CPU Usage) nos tenants vítimas.
- Padrões de causalidade dirigidos do tenant ruidoso para os demais, com consistência elevada entre rounds.
- Correlações que, embora moderadas, oferecem contexto adicional sobre acoplamentos durante fases com ruído.

Conclusão: as análises indicam a presença do fenômeno de noisy neighbor nesta base, com maior intensidade em sinais relacionados a CPU e em fases combinadas de ruído.

## Checklist prático para apresentação

1) Comece pelos gráficos de impacto por métrica:
   - Quais fases afastam mais as barras de zero? Quais tenants são mais afetados?
2) Cruze com os heatmaps de correlação dessas fases:
   - Existem blocos mais intensos envolvendo os mesmos tenants?
3) Valide direção nos heatmaps de causalidade:
   - Há setas do tenant-nsy para as vítimas? Esse padrão se repete em várias fases/rounds?
4) Resuma com consistência:
   - Destaque links com maior taxa de recorrência como evidência mais forte.

## Limitações e próximos passos

- Correlação não implica causalidade; use-a como triagem.
- Causalidade depende de escolhas de defasagens e pressupostos de estacionariedade; recomenda-se análise de sensibilidade.
- Melhorias sugeridas:
  - Adicionar intervalos de confiança às barras de impacto.
  - Anotar significância nos gráficos de causalidade (ex.: p<0,05) quando aplicável.
  - Exportar tabelas compactas (ex.: LaTeX) a partir dos CSVs tidy para uso em artigos/slides.

---

Referências dos artefatos gerados (padrões atuais):
- Figuras
  - Impacto: `figs/impact_aggregated_*.png`
  - Correlação: `figs/correlation_heatmap_{metricaSlug|all}_{fase}.png`
  - Causalidade: `figs/causality_heatmap_{Métrica}_{fase}.png`, `figs/causality_consistency_top15.png`
- CSVs
  - ECDF: `figs/csv/ecdf_{Métrica}.csv`
  - Boxplot (stats): `figs/csv/box_stats_{Métrica}.csv`
  - Impacto (agregado): `figs/csv/impact_aggregated_stats.csv`
  - Correlação (matriz): `figs/csv/correlation_matrix_{metricaSlug|all}_{fase}.csv`
  - Causalidade (matriz): `figs/csv/causality_matrix_{Métrica}_{fase}.csv`
  - Causalidade (Top-15): `figs/csv/causality_consistency_top15.csv`

Este documento resume o estado atual e pode ser ajustado conforme novas execuções/experimentos.
