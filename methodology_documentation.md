# Documentação Metodológica para Análise Multi-tenant

Este documento detalha as escolhas metodológicas e parâmetros técnicos utilizados no pipeline de análise de séries temporais multi-tenant.

## Parâmetros e Configurações Gerais

### Limiares e Configurações de Análise

- **Correlação**:
  - Métodos implementados: Pearson, Kendall e Spearman
  - Método padrão: Pearson (paramétrico)
  - Tratamento de dados faltantes: Interpolação linear

- **Cross-Correlation (CCF)**:
  - Defasagem máxima (`max_lag`): 20 períodos
  - Normalização: Z-score (média = 0, desvio padrão = 1)
  - Intervalo de confiança: 95% (1.96/sqrt(N))

- **Detecção de Anomalias**:
  - Método: Z-score com janela móvel
  - Tamanho da janela (`window_size`): 10 períodos
  - Limiar para anomalias (`threshold`): 2.0 desvios padrão 

- **Causalidade de Granger**:
  - Limiar de significância (`threshold`): 0.05 (5%)
  - Defasagem máxima: determinada através de AIC/BIC ou defasagem de máxima CCF
  - Teste utilizado: teste F para comparação de modelos aninhados

- **Transfer Entropy (TE)**:
  - Biblioteca: pyinform
  - Discretização: 8 bins (valor padrão)
  - Histórico considerado (k): 1 período
  - Normalização antes da discretização

## Justificativa das Escolhas Metodológicas

### 1. Correlação vs. Causalidade

Utilizamos tanto métricas de correlação quanto de causalidade para oferecer uma visão mais completa:

- **Correlação (Pearson)**: Mede associação linear, simétrica e não causal. É mais intuitiva e conhecida.
- **Causalidade de Granger**: Testa se os valores passados de uma série ajudam a prever os valores futuros de outra, além da informação contida no próprio passado da série prevista.
- **Transfer Entropy**: Métrica baseada em teoria da informação, não paramétrica, que captura dependências estatísticas lineares e não lineares.

### 2. Escolha da Transfer Entropy sobre outros métodos

A Transfer Entropy foi escolhida pelos seguintes motivos:

- É uma medida não paramétrica (não assume linearidade como Granger)
- É baseada na teoria da informação e captura dependências complexas
- Complementa bem o teste de Granger que assume modelo linear

### 3. Métricas Compostas para Identificação de "Tenant Barulhento"

A identificação do tenant com maior impacto no sistema utiliza uma combinação ponderada de:

- Escore de impacto causal (50%): baseado em p-values de Granger e valores de TE
- Força de correlação (30%): força média das correlações com outros tenants
- Variação entre fases (20%): magnitude da mudança de comportamento do tenant entre fases

### 4. Visualizações e Representações Visuais

As visualizações foram selecionadas para auxiliar a interpretação:

- **Grafos direcionados**: Para representar visualmente relações causais
- **Heatmaps**: Para matrizes de correlação e causalidade
- **Séries temporais com anomalias destacadas**: Para identificação de eventos importantes

## Metodologia de Análise Consolidada para Experimentos Multi-Round

Para experimentos com múltiplos rounds, implementamos abordagens específicas para garantir resultados estatisticamente robustos e identificar com precisão padrões consistentes versus anomalias pontuais.

### 1. Análise de Consistência Entre Rounds

Esta metodologia avalia a consistência dos comportamentos observados entre diferentes rounds do mesmo experimento.

- **Coeficiente de Variação (CV)**: Calculado como (desvio padrão/média)*100 para cada métrica entre rounds
  - CV < 15%: Alta consistência
  - CV entre 15% e 30%: Consistência média 
  - CV > 30%: Baixa consistência

- **Teste de Friedman**: Utilizado para avaliar se há diferenças estatisticamente significativas no comportamento dos tenants entre rounds
  - p-valor < 0.05: Existem diferenças significativas entre rounds
  - p-valor ≥ 0.05: Não há evidência de diferença significativa entre rounds

- **Visualização**: Boxplots agrupados por round para cada métrica, permitindo a inspeção visual da variabilidade

### 2. Análise de Robustez de Causalidade

Esta metodologia examina a consistência das relações causais identificadas em diferentes rounds, distinguindo relações causais robustas de correlações espúrias.

- **Meta-análise de Transfer Entropy (TE)**: Combinação dos resultados de TE entre diferentes rounds
  - Período de observação: Todos os rounds disponíveis do experimento
  - Limiar de significância: 0.05 (padrão)

- **Métrica de Robustez**: R = N_significativo / N_total
  - R > 0.75: Relação causal altamente robusta
  - 0.5 ≤ R ≤ 0.75: Relação causal moderadamente robusta
  - 0.25 ≤ R < 0.5: Relação causal fracamente robusta
  - R < 0.25: Relação causal não robusta (possivelmente espúria)

- **Registro de Causalidade**: Formato "(robustez) (contagem/total) [média TE]"
  - Exemplo: "0.67 (2/3) [0.123]" indica que a relação causal foi significativa em 2 de 3 rounds, com robustez 0.67 e valor médio de TE de 0.123

### 3. Análise de Divergência de Comportamento

Esta metodologia identifica padrões anômalos em rounds específicos, permitindo detectar execuções com condições incomuns ou problemas experimentais.

- **Distância de Kullback-Leibler (KL)**: Medida da diferença entre distribuições de métricas entre rounds
  - Fórmula simétrica: KL(P||Q) + KL(Q||P) / 2
  - Normalização: Histogramas com 20 bins, densidade normalizada

- **Detecção de Outliers**: Usando Z-score sobre as distâncias médias
  - Limiar: |Z| > 2 para identificar rounds significativamente divergentes 
  - Visualização: Dendrogramas de similaridade entre rounds usando clustering hierárquico

### 4. Agregação de Consenso

Esta metodologia combina resultados de múltiplos rounds para produzir um veredicto consolidado sobre o comportamento do sistema.

- **Votação por Maioria**: Para classificações categóricas (ex: tenant barulhento)
  - Limiar: >50% dos rounds para classificação positiva 
  - Nível de confiança: Percentual de concordância entre rounds

- **Média Ponderada**: Para métricas numéricas (ex: scores de impacto)
  - Peso de cada round: Igual por padrão, ajustável conforme qualidade do round

- **Consolidação de Relações**: Critério de frequência mínima
  - Relação significativa: Presente em pelo menos 50% dos rounds
  - Recomendações: Presentes em pelo menos 1/3 dos rounds

### 5. Visualizações de Consistência entre Rounds

As seguintes visualizações são utilizadas para representar adequadamente os resultados consolidados:

- **Gráficos de Barras com Intervalo de Confiança (95%)**: Média ± IC para métricas entre rounds
- **Dendrogramas de Similaridade**: Agrupamento hierárquico de rounds por similaridade de comportamento
- **Heatmaps Comparativos**: Visualização lado-a-lado das matrizes de correlação/causalidade entre rounds
- **Gráficos de Radar**: Visualização multidimensional de métricas entre diferentes rounds

Estas metodologias em conjunto garantem uma avaliação robusta dos dados em experimentos multi-round, permitindo conclusões mais confiáveis sobre o comportamento dos tenants e suas interações.

## Configurações Técnicas

### Performance e Otimização

- **Cache**: Funções computacionalmente intensas utilizam `@lru_cache` para melhorar desempenho
- **Formato de armazenamento**: Parquet para melhor compressão e velocidade de leitura
- **Normalização de dados**: Aplicada antes de análises para evitar viés por escalas diferentes

### Tratamento de Valores Ausentes

- **Interpolação**: Interpolação linear para falhas pontuais
- **Imputação por média**: Utilizada quando a interpolação não é possível

### Discretização para Transfer Entropy

- **Método**: Quantis uniformes (bins de igual tamanho após normalização)
- **Número de bins**: 8 (compromisso entre precisão e problema de dimensionalidade)

## Limitações Conhecidas

- Transfer Entropy requer séries temporais relativamente longas para estimativas confiáveis
- A causalidade inferida é sempre estatística, não necessariamente causal no sentido estrito
- O pipeline assume regularidade temporal nas séries (intervalos constantes entre medições)

## Roadmap Metodológico

Melhorias futuras planejadas para o pipeline:

1. Implementação de testes de significância estatística para valores de TE via bootstrapping
2. Análises com janelas deslizantes para capturar mudanças de dinâmica temporal
3. Incorporação de métodos de causalidade baseados em modelo (ex: VAR estrutural)
