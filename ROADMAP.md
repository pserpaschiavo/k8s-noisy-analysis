# Roadmap Estratégico de Refatoração

Este documento serve como um guia de alto nível para as fases de trabalho no pipeline de análise, permitindo um acompanhamento claro do progresso.

---

### Fase 1: Fundação e Estabilidade

*Objetivo: Estabilizar o pipeline, corrigir problemas críticos e estabelecer uma base sólida para futuras análises.*

- [x] **Definir Objetivos e Plano de Refatoração:**
  - **Status:** Concluído.
  - **Artefatos:** `PLANO_DE_REFATORACAO.md`.

- [x] **Corrigir Alertas `SettingWithCopyWarning`:**
  - **Status:** Concluído.
  - **Local:** `src/visualization/plots.py`.
  - **Descrição:** Refatorar o código para usar o acessador `.loc` do Pandas, eliminando alertas e garantindo a robustez do tratamento de dados.

- [x] **Implementar Parametrização do Pipeline:**
  - **Status:** Concluído.
  - **Descrição:** Modificar `run_pipeline.py` para aceitar argumentos de linha de comando que permitam selecionar quais fases da análise (descritiva, correlação, impacto) e quais rodadas executar. Isso trará um ganho de performance imediato.

---

### Fase 2: Análise de Impacto e Geração de Artefatos

*Objetivo: Desenvolver a lógica central para quantificar o "noisy neighbour" e gerar os artefatos necessários para publicações acadêmicas.*

- [x] **Desenvolver Módulo de Análise de Impacto:**
  - **Status:** Concluído.
  - **Local:** Criar `src/analysis_impact.py`.
  - **Descrição:** Implementar a lógica de análise comparativa entre a fase `Baseline` e as fases de ruído.

- [x] **Implementar Métricas de Impacto e Testes Estatísticos:**
  - **Status:** Concluído.
  - **Descrição:** Adicionar cálculos de variação percentual, volatilidade (desvio padrão) e testes de hipótese (ex: t-test de Student) para validar a significância estatística dos resultados.

- [x] **Gerar CSVs e Plots Focados:**
  - **Status:** Concluído.
  - **Descrição:** Criar funções para exportar os resultados da análise de impacto em formato CSV e gerar plots estáticos de alta qualidade (PNG/PDF) para uso em artigos.

---

### Fase 3: Otimização e Refinamento

*Objetivo: Melhorar a qualidade do código, remover redundâncias e otimizar a performance geral do pipeline.*

- [x] **Refatorar Módulos de Análise Legados:**
  - **Status:** Concluído.
  - **Descrição:** Migrar a lógica dos módulos de análise para a nova estrutura de `PipelineStage`, separando claramente a lógica de cálculo da lógica de visualização.
    - [x] **Análise de Impacto:** Módulo criado e integrado.
    - [x] **Análise Descritiva:** Módulo de visualização (`descriptive_plots.py`) criado e refatorado.
    - [x] **Análise de Correlação:** Módulo de visualização (`correlation_plots.py`) criado e refatorado.
    - [x] **Análise de Causalidade:** Módulo de visualização (`causality_plots.py`) criado e refatorado.
    - [x] **Análise de Comparação de Fases:** Refatorar o módulo de visualização (`phase_comparison_plots.py`).

- [x] **Remover Código Obsoleto:**
  - **Status:** Concluído.
  - **Descrição:** Após a refatoração, identificar e remover funções e scripts que não são mais utilizados.

- [x] **Otimização de Performance:**
  - **Status:** Concluído.
  - **Descrição:** Investigar e implementar otimizações de performance, como o uso de `Polars` ou `Dask` para manipulação de grandes DataFrames, se necessário.

### Fase 4: Documentação e Finalização (A fazer)

*Objetivo: Garantir que toda a lógica do pipeline esteja bem documentada e que o sistema esteja pronto para manutenção e futuras expansões.*

- [ ] **Documentar Pipeline e Análises:**
  - **Status:** A fazer.
  - **Descrição:** Criar ou atualizar a documentação do pipeline, incluindo a descrição detalhada de cada fase da análise, parâmetros utilizados e interpretação dos resultados.

- [ ] **Treinamento e Transferência de Conhecimento:**
  - **Status:** A fazer.
  - **Descrição:** Realizar sessões de treinamento para a equipe envolvida, garantindo que todos compreendam a nova estrutura do código e como utilizá-lo efetivamente.

- [ ] **Planejamento de Futuras Melhorias:**
  - **Status:** A fazer.
  - **Descrição:** Com base no aprendizado das fases anteriores, planejar melhorias e novas funcionalidades para o pipeline, priorizando aquelas que trarão maior impacto positivo na qualidade da análise.

---

### Fase 5: Análise Multi-Round e Consolidação de Resultados

*Objetivo: Agregar os resultados das múltiplas rodadas de execução para obter uma visão estatística consolidada do comportamento dos tenants sob o efeito de "noisy neighbours". O foco é gerar artefatos que demonstrem a estabilidade e a variabilidade dos resultados.*

- [x] **Desenvolver Módulo de Agregação de Resultados:**
  - **Status:** Concluído.
  - **Local:** Criar `src/analysis_multi_round.py`.
  - **Descrição:** Implementar a lógica para carregar e consolidar os resultados de todas as rodadas (ex: arquivos CSV de impacto e métricas).

- [x] **Calcular Estatísticas Agregadas:**
  - **Status:** Concluído.
  - **Descrição:** Calcular métricas estatísticas descritivas sobre os resultados agregados, como média, mediana, desvio padrão e intervalos de confiança para as principais métricas de impacto.

- [x] **Gerar Visualizações Multi-Round:**
  - **Status:** Concluído.
  - **Local:** Criar `src/visualization/multi_round_plots.py`.
  - **Descrição:** Desenvolver um conjunto de visualizações agregadas para demonstrar a robustez e a variabilidade dos resultados ao longo de N rodadas.

### Fase 6: Validação e Execução Multi-Round

*Objetivo: Executar o pipeline completo para múltiplas rodadas, gerar os artefatos de saída corretos e validar a análise consolidada para extrair conclusões robustas.*

- [x] **Adaptar Script de Execução para Múltiplas Rodadas:**
  - **Status:** Concluído.
  - **Descrição:** Modificar `run_pipeline.py` ou criar um script wrapper para iterar sobre as rodadas definidas na configuração, salvando os resultados em diretórios específicos por rodada (ex: `outputs/sfi2-paper-analysis/round-1`, `outputs/sfi2-paper-analysis/round-2`).

- [x] **Executar o Pipeline Completo:**
  - **Status:** Concluído.
  - **Descrição:** Realizar uma execução completa com as múltiplas rodadas para gerar os dados necessários para a análise consolidada.

- [x] **Validar Análise Multi-Round:**
  - **Status:** Concluído.
  - **Descrição:** Verificar se a `MultiRoundAnalysisStage` processa corretamente os resultados das múltiplas rodadas e gera os artefatos agregados (CSVs e plots).

- [x] **Analisar Resultados Consolidados:**
  - **Status:** Concluído.
  - **Descrição:** Interpretar os resultados agregados para extrair conclusões sobre a estabilidade e variabilidade do impacto dos "noisy neighbours".
