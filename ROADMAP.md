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

### Fase 3: Otimização e Refinamento (Concluída)

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

- [ ] **Otimização de Performance:**
  - **Status:** A fazer.
  - **Descrição:** Investigar e implementar otimizações de performance, como o uso de `Polars` ou `Dask` para manipulação de grandes DataFrames, se necessário.

### Fase 4: Documentação e Finalização (Estamos aqui)

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
