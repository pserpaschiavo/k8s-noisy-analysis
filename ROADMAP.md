# Roadmap Estratégico de Refatoração

Este documento serve como um guia de alto nível para as fases de trabalho no pipeline de análise, permitindo um acompanhamento claro do progresso.

---

### Fase 1: Fundação e Estabilidade (Estamos aqui)

*Objetivo: Estabilizar o pipeline, corrigir problemas críticos e estabelecer uma base sólida para futuras análises.*

- [x] **Definir Objetivos e Plano de Refatoração:**
  - **Status:** Concluído.
  - **Artefatos:** `PLANO_DE_REFATORACAO.md`.

- [ ] **Corrigir Alertas `SettingWithCopyWarning`:**
  - **Status:** Pendente.
  - **Local:** `src/visualization/plots.py`.
  - **Descrição:** Refatorar o código para usar o acessador `.loc` do Pandas, eliminando alertas e garantindo a robustez do tratamento de dados.

- [ ] **Implementar Parametrização do Pipeline:**
  - **Status:** Pendente.
  - **Descrição:** Modificar `run_pipeline.py` para aceitar argumentos de linha de comando que permitam selecionar quais fases da análise (descritiva, correlação, impacto) e quais rodadas executar. Isso trará um ganho de performance imediato.

---

### Fase 2: Análise de Impacto e Geração de Artefatos

*Objetivo: Desenvolver a lógica central para quantificar o "noisy neighbour" e gerar os artefatos necessários para publicações acadêmicas.*

- [ ] **Desenvolver Módulo de Análise de Impacto:**
  - **Status:** Pendente.
  - **Local:** Criar `src/analysis_impact.py`.
  - **Descrição:** Implementar a lógica de análise comparativa entre a fase `Baseline` e as fases de ruído.

- [ ] **Implementar Métricas de Impacto e Testes Estatísticos:**
  - **Status:** Pendente.
  - **Descrição:** Adicionar cálculos de variação percentual, volatilidade (desvio padrão) e testes de hipótese (ex: t-test de Student) para validar a significância estatística dos resultados.

- [ ] **Gerar CSVs e Plots Focados:**
  - **Status:** Pendente.
  - **Descrição:** Criar funções para exportar os resultados da análise de impacto em formato CSV e gerar plots estáticos de alta qualidade (PNG/PDF) para uso em artigos.

---

### Fase 3: Otimização e Refinamento

*Objetivo: Otimizar o código legado, remover partes obsoletas e garantir a manutenibilidade do pipeline a longo prazo.*

- [ ] **Refatorar Módulos de Análise Legados:**
  - **Status:** Pendente.
  - **Locais:** `analysis_correlation.py`, `analysis_descriptive.py`.
  - **Descrição:** Otimizar os cálculos e a geração de plots nesses módulos, alinhando-os com a nova estrutura parametrizada.

- [ ] **Revisar e Remover Código Obsoleto:**
  - **Status:** Pendente.
  - **Descrição:** Realizar uma varredura completa no projeto para identificar e remover scripts, funções e módulos que não são mais utilizados após a refatoração.
