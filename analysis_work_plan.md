# Plano de Trabalho para Análise de Séries Temporais Multi-Tenant

O objetivo é investigar a co-variação, relações causais e flutuações temporais das métricas entre diferentes tenants e fases experimentais (baseline, ataque, recuperação), utilizando ferramentas estatísticas básicas, interpretáveis e confiáveis.

## Status do Projeto (Atualizado em Junho/2025)

- ✅ **Concluído**: Estrutura principal do projeto implementada, ingestão de dados, segmentação, persistência, componentes de análise descritiva, correlação e causalidade básicos.
- 🔄 **Em andamento**: Refinamento do módulo de Causalidade com Transfer Entropy, testes unitários completos.
- ❌ **Pendente**: Consolidação da metodologia de análise inter-tenant, relatórios comparativos entre fases experimentais, documentação detalhada das escolhas metodológicas, análises com janelas móveis.

## Diretrizes Gerais para o Desenvolvimento do Pipeline

- O DataFrame consolidado em formato "long" deve ser a única fonte de verdade para todas as análises. Todas as transformações e segmentações devem partir dele.
- Todas as funções de carregamento, transformação e exportação de dados devem ser modulares, testáveis e possuir logging detalhado de etapas e erros.
- O pipeline deve ser desenvolvido de forma incremental: comece com um subconjunto pequeno de dados/métricas, garanta testes e outputs corretos, depois expanda.
- O código deve ser escrito em inglês (nomes, docstrings, comentários). Documentação geral pode ser em português.
- Toda lógica de ingestão, validação e transformação de dados deve ser separada da lógica de análise e visualização.
- Recomenda-se a criação de um módulo central de ingestão de dados (ex: `data_ingestion.py`) responsável por:
    - Navegar na estrutura de diretórios.
    - Carregar arquivos CSV.
    - Validar e padronizar colunas e tipos.
    - Gerar logs de inconsistências e erros.
    - Retornar o DataFrame "long" padronizado.
- Implementar testes unitários para cada etapa crítica (carregamento, transformação, análise, visualização).
- Utilizar um arquivo de configuração central (ex: `config.yaml` ou `.py`) para caminhos, métricas, parâmetros de análise e exportação.
- Outputs (plots, tabelas, arquivos) devem ser versionados e organizados por experimento, round e fase.
- Documentar todas as decisões metodológicas e parâmetros relevantes em arquivos Markdown.

## Diretrizes para Estruturação e Persistência de DataFrames

- O DataFrame consolidado em formato "long" é o padrão e fonte única de verdade para todo o pipeline. Todas as operações de ingestão, validação, limpeza e transformação devem resultar nesse formato.
- Subdatasets no formato "wide" podem ser gerados sob demanda, a partir do DataFrame long, para análises específicas (correlação, causalidade, visualizações comparativas), mas nunca devem substituir o long como fonte principal.
- Recomenda-se fortemente a persistência dos DataFrames processados (long e, se necessário, wide) em formatos eficientes e portáveis (Parquet preferencialmente, ou CSV/Feather), organizados por experimento, round e fase. Isso facilita reuso, integração com notebooks (Jupyter) e compartilhamento com outros times ou ferramentas.
- O pipeline deve prover funções utilitárias para salvar e carregar datasets processados, garantindo reprodutibilidade e agilidade no desenvolvimento.

## Fase 1: Preparação e Estratégia de Dados

1.  **Definição da Estrutura dos Dados de Entrada:**
    *   1.1. ✅ Confirmar e documentar as colunas essenciais no DataFrame de entrada (ex: `timestamp`, `tenant_id`, `metric_name`, `metric_value`, `experimental_phase`). 
    *   1.2. ✅ Validar os tipos de dados e a consistência dos identificadores (tenants, métricas, fases). 

2.  **Estratégia de Carregamento, Formatos de DataFrame e Segmentação de Dados:**
    *   2.1. ✅ Implementar ou refinar a lógica para carregar os dados brutos e consolidá-los em um DataFrame principal em **formato "long"**. 
        *   Este DataFrame conterá colunas como: `timestamp`, `metric_value`, `metric_name`, `tenant_id`, `experimental_phase`, `round_id`, `experiment_id`. 
    *   2.2. ✅ Desenvolver funções para segmentar o DataFrame principal (formato "long") por: 
        *   Fase experimental (Baseline, Ataque, Recuperação).
        *   Tenant individual.
        *   Métrica específica.
        *   Combinações destes (ex: dados de uma métrica específica para um tenant em uma fase).
    *   2.3. ✅ **Estratégia para DataFrames em formato \"wide\":**
        *   Serão gerados em memória, dinamicamente, a partir do DataFrame \"long\", conforme a necessidade de cada módulo de análise (ex: para cálculo de correlações ou causalidade entre múltiplas séries temporais).
        *   Estes DataFrames \\\"wide\\\" também serão considerados para exportação, permitindo análises futuras ou o uso por parsers específicos.
        *   ✅ **Implementar `get_wide_format_for_analysis` para transformar dados longos em largos para uma métrica, fase e round específicos. 
    *   2.4. ✅ **Exportação dos DataFrames Processados:**
        *   Implementar a funcionalidade para exportar o DataFrame consolidado em formato "long" (após carregamento e pré-processamento inicial) para um formato de arquivo eficiente (ex: Parquet). 
            *   Objetivo: otimizar o desempenho em análises subsequentes e facilitar a interoperabilidade com outras ferramentas ou processos.
        *   Implementar a funcionalidade para exportar os DataFrames em formato "wide" gerados para formatos de arquivo eficientes (ex: Parquet ou CSV, a ser definido). 
            *   Objetivo: permitir análises futuras ou o uso por parsers específicos que possam necessitar deste formato.

3.  **Configuração e Aplicação de Otimização de Dados:**
    *   3.1. 🔄 Revisar e ajustar otimização para análise descritiva. 
    *   3.2. 🔄 Revisar e ajustar otimização para análise de correlação. 
    *   3.3. 🔄 Revisar e ajustar otimização para análise de causalidade. 
4.  **Definição do Processo de Seleção de Variáveis e Pares para Análise:**
    *   4.1. ✅ Estabelecer critérios para selecionar as métricas de interesse (ex: CPU, memória, latência). 
    *   4.2. ✅ Definir como os pares de tenants serão selecionados para análises comparativas (inter-tenant). 

# Sequência Recomendada para o Desenvolvimento do Pipeline

## 1. Estruturação Inicial e Organização do Projeto

1. **Definição e Criação da Estrutura de Diretórios:**
    - `data/raw/` — Dados brutos extraídos (ex: CSVs originais).
    - `data/processed/` — DataFrames processados (long/wide) em Parquet/CSV.
    - `outputs/plots/` — Gráficos gerados por fase, experimento, round.
    - `outputs/tables/` — Tabelas e resumos exportados.
    - `notebooks/` — Jupyter Notebooks para exploração e validação.
    - `config/` — Arquivos de configuração (YAML, JSON, etc).
    - `logs/` — Logs de execução e validação.
    - `src/` — Código-fonte do pipeline (módulos de ingestão, análise, visualização, etc).

## 2. Desenvolvimento Incremental do Pipeline

### 2.1. Funcionalidades Fundamentais

1. **Configuração Centralizada:**
    - Criar arquivo de configuração (ex: `config.yaml`) para caminhos, métricas, parâmetros globais.
2. **Ingestão e Validação de Dados:**
    - Implementar módulo/função para navegar na estrutura de diretórios, carregar CSVs, validar e padronizar colunas/tipos.
    - Gerar logs de inconsistências e erros.
    - Consolidar tudo em um DataFrame long padronizado.
3. **Persistência de DataFrames:**
    - Implementar funções utilitárias para salvar/carregar DataFrames long (e wide, se necessário) em Parquet/CSV.
    - Organizar arquivos por experimento, round e fase.
4. **Testes Unitários Básicos:**
    - Testar ingestão, validação e persistência.

### 2.2. Segmentação e Exportação

1. **Funções de Segmentação:**
    - Permitir filtragem do DataFrame long por fase, tenant, métrica, etc.
2. **Geração de DataFrames Wide sob Demanda:**
    - Implementar função para converter long→wide para análises específicas.
3. **Exportação de Subdatasets:**
    - Exportar subconjuntos relevantes para uso em notebooks ou outras ferramentas.

### 2.3. Análise Descritiva

1. **Módulo Descritivo:**
    - Calcular estatísticas básicas (média, desvio padrão, skewness, kurtosis) por série temporal.
    - Gerar plots simples (séries temporais, histogramas, ACF).
    - Exportar tabelas resumo.
2. **Testes Unitários:**
    - Testar cálculos e geração de plots/tabelas.

### 2.4. Módulos Avançados

1. **Correlação e Covariância:**
    - Implementar cálculos e visualizações de correlação/covariância entre tenants/métricas.
2. **Causalidade:**
    - Implementar testes de Granger e Transfer Entropy, visualizações e tabelas.
3. **Comparação entre Fases:**
    - Lógica para comparar resultados entre baseline, ataque e recuperação.
4. **Janelas Móveis (Opcional):**
    - Adaptação dos módulos para análises com janelas móveis.

### 2.5. Consolidação, Relatórios e Iteração

1. **Agregação de Insights:**
    - Combinar resultados dos módulos para formar narrativa coesa.
2. **Tabela Final Comparativa:**
    - Gerar ranking e métricas de influência inter-tenant.
3. **Documentação e Justificativas:**
    - Registrar parâmetros, decisões e métodos.
4. **Execução, Debugging e Iteração:**
    - Rodar pipeline completo, refinar e iterar conforme resultados.

## Fase 2: Implementação Detalhada dos Módulos de Análise

Cada módulo seguirá a arquitetura `BaseModule`, `BaseAnalyzer`, `BaseVisualizer`.

**2.1. Módulo de Análise Descritiva (em `src/analysis_descriptive.py`)**
    *   2.1.1. **Estrutura do Módulo:**
        *   ✅ Implementar funções de análise descritiva.
        *   ✅ Implementar funções de visualização.
    *   2.1.2. **Cálculos de Estatísticas Descritivas:** 
        *   ✅ Implementar função para calcular estatísticas descritivas (média, desvio padrão, etc.) por série temporal (tenant, métrica, fase).
    *   2.1.3. **Visualizações:** 
        *   ✅ Implementar função para gerar plots de séries temporais individuais. 
        *   ✅ Implementar função para gerar plots de autocorrelação (ACF) para cada métrica/tenant. 
    *   2.1.4. **Tabelas de Resultados:** 
        *   ✅ Implementar função para gerar tabela resumo das estatísticas descritivas.
    *   2.1.5. **Testes Unitários:** 
        *   🔄 Implementados arquivos básicos de teste `test_analysis_descriptive.py`.
        *   🔄 Expandir cobertura de testes para casos edge.

**2.2. Módulo de Análise de Correlação e Covariância (em `src/analysis_correlation.py`)**
    *   2.2.1. **Estrutura do Módulo:**
        *   ✅ Implementar funções de análise de correlação.
        *   ✅ Implementar funções de visualização para correlação.
    *   2.2.2. **Cálculos de Correlação (por fase experimental):**
        *   ✅ Implementar função para calcular matrizes de correlação (Pearson, Kendall, Spearman) entre métricas de tenants distintos.
        *   ✅ Implementar função para calcular matriz de covariância (com dados padronizados).
        *   🔄 Implementar função para calcular Correlação Cruzada com Defasagem (CCF) entre pares de séries.
    *   2.2.3. **Visualizações (por fase experimental):**
        *   ✅ Implementar função para gerar heatmaps das matrizes de correlação.
        *   ✅ Implementar função para gerar heatmap da matriz de covariância padronizada.
        *   🔄 Implementar função para gerar gráficos de CCF.
        *   🔄 Implementar função para gerar lag plots.
    *   2.2.4. **Tabelas de Resultados:**
        *   ✅ Implementar função para gerar tabelas das matrizes de correlação e covariância.
    *   2.2.5. **Testes Unitários:**
        *   🔄 Implementados arquivos básicos de teste `test_analysis_correlation.py`.
        *   🔄 Expandir cobertura de testes para casos edge.

**2.3. Módulo de Análise de Causalidade (em `src/analysis_causality.py`)**
    *   2.3.1. **Estrutura do Módulo (Reconstrução/Criação):**
        *   ✅ Implementar funções para análise de causalidade abrangendo Granger e Transfer Entropy.
        *   ✅ Implementar funções de visualização para causalidade.
        *   ✅ Integração com outras funcionalidades do pipeline.
    *   2.3.2. **Implementação da Análise de Causalidade de Granger:**
        *   ✅ Implementar função para aplicar testes de Causalidade de Granger para pares de séries temporais.
        *   ✅ Integrar lógica para determinar `max_lags` (pode usar CCF do módulo de correlação ou critérios como AIC/BIC).
        *   ✅ Assegurar a coleta e o armazenamento adequado dos p-values e estatísticas do teste.
    *   2.3.3. **Implementação da Análise de Transfer Entropy:**
        *   🔄 Selecionar e integrar a biblioteca Python para Transfer Entropy (ex: `pyinform`, ou outra). Adicionar ao `requirements.txt`.
        *   🔄 Implementar função para calcular Transfer Entropy para pares de séries temporais.
        *   🔄 Assegurar a coleta e o armazenamento adequado dos valores de TE.
    *   2.3.4. **Implementação das Visualizações:**
        *   ✅ Implementar função para gerar plots dos resultados da Causalidade de Granger (ex: heatmap de p-values).
        *   🔄 Implementar função para gerar plots dos resultados da Transfer Entropy (ex: heatmap de valores TE).
        *   ✅ Implementar função para gerar visualização em grafo (usando NetworkX) para ilustrar relações de influência.
        *   🔄 Garantir que a legenda dos grafos multi-métrica seja contextual e automática: priorizar p-valor (Granger real) se disponível, senão TE, para máxima clareza interpretativa.
        *   ✅ Outputs organizados e reprodutíveis, com legendas e títulos informativos.
    *   2.3.5. **Geração de Tabelas de Resultados:**
        *   ✅ Implementar função para criar tabela consolidada de scores de causalidade (p-values Granger).
        *   🔄 Implementar função para criar uma matriz de influência cruzada resumida com scores de TE.
    *   2.3.6. **Testes Unitários e Integração:**
        *   🔄 Implementados arquivos básicos de teste `test_analysis_causality.py`.
        *   ❌ Falta testar funções de cálculo de TE com dados sintéticos ou subconjuntos.
        *   🔄 Testar geração de plots e tabelas para TE.

## Fase 3: Consolidação, Interpretação e Geração de Relatórios

1.  **Desenvolvimento da Metodologia de Agregação de Insights:**
    *   3.1.1. ❌ Definir como os resultados das análises descritiva, de correlação/covariância e de causalidade serão combinados para formar uma narrativa coesa.
    *   3.1.2. ❌ Estabelecer critérios para identificar o "tenant barulhento" e quantificar sua influência.
2.  **Implementação da Geração da Tabela Final de Comparativo Inter-Tenant:**
    *   3.2.1. ❌ Projetar a estrutura da tabela final, incluindo as métricas de influência e ranking.
    *   3.2.2. ❌ Implementar a lógica para popular esta tabela, utilizando os resultados armazenados pelos Analyzers.
3.  **Documentação Detalhada das Escolhas Metodológicas:**
    *   3.3.1. 🔄 Registrar todos os parâmetros utilizados (ex: `max_lags` para Granger, limiares de significância, janelas de CCF).
    *   3.3.2. ❌ Justificar as escolhas de bibliotecas e métodos.
4.  **Análise Comparativa dos Resultados Entre Fases Experimentais:**
    *   3.4.1. ❌ Implementar lógica para comparar os resultados (correlações, causalidade, etc.) entre as fases de baseline, ataque e recuperação.
    *   3.4.2. ❌ Preparar visualizações que destaquem essas mudanças.
5.  **Avaliação e Implementação (Opcional) de Análises com Janelas Móveis:**
    *   3.5.1. ❌ Se decidido, adaptar os módulos de Correlação e Causalidade para operar com janelas móveis.
    *   3.5.2. ❌ Definir o tamanho da janela e o passo (step).
    *   3.5.3. ❌ Implementar visualizações para os resultados de janelas móveis (ex: evolução da correlação/causalidade ao longo do tempo).
    *   3.5.4. ❌ Realizar testes específicos para as funcionalidades de janelas móveis.

## Fase 4: Execução, Debugging e Iteração

1.  **Configuração do Script Principal de Análise:**
    *   4.1.1. ✅ Garantir que o script (`main.py`) possa carregar os dados, instanciar os módulos de análise e executar as análises em sequência.
    *   4.1.2. ✅ Implementar a lógica para salvar todos os outputs (plots, tabelas) de forma organizada.
2.  **Execução com Dados de Demonstração (`demo-data`):**
    *   4.2.1. ✅ Rodar o pipeline completo com os dados de demonstração.
    *   4.2.2. 🔄 Verificar a corretude dos resultados parciais e finais.
3.  **Debugging e Refinamento dos Módulos:**
    *   4.3.1. 🔄 Corrigir bugs identificados durante a execução.
    *   4.3.2. 🔄 Refinar parâmetros e lógicas com base nos resultados observados.
4.  **Análise dos Resultados Iniciais e Iteração:**
    *   4.4.1. 🔄 Interpretar os primeiros resultados completos.
    *   4.4.2. 🔄 Com base na interpretação, decidir sobre ajustes nos métodos, parâmetros ou visualizações.
    *   4.4.3. 🔄 Repetir etapas de execução e análise conforme necessário.

## Observações e Filosofia da Análise Inter-Tenant

- A identificação de tenants que causam contenção de recursos ou degradação será feita exclusivamente de forma data-driven, a partir dos resultados das técnicas estatísticas e de causalidade implementadas (correlação, causalidade, influência cruzada, etc.).
- Não há pré-julgamento sobre quem é o "tenant barulhento" ou malicioso: a descoberta será imparcial e baseada em evidências extraídas dos dados.
- A ausência de determinados tenants em certas fases (baseline, ataque, recuperação) é esperada e deve ser tratada naturalmente pelo pipeline, sem gerar erro ou viés. Todas as análises devem considerar apenas os tenants presentes em cada contexto/fase.
- O pipeline deve ser robusto para lidar com a presença/ausência de tenants e arquivos de métricas em cada fase, e as funções de ingestão devem registrar (logar) essas ausências para rastreabilidade.

## Sugestão para o Desenvolvimento do Pipeline

- Desenvolva o pipeline de forma incremental, começando pela ingestão e validação dos dados, garantindo que o DataFrame long seja corretamente consolidado mesmo com ausências de tenants/fases.
- Implemente funções utilitárias para:
    - Listar todos os tenants presentes em cada fase/round/experimento.
    - Registrar ausências de tenants ou métricas esperadas (logging).
    - Gerar DataFrames segmentados por fase, tenant, métrica, etc.
- Priorize a modularidade: separe claramente ingestão, transformação, análise e visualização.
- Implemente testes unitários para garantir que a ingestão lida corretamente com casos de ausência de dados.
- Considere criar um notebook de exploração inicial para validar a consolidação dos dados e a robustez do pipeline antes de avançar para análises mais complexas.

## Estrutura de Dados de Entrada (Alinhada ao Experimento Original)

- Os dados de entrada devem ser organizados conforme exportação do experimento original, seguindo a estrutura de diretórios encontrada em `demo-data/`.
- Cada experimento pode conter um ou múltiplos rounds (ex: `demo-experiment-1-round/`, `demo-experiment-3-rounds/`).
- Dentro de cada round (`round-1/`, `round-2/`, etc.), existem três fases sequenciais: `1 - Baseline/`, `2 - Attack/`, `3 - Recovery/`.
- Dentro de cada fase, os subdiretórios representam os tenants (ex: `tenant-a/`, `tenant-b/`, ...), além de possíveis diretórios auxiliares (ex: `ingress-nginx/`, `active/`, etc.).
- Cada diretório de tenant contém arquivos CSV de métricas, cada um com duas colunas: `timestamp,value`.
- O pipeline deve:
    - Navegar recursivamente por todos os experimentos, rounds e fases.
    - Identificar corretamente o experimento, round, fase e tenant para cada arquivo de métrica.
    - Ignorar diretórios que não seguem o padrão de tenant (ex: `ingress-nginx/`, `active/`, etc.), conforme critérios definidos no contexto do projeto.
    - Consolidar todos os dados em um DataFrame long padronizado, adicionando as colunas: `timestamp`, `metric_value`, `metric_name`, `tenant_id`, `experimental_phase`, `round_id`, `experiment_id`.
    - Garantir a consistência dos tipos e valores categóricos durante a ingestão.
- Essa lógica deve ser implementada no módulo central de ingestão de dados, garantindo flexibilidade para diferentes estruturas de experimentos e rounds.

## Lacunas e Oportunidades de Melhoria (Adicionado em Junho/2025)

Após análise da implementação atual e comparação com o plano original, foram identificadas as seguintes lacunas e oportunidades de melhoria:

1. **Implementação do Transfer Entropy**:
   - ✅ Estrutura base implementada no módulo `analysis_causality.py`
   - 🔄 A aplicação do Transfer Entropy está em andamento, conforme evidenciado pelo arquivo `debug_te_attack.out` 
   - ❌ Falta integrar plenamente a biblioteca para cálculos de TE (ex: `pyinform`) com documentação adequada
   - ❌ Necessidade de refinamento da visualização contextual dos grafos de causalidade

2. **Completude dos Testes**:
   - 🔄 Arquivos de teste criados (`test_analysis_causality.py`, `test_analysis_correlation.py`, etc.)
   - ❌ Cobertura de testes insuficiente para garantir robustez do pipeline
   - ❌ Falta testes para casos extremos (ausência de dados, inconsistências)

3. **Análises Comparativas entre Fases**:
   - ✅ Pipeline gera visualizações separadas por fase experimental
   - ❌ Falta implementação da lógica para comparar resultados entre fases
   - ❌ Ausência de visualizações específicas para comparação de baseline/ataque/recuperação

4. **Documentação das Escolhas Metodológicas**:
   - 🔄 Estrutura básica de arquivos Markdown criada
   - ❌ Falta registro detalhado de parâmetros estatísticos utilizados
   - ❌ Ausência de justificativas para escolhas metodológicas

5. **Relatórios e Consolidação de Insights**:
   - ❌ Falta implementação da metodologia de agregação de insights
   - ❌ Ausência de tabela final comparativa inter-tenant
   - ❌ Necessidade de estruturar relatórios automatizados

6. **Janelas Móveis**:
   - ❌ Fase avançada não iniciada
   - ❌ Adaptação para análises temporais dinâmicas

7. **Dependências e Integração**:
   - ❌ `NetworkX` não está no `requirements.txt` mas é usado para visualizações em grafo
   - ❌ Biblioteca para Transfer Entropy não especificada no `requirements.txt`

## Prioridades para Próximos Passos (Junho/2025)

As seguintes prioridades foram identificadas para concluir o projeto com sucesso:

### Prioridade Alta (Imediata) ✅
1. **Completar Implementação do Transfer Entropy** ✅:
   - ✅ Finalizar integração da biblioteca de TE
   - ✅ Garantir armazenamento adequado dos valores
   - ✅ Completar testes unitários específicos para TE

2. **Atualizar `requirements.txt`** ✅:
   - ✅ Adicionar `networkx` e biblioteca para TE (ex: `pyinform`)
   - ✅ Especificar versões compatíveis

3. **Consolidar Testes Unitários Críticos** ✅:
   - ✅ Focar em testes para ingestão de dados, causalidade e exportação
   - ✅ Garantir cobertura para casos edge de ausência de dados

### Prioridade Média (Próximas 2-3 semanas) ✅
1. **Implementar Comparação entre Fases Experimentais** ✅:
   - ✅ Desenvolver lógica para comparar métricas entre baseline/ataque/recovery
   - ✅ Criar visualizações específicas para destacar mudanças

2. **Desenvolver Metodologia de Agregação de Insights** ✅:
   - ✅ Definir e implementar critérios para identificação de "tenant barulhento"
   - ✅ Estruturar tabela final comparativa

3. **Documentar Escolhas Metodológicas** ✅:
   - ✅ Registrar parâmetros utilizados (ex: `max_lags`, thresholds)
   - ✅ Justificar escolhas de bibliotecas e métodos

### Prioridade Baixa (Após concluir anteriores) 🔄
1. **Análises com Janelas Móveis** ✅:
   - ✅ Adaptar módulos para análise temporal dinâmica
   - ✅ Implementar visualizações específicas
   - ✅ Testar e validar a execução completa do pipeline com janelas móveis

2. **Refinamentos Estéticos e Usabilidade** 🔄:
   - ✅ Melhorar formatação de gráficos (estilo tableau-colorblind10)
   - 🔄 Adicionar opções de personalização de visualizações
   - ✅ Aprimorar mensagens de log e feedback

3. **Documentação para Usuários Finais** 🔄:
   - 🔄 Tutorial de uso do pipeline
   - 🔄 Guia de interpretação dos resultados

