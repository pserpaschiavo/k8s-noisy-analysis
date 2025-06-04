# Plano de Trabalho para Análise de Séries Temporais Multi-Tenant

O objetivo é investigar a co-variação, relações causais e flutuações temporais das métricas entre diferentes tenants e fases experimentais (baseline, ataque, recuperação), utilizando ferramentas estatísticas básicas, interpretáveis e confiáveis.

## Status do Projeto (Atualizado em 4 de Junho/2025)

- ✅ **Concluído**: Estrutura principal do projeto implementada, ingestão de dados (incluindo suporte a carregamento direto de Parquet), segmentação, persistência, componentes de análise descritiva, correlação e causalidade básicos, agregação de insights, análise multi-round, ingestão direta de arquivos parquet com resolução de caminhos relativos e absolutos. Correções de erros no pipeline, incluindo problemas com dict comparisons no teste de Granger, uso obsoleto de Series.fillna no módulo de causalidade e problemas no estágio de agregação de insights.
- 🔄 **Em andamento**: Refinamento do módulo de Causalidade com Transfer Entropy, testes unitários completos, análises com janelas móveis, documentação detalhada, visualizações ausentes/incompletas.
- ❌ **Pendente**: Relatórios comparativos entre fases experimentais, integração completa de todos os componentes, documentação para usuários finais.

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
- ✅ **Suporte a ingestão direta de parquet**: O pipeline agora oferece suporte para ingerir diretamente um arquivo parquet criado em análises anteriores, economizando tempo de processamento e permitindo retomar análises de onde pararam. Foi implementada uma hierarquia de prioridades para carregamento de dados: (1) arquivo parquet específico via configuração, (2) arquivo parquet consolidado existente, (3) processamento de dados brutos.

## Sistema de Configuração YAML Abrangente

Para garantir uma configuração flexível e completa do pipeline, implementou-se um sistema baseado em arquivos YAML capazes de controlar todos os aspectos da análise:

1. **Configuração de Ingestão de Dados**
   - ✅ `data_root`: Caminho base para os dados brutos
   - ✅ `processed_data_dir`: Diretório para dados processados
   - ✅ `output_dir`: Diretório para outputs (plots, relatórios)
   - ✅ `input_parquet_path`: Caminho para um arquivo parquet existente
   - ✅ `output_parquet_name`: Nome personalizado para o arquivo parquet gerado
   - ✅ `selected_metrics`, `selected_tenants`, `selected_rounds`: Filtros de dados

2. **Configuração de Parâmetros de Análise**
   - ✅ `correlation.methods`: Métodos de correlação a serem usados
   - ✅ `causality.granger_max_lag`: Lag máximo para teste de Granger
   - ✅ `causality.granger_threshold`: Threshold para significância estatística
   - ✅ `causality.transfer_entropy_bins`: Número de bins para cálculo de Transfer Entropy

3. **Configuração de Visualização**
   - ✅ `plots.figure_size`: Dimensões das figuras geradas
   - ✅ `plots.style`: Estilo de visualização (ex: tableau-colorblind10)
   - ✅ `plots.dpi`: Resolução das figuras

4. **Configuração de Análise Avançada**
   - ✅ `sliding_window.window_size`: Tamanho da janela para análise deslizante
   - ✅ `sliding_window.step_size`: Passo para janela deslizante
   - ✅ `multi_round.consistency_threshold`: Threshold para consistência entre rounds

5. **Extensibilidade**
   - ✅ O sistema de parse de configuração está estruturado para permitir fácil adição de novos parâmetros
   - ✅ Cada módulo de análise pode acessar suas configurações específicas do arquivo YAML

Este sistema de configuração centralizada via YAML oferece vantagens significativas:
1. Reprodutibilidade: todos os parâmetros usados em uma análise podem ser documentados e reutilizados
2. Flexibilidade: alteração de parâmetros sem necessidade de editar código
3. Experimentação: facilidade para testar diferentes configurações
4. Transparência: todos os parâmetros explícitos em um único arquivo

## Guia de Uso do Sistema de Configuração YAML

Para maximizar a flexibilidade e reprodutibilidade das análises, o pipeline foi projetado para ser amplamente configurável via arquivos YAML. Este guia descreve como utilizar esse sistema de configuração.

### Arquivo de Configuração Padrão

O arquivo de configuração principal está localizado em `config/pipeline_config.yaml` e contém todas as configurações necessárias para controlar o comportamento do pipeline. Abaixo está um exemplo de configuração completa:

```yaml
# Configuração do Pipeline de Análise Multi-Tenant

# Diretórios de entrada/saída
data_root: /path/to/data/demo-experiment-1-round
processed_data_dir: /path/to/data/processed
output_dir: /path/to/outputs

# Configuração de ingestão de dados
input_parquet_path: /path/to/existing/dataframe.parquet  # Opcional: usar arquivo parquet existente
output_parquet_name: consolidated_long.parquet  # Nome do arquivo parquet a ser gerado

# Seleção de dados
selected_metrics:
  - cpu_usage
  - memory_usage
  
selected_tenants:
  - tenant-a
  - tenant-b
  - tenant-c
  - tenant-d
  
selected_rounds:
  - round-1

# Parâmetros de análise
correlation:
  methods:
    - pearson
    - spearman
  time_lag: 5  # Lag máximo para correlação cruzada (minutos)
  
causality:
  granger_max_lag: 5
  granger_threshold: 0.05
  transfer_entropy_bins: 8
  
# Opções de visualização
plots:
  figure_size: [10, 6]
  style: tableau-colorblind10
  dpi: 300
  
# Configuração para análise com janelas deslizantes
sliding_window:
  window_size: 5min
  step_size: 1min
  
# Configuração para análise multi-round
multi_round:
  consistency_threshold: 0.7
  robustness_threshold: 0.05
```

### Comandos de Execução

O pipeline pode ser executado utilizando diferentes scripts, todos configuráveis via YAML:

1. **Pipeline Padrão (processamento de dados brutos):**
   ```bash
   python run_pipeline.py --config config/pipeline_config.yaml
   ```

2. **Pipeline com Ingestão de Parquet Existente:**
   ```bash
   # Via configuração YAML (adicionar ao arquivo config/pipeline_config.yaml):
   # input_parquet_path: /path/to/existing/dataframe.parquet
   # output_parquet_name: custom_output_name.parquet
   
   # Ou via argumento de linha de comando (sobrescreve a configuração YAML):
   python run_pipeline.py --config config/pipeline_config.yaml --input-parquet-path /path/to/existing.parquet
   ```

3. **Pipeline Unificado (todos os tipos de análises):**
   ```bash
   python -m src.run_unified_pipeline --config config/pipeline_config.yaml
   ```

4. **Execução com Seleção Específica via CLI:**
   ```bash
   python run_pipeline.py --config config/pipeline_config.yaml --selected-metrics cpu_usage memory_usage --selected-tenants tenant-a tenant-c
   ```

5. **Testar Funcionalidade de Carregamento de Parquet:**
   ```bash
   # Teste específico para validar a ingestão de parquet:
   python test_parquet_loading.py
   ```

### Extensão do Sistema de Configuração

O sistema de configuração foi projetado para ser facilmente extensível. Para adicionar novos parâmetros:

1. **Adicionar ao Arquivo YAML:**
   Basta adicionar a nova chave e valor ao arquivo `pipeline_config.yaml`.

2. **Atualizar o Parser de Configuração:**
   Adicionar uma função getter em `parse_config.py`:
   ```python
   def get_new_parameter(config: dict) -> Optional[Any]:
       return config.get('new_parameter')
   ```

3. **Acessar no Código:**
   ```python
   new_parameter = context.get('config', {}).get('new_parameter', default_value)
   ```

Este sistema de configuração via YAML oferece uma maneira robusta, flexível e transparente de controlar todos os aspectos do pipeline, desde a ingestão de dados até a geração de visualizações e relatórios.

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
    *   2.5. ✅ **Ingestão Direta de DataFrames Parquet:**
        *   ✅ Implementada função `load_from_parquet()` no módulo `data_ingestion.py` para carregar diretamente arquivos parquet de análises anteriores.
        *   ✅ Adicionado parâmetro `input_parquet_path` ao arquivo de configuração YAML para especificar o caminho do parquet a ser carregado.
        *   ✅ Implementado parâmetro `output_parquet_name` para controlar o nome do arquivo parquet gerado pelo pipeline.
        *   ✅ Atualizado `DataIngestionStage` para primeiro tentar carregar do arquivo parquet especificado, depois verificar se há um arquivo consolidado existente e, por fim, processar dados brutos.
        *   ✅ Adicionado argumento de linha de comando `--input-parquet-path` em todos os scripts relevantes.
        *   ✅ Adicionadas funções `get_input_parquet_path()` e `get_output_parquet_name()` em `parse_config.py` para gerenciar configurações relacionadas a Parquet.
        *   ✅ Implementada lógica de resolução de caminhos para lidar com caminhos relativos e absolutos.
        *   ✅ Testada e validada a funcionalidade através do script `test_parquet_loading.py`.

3.  **Configuração e Aplicação de Otimização de Dados:**
    *   3.1. ✅ Revisar e ajustar otimização para análise descritiva. 
    *   3.2. ✅ Revisar e ajustar otimização para análise de correlação. 
    *   3.3. ✅ Revisar e ajustar otimização para análise de causalidade. 
4.  **Definição do Processo de Seleção de Variáveis e Pares para Análise:**
    *   4.1. ✅ Estabelecer critérios para selecionar as métricas de interesse (ex: CPU, memória, latência). 
    *   4.2. ✅ Definir como os pares de tenants serão selecionados para análises comparativas (inter-tenant). 

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
        *   ✅ Implementar função para calcular Correlação Cruzada com Defasagem (CCF) entre pares de séries.
    *   2.2.3. **Visualizações (por fase experimental):**
        *   ✅ Implementar função para gerar heatmaps das matrizes de correlação.
        *   ✅ Implementar função para gerar heatmap da matriz de covariância padronizada.
        *   ✅ Implementar função para gerar gráficos de CCF.
        *   ✅ Implementar função para gerar lag plots.
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
        *   ✅ Selecionar e integrar a biblioteca Python para Transfer Entropy (ex: `pyinform`, ou outra). Adicionar ao `requirements.txt`.
        *   ✅ Implementar função para calcular Transfer Entropy para pares de séries temporais.
        *   ✅ Assegurar a coleta e o armazenamento adequado dos valores de TE.
    *   2.3.4. **Implementação das Visualizações:**
        *   ✅ Implementar função para gerar plots dos resultados da Causalidade de Granger (ex: heatmap de p-values).
        *   ✅ Implementar função para gerar plots dos resultados da Transfer Entropy (ex: heatmap de valores TE).
        *   ✅ Implementar função para gerar visualização em grafo (usando NetworkX) para ilustrar relações de influência.
        *   ✅ Garantir que a legenda dos grafos multi-métrica seja contextual e automática: priorizar p-valor (Granger real) se disponível, senão TE, para máxima clareza interpretativa.
        *   ✅ Outputs organizados e reprodutíveis, com legendas e títulos informativos.
    *   2.3.5. **Geração de Tabelas de Resultados:**
        *   ✅ Implementar função para criar tabela consolidada de scores de causalidade (p-values Granger).
        *   ✅ Implementar função para criar uma matriz de influência cruzada resumida com scores de TE.
    *   2.3.6. **Testes Unitários e Integração:**
        *   🔄 Implementados arquivos básicos de teste `test_analysis_causality.py`.
        *   ❌ Falta testar funções de cálculo de TE com dados sintéticos ou subconjuntos.
        *   🔄 Testar geração de plots e tabelas para TE.

## Fase 3: Consolidação, Interpretação e Geração de Relatórios

1.  **Desenvolvimento da Metodologia de Agregação de Insights:**
    *   3.1.1. ✅ Definir como os resultados das análises descritiva, de correlação/covariância e de causalidade serão combinados para formar uma narrativa coesa.
    *   3.1.2. ✅ Estabelecer critérios para identificar o "tenant barulhento" e quantificar sua influência.
2.  **Implementação da Geração da Tabela Final de Comparativo Inter-Tenant:**
    *   3.2.1. ✅ Projetar a estrutura da tabela final, incluindo as métricas de influência e ranking.
    *   3.2.2. ✅ Implementar a lógica para popular esta tabela, utilizando os resultados armazenados pelos Analyzers.
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
6.  **Análise Consolidada para Experimentos Multi-Round:**
    *   3.6.1. 🔄 Implementar metodologia de análise de consistência entre rounds para identificar padrões persistentes vs. pontuais.
    *   3.6.2. 🔄 Desenvolver análise de robustez de causalidade para distinguir relações causais robustas de correlações espúrias.
    *   3.6.3. 🔄 Criar sistema de análise de divergência de comportamento para identificar rounds anômalos.
    *   3.6.4. 🔄 Implementar agregação de consenso para produzir veredictos consolidados sobre o comportamento do sistema.
    *   3.6.5. 🔄 Desenvolver visualizações de consistência entre rounds (gráficos com intervalos de confiança, heatmaps, dendrogramas).

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

## Lacunas e Oportunidades de Melhoria (Atualizado em 4 de Junho/2025)

Após análise da implementação atual e comparação com o plano original, foram identificadas as seguintes lacunas e oportunidades de melhoria:

1. **Implementação do Transfer Entropy**:
   - ✅ Estrutura base implementada no módulo `analysis_causality.py`
   - ❌ Necessidade de refinamento da visualização contextual dos grafos de causalidade

2. **Completude dos Testes**:
   - 🔄 Arquivos de teste criados (`test_analysis_causality.py`, `test_analysis_correlation.py`, etc.)
   - ❌ Falta testes para casos extremos (ausência de dados, inconsistências)

3. **Análises Comparativas entre Fases**:
   - ✅ Pipeline gera visualizações separadas por fase experimental
   - ❌ Ausência de visualizações específicas para comparação de baseline/ataque/recuperação

4. **Documentação das Escolhas Metodológicas**:
   - 🔄 Estrutura básica de arquivos Markdown criada
   - ❌ Ausência de justificativas para escolhas metodológicas

5. **Relatórios e Consolidação de Insights**:
   - ✅ Implementação da metodologia de agregação de insights
   - ✅ Estruturação de relatórios automatizados
   - ✅ Correção de erros críticos no estágio de agregação de insights para lidar com diferentes formatos de dados e evitar erros de tipo

6. **Janelas Móveis**:
   - ✅ Módulo implementado em `analysis_sliding_window.py` com funcionalidades completas
   - ✅ Disponível via pipeline dedicado (`pipeline_with_sliding_window.py`)
   - ✅ Correção do erro de comparação de dicionários na análise de causalidade de Granger
   - 🔄 Pipeline executando sem erros, mas algumas visualizações podem estar ausentes

7. **Análise Consolidada para Experimentos Multi-Round**:
   - ✅ Implementação de metodologias específicas para análise entre rounds
   - ✅ Avaliação de consistência entre diferentes execuções do experimento
   - ✅ Métricas de robustez para relações causais identificadas
   - 🔄 Visualizações implementadas e o pipeline está executando sem erros

8. **Dependências e Integração**:
   - ✅ `NetworkX` adicionado ao `requirements.txt` para visualizações em grafo
   - ✅ Biblioteca `pyinform` para Transfer Entropy especificada no `requirements.txt`
   - ✅ Correção do uso obsoleto de Series.fillna no módulo de causalidade
   
9. **Visualizações Ausentes/Incompletas**:
   - 🔄 Plots de correlação - pipeline executando sem erros, verificar geração
   - 🔄 Visualizações de séries temporais combinadas - pipeline executando sem erros, verificar geração
   - 🔄 Plots de detecção de anomalias - pipeline executando sem erros, verificar geração
   - 🔄 Visualizações de janelas deslizantes - pipeline executando sem erros, verificar geração

10. **Arquitetura do Pipeline**:
    - ❌ Múltiplas implementações de pipeline (`pipeline.py`, `pipeline_new.py`, `pipeline_with_sliding_window.py`)
    - ❌ Falta de sistema unificado para configuração e execução
    - ❌ Ausência de mecanismos de cache para evitar recálculos desnecessários

## Prioridades para Próximos Passos (Atualizado em 4 de Junho/2025)

As seguintes prioridades foram identificadas para concluir o projeto com sucesso:

### Prioridade Alta (Imediata)
1. **Verificar Geração de Visualizações** 🔄:
   - 🔄 Executar o pipeline unificado com todos os módulos habilitados para verificar a correta geração das visualizações
   - 🔄 Verificar se os plots de correlação estão sendo gerados corretamente
   - 🔄 Confirmar a geração de visualizações de séries temporais combinadas de todas as fases
   - 🔄 Confirmar a integração completa da detecção de anomalias ao fluxo principal do pipeline

2. **Completar Análise Multi-Round** 🔄:
   - 🔄 Verificar a integração completa do módulo `analysis_multi_round.py`
   - 🔄 Validar a geração de visualizações de consistência e robustez entre rounds
   - 🔄 Documentar resultados e insights gerados por esta análise

3. **Correções Críticas no Pipeline** ✅:
   - ✅ Desenvolver script utilitário para verificação da geração de todas as visualizações esperadas (`src/run_unified_pipeline.py`)
   - ✅ Corrigir erros críticos no estágio de agregação de insights
   - ✅ Corrigir erros de comparação de dicionários na análise de janelas deslizantes
   - ✅ Atualizar uso obsoleto de Series.fillna no módulo de causalidade

### Prioridade Média (Semanas 2-3 de Junho/2025)
1. **Consolidação da Arquitetura do Pipeline** ❌:
   - ❌ Unificar os múltiplos arquivos de pipeline em uma implementação modular baseada em plugins
   - ❌ Implementar sistema de configuração centralizado com validação
   - ❌ Desenvolver CLI unificada para controle granular da execução

2. **Documentação Técnica** ❌:
   - ❌ Documentar detalhadamente todas as visualizações geradas pelo sistema
   - ❌ Criar guia técnico sobre como adicionar novos tipos de análise ao pipeline
   - ❌ Documentar configurações e parâmetros disponíveis

3. **Refatoração de Código** ❌:
   - ❌ Padronizar interface dos diferentes módulos de análise
   - ❌ Melhorar sistema de logging para facilitar depuração
   - ❌ Remover código duplicado entre as diferentes implementações do pipeline

### Prioridade Baixa (Julho-Agosto/2025)
1. **Otimizações de Desempenho** ❌:
   - ❌ Implementar sistema de cache para resultados intermediários
   - ❌ Adicionar suporte para paralelização em estágios computacionalmente intensivos
   - ❌ Otimizar uso de memória para conjuntos de dados grandes

2. **Extensibilidade e Interface** ❌:
   - ❌ Desenvolver sistema de plugins para facilitar adição de novas análises
   - ❌ Considerar implementação de interface web simples para visualização de resultados
   - ❌ Criar mecanismos para exportação de resultados em diferentes formatos

3. **Testes e CI/CD** ❌:
   - ❌ Implementar testes unitários e de integração
   - ❌ Configurar pipeline de CI/CD para validação automática
   - ❌ Desenvolver casos de teste com diferentes configurações de experimentos
   - ✅ Estruturar tabela final comparativa
   - ✅ Implementar visualizações comparativas inter-tenant

3. **Documentar Escolhas Metodológicas** ✅:
   - ✅ Registrar parâmetros utilizados (ex: `max_lags`, thresholds)
   - ✅ Justificar escolhas de bibliotecas e métodos

### Prioridade Baixa (Após concluir anteriores) 🔄
1. **Análises com Janelas Móveis** ✅:
   - ✅ Adaptar módulos para análise temporal dinâmica
   - ✅ Testar e validar a execução completa do pipeline com janelas móveis

2. **Análise Consolidada para Experimentos Multi-Round** 🔄:
   - 🔄 Implementar análise de consistência entre rounds
   - 🔄 Desenvolver metodologia de robustez para causalidade
   - 🔄 Criar sistema de agregação de consenso entre rounds
   - 🔄 Implementar visualizações específicas para comparação entre rounds

3. **Refinamentos Estéticos e Usabilidade** 🔄:
   - ✅ Melhorar formatação de gráficos (estilo tableau-colorblind10)
   - ✅ Aprimorar mensagens de log e feedback

4. **Documentação para Usuários Finais** 🔄:
   - 🔄 Tutorial de uso do pipeline
   - 🔄 Guia de interpretação dos resultados

## Otimizações do Pipeline e Correções de Visualizações (Adicionado em Junho/2025)

Com base na análise do estado atual da implementação e no levantamento de plots não gerados ou incompletos, identificamos as seguintes oportunidades de melhoria organizadas em fases progressivas:

### Fase 1: Correção e Integração de Visualizações Existentes (Prioridade Alta)

1. **Execução de Visualizações Implementadas mas Não Geradas:**
   - ❌ Executar o pipeline com janelas deslizantes para gerar plots de correlação ao longo do tempo
   - ❌ Garantir a geração de plots de correlação ausentes (apenas correlação, já que covariância está sendo gerada)
   - ❌ Executar módulo de análise multi-round para gerar visualizações de consistência entre rounds
   - ❌ Verificar ambiente de execução para garantir que as dependências para Transfer Entropy estão disponíveis

2. **Correção de Problemas na Geração de Visualizações:**
   - ❌ Investigar e corrigir problemas na geração de plots de séries temporais combinadas de todas as fases
   - ❌ Adicionar chamadas para funções de detecção e visualização de anomalias no pipeline principal

### Fase 2: Unificação e Modularização do Pipeline (Prioridade Média)

1. **Consolidação dos Arquivos de Pipeline:**
   - ❌ Criar um framework de pipeline unificado que substitua os múltiplos arquivos atuais (`pipeline.py`, `pipeline_new.py`, `pipeline_with_sliding_window.py`)
   - ❌ Implementar sistema de estágios de pipeline como plugins carregáveis baseados em configuração
   - ❌ Garantir compatibilidade com o pipeline existente durante a transição

2. **Centralização de Configurações:**
   - ❌ Criar um sistema de configuração central baseado em YAML mais abrangente
   - ❌ Parametrizar todos os limiares, janelas e opções atualmente hardcoded no código
   - ❌ Adicionar documentação inline para todos os parâmetros configuráveis

3. **Interface de Linha de Comando (CLI) Unificada:**
   - ❌ Desenvolver CLI integrada para controlar todos os aspectos da execução do pipeline
   - ❌ Implementar opções de execução específicas (apenas descritiva, apenas correlação, etc.)
   - ❌ Adicionar suporte para execução de estágios específicos ou combinações de estágios

### Fase 3: Otimizações de Desempenho e Usabilidade (Prioridade Baixa)

1. **Sistema de Cache Inteligente:**
   - ❌ Implementar sistema de cache baseado em hash para evitar recálculos desnecessários
   - ❌ Adicionar invalidação seletiva de cache para recomputar apenas o necessário
   - ❌ Persistir resultados intermediários em formatos eficientes

2. **Paralelização de Análises Independentes:**
   - ❌ Identificar operações paralelizáveis (análises entre diferentes métricas, rounds, etc.)
   - ❌ Implementar paralelização com multiprocessing ou threading onde aplicável
   - ❌ Adicionar controle de concorrência e dependências entre tarefas do pipeline

3. **Interface Web Simples (Opcional):**
   - ❌ Criar interface web básica para visualizar resultados e configurar execuções
   - ❌ Implementar dashboard para monitoramento de execuções longas
   - ❌ Adicionar capacidade de salvar e compartilhar configurações

### Plano de Implementação Progressivo

Para garantir um progresso contínuo e tangível, recomendamos a seguinte abordagem:

1. **Sprint 1 (1 semana):**
   - Focar na Fase 1 para garantir que todas as visualizações implementadas estão funcionando corretamente
   - Executar `python -m src.pipeline_with_sliding_window` para gerar os plots de janelas deslizantes
   - Corrigir problemas imediatos de geração de plots
   
2. **Sprint 2 (2 semanas):**
   - Iniciar a consolidação do pipeline conforme a Fase 2
   - Desenvolver o novo framework de estágios como plugins
   - Implementar configuração central baseada em YAML
   
3. **Sprint 3 (2 semanas):**
   - Finalizar a transição para o pipeline unificado
   - Implementar CLI integrada
   - Testar e validar com diferentes configurações
   
4. **Sprint 4 (conforme disponibilidade):**
   - Implementar otimizações da Fase 3
   - Focar em sistemas de cache e paralelização
   - Considerar interface web se o tempo permitir

Este plano equilibra a necessidade de correções imediatas com melhorias arquiteturais de longo prazo, garantindo que o sistema continue funcionando enquanto é progressivamente aprimorado.

## Otimizações do Pipeline e Correções de Visualizações (Junho/2025)

Com base no levantamento realizado em 03/06/2025, identificamos uma série de visualizações que estão implementadas no código mas não estão sendo geradas na última execução do pipeline. Também foram identificadas oportunidades de otimização da arquitetura do pipeline para torná-lo mais modular, eficiente e fácil de manter.

### Visualizações Implementadas vs. Geradas

| Tipo de Visualização | Status | Localização da Implementação | Problema Identificado |
|----------------------|--------|------------------------------|------------------------|
| Plots de correlação | ❌ Não Gerado | `analysis_correlation.py` | Apenas visualizações de covariância estão sendo geradas |
| Plots de janela deslizante | ❌ Não Gerado | `analysis_sliding_window.py` | Módulo implementado, mas pipeline dedicado não executado |
| Visualização de séries temporais combinadas | ❌ Não Gerado | `analysis_descriptive.py` | Função implementada mas não chamada no pipeline principal |
| Plots de detecção de anomalias | ❌ Não Gerado | `analysis_descriptive.py` | Função implementada mas não integrada ao pipeline |
| Visualizações de análise multi-round | ❌ Não Gerado | `analysis_multi_round.py` | Estágio incluído no pipeline com janelas deslizantes, mas não no principal |

### Plano de Otimização do Pipeline

#### Fase 1: Verificação de Visualizações e Resolução de Erros Críticos (Atualizado em 4 de Junho/2025)

1. **Execução do Pipeline Unificado** ✅:
   - ✅ Script unificado desenvolvido em `src/run_unified_pipeline.py` para executar todas as análises
   - ✅ Comando de execução: `python -m src.run_unified_pipeline --config config/pipeline_config.yaml`
   - ✅ Script verificando automaticamente quais visualizações foram geradas e quais estão faltando
   - ✅ Opções para desativar análises específicas implementadas: `--no-sliding-window` ou `--no-multi-round`
   - ✅ Correção de erros críticos que impediam a execução completa do pipeline
      - ✅ Corrigidos erros na análise de janelas deslizantes (comparação de dicionários)
      - ✅ Corrigida a manipulação de Series pandas no módulo de causalidade
      - ✅ Corrigido o estágio de agregação de insights para lidar com diferentes formatos de dados

2. **Validação dos Plots de Correlação** 🔄:
   - 🔄 Pipeline executando sem erros, verificar a geração completa dos plots
   - 🔄 Verificar se as visualizações de correlação estão sendo geradas corretamente
   - 🔄 Confirmar que o diretório de saída existe e que os arquivos estão sendo salvos adequadamente

3. **Integração da Detecção de Anomalias** 🔄:
   - 🔄 Pipeline executando sem erros, verificar a geração dos plots de anomalias
   - 🔄 Confirmar que as funções de detecção de anomalias estão sendo chamadas corretamente
   - 🔄 Verificar se os plots de anomalias estão sendo salvos no diretório adequado

#### Fase 2: Consolidação da Arquitetura (Junho/2025 - Semanas 2-3)

1. **Unificação dos Arquivos de Pipeline**:
   - ❌ Criar um framework de pipeline unificado que substitua os múltiplos arquivos atuais (`pipeline.py`, `pipeline_new.py`, `pipeline_with_sliding_window.py`)
   - ❌ Implementar sistema de estágios de pipeline como plugins carregáveis baseados em configuração
   - ❌ Garantir compatibilidade com o pipeline existente durante a transição

2. **Centralização de Configurações**:
   - ❌ Criar um sistema de configuração central baseado em YAML mais abrangente
   - ❌ Parametrizar todos os limiares, janelas e opções atualmente hardcoded no código
   - ❌ Adicionar documentação inline para todos os parâmetros configuráveis

3. **Interface de Linha de Comando (CLI) Unificada**:
   - ❌ Desenvolver CLI integrada para controlar todos os aspectos da execução do pipeline
   - ❌ Implementar opções de execução específicas (apenas descritiva, apenas correlação, etc.)
   - ❌ Adicionar suporte para execução de estágios específicos ou combinações de estágios

#### Fase 3: Otimizações de Desempenho e Usabilidade (Prioridade Baixa)

1. **Sistema de Cache Inteligente:**
   - ❌ Implementar sistema de cache baseado em hash para evitar recálculos desnecessários
   - ❌ Adicionar invalidação seletiva de cache para recomputar apenas o necessário
   - ❌ Persistir resultados intermediários em formatos eficientes

2. **Paralelização de Análises Independentes:**
   - ❌ Identificar operações paralelizáveis (análises entre diferentes métricas, rounds, etc.)
   - ❌ Implementar paralelização com multiprocessing ou threading onde aplicável
   - ❌ Adicionar controle de concorrência e dependências entre tarefas do pipeline

3. **Interface Web Simples (Opcional):**
   - ❌ Criar interface web básica para visualizar resultados e configurar execuções
   - ❌ Implementar dashboard para monitoramento de execuções longas
   - ❌ Adicionar capacidade de salvar e compartilhar configurações

