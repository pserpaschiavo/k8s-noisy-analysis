# Contexto do Projeto e Decisões para Análise de Séries Temporais Multi-Tenant

Este documento resume as principais informações, decisões e o estado atual do planejamento para o projeto de análise de séries temporais em um cluster Kubernetes multi-tenant. Ele serve como um guia para a continuação dos trabalhos, especialmente em caso de interrupções.

## 1. Objetivo Principal

Investigar a co-variação, relações causais e flutuações temporais das métricas entre diferentes tenants (namespaces `tenant-*`) e fases experimentais (baseline, ataque, recuperação) em um cluster Kubernetes. O foco é utilizar ferramentas estatísticas básicas, interpretáveis e confiáveis para gerar insights sobre o efeito de "tenants barulhentos" e quantificar sua influência.

## 2. Plano de Trabalho Detalhado

O plano de trabalho granular, com todas as fases e etapas detalhadas, está documentado em:
- `analysis_work_plan.md`

Este plano deve ser o guia principal para a implementação das funcionalidades de análise.

## 3. Características e Estrutura dos Dados de Entrada

- **Origem dos Dados:** Métricas coletadas de queries do Prometheus.
- **Formato dos Arquivos CSV Individuais:** Cada arquivo de métrica (ex: `cpu_usage.csv`) contém duas colunas: `timestamp,value`.
    - Exemplo: `/home/phil/Projects/k8s-noisy-detection/demo-data/demo-experiment-3-rounds/round-1/1 - Baseline/tenant-a/cpu_usage.csv`
- **Padronização de Nomes de Arquivos:** Dentro dos diretórios de cada tenant (ex: `tenant-a/`, `tenant-b/`), os arquivos de métricas comuns (CPU, memória, etc.) possuem nomes padronizados. Arquivos com nomes divergentes são resultados de queries específicas e podem ser tratados separadamente ou em uma fase posterior.
- **Escopo dos Tenants para Análise:** A análise inter-tenant prioritariamente focará nos diretórios que seguem o padrão `tenant-*`. Outros diretórios (ex: `ingress-nginx/`, `active/`) não serão considerados como "tenants" para esta análise comparativa inicial.
- **Estrutura de Diretórios:** Os dados estão organizados em uma estrutura hierárquica que reflete o experimento, o round, a fase experimental e o tenant, conforme visualizado na estrutura do workspace (ex: `demo-data/demo-experiment-3-rounds/round-1/1 - Baseline/tenant-a/`).

### 3.1. Estrutura do DataFrame Consolidado (Formato Longo)

- O DataFrame "long" é o formato padrão e central de toda a análise. Todas as etapas de ingestão, validação, limpeza e transformação devem resultar em um único DataFrame long, que servirá como fonte de verdade para todo o pipeline.
- O DataFrame long deve conter obrigatoriamente as colunas: `timestamp` (datetime), `metric_value` (float), `metric_name` (str), `tenant_id` (str), `experimental_phase` (str), `round_id` (str), `experiment_id` (str).
- Subdatasets no formato "wide" podem ser gerados sob demanda, a partir do DataFrame long, para análises específicas (ex: correlação, causalidade, visualizações comparativas). Nunca devem substituir o long como fonte principal.
- Recomenda-se fortemente a persistência dos DataFrames processados (long e, se necessário, wide) em formatos eficientes e portáveis (Parquet preferencialmente, ou CSV/Feather), organizados por experimento, round e fase. Isso facilita reuso, integração com notebooks (Jupyter) e compartilhamento com outros times ou ferramentas.
- O pipeline deve prover funções utilitárias para salvar e carregar datasets processados, garantindo reprodutibilidade e agilidade no desenvolvimento.
- Outputs intermediários e finais devem ser exportados em formatos eficientes (Parquet/CSV), organizados por experimento, round e fase.
- Toda lógica de ingestão, transformação, análise e visualização deve ser modular e testável.
- O pipeline deve ser desenvolvido incrementalmente, começando com um subconjunto reduzido de dados/métricas.
- Utilizar um arquivo de configuração central para caminhos, métricas e parâmetros.
- Todo código, docstrings e nomes devem ser em inglês. Documentação geral pode ser em português.
- Decisões metodológicas e parâmetros relevantes devem ser documentados em Markdown.

## 4. Decisões Arquiteturais Chave

- **Módulos de Análise:** A arquitetura seguirá o padrão `BaseModule`, `BaseAnalyzer`, e `BaseVisualizer` definido em `k8s_noisy_detection/analysis/base.py`.
- **Unificação do Módulo de Causalidade:**
    - As análises de Causalidade de Granger e Transfer Entropy serão implementadas e orquestradas dentro de um único módulo Python: `k8s_noisy_detection/analysis/causality.py`.
    - Este módulo conterá as classes `CausalityModule(BaseModule)`, `CausalityAnalyzer(BaseAnalyzer)`, e `CausalityVisualizer(BaseVisualizer)`.
    - Os arquivos `granger.py` e `transfer_entropy.py` existentes na pasta `k8s_noisy_detection/analysis/` serão refatorados/integrados em `causality.py`, ou suas lógicas serão reimplementadas conforme necessário dentro da nova estrutura unificada. O arquivo `causality.py` precisará ser (re)criado.
- **Otimização de Dados:** A classe `DataOptimizer` em `k8s_noisy_detection/analysis/base.py` será utilizada para preparar os dados especificamente para cada tipo de análise (descritiva, correlação, causalidade).

## 5. Próximos Passos Imediatos (Conforme `analysis_work_plan.md`)

1.  **Fase 1: Preparação e Estratégia de Dados:**
    *   Iniciar com o item "1.1. Confirmar e documentar as colunas essenciais no DataFrame de entrada".
    *   Prosseguir com a implementação da lógica de carregamento e segmentação de dados, levando em consideração as características dos dados mencionadas acima.
    *   Revisar e testar os métodos da classe `DataOptimizer`.

## 6. Language Standard for Development

- **Code and Docstrings:** All new and modified Python code, including docstrings, comments, and variable names, should be written in English to ensure consistency and broader understanding.
- **Markdown Documents:** Project documentation files (like this one and `analysis_work_plan.md`) can be maintained in Portuguese as per current practice, but code-related elements within them (e.g., references to function names) should reflect the English standard used in the codebase.

Este documento deve ser atualizado caso novas decisões importantes sejam tomadas ou o contexto do projeto mude significativamente.

## 7. Critérios para Seleção de Métricas

Os seguintes critérios foram estabelecidos para guiar a seleção de métricas para a análise de vizinho barulhento:

1.  **Relevância para os Objetivos da Análise**:
    *   As métricas devem ser pertinentes à identificação e caracterização de fenômenos de vizinho barulhento.
    *   O pipeline de análise (`main.py`) suporta a seleção de métricas definidas pelo usuário através do argumento `--selected-metrics`. Por padrão, processa todas as métricas listadas em `METRICS_CONFIG`.
    *   O foco inicial será em métricas fundamentais de utilização de recursos (ex: CPU, memória, I/O de rede, I/O de disco).
    *   *(Consideração Futura)*: O conceito de atribuir pesos às métricas para variar sua importância pode ser explorado posteriormente, o que pode exigir ajustes na lógica de seleção e processamento de métricas.

2.  **Qualidade e Disponibilidade dos Dados**:
    *   **Completude**: As métricas devem ter uma cobertura de dados robusta entre diferentes tenants e fases experimentais.
        *   É uma característica esperada que um tenant ativamente sob ataque não possua dados para as fases "baseline" ou "recovery" referentes às suas próprias métricas diretamente induzidas pelo ataque. Isso é inerente à configuração experimental.
    *   **Consistência e Integridade**: As métricas serão avaliadas quanto a anomalias como timestamps duplicados, valores conflitantes para timestamps idênticos ou lacunas significativas. Etapas de pré-processamento serão definidas para lidar com isso (ex: média de duplicatas, regras para valores conflitantes, estratégias para imputação ou exclusão).
    *   **Granularidade**: O intervalo atual de coleta de dados de 5 segundos é considerado apropriado para a análise.
    *   **Tratamento de Valores Ausentes**: Uma estratégia clara para gerenciar pontos de dados ausentes será estabelecida (ex: interpolação, imputação estatística ou, se a esparsidade dos dados for muito alta, exclusão da métrica ou do conjunto de dados específico).

3.  **Acionabilidade (Perspectiva de Longo Prazo)**:
    *   Embora o objetivo principal atual seja a detecção e análise, será dada preferência a métricas que possam informar futuras análises de causa raiz ou o desenvolvimento de estratégias de mitigação.

4.  **Escopo Inicial para Desenvolvimento da Metodologia**:
    *   Para facilitar o desenvolvimento inicial e a validação dos padrões de análise, o trabalho começará com um subconjunto focado de métricas e dados.
    *   O ponto de partida proposto é a métrica `cpu_usage` para os tenants 'a', 'b', 'c', e 'd', utilizando dados do conjunto `demo-experiment-1-round`. Isso permitirá o refinamento do tratamento de dados e das técnicas analíticas antes de escalar para um conjunto mais amplo de métricas.

## 8. Critérios para Seleção de Pares de Tenants para Análise Comparativa

A seleção de pares de tenants para análise comparativa inter-tenant será guiada pelos seguintes princípios, considerando o ambiente de cluster single-node e os objetivos do experimento:

1.  **Coexistência Temporal e Fases Experimentais**:
    *   A análise primária considerará tenants que coexistem durantes as mesmas fases experimentais (Baseline, Ataque, Recuperação).
    *   O cluster é single-node, implicando que todos os tenants ativos compartilham os mesmos recursos físicos subjacentes.

2.  **Identificação de "Agressores" e "Vítimas" (Conforme Desenho Experimental)**:
    *   O desenho experimental inclui um "tenant barulhento" (agressor) designado, que introduz cargas de trabalho específicas para criar contenção de recursos.
    *   Outros tenants foram desenhados para serem sensíveis a tipos específicos de pressão de recursos (ex: CPU, memória, rede, disco) e são considerados "vítimas" potenciais.
    *   **Pares Prioritários**:
        *   O "tenant barulhento" será pareado com cada um dos "tenants sensíveis" para análise direta de impacto.
        *   Os "tenants sensíveis" serão pareados entre si para observar interações indiretas e o impacto compartilhado da contenção de recursos.

3.  **Análise Abrangente para Descoberta de Impactos**:
    *   Para capturar o comportamento global do sistema e potenciais impactos indiretos ou não antecipados, uma análise inicial "todos-com-todos" (all-pairs) será considerada para os tenants que se enquadram no escopo de análise (padrão `tenant-*`) dentro de cada fase experimental.
    *   Esta abordagem visa inferir o impacto direto e indireto entre tenants e observar as dinâmicas de recuperação pós-contenção.

4.  **Foco na Dinâmica de Contenção e Recuperação**:
    *   A seleção de pares e as análises subsequentes se concentrarão em entender como a contenção de recursos (durante a fase de Ataque) afeta os tenants e como o sistema e os tenants individuais se recuperam após a remoção da contenção.

5.  **Disponibilidade de Dados**:
    *   Para qualquer par selecionado, deve haver dados suficientes e sobrepostos para ambos os tenants nas métricas e fases experimentais de interesse para permitir uma análise estatística robusta.

### Observações e Filosofia da Análise Inter-Tenant

- A identificação de tenants que causam contenção de recursos ou degradação será feita exclusivamente de forma data-driven, a partir dos resultados das técnicas estatísticas e de causalidade implementadas (correlação, causalidade, influência cruzada, etc.).
- Não há pré-julgamento sobre quem é o "tenant barulhento" ou malicioso: a descoberta será imparcial e baseada em evidências extraídas dos dados.
- A ausência de determinados tenants em certas fases (baseline, ataque, recuperação) é esperada e deve ser tratada naturalmente pelo pipeline, sem gerar erro ou viés. Todas as análises devem considerar apenas os tenants presentes em cada contexto/fase.
- O pipeline deve ser robusto para lidar com a presença/ausência de tenants e arquivos de métricas em cada fase, e as funções de ingestão devem registrar (logar) essas ausências para rastreabilidade.

### Sugestão para o Desenvolvimento do Pipeline

- Desenvolva o pipeline de forma incremental, começando pela ingestão e validação dos dados, garantindo que o DataFrame long seja corretamente consolidado mesmo com ausências de tenants/fases.
- Implemente funções utilitárias para:
    - Listar todos os tenants presentes em cada fase/round/experimento.
    - Registrar ausências de tenants ou métricas esperadas (logging).
    - Gerar DataFrames segmentados por fase, tenant, métrica, etc.
- Priorize a modularidade: separe claramente ingestão, transformação, análise e visualização.
- Implemente testes unitários para garantir que a ingestão lida corretamente com casos de ausência de dados.
- Considere criar um notebook de exploração inicial para validar a consolidação dos dados e a robustez do pipeline antes de avançar para análises mais complexas.

## 9. Visualização de Causalidade Multi-Métrica

- Os grafos de causalidade multi-métrica agora possuem legenda contextual automática: se houver pelo menos uma matriz de p-valor (Granger real), a legenda prioriza p-valor; caso contrário, utiliza Transfer Entropy (TE). Isso garante máxima clareza interpretativa e padronização dos outputs.
- Outputs de visualização são organizados por técnica e fase, com legendas e títulos informativos para facilitar a análise comparativa e a reprodutibilidade.
