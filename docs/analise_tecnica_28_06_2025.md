# Análise Técnica do Pipeline K8s-Noisy-Analysis
**Data: 28 de Junho de 2025**

## 1. Visão Geral do Estado Atual

Após uma análise detalhada do codebase, identifiquei o estado atual do pipeline e os problemas específicos nas visualizações que precisam ser corrigidos.

### 1.1 Arquitetura do Sistema

O sistema está estruturado em três componentes principais:

1. **Pipeline Principal** (`run_pipeline.py`): Orquestra a execução completa de todas as etapas de análise.

2. **Análise Multi-Round** (`run_multi_round_analysis.py`): Focado na comparação e consolidação de dados entre diferentes rounds experimentais.

3. **Módulos de Visualização**:
   - `src/visualization/plots.py`: Funções básicas de visualização
   - `src/visualization/advanced_plots.py`: Visualizações avançadas para análise multi-round

### 1.2 Funcionalidades Implementadas

✅ **Implementadas e funcionando**:
- Ingestão de dados experimentais
- Processamento de DataFrames
- Análises descritivas básicas
- Time series consolidados (`generate_consolidated_timeseries()` em `advanced_plots.py`)

❌ **Implementadas com problemas**:
- Grafos de correlação
- Heatmaps de causalidade (geração intermitente)
- Séries temporais específicas

⏳ **Pendentes de implementação**:
- Verificação de integridade de dados
- Meta-análise estatística avançada
- Dashboards interativos

## 2. Diagnóstico de Problemas

### 2.1 Problema: Grafos de Correlação

**Sintoma**: A função `plot_aggregated_correlation_graph` é referenciada no plano de trabalho, mas não está implementada em nenhum lugar do codebase.

**Análise**: Uma busca completa pelo repositório não encontrou a implementação desta função, apenas referências a ela.

**Solução Técnica**: Precisamos implementar esta função em `src/visualization/plots.py` ou `src/visualization/advanced_plots.py`, com base nas funções existentes de correlação.

### 2.2 Problema: Séries Temporais

**Sintoma**: Algumas visualizações de séries temporais não estão sendo geradas, apesar da função `generate_consolidated_timeseries()` estar funcionando.

**Análise**: A função existe e foi implementada com sucesso, mas aparentemente há problemas na forma como está sendo chamada ou nos dados fornecidos a ela.

**Possíveis causas**:
- Filtragem incorreta de dados antes da chamada
- Problemas de manipulação de datas nas séries temporais
- Incompatibilidade entre os formatos esperados pelos diferentes componentes

### 2.3 Problema: Heatmaps de Causalidade

**Sintoma**: Os heatmaps de causalidade são gerados de forma intermitente, às vezes funcionando e às vezes não.

**Análise**: A função `plot_causality_heatmap` está implementada e é chamada corretamente, mas parece haver problema na consistência dos dados ou na robustez da função.

**Possíveis causas**:
- Ausência de validação de dados antes da plotagem
- Problemas de formatação nas matrizes de causalidade
- Erros não tratados durante a geração dos gráficos

## 3. Plano de Ação Técnico

### 3.1 Corrigir Grafos de Correlação

1. **Implementar a função `plot_aggregated_correlation_graph`** em `src/visualization/advanced_plots.py`:
   - Função deve receber dados de correlação de múltiplos rounds
   - Usar NetworkX para gerar grafo com arestas ponderadas pela correlação
   - Implementar esquema de cores consistente
   - Incluir threshold configurável

2. **Integrar chamada da função** em `src/analysis_multi_round.py`:
   - Adicionar chamada no método apropriado da classe `MultiRoundAnalysisStage`
   - Garantir passagem correta dos parâmetros

### 3.2 Restaurar Geração de Séries Temporais

1. **Adicionar logging detalhado** na função de geração de séries temporais:
   - Adicionar validação explícita do DataFrame antes da plotagem
   - Logging de formato e estrutura dos dados recebidos

2. **Normalizar manipulação de timestamps**:
   - Verificar consistência de formato de datas
   - Implementar conversão robusta para evitar problemas de fuso horário

### 3.3 Resolver Inconsistências de Heatmaps de Causalidade

1. **Reforçar validação de dados** em `plot_causality_heatmap`:
   - Adicionar verificações de matriz vazia
   - Validar formato e dimensões da matriz
   - Tratar casos de ausência de relações causais significativas

2. **Implementar escala adaptativa**:
   - Ajustar limites da colorbar dinamicamente
   - Normalizar valores para comparação entre diferentes fases

### 3.4 Implementar Verificação de Integridade de Dados

1. **Criar função `validate_data_for_visualization`** em um novo módulo `src/validation.py`:
   - Validar estrutura do DataFrame
   - Verificar missing values
   - Confirmar consistência temporal
   - Validar disponibilidade de todas as colunas necessárias

2. **Integrar validação** em todas as funções de plotagem:
   - Adicionar chamada de validação no início de cada função
   - Retornar mensagens de erro claras e acionáveis

## 4. Conclusão

O pipeline possui uma estrutura sólida, com a maioria dos componentes funcionando corretamente. Os problemas identificados estão principalmente relacionados às visualizações mais avançadas e à validação de dados.

As correções propostas são específicas e focadas, podendo ser implementadas em um prazo curto (1-2 dias), sem necessidade de reestruturação profunda do sistema. A implementação de verificações de dados mais robustas não só corrigirá os problemas atuais, como também aumentará a resiliência do sistema a longo prazo.

Recomendo iniciar as correções pela ordem de prioridade identificada nos documentos de planejamento:
1. Grafos de Correlação
2. Séries Temporais
3. Heatmaps de Causalidade
4. Validação de Dados
