# Atualização do Plano de Trabalho - Julho/2025

## Status Atual do Projeto

- ✅ **Concluído**: Estrutura principal do projeto implementada, ingestão de dados (incluindo suporte a carregamento direto de Parquet), segmentação, persistência, componentes de análise descritiva, correlação e causalidade básicos, agregação de insights, análise multi-round, ingestão direta de arquivos parquet com resolução de caminhos relativos e absolutos. Correções de erros no pipeline, incluindo problemas com dict comparisons no teste de Granger, uso obsoleto de Series.fillna no módulo de causalidade e problemas no estágio de agregação de insights. Implementação do suporte a `experiment_folder` para especificar experimentos específicos dentro de data_root.
- 🔄 **Em andamento**: Refinamento do módulo de Causalidade com Transfer Entropy, testes unitários completos, análises com janelas móveis, visualizações ausentes/incompletas.
- ❌ **Pendente**: Relatórios comparativos entre fases experimentais, integração completa de todos os componentes.

## Implementações Recentes (Junho/2025)

### 1. Suporte Completo ao Parâmetro `experiment_folder`

- ✅ **Implementado no núcleo do pipeline**: Integrado ao `DataIngestionStage` para permitir a especificação de experimentos específicos.
- ✅ **Funções auxiliares**: Adicionadas `get_experiment_folder` e `get_experiment_dir` em `src/parse_config.py`.
- ✅ **Prevenção de duplicação de caminhos**: Lógica robusta para evitar problemas com concatenação de caminhos.
- ✅ **Propagação de flags**: Implementado sistema para rastrear quando o experiment_folder é aplicado.
- ✅ **Configuração padrão**: Adicionada constante `DEFAULT_EXPERIMENT_FOLDER` em `src/config.py`.

### 2. Scripts e Documentação de Suporte

- ✅ **Scripts de conveniência**: Criados `run_pipeline_with_experiment.py` e `run_pipeline_3_rounds.py`.
- ✅ **Documentação abrangente**: Criados `docs/experiment_folder_guide.md` e `docs/README_experiment_folder.md`.
- ✅ **Testes**: Desenvolvidos `test_experiment_folder.py`, `debug_experiment_folder.py` e `src/test_experiment_folder_parameter.py`.
- ✅ **Configuração para testes**: Criado `config/pipeline_config_3rounds.yaml`.
- ✅ **Automatização**: Script `make_scripts_executable.sh` para garantir permissões de execução.

## Prioridades para o Próximo Ciclo (Julho-Agosto/2025)

### Prioridade Alta

1. **Consolidação do Pipeline Unificado**:
   - ❌ Unificar as diferentes implementações do pipeline (`pipeline.py`, `pipeline_new.py`, `pipeline_with_sliding_window.py`) em uma arquitetura modular baseada em plugins.
   - ❌ Criar sistema de configuração central para controlar quais módulos de análise são ativados.
   - ❌ Implementar mecanismo de dependência entre estágios do pipeline para garantir execução correta.

2. **Correção e Verificação de Visualizações**:
   - ❌ Confirmar que todas as visualizações implementadas estão sendo geradas corretamente.
   - ❌ Investigar por que apenas visualizações de covariância estão sendo geradas (faltando correlação).
   - ❌ Integrar plots de janela deslizante ao pipeline principal.

3. **Integração Completa do Parâmetro `experiment_folder`**:
   - ❌ Estender suporte para todos os scripts e modos de análise (incluindo janelas deslizantes e análise multi-round).
   - ❌ Implementar validação para garantir que o diretório de experimento existe antes da execução.
   - ❌ Adicionar logging detalhado sobre o experimento selecionado.

### Prioridade Média

1. **Melhoria dos Testes Unitários**:
   - ❌ Expandir cobertura de testes para casos extremos e edge cases.
   - ❌ Implementar testes de integração para o pipeline completo.
   - ❌ Criar ambiente de teste automatizado para validação contínua.

2. **Documentação Técnica**:
   - ❌ Criar documentação detalhada sobre arquitetura do pipeline e fluxo de dados.
   - ❌ Documentar todos os parâmetros de configuração disponíveis.
   - ❌ Desenvolver guias passo-a-passo para análises comuns.

3. **Otimização de Desempenho**:
   - ❌ Implementar sistema de cache para resultados intermediários.
   - ❌ Adicionar paralelização para análises independentes.
   - ❌ Otimizar uso de memória para conjuntos de dados grandes.

### Prioridade Baixa

1. **Interface de Usuário Melhorada**:
   - ❌ Desenvolver interface de linha de comando mais amigável.
   - ❌ Considerar implementação de interface web simples para visualização de resultados.
   - ❌ Criar sistema de alertas para anomalias detectadas.

2. **Expansão de Funcionalidades Analíticas**:
   - ❌ Adicionar novos métodos de análise de causalidade.
   - ❌ Implementar detecção de anomalias mais sofisticada.
   - ❌ Desenvolver capacidade de previsão baseada em séries temporais.

3. **Interoperabilidade com Outros Sistemas**:
   - ❌ Criar exportadores para formatos comuns (JSON, CSV, etc.).
   - ❌ Desenvolver APIs para integração com outros sistemas.
   - ❌ Implementar mecanismos para importar dados de fontes diversas.

## Roadmap Técnico

### Fase 1: Consolidação e Estabilização (Julho/2025)
1. Semana 1-2: Unificação do pipeline e correção de visualizações
2. Semana 3-4: Testes extensivos e documentação técnica

### Fase 2: Otimização e Extensibilidade (Agosto/2025)
1. Semana 1-2: Implementação do sistema de cache e paralelização
2. Semana 3-4: Desenvolvimento da arquitetura baseada em plugins

### Fase 3: Experiência do Usuário e Recursos Avançados (Setembro/2025)
1. Semana 1-2: Melhoria da interface de usuário
2. Semana 3-4: Adição de funcionalidades analíticas avançadas

## Considerações Arquiteturais

Para avançar em direção a uma arquitetura mais modular e extensível, recomendamos:

1. **Arquitetura Baseada em Plugins**: 
   - Definir interface clara para módulos de análise
   - Criar sistema de registro e descoberta de plugins
   - Permitir carregamento dinâmico de módulos

2. **Sistema de Orquestração**:
   - Implementar mecanismo para definir fluxos de análise
   - Gerenciar dependências entre estágios do pipeline
   - Permitir execução parcial e retomada

3. **Infraestrutura de Persistência**:
   - Desenvolver sistema de cache inteligente
   - Implementar versionamento de resultados
   - Garantir rastreabilidade completa

Esta atualização do plano de trabalho reflete os avanços recentes com o parâmetro `experiment_folder` e estabelece prioridades claras para os próximos meses, com foco na consolidação da arquitetura, ampliação da cobertura de testes e melhoria das visualizações.
