# Plano de Trabalho para Análise de Séries Temporais Multi-Tenant (Atualizado em 5 de Junho/2025)

O objetivo é investigar a co-variação, relações causais e flutuações temporais das métricas entre diferentes tenants e fases experimentais (baseline, ataque, recuperação), utilizando ferramentas estatísticas básicas, interpretáveis e confiáveis.

## Status do Projeto (Atualizado em 5 de Junho/2025)

- ✅ **Concluído**: 
  - Estrutura principal do projeto implementada.
  - Ingestão de dados (incluindo suporte a carregamento direto de Parquet).
  - Segmentação, persistência, componentes de análise descritiva.
  - Correlação e causalidade básicos, agregação de insights, análise multi-round.
  - Ingestão direta de arquivos parquet com resolução de caminhos relativos e absolutos.
  - Correções de erros no pipeline, incluindo problemas com dict comparisons no teste de Granger, uso obsoleto de Series.fillna no módulo de causalidade.
  - Implementação do suporte a `experiment_folder` para especificar experimentos específicos dentro de data_root.
  - **Recente**: Implementação completa de correlação cruzada (CCF) em `pipeline.py` e `pipeline_new.py`.
  - **Recente**: Correção do problema no estágio de agregação de insights.
  - **Recente**: Atualização do script de organização de visualizações para incluir plots de correlação cruzada.
  - **Recente**: Documentação detalhada sobre correlação cruzada em `docs/correlacao_cruzada.md`.

- 🔄 **Em andamento**: 
  - Unificação do pipeline em arquitetura baseada em plugins.
  - Refinamento do módulo de Causalidade com Transfer Entropy.
  - Testes unitários completos.
  - Relatórios comparativos entre fases experimentais.

- ❌ **Pendente**: 
  - Sistema de cache para resultados intermediários.
  - Paralelização de análises independentes.
  - Integração avançada entre correlação cruzada e detecção de anomalias.

## Realizações para a Apresentação (Concluídas em 5 de Junho/2025)

As seguintes tarefas foram concluídas com sucesso para suportar a apresentação:

1. **Implementação de Correlação Cruzada (CCF)**:
   - ✅ Implementado cálculo de correlação cruzada (CCF) para análise de relações temporais entre tenants.
   - ✅ Adicionada geração de visualizações de CCF mostrando defasagem (lag) entre séries temporais.
   - ✅ Integradas visualizações no pipeline principal e no pipeline_new.

2. **Correção do Estágio de Agregação de Insights**:
   - ✅ Resolvido o erro "Dados necessários para agregação de insights não disponíveis".
   - ✅ Implementada geração robusta de insights mesmo com dados parciais ou incompletos.
   - ✅ Corrigidos erros de tipagem que causavam falhas na serialização para JSON.

3. **Organização de Visualizações para Apresentação**:
   - ✅ Atualizado script `organize_presentation_visualizations.py` para incluir plots de CCF.
   - ✅ Organizada a estrutura de diretórios para facilitar a apresentação.
   - ✅ Plots de CCF para a fase de ataque separados em pasta dedicada.

4. **Documentação Completa**:
   - ✅ Criado documento explicativo sobre correlação cruzada com guia de interpretação.
   - ✅ Documentados os resultados e as melhorias implementadas em `docs/melhorias_implementadas.md`.

## Próximos Passos (Roadmap pós-Apresentação)

Após a conclusão bem-sucedida das tarefas para a apresentação, recomenda-se focar nos seguintes próximos passos:

### Prioridade Alta (Junho-Julho/2025)

1. **Análise Avançada de Correlação Cruzada**:
   - Integrar CCF com detecção de anomalias para identificação mais precisa de tenants barulhentos.
   - Implementar análise estatística dos lags para determinar tempo médio de propagação de efeitos.
   - Criar visualizações comparativas de CCF entre diferentes fases experimentais.

2. **Unificação do Pipeline**:
   - Desenvolver arquitetura modular baseada em plugins.
   - Migrar progressivamente estágios existentes para a nova arquitetura.
   - Implementar sistema de configuração unificado.

3. **Conclusão dos Testes Unitários**:
   - Finalizar testes para o módulo de CCF.
   - Adicionar testes para a agregação de insights robusta.
   - Implementar testes de integração para o pipeline completo.

### Prioridade Média (Julho-Agosto/2025)

1. **Otimização de Desempenho**:
   - Implementar sistema de cache para resultados intermediários.
   - Adicionar paralelização para análises independentes.
   - Otimizar uso de memória em conjuntos de dados grandes.

2. **Melhorias na Visualização de Dados**:
   - Desenvolver dashboards interativos (opcional).
   - Criar visualizações comparativas mais avançadas entre fases.
   - Implementar exportação para múltiplos formatos.

3. **Expansão da Documentação**:
   - Criar tutoriais e guias passo-a-passo.
   - Documentar arquitetura técnica completa.
   - Adicionar exemplos de casos de uso reais.

### Prioridade Baixa (Setembro-Outubro/2025)

1. **Interface de Usuário Melhorada**:
   - Desenvolver CLI unificada para controle do pipeline.
   - Considerar interface web simples para visualização de resultados.
   - Implementar sistema de notificações para anomalias detectadas.

2. **Integração com Sistemas Externos**:
   - Desenvolver APIs para integração com ferramentas de monitoramento.
   - Adicionar suporte para exportação para sistemas de BI.
   - Implementar mecanismos de alertas em tempo real.

## Conclusão das Demandas para Apresentação

Todas as demandas críticas para a apresentação foram concluídas com sucesso. O sistema agora:

1. Gera plots de correlação cruzada para todos os pares de tenants, permitindo identificar relações temporais e defasagem entre métricas de diferentes tenants.
2. Agrega insights de forma robusta, mesmo na presença de dados incompletos.
3. Organiza automaticamente as visualizações relevantes para a apresentação.
4. Fornece documentação detalhada sobre as funcionalidades implementadas.

A apresentação poderá demonstrar como o pipeline identifica relações de causa e efeito entre tenants, com ênfase nos plots de correlação cruzada que mostram quais tenants influenciam outros e com qual defasagem temporal.
