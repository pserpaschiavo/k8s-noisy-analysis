# Plano de Trabalho para Análise de Séries Temporais Multi-Tenant (Atualizado em 6 de Junho/2025)

O objetivo é investigar a co-variação, relações causais e flutuações temporais das métricas entre diferentes tenants e fases experimentais (baseline, ataque, recuperação), utilizando ferramentas estatísticas básicas, interpretáveis e confiáveis.

## Status do Projeto (Atualizado em 6 de Junho/2025)

- ✅ **Concluído**: 
  - Estrutura principal do projeto implementada.
  - Ingestão de dados (incluindo suporte a carregamento direto de Parquet).
  - Segmentação, persistência, componentes de análise descritiva.
  - Correlação e causalidade básicos, agregação de insights, análise multi-round.
  - Ingestão direta de arquivos parquet com resolução de caminhos relativos e absolutos.
  - Correções de erros no pipeline, incluindo problemas com dict comparisons no teste de Granger, uso obsoleto de Series.fillna no módulo de causalidade.
  - Implementação do suporte a `experiment_folder` para especificar experimentos específicos dentro de data_root.
  - Implementação completa de correlação cruzada (CCF) em `pipeline.py` e `pipeline_new.py`.
  - Correção do problema no estágio de agregação de insights.
  - Atualização do script de organização de visualizações para incluir plots de correlação cruzada.
  - Documentação detalhada sobre correlação cruzada em `docs/correlacao_cruzada.md`.
  - **Recente**: Implementação de visualizações de grafos de causalidade melhoradas, corrigindo o problema de nós escondidos atrás de arestas.
  - **Recente**: Implementação de visualização consolidada multi-métrica mostrando relações de causalidade entre diferentes métricas em um único grafo.
  - **Recente**: Documentação sobre melhorias na visualização de causalidade em `docs/melhorias_visualizacao_causalidade.md`.

- 🔄 **Em andamento**: 
  - Unificação do pipeline em arquitetura baseada em plugins.
  - Refinamento do módulo de Causalidade com Transfer Entropy.
  - Testes unitários completos.
  - Relatórios comparativos entre fases experimentais.
  - Integração avançada entre correlação cruzada e detecção de anomalias.

- ❌ **Pendente**: 
  - Sistema de cache para resultados intermediários.
  - Paralelização de análises independentes.
  - Análise estatística de lags para determinar tempo médio de propagação de efeitos.

## Realizações para a Apresentação (Concluídas em 6 de Junho/2025)

1. **Análise Multi-Tenant**: 
   - Implementação completa da pipeline de análise de séries temporais.
   - Segmentação avançada de dados por fase experimental.

2. **Correlação e Causalidade**: 
   - Matrizes de correlação entre tenants.
   - Análise de causalidade de Granger.
   - Visualização baseada em grafos para relações causais.
   - **NOVO**: Grafos de causalidade multi-métrica consolidados.
   - **NOVO**: Visualização melhorada com nós visíveis sobre as arestas.

3. **Correlação Cruzada**:
   - Análise CCF completa para todas as métricas e fases.
   - Visualização de correlação cruzada com lags para identificar padrões temporais.
   - 61 gráficos CCF gerados para análise detalhada de métricas.

4. **Materiais de Apresentação**:
   - Organização automatizada de visualizações para apresentação.
   - Guia para demonstração dos resultados.
   - Verificação automática dos materiais para garantir consistência.

5. **Documentação**:
   - Documentação sobre correlação cruzada e interpretação de lags.
   - **NOVO**: Documentação sobre visualizações de causalidade melhoradas.
   - Guia de uso das novas funcionalidades.

## Próximos Passos

1. **Melhorias de Detecção de Noisy-Neighbor**:
   - Integrar CCF com análise de causalidade para melhor identificação de tenants causadores de interferência.
   - Desenvolver índice de interferência baseado em múltiplas métricas.

2. **Otimização**:
   - Implementar cache e paralelização para análise de grandes conjuntos de dados.
   - Completar a arquitetura de plugins para extensibilidade.

3. **Visualização Avançada**:
   - Dashboard interativo para exploração de resultados.
   - Gráficos de comparação entre fases experimentais.

4. **Análise Estatística de Propagação**:
   - Desenvolver métricas de tempo médio de propagação baseadas em lags significativos de CCF.
   - Criar matriz de tempo de propagação entre tenants.
