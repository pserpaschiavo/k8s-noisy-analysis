# Implementação do Plano de Trabalho: Resumo e Próximos Passos

## O que foi implementado até agora:

1. **Função `extract_effect_sizes()`** para extração de tamanhos de efeito:
   - Implementada no módulo `src/effect_size.py`
   - Calcula Cohen's d, Eta-squared e p-valores
   - Suporte para cache e processamento paralelo
   - Avaliação de qualidade dos resultados

2. **Método `_extract_effect_sizes()`** na classe `MultiRoundAnalysisStage`:
   - Integra a função de extração ao pipeline existente
   - Utiliza configurações do arquivo YAML
   - Salva resultados em CSV

3. **Função `extract_phase_correlations()`** para extração de correlações intra-fase:
   - Implementada no módulo `src/phase_correlation.py`
   - Calcula correlações entre tenants dentro de cada fase e round
   - Implementação da função `analyze_correlation_stability()` para análise de estabilidade
   - Suporte para cache e processamento paralelo
   - Avaliação da força e qualidade das correlações

4. **Método `_extract_phase_correlations()`** na classe `MultiRoundAnalysisStage`:
   - Integra a função de correlação intra-fase ao pipeline
   - Gera relatório de estabilidade das correlações
   - Salva resultados em CSV e análises em Markdown

5. **Função `aggregate_effect_sizes()`** para agregação de tamanhos de efeito:
   - Implementada no módulo `src/effect_aggregation.py`
   - Calcula média e desvio padrão dos tamanhos de efeito
   - Implementação inicial para intervalos de confiança (IC95%)
   - Estrutura para combinação de p-valores
   - Classificação de confiabilidade dos resultados

6. **Método `_aggregate_effect_sizes()`** na classe `MultiRoundAnalysisStage`:
   - Integra a função de agregação ao pipeline existente
   - Gera resumo das estatísticas agregadas
   - Salva resultados em CSV e resumos em Markdown

7. **Implementação inicial das visualizações**:
   - Módulo `src/visualization/effect_plots.py` com funções para:
     - `generate_effect_size_heatmap()` (implementação parcial)
     - `plot_effect_error_bars()` (implementação parcial)
     - `plot_effect_scatter()` (implementação parcial)
     - `generate_effect_forest_plot()` (implementação parcial)

8. **Atualização do arquivo de configuração**:
   - Adicionadas opções para análise multi-round no `pipeline_config_sfi2.yaml`
   - Configuração para métodos de tamanho de efeito
   - Configuração para correlação intra-fase (método, limiar, periodicidade)
   - Opções para meta-análise e visualização

9. **Documentação**:
   - Criado arquivo `docs/multi_round_analysis.md` com detalhes da implementação
   - Atualizado o plano de trabalho com o status atual e próximos passos

## Próximos passos:

### 1. Finalizar `aggregate_effect_sizes()`
- Testar e refinar o cálculo de IC95% usando bootstrapping
- Completar a implementação dos métodos de combinação de p-valores (Fisher/Stouffer)
- Adicionar mais métricas de estabilidade e robustez

### 2. Completar visualizações
- Finalizar `generate_effect_size_heatmap()` para visualizar tamanhos de efeito médios
- Completar `plot_effect_error_bars()` para visualizar IC95%
- Terminar `plot_effect_scatter()` para gráficos de dispersão multivariados
- Implementar `generate_effect_forest_plot()` para visualização estilo meta-análise
- Desenvolver visualizações para correlações intra-fase:
  - Redes de correlação
  - Heatmaps de correlação
  - Gráficos de estabilidade

### 3. Implementar análises de robustez
- Desenvolver `perform_robustness_analysis()` para análise leave-one-out
- Implementar testes de sensibilidade variando limiares de significância
- Gerar insights automáticos
- Integrar análises de robustez com visualizações

### 4. Integração completa com o pipeline
- Adicionar os novos métodos ao fluxo de execução
- Configurar geração de relatórios consolidados
- Implementar cache inteligente para evitar reprocessamentos

## Como executar o código atual:

Para testar a nova funcionalidade de extração de tamanhos de efeito e correlações intra-fase:

```bash
python run_multi_round_analysis.py --config config/pipeline_config_sfi2.yaml
```

Esta execução já incluirá a extração de tamanhos de efeito e correlações intra-fase, salvando os resultados em CSV.

## Conclusão

Esta implementação inicial foca na primeira etapa do plano de trabalho (Extração de Tamanhos de Efeito e Correlações Intra-Fase). As próximas implementações seguirão o plano de priorização definido no documento de trabalho.
