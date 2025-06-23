# Atualização de Status - 23/06/2025

Este documento resume o estado atual do codebase e define as próximas tarefas.

## Tarefas Concluídas Hoje (23/06)

- **Correção dos Shaders das Fases nos Lineplots:** O problema com o sombreamento que indica as fases experimentais nos gráficos de série temporal foi resolvido. A lógica em `src/visualization/plots.py` foi ajustada para normalizar os nomes das fases antes de buscar as cores na configuração, garantindo que a cor correta seja aplicada e que a visualização seja clara.

## Próximas Tarefas (para 24/06)

- **Prioridade:** Avaliar a funcionalidade da análise entre rounds para todos os tipos de resultados (descritivo, correlação, causalidade) e garantir que está operando corretamente.
- **Implementar Heatmaps de Causalidade:** Integrar a função `plot_causality_heatmap` ao pipeline para gerar os mapas de calor dos resultados de causalidade.
- **Verificar Gráficos de Barra e Boxplots:** Investigar a ausência de dados em alguns dos barplots e boxplots gerados.
- **Gerar Relatório de Sumário Executivo:** Criar um relatório em Markdown que consolide os insights mais importantes de todas as análises para facilitar a interpretação dos resultados.
