# Tarefas para o Próximo Dia (22/06/2025)

- **Implementar Heatmaps para Análise de Causalidade:** Adicionar a função `plot_causality_heatmap` em `src/visualization/plots.py` e integrá-la ao pipeline para gerar heatmaps dos resultados de causalidade (Granger e TE).
- **Melhorar Visualizações:**
    - Acertar títulos, labels e legendas para a melhor visualização e legibilidade.
    - Adicionar marcadores diferentes para os plots das séries temporais de acordo com o tenant.
    - Ajustar os plots dos grafos de causalidade para melhor clareza.
- **Verificar Barplots e Boxplots:** Investigar a ausência de dados em alguns gráficos de barra e boxplots.
- **Corrigir Cores dos Tenants:** Garantir que cada tenant tenha uma cor única e consistente em todos os gráficos para evitar ambiguidades.
- **Avaliar Análise Multi-Round:** Avaliar a funcionalidade da análise entre rounds para todos os tipos de resultados (descritivo, correlação, causalidade).
- **Verificar Agregação de Rounds:** Pesquisar no código por uma funcionalidade existente que agregue os resultados entre os diferentes rounds do experimento.
- **Calcular Impacto da Interferência:** Implementar ou utilizar uma métrica para calcular o quanto de interferência cada tenant sofreu durante os experimentos.
- **Implementar Coleta de Matrizes de Causalidade:** Completar a lógica em `src/analysis_multi_round.py` para agregar as matrizes de causalidade de cada round, permitindo a análise de robustez.
- **Finalizar Análise de Janela Deslizante:** Concluir a implementação das funções de análise de correlação e causalidade em `src/analysis_sliding_window.py` para permitir a análise da evolução temporal das métricas.
- **Gerar Relatório de Sumário Executivo:** Criar um relatório em Markdown que consolide os insights mais importantes de todas as análises (descritiva, correlação, causalidade, etc.) em um único documento, facilitando a interpretação dos resultados gerais do experimento.
