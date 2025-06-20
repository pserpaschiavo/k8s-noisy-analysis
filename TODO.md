# Tarefas para Amanhã (21/06/2025)

- **Verificar Barplots e Boxplots:** Investigar a ausência de dados em alguns gráficos de barra e boxplots.
- **Corrigir Cores dos Tenants:** Garantir que cada tenant tenha uma cor única e consistente em todos os gráficos para evitar ambiguidades.
- **Verificar Agregação de Rounds:** Pesquisar no código por uma funcionalidade existente que agregue os resultados entre os diferentes rounds do experimento.
- **Calcular Impacto da Interferência:** Implementar ou utilizar uma métrica para calcular o quanto de interferência cada tenant sofreu durante os experimentos.
- **Implementar Coleta de Matrizes de Causalidade:** Completar a lógica em `src/analysis_multi_round.py` para agregar as matrizes de causalidade de cada round, permitindo a análise de robustez.
- **Finalizar Análise de Janela Deslizante:** Concluir a implementação das funções de análise de correlação e causalidade em `src/analysis_sliding_window.py` para permitir a análise da evolução temporal das métricas.
