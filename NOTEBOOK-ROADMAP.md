# Plano de Ação para Notebooks de Análise de Resultados

Este documento descreve o plano para a criação de uma série de notebooks Jupyter para a apresentação dos resultados do pipeline de análise.

## Sequência de Notebooks Proposta

1.  **`01_Introducao_e_Carga_de_Dados.ipynb`**:
    *   **Objetivo**: Apresentar o conjunto de dados, as métricas e os *tenants* selecionados.
    *   **Conteúdo**: Carregar os dados processados (o arquivo parquet `sfi2_paper.parquet`) e exibir informações básicas, como o número de registros, rounds e fases de experimento.

2.  **`02_Analise_Descritiva.ipynb`**:
    *   **Objetivo**: Fornecer uma visão geral das características dos dados.
    *   **Conteúdo**: Gerar estatísticas descritivas (média, mediana, desvio padrão) e visualizações como histogramas e boxplots para as métricas selecionadas.

3.  **`03_Analise_de_Correlacao.ipynb`**:
    *   **Objetivo**: Investigar as relações entre as diferentes métricas.
    *   **Conteúdo**: Calcular e visualizar as matrizes de correlação (Pearson e Spearman), destacando as correlações mais significativas com base no limiar (`threshold`) definido no arquivo de configuração.

4.  **`04_Analise_de_Causalidade.ipynb`**:
    *   **Objetivo**: Explorar as relações de causa e efeito entre as variáveis.
    *   **Conteúdo**: Aplicar a análise de causalidade de Granger e visualizar os resultados em grafos, conforme as configurações.

5.  **`05_Analise_Multi-Round.ipynb`**:
    *   **Objetivo**: Comparar os resultados entre os diferentes rounds de experimentos.
    *   **Conteúdo**: Analisar a estabilidade das correlações, o tamanho do efeito das anomalias e agregar os resultados usando meta-análise.

6.  **`06_Relatorio_Final.ipynb`**:
    *   **Objetivo**: Consolidar e apresentar os principais *insights* obtidos.
    *   **Conteúdo**: Um resumo executivo com as conclusões mais importantes de cada análise, gerando um relatório final.
