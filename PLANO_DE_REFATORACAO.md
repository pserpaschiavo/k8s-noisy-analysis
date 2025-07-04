# Plano de Refatoração do Pipeline de Análise

## 1. Visão Geral e Objetivos

O pipeline atual tornou-se excessivamente complexo e lento, perdendo o foco nos objetivos principais da análise. O objetivo desta refatoração é simplificar o pipeline, otimizar sua performance e torná-lo mais modular e fácil de manter.

## 2. Pontos de Melhoria e Ações Propostas

### 2.1. Performance e Pipeline "Travado"

**Problema:** A execução é muito longa devido à geração massiva e repetitiva de análises e plots para cada rodada e métrica.

**Soluções:**

- **Modularizar o Pipeline:** Separar as etapas de análise (descritiva, correlação, etc.) em módulos independentes. Atualmente, tudo parece ser executado em uma única sequência.
- **Parametrizar a Execução:** Permitir que o usuário escolha quais análises, métricas e rodadas executar através de parâmetros na linha de comando ou em um arquivo de configuração. Isso evita reprocessar tudo a cada execução.
- **Otimizar Geração de Artefatos (Plots e CSVs):** A geração de artefatos para cada análise é um gargalo.
    - **Ação:** Manter a geração de plots estáticos (PNG, PDF) para uso em publicações, mas otimizar as funções em `src/visualization/plots.py` para serem mais eficientes e customizáveis (e.g., ajustar títulos, labels, DPI).
    - **Ação:** Implementar uma exportação de dados consolidada em formato CSV. Para cada análise principal (ex: impacto por fase), gerar um CSV limpo e bem formatado contendo os resultados, facilitando o uso em outras ferramentas e a reproducibilidade.
- **Remover Análises Redundantes:** Avaliar a real necessidade de todas as análises atuais. Por exemplo, a análise de causalidade de Granger é computacionalmente cara. Precisamos dela em todas as execuções?

### 2.2. Corrigir Alertas `SettingWithCopyWarning`

**Problema:** O código em `src/visualization/plots.py` está modificando cópias de DataFrames, o que pode levar a erros.

**Ação:**

- **Refatorar o Código:** Iremos percorrer o arquivo `src/visualization/plots.py` e corrigir as operações de atribuição para usar o acessador `.loc` do Pandas, como recomendado na mensagem de erro. Exemplo:
  ```python
  # Antes
  subset['timestamp'] = pd.to_datetime(subset['timestamp'])
  # Depois
  subset.loc[:, 'timestamp'] = pd.to_datetime(subset['timestamp'])
  ```

### 2.3. Simplificação e Foco

**Problema:** O pipeline cresceu sem um direcionamento claro, incluindo análises que talvez não sejam mais relevantes.

**Ações:**

- **Redefinir os Objetivos:** O objetivo principal do pipeline é **investigar o fenômeno de "noisy neighbours" em um cluster Kubernetes multi-tenant, utilizando ferramentas estatísticas clássicas como análise de correlação e causalidade para identificar e quantificar a interferência entre tenants.**
- **Calcular o Impacto:** Desenvolver métricas e análises para quantificar o impacto sofrido por um tenant devido à interferência de outros tenants no mesmo nó. Isso envolve comparar o desempenho do tenant (ex: latência, vazão) com e sem a presença do "ruído" gerado pelos vizinhos.
- **Priorizar Análises:** Com base nos objetivos, definir um conjunto "core" de análises e visualizações essenciais.
- **Remover Código Morto:** Identificar e remover módulos e scripts que não são mais utilizados. A estrutura do projeto é grande, e provavelmente existem partes obsoletas.

## 3. Próximos Passos Sugeridos

1.  **[Curto Prazo]** Corrigir os `SettingWithCopyWarning` em `src/visualization/plots.py` para limpar os logs e garantir a robustez do código.
2.  **[Curto Prazo]** Implementar a parametrização do pipeline para permitir a execução de etapas específicas, oferecendo um alívio imediato na lentidão.
3.  **[Médio Prazo]** Iniciar a refatoração dos módulos de análise, focando em `analysis_correlation.py` e `analysis_descriptive.py` para otimizar os cálculos.
4.  **[Longo Prazo]** Desenvolver um protótipo de dashboard interativo para substituir a geração massiva de plots.
5.  **[Contínuo]** Revisar e remover código obsoleto.

Este plano é um ponto de partida. Podemos ajustá-lo conforme nossa discussão.
