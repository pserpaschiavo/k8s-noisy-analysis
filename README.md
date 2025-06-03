# Sistema de Análise de Séries Temporais Multi-Tenant

Este sistema oferece um pipeline estruturado para análise de séries temporais multi-tenant de métricas do Kubernetes, com foco em identificação de "vizinhos barulhentos" e análise de causalidade entre tenants.

## Estrutura do Pipeline

O pipeline é organizado em estágios sequenciais, facilmente extensíveis:

1. **Ingestão de Dados**: Carrega dados brutos dos experimentos
2. **Exportação de DataFrames**: Salva dados consolidados em formato eficiente
3. **Análise Descritiva**: Calcula estatísticas básicas e gera visualizações
4. **Análise de Correlação**: Examina relações entre métricas de diferentes tenants
5. **Análise de Causalidade**: Investiga relações de causalidade usando Granger e Transfer Entropy
6. **Comparação entre Fases**: Compara métricas entre diferentes fases experimentais (baseline, ataque, recuperação)
7. **Geração de Relatórios**: Consolida resultados em um relatório com identificação de tenants com maior impacto

## Estrutura do Projeto

- **src/**: Código-fonte do projeto
  - **pipeline.py**: Sistema principal de orquestração do pipeline
  - **data_ingestion.py**: Módulo para ingestão e consolidação de dados
  - **data_export.py**: Módulo para exportação e carregamento de DataFrames
  - **data_segment.py**: Utilitários para segmentação e transformação de dados
  - **analysis_descriptive.py**: Análises descritivas e visualizações
  - **analysis_correlation.py**: Análises de correlação e covariância
  - **analysis_causality.py**: Análises de causalidade (Granger e Transfer Entropy)
  - **analysis_phase_comparison.py**: Análises comparativas entre fases experimentais
  - **report_generation.py**: Geração de relatórios consolidados e identificação de tenants "barulhentos"
  - **test_*.py**: Testes unitários dos componentes
- **config/**: Arquivos de configuração
  - **pipeline_config.yaml**: Configuração principal do pipeline
  - **parse_config.yaml**: Configuração de seleção de métricas/tenants
- **demo-data/**: Dados de exemplo para análise
- **data/processed/**: Dados processados em formatos eficientes (Parquet)
- **outputs/**: Resultados das análises
  - **plots/**: Visualizações geradas (descritivas, correlação, causalidade, etc.)
  - **reports/**: Relatórios consolidados em formato Markdown

## Instalação

```bash
# Clonar repositório
git clone https://github.com/seu-usuario/gpt-nn-analysis.git
cd gpt-nn-analysis

# Configurar ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
# .\venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt
```

## Uso Básico

Para executar o pipeline completo com as configurações padrão:

```bash
./run_pipeline.py
```

## Execução do Pipeline

O pipeline oferece uma interface de linha de comando simples e flexível:

```bash
# Execução básica com configurações padrão
./run_pipeline.py

# Execução com arquivo de configuração específico
./run_pipeline.py --config config/pipeline_config.yaml

# Seleção de métricas e tenants específicos
./run_pipeline.py --selected-metrics cpu_usage memory_usage --selected-tenants tenant-a tenant-b

# Definir caminhos personalizados
./run_pipeline.py --data-root /path/to/data --output-dir /path/to/outputs
```

### Opções de Linha de Comando

```bash
./run_pipeline.py --help
```

Exemplos de uso:

```bash
# Executar apenas com métricas específicas
./run_pipeline.py --metrics cpu_usage memory_usage

# Executar apenas para tenants específicos
./run_pipeline.py --tenants tenant-a tenant-b

# Pular estágios específicos
./run_pipeline.py --skip-stages data_ingestion data_segmentation

# Carregar dados previamente processados
./run_pipeline.py --load-data --data-path data/processed/consolidated_long.parquet

# Modo de debug
./run_pipeline.py --debug
```

## Configuração

O comportamento do pipeline é controlado através do arquivo `config/pipeline_config.yaml`:

```yaml
# Diretórios de entrada/saída
data_root: /path/to/data
processed_data_dir: /path/to/processed
output_dir: /path/to/outputs

# Seleção de dados
selected_metrics:
  - cpu_usage
  - memory_usage
  
selected_tenants:
  - tenant-a
  - tenant-b
  
# Parâmetros de análise
correlation:
  methods:
    - pearson
    - spearman
  
causality:
  granger_max_lag: 5
  granger_threshold: 0.05
  transfer_entropy_bins: 8
```

## Extensibilidade

O sistema foi projetado para ser facilmente extensível:

1. **Adicionar novos estágios**: Criar uma classe que herde de `PipelineStage` e implementar o método `_execute_implementation`
2. **Personalizar fluxo**: Usar métodos como `pipeline.add_stage()` para modificar o pipeline
3. **Criar estágios personalizados**: Implementar etapas específicas conforme necessário

## Estrutura de Dados

O sistema utiliza um DataFrame "long" como fonte única de verdade, com as seguintes colunas:

- `timestamp`: Momento da medição (datetime)
- `metric_value`: Valor da métrica (float)
- `metric_name`: Nome da métrica (string)
- `tenant_id`: Identificador do tenant (string)
- `experimental_phase`: Fase experimental - Baseline, Attack, Recovery (string) 
- `round_id`: Identificador do round (string)
- `experiment_id`: Identificador do experimento (string)

## Análises Implementadas

### 1. Análise Descritiva
- Estatísticas básicas por tenant e fase
- Visualizações temporais das métricas
- Boxplots comparativos
- Gráficos de barras por fase

### 2. Análise de Correlação
- Matriz de correlação entre tenants
- Matriz de covariância
- Heatmaps para visualização
- Agrupamento hierárquico opcional

### 3. Análise de Causalidade
- **Causalidade de Granger**: Testa se valores passados de um tenant ajudam a prever valores futuros de outro
- **Transfer Entropy (TE)**: Mede transferência de informação não-linear entre séries temporais
- Visualização em forma de grafo direcionado

### 4. Comparação entre Fases
- Análise comparativa de métricas entre baseline, ataque e recuperação
- Visualização de variação percentual
- Identificação de tendências e anomalias

### 5. Identificação de Tenants "Barulhentos"
- Score composto baseado em múltiplos critérios:
  - Impacto causal (50%)
  - Força de correlação (30%)
  - Variação entre fases (20%)
- Ranking de tenants por impacto no ambiente
- Relatório detalhado com justificativas

## Executando Testes

Para verificar a implementação do Transfer Entropy:

```bash
python src/test_transfer_entropy.py
```

Para executar todos os testes unitários:

```bash
python -m pytest src/test_*.py -v
```

## Relatório Final

Após a execução do pipeline, um relatório markdown é gerado em `outputs/reports/` com:

1. Ranking de tenants por impacto
2. Tabela comparativa de métricas
3. Links para visualizações geradas
4. Explicação da metodologia

Este relatório é ideal para entender quais tenants têm maior impacto no ambiente e por quê, com base em evidências quantitativas.

## Relatórios

Após a execução, um relatório HTML será gerado em `outputs/reports/` com links para todas as visualizações e resultados.
