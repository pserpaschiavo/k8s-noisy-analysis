# Análise de Dados com Parquet

Este documento descreve como usar os arquivos Parquet gerados a partir dos dados CSV do experimento SFI2.

## Arquivos Disponíveis

- `data/processed/sfi2_paper_consolidated.parquet`: Arquivo Parquet consolidado com todos os dados de todos os rounds e fases.
- `data/processed/extracted/`: Diretório contendo extrações específicas dos dados.

## Estrutura dos Dados

O arquivo Parquet consolidado contém as seguintes colunas:

- `timestamp`: Timestamp da medição
- `value`: Valor da métrica
- `round_id`: Identificador do round (round-1, round-2, round-3)
- `phase`: Fase do experimento (1 - Baseline, 2 - CPU Noise, etc.)
- `component`: Componente relacionado à métrica
- `metric`: Nome da métrica

## Como Usar

### Com Python/Pandas

```python
import pandas as pd

# Carregar o arquivo Parquet consolidado
df = pd.read_parquet('data/processed/sfi2_paper_consolidated.parquet')

# Filtrar para uma métrica específica
nginx_connections = df[df['metric'] == 'nginx_connections']

# Agrupar por fase e calcular estatísticas
stats_by_phase = nginx_connections.groupby(['round_id', 'phase'])['value'].agg(
    ['mean', 'std', 'min', 'max']
)
```

### Com o Notebook de Análise

Utilize o notebook `análise_parquet.ipynb` para realizar análises interativas dos dados. O notebook contém exemplos de:

- Carregamento e limpeza de dados
- Visualização de métricas por fase e round
- Análise de impacto dos diferentes tipos de ruído
- Correlação entre métricas
- Análise de séries temporais

### Com o Script de Extração

O script `scripts/extract_parquet_data.py` permite extrair subconjuntos específicos dos dados:

```bash
# Extrair dados de uma métrica específica
python scripts/extract_parquet_data.py --metric nginx_connections

# Extrair dados de uma fase específica e gerar estatísticas
python scripts/extract_parquet_data.py --phase "6 - Combined Noise" --stats

# Extrair dados em formato CSV
python scripts/extract_parquet_data.py --metric postgres_connections --format csv

# Extrair dados com múltiplos filtros
python scripts/extract_parquet_data.py --metric nginx_cpu_usage --round round-3 --phase "2 - CPU Noise"
```

## Vantagens do Formato Parquet

1. **Performance**: Carregamento e consulta mais rápidos que CSV.
2. **Espaço**: Compressão eficiente dos dados.
3. **Preservação de Tipos**: Os tipos de dados são mantidos corretamente.
4. **Compatibilidade**: Funciona bem com ferramentas como Pandas, Spark, Dask, etc.

## Ferramentas de Visualização Recomendadas

- **Python**: Matplotlib, Seaborn, Plotly
- **Ferramentas Externas**: Power BI, Tableau, Apache Superset

## Exemplos de Análises

### Comparação entre Fases

Compare como uma métrica se comporta entre diferentes fases do experimento:

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Filtrar para uma métrica específica
metric_data = df[df['metric'] == 'nginx_cpu_usage']

# Criar boxplot por fase
plt.figure(figsize=(12, 6))
sns.boxplot(x='phase', y='value', data=metric_data)
plt.xticks(rotation=45)
plt.title('Uso de CPU do Nginx por Fase')
plt.tight_layout()
plt.show()
```

### Análise de Impacto

Calcule o impacto percentual de cada tipo de ruído em relação à linha de base:

```python
# Calcular média por fase
baseline = metric_data[metric_data['phase'] == '1 - Baseline']['value'].mean()
phases = metric_data['phase'].unique()

for phase in phases:
    if phase == '1 - Baseline':
        continue
    phase_mean = metric_data[metric_data['phase'] == phase]['value'].mean()
    impact = ((phase_mean / baseline) - 1) * 100
    print(f"{phase}: {impact:.2f}% de impacto")
```
