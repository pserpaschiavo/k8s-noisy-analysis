# Guia de Uso do Parâmetro `experiment_folder`

## Visão Geral

O parâmetro `experiment_folder` permite especificar qual pasta de experimento usar dentro do diretório raiz de dados (`data_root`). Isso facilita o gerenciamento de múltiplos experimentos organizados em subdiretórios diferentes.

## Estrutura de Diretórios Suportada

O sistema suporta duas organizações principais de dados:

### 1. Experimento no Diretório Raiz:
```
data_root/
├── round-1/
│   ├── Baseline/
│   │   ├── tenant-a/
│   │   └── tenant-b/
│   └── Attack/
└── round-2/
```

### 2. Múltiplos Experimentos em Subdiretórios (modelo recomendado):
```
data_root/
├── experiment1/
│   ├── round-1/
│   └── round-2/
└── experiment2/
    ├── round-1/
    └── round-2/
```

O parâmetro `experiment_folder` é útil principalmente para a segunda estrutura.

## Como Usar

### 1. No Arquivo de Configuração

Adicione o parâmetro `experiment_folder` ao arquivo de configuração YAML:

```yaml
# Diretórios de entrada/saída
data_root: /home/user/data
experiment_folder: demo-experiment-3-rounds
processed_data_dir: /home/user/processed
output_dir: /home/user/outputs/demo-experiment-3-rounds
```

### 2. Executando o Pipeline

Há duas maneiras de executar o pipeline com suporte para `experiment_folder`:

#### A. Usar o Script Wrapper Dedicado:

```bash
./run_pipeline_with_experiment.py --config config/pipeline_config_3rounds.yaml
```

Este script:
- Carrega o arquivo de configuração
- Extrai o parâmetro `experiment_folder`
- Combina com `data_root` para formar o caminho completo
- Executa o pipeline com o caminho correto

#### B. Script Personalizado para Experimento Específico:

```bash
./run_pipeline_3_rounds.py
```

Este script é configurado especificamente para o experimento de 3 rounds, oferecendo uma interface simplificada.

## Como o Parâmetro é Processado

O processamento do parâmetro `experiment_folder` ocorre em dois locais:

1. **Nível de Pipeline**: Através do patch aplicado à função `Pipeline.run()` em `pipeline_experiment_folder.py`, que:
   - Define `data_root` = `data_root` + `experiment_folder`
   - Define a flag `experiment_folder_applied` = `True`

2. **Nível de Estágio de Ingestão**: Em `DataIngestionStage._execute_implementation()`:
   - Verifica a flag `experiment_folder_applied`
   - Se `experiment_folder` está definido e a flag é falsa, aplica o parâmetro
   - Propaga a flag para os próximos estágios

## Funções Auxiliares

O módulo `parse_config.py` contém funções úteis:

- `get_experiment_folder(config)`: Retorna o valor do parâmetro
- `get_experiment_dir(config)`: Combina `data_root` + `experiment_folder`

## Depuração

Para depurar problemas com o parâmetro `experiment_folder`, use os scripts:

- `debug_experiment_folder.py`: Para verificar a configuração
- `test_experiment_folder.py`: Para validar o caminho do experimento

## Melhores Práticas

1. Use sempre nomes únicos para os diretórios de experimentos
2. Mantenha a mesma estrutura dentro de cada experimento (rounds, fases, tenants)
3. Use o script wrapper quando possível para evitar problemas
4. Especifique também diretórios de saída distintos para cada experimento
