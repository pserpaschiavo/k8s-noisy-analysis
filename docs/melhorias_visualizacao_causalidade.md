# Melhorias na Visualização de Causalidade

## Problemas Corrigidos

1. **Nós ocultos atrás de arestas**: Nas visualizações originais, algumas vezes os nós ficavam escondidos atrás de arestas, dificultando a visualização. Isso ocorria porque o Matplotlib não respeitava adequadamente a ordem de desenho (z-order) dos elementos.

2. **Falta de visualização consolidada multi-métrica**: Não havia uma forma padrão de visualizar a relação causal entre diferentes métricas em um único grafo.

## Melhorias Implementadas

### 1. Visualização Aprimorada de Grafos de Causalidade

- **Z-order controlado**: Implementação de controle de z-order para garantir que os nós sempre apareçam na frente das arestas.
- **Legibilidade aprimorada**: Aumentamos o tamanho dos nós, melhoramos o contraste de cores e ajustamos o tamanho e posicionamento dos rótulos.
- **Estética aprimorada**: Bordas mais suaves, cores consistentes e layout mais equilibrado.

### 2. Gráfico Consolidado Multi-Métrica

- **Visualização de múltiplas métricas**: Criamos uma função que permite visualizar a relação causal de múltiplas métricas em um único grafo, usando cores diferentes para cada métrica.
- **Legendas explicativas**: Adição de legendas que explicam o significado das cores e dos valores de limiar (threshold).
- **Detecção automática do tipo de matriz**: Detecção inteligente se a matriz representa p-valores (Granger) ou valores de Transfer Entropy (TE).

### 3. Integração com o Pipeline

- As novas visualizações foram integradas ao pipeline existente, mantendo compatibilidade com o código legado.
- Adição de funções para gerar automaticamente visualizações consolidadas para cada combinação de experimento, fase e round.
- As visualizações originais são mantidas, e as melhoradas são geradas em diretórios específicos.

## Como Usar

### Visualizações Individuais Melhoradas

As visualizações melhoradas são geradas automaticamente em:
```
outputs/plots/causality/improved/
```

### Visualizações Consolidadas Multi-Métrica

As visualizações consolidadas são geradas em:
```
outputs/plots/causality/consolidated/
```

### Uso Manual

Para gerar visualizações manualmente, use as funções:

```python
from src.improved_causality_graph import plot_improved_causality_graph, plot_consolidated_causality_graph

# Para um grafo individual
plot_improved_causality_graph(
    causality_matrix,
    output_path,
    threshold=0.05,
    directed=True,
    metric="cpu_usage"
)

# Para um grafo consolidado multi-métrica
plot_consolidated_causality_graph(
    {
        "cpu_usage": cpu_matrix,
        "memory_usage": memory_matrix,
        "disk_io": disk_matrix
    },
    output_path,
    threshold=0.05,
    directed=True,
    phase="1 - Baseline",
    round_id="round-1"
)
```

## Interpretação das Visualizações

### Visualização Individual

- **Nós**: Representam tenants (inquilinos) no sistema.
- **Arestas**: Indicam relação causal entre tenants.
- **Espessura das arestas**: Representa a força da relação causal.
- **Direção das arestas**: Indica a direção da causalidade (de causa para efeito).

### Visualização Consolidada

- **Cores das arestas**: Cada cor representa uma métrica diferente.
- **Espessura das arestas**: Representa a força da relação causal.
- **Nós compartilhados**: Um mesmo tenant pode ter relações causais em diferentes métricas.

## Exemplos

Exemplos de visualizações geradas podem ser encontrados nos diretórios mencionados acima. Para gerar exemplos de teste, execute:

```bash
python test_causality_visualizations.py
```
