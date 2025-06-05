# Implementação de Tempo Relativo nas Visualizações

## Visão Geral

Esta implementação modifica todas as funções de visualização do pipeline para exibir o tempo em segundos desde o início de cada fase (tempo relativo), em vez de timestamps absolutos. Esta mudança melhora significativamente a legibilidade e interpretabilidade dos gráficos, facilitando a comparação entre diferentes fases e experimentos.

Anteriormente, algumas visualizações alternavam entre segundos e minutos dependendo da duração dos dados. Agora, todas as visualizações usam consistentemente segundos como unidade para o eixo X, independentemente da duração da fase ou experimento.

## Arquivos Modificados

1. `analysis_descriptive.py`: 
   - Função `plot_metric_timeseries_multi_tenant`
   - Função `plot_anomalies`
   - Função `plot_metric_timeseries_multi_tenant_all_phases`

2. `analysis_anomaly.py`:
   - Função `plot_anomalies`

3. `analysis_sliding_window.py`:
   - Função `plot_sliding_window_correlation`
   - Função `plot_sliding_window_causality`

## Principais Modificações

### Cálculo de Tempo Relativo
Em cada função de visualização, implementamos o cálculo do tempo relativo:

```python
# Calcula o timestamp de início da fase
phase_start = subset['timestamp'].min()

# Converte timestamps para segundos relativos
elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
```

### Tratamento de Tipos de Dados
Adicionamos conversões explícitas para garantir compatibilidade com Matplotlib:

```python
# Conversão segura para tipos numéricos compatíveis
x_plot_data = np.array(elapsed_times, dtype=float)
y_plot_data = pd.to_numeric(values).to_numpy(dtype=float, na_value=np.nan)
```

### Padronização de Rótulos
Padronizamos todos os rótulos do eixo X para usar segundos em todas as visualizações:

```python
# Sempre usar segundos para consistência
time_unit = 1  # Usar sempre segundos
x_label = "Segundos desde o início da fase"

# Aplicar ao eixo X em todas as visualizações
ax.set_xlabel(x_label)
```

Também atualizamos todas as unidades de tempo em janelas deslizantes para usar segundos:

```python
# Antes
window_size='5min'
step_size='1min'

# Depois
window_size='300s'  # 5min convertido para segundos
step_size='60s'     # 1min convertido para segundos
```

## Benefícios

1. **Melhor Interpretabilidade**: Os gráficos agora mostram quanto tempo se passou desde o início da fase, tornando mais fácil entender a sequência e duração dos eventos.

2. **Comparação Facilitada**: É possível comparar diretamente diferentes fases e experimentos, pois todos começam do "tempo zero".

3. **Visualização Mais Limpa**: Elimina a complexidade de ler timestamps completos nos rótulos dos eixos.

4. **Análise Quantitativa**: Facilita a medição precisa do tempo entre eventos ou anomalias dentro da mesma visualização.

## Validação

A implementação foi validada executando o pipeline completo com a configuração de 3 rounds:

```bash
python -m src.run_unified_pipeline --config config/pipeline_config_3rounds.yaml
```

Os resultados mostram que todas as 798 visualizações foram geradas corretamente com o formato de tempo relativo.

## Próximos Passos

1. **Opção de Configuração**: Adicionar um parâmetro de configuração para permitir alternar entre tempo relativo e absoluto.

2. **Anotações Adicionais**: Considerar adicionar marcadores ou anotações em pontos-chave nas visualizações.

3. **Zoom Interativo**: Explorar opções para visualizações interativas que permitam ampliar regiões específicas do tempo.
