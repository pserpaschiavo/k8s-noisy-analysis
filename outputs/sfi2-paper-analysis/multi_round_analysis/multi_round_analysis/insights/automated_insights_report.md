# Resumo Executivo: Análise Multi-Round

A análise consolidou dados de múltiplos rounds experimentais para identificar efeitos consistentes e robustos.

## Principais Descobertas:
- De 48 combinações analisadas, 46 (95.8%) mostram efeitos estatisticamente significativos.
- 42 (87.5%) demonstram alta robustez;
- 19 correlações fortes (|r| > 0.7) identificadas;
- As métricas mais impactadas são 'memory_usage', 'cpu_usage'.
- As fases com maior impacto são '5 - Disk Noise', '4 - Network Noise'.

## Recomendações Principais:
- Concentre a atenção nas métricas com efeitos significativos e de alta magnitude, especialmente aqueles com alta confiabilidade entre rounds.
- Para resultados com baixa robustez, considere realizar rounds adicionais para aumentar a confiabilidade ou investigar fatores específicos que podem estar causando instabilidade nos resultados.

## Próximos Passos Sugeridos:
1. Investigar anomalias detectadas, especialmente efeitos com alta significância mas baixa robustez.
2. Aprofundar análise nas métricas e fases de maior impacto.
3. Implementar monitoramento contínuo baseado nos padrões identificados.

## Detalhamento dos Tamanhos de Efeito

De 48 combinações analisadas, 46 (95.8%) apresentaram efeitos estatisticamente significativos (p < 0.05).
Efeitos de magnitude 'large': 16 (33.3%).
Efeitos de magnitude 'medium': 20 (41.7%).
Efeitos de magnitude 'small': 8 (16.7%).
Efeitos de magnitude 'negligible': 4 (8.3%).

Principais efeitos significativos:
- 7 - Recovery causa aumento significativo em 'memory_usage' para tenant tenant-ntk (Cohen's d = 1.65, p = 0.0000).
- 3 - Memory Noise causa aumento significativo em 'memory_usage' para tenant tenant-ntk (Cohen's d = 1.50, p = 0.0000).
- 6 - Combined Noise causa aumento significativo em 'memory_usage' para tenant tenant-ntk (Cohen's d = 1.40, p = 0.0000).
- 5 - Disk Noise causa aumento significativo em 'memory_usage' para tenant tenant-ntk (Cohen's d = 1.26, p = 0.0000).
- 4 - Network Noise causa aumento significativo em 'memory_usage' para tenant tenant-ntk (Cohen's d = 1.23, p = 0.0000).

A fase '4 - Network Noise' é a mais impactante, com 8 métricas significativamente afetadas.
O tenant 'tenant-cpu' é o mais afetado, com 12 métricas significativamente impactadas.
A métrica 'memory_usage' é a mais frequentemente impactada, com efeitos significativos em 24 combinações de fase/tenant.

38 (79.2%) dos efeitos têm alta confiabilidade entre rounds.

## Análise de Robustez

De 48 efeitos analisados quanto à robustez, 42 (87.5%) apresentam alta robustez, 3 (6.2%) média robustez, e 3 (6.2%) baixa robustez.
42 (87.5%) dos efeitos são robustos tanto em magnitude quanto em significância estatística.

Foram identificados 3 efeitos cuja significância é sensível à remoção de rounds específicos:
- 'cpu_usage' em 2 - CPU Noise (tenant tenant-mem) é sensível aos rounds: round-1
- 'cpu_usage' em 3 - Memory Noise (tenant tenant-mem) é sensível aos rounds: round-2
- 'cpu_usage' em 7 - Recovery (tenant tenant-ntk) é sensível aos rounds: round-2
O round 'round-2' é o mais frequentemente responsável por alterações na significância (2 efeitos).

## Correlações Intra-fase

De 395 correlações analisadas, 19 (4.8%) são fortes (|r| > 0.7).
279 (70.6%) das correlações são positivas e 116 (29.4%) são negativas.

A fase '1 - Baseline' exibe as correlações mais fortes em média (|r| = 0.51).
A métrica 'cpu_usage' exibe as correlações mais fortes em média (|r| = 0.30).

Principais correlações intra-fase:
- Em '7 - Recovery', tenant-pair tenant-mem:tenant-nsy é negativamente correlacionado (r = -0.96) para a métrica 'memory_usage'.
- Em '7 - Recovery', tenant-pair tenant-nsy:tenant-ntk é positivamente correlacionado (r = 0.93) para a métrica 'memory_usage'.
- Em '1 - Baseline', tenant-pair tenant-dsk:tenant-mem é positivamente correlacionado (r = 0.92) para a métrica 'memory_usage'.

## Anomalias Detectadas

Detectados 3 efeitos com baixa robustez mas alta significância estatística (p < 0.01):
- 'cpu_usage' em 2 - CPU Noise (tenant tenant-mem) com p=0.0003 mas robustez Baixa.
- 'cpu_usage' em 3 - Memory Noise (tenant tenant-mem) com p=0.0042 mas robustez Baixa.
- 'cpu_usage' em 7 - Recovery (tenant tenant-ntk) com p=0.0038 mas robustez Baixa.

Detectados 8 efeitos com alta variabilidade entre rounds (CV > 0.5):
- 'cpu_usage' em 2 - CPU Noise (tenant tenant-mem), CV=0.84
- 'cpu_usage' em 3 - Memory Noise (tenant tenant-mem), CV=1.63
- 'cpu_usage' em 4 - Network Noise (tenant tenant-dsk), CV=0.70

Detectados 56 pares de correlação que mudam de sinal entre rounds:
- 'cpu_usage' em 1 - Baseline para tenant-dsk:tenant-ntk: 1 correlações positivas e 2 negativas entre rounds.
- 'cpu_usage' em 1 - Baseline para tenant-mem:tenant-ntk: 1 correlações positivas e 2 negativas entre rounds.
- 'cpu_usage' em 2 - CPU Noise para tenant-cpu:tenant-dsk: 2 correlações positivas e 1 negativas entre rounds.

## Recomendações Detalhadas

1. Concentre a atenção nas métricas com efeitos significativos e de alta magnitude, especialmente aqueles com alta confiabilidade entre rounds.
2. Para resultados com baixa robustez, considere realizar rounds adicionais para aumentar a confiabilidade ou investigar fatores específicos que podem estar causando instabilidade nos resultados.
3. Explore mais profundamente as fases com correlações fortes entre tenants, pois podem indicar comportamentos sistêmicos importantes ou oportunidades de otimização conjunta.
4. Investigue com cautela os efeitos que apresentam alta significância estatística mas baixa robustez, pois podem representar falsos positivos ou condições específicas que não generalizam.
5. Revisite as correlações que mudam de sinal entre rounds para identificar possíveis fatores externos ou condições específicas que alteram fundamentalmente as relações entre tenants.
6. Implemente monitoramento contínuo para as métricas e tenants identificados como mais sensíveis às fases experimentais.
7. Estabeleça limiares de alerta baseados nos tamanhos de efeito observados para detectar anomalias em tempo real.
8. Consolide os descobrimentos em um modelo preditivo para antecipar o comportamento do sistema em diferentes cenários.
