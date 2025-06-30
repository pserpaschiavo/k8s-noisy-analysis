# Resumo da Estabilidade das Correlações Intra-Fase

Gerado em: 2025-06-30 14:48:25

## Estatísticas Gerais

- Total de pares analisados: 26
- Pares com correlação estável: 9
- Pares com correlação instável: 17

## Correlações Estáveis por Métrica e Fase

### cpu_usage - 1 - Baseline

- **tenant-dsk:tenant-mem**: 0.866 ± 0.019 (low variability, 3 rounds)

### cpu_usage - 4 - Network Noise

- **tenant-mem:tenant-nsy**: 0.533 ± 0.020 (low variability, 2 rounds)

### cpu_usage - 6 - Combined Noise

- **tenant-dsk:tenant-nsy**: 0.510 ± 0.006 (low variability, 2 rounds)

### memory_usage - 1 - Baseline

- **tenant-dsk:tenant-mem**: 0.899 ± 0.028 (low variability, 2 rounds)
- **tenant-mem:tenant-ntk**: 0.775 ± 0.006 (low variability, 2 rounds)
- **tenant-dsk:tenant-ntk**: 0.743 ± 0.164 (medium variability, 3 rounds)
- **tenant-cpu:tenant-ntk**: -0.716 ± 0.160 (medium variability, 3 rounds)

### memory_usage - 7 - Recovery

- **tenant-mem:tenant-nsy**: -0.125 ± 1.180 (high variability, 2 rounds)
- **tenant-cpu:tenant-nsy**: 0.000 ± 1.117 (high variability, 2 rounds)


## Variabilidade da Correlação por Métrica e Fase

| Métrica | Fase | Desvio Padrão Médio | CV Médio | % Pares Estáveis |
|---------|------|---------------------|----------|------------------|
| cpu_usage | 1 - Baseline | 0.019 | 0.022 | 25.0% |
| cpu_usage | 4 - Network Noise | 0.020 | 0.037 | 50.0% |
| cpu_usage | 5 - Disk Noise | nan | nan | 0.0% |
| cpu_usage | 6 - Combined Noise | 0.006 | 0.012 | 50.0% |
| cpu_usage | 7 - Recovery | nan | nan | 0.0% |
| memory_usage | 1 - Baseline | 0.090 | 0.121 | 66.7% |
| memory_usage | 4 - Network Noise | nan | nan | 0.0% |
| memory_usage | 7 - Recovery | 1.149 | 2320.837 | 40.0% |
