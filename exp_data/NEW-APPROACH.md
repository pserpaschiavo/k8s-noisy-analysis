# Nova Abordagem para Experimentos de Noisy Neighbors

Este documento descreve a nova abordagem implementada na branch `refactor/new-approach` para o experimento de Noisy Neighbors no Kubernetes.

## Estrutura de Fases do Experimento

A nova abordagem implementa um fluxo de experimento com 7 fases distintas:

1. **Fase de Baseline**: Todos os tenants em funcionamento normal sem ruído.
2. **Fase de Ruído de CPU**: Aplicação de estresse específico de CPU.
3. **Fase de Ruído de Memória**: Aplicação de estresse específico de memória.
4. **Fase de Ruído de Rede**: Aplicação de estresse específico de rede.
5. **Fase de Ruído de Disco**: Aplicação de estresse específico de disco (I/O).
6. **Fase de Ruído Combinado**: Aplicação de estresse em todos os recursos simultaneamente.
7. **Fase de Recuperação**: Remoção de todos os ruídos para observar a recuperação dos workloads.

## Estrutura dos Tenants

### Tenant CPU (tenant-cpu)
- **Aplicação principal**: Sysbench CPU
  - Configuração para cálculos de números primos
  - Configuração de threads múltiplos
  - Sistema de medição de tempo de execução

### Tenant Memória (tenant-mem)
- **Aplicação principal**: Redis + Redis Benchmark
  - Configuração do servidor Redis com dataset
  - Implementação de benchmark para operações SET/GET
  - Sistema de medição de latência e throughput

### Tenant Rede (tenant-ntk)
- **Aplicação principal**: Nginx + Wrk2
  - Configuração do servidor web
  - Implementação do gerador de carga HTTP
  - Sistema de medição de latência precisa

### Tenant Disco (tenant-dsk)
- **Aplicação principal**: PostgreSQL + pgbench
  - Configuração do banco de dados
  - Implementação de benchmark TPC-B
  - Sistema de medição de transações/segundo

### Tenant Ruidoso (tenant-nsy)
- **Aplicação principal**: Stress-ng
  - Configuração para estresse de CPU
  - Configuração para estresse de Memória
  - Configuração para estresse de Rede
  - Configuração para estresse de Disco
  - Sistema de controle de intensidade do ruído

## Executando o Experimento

### Comando básico:
```bash
./run-experiment.sh
```

### Com opções personalizadas:
```bash
./run-experiment.sh --name experimento-personalizado --rounds 2 --baseline 180 --noise 240 --recovery 180
```

### Opções disponíveis:

- `--name`: Nome do experimento
- `--rounds`: Número de rounds completos
- `--baseline`: Duração da fase de baseline em segundos
- `--noise`: Duração de cada fase de ruído em segundos
- `--recovery`: Duração da fase de recuperação em segundos
- `--interval`: Intervalo de coleta de métricas em segundos
- `--limited-resources`: Executar com recursos limitados
- `--non-interactive`: Executar sem prompts de confirmação

## Resultados e Métricas

Os resultados são armazenados em diretórios específicos dentro de `results/` e incluem:

- Métricas de todos os tenants durante cada fase
- Logs do experimento
- Informações resumidas sobre cada fase e round

## Visualização de Resultados

Use o Grafana para visualizar os dados coletados:

```bash
kubectl -n monitoring port-forward svc/prometheus-grafana 3000:80
```

Acesse http://localhost:3000 (usuário: admin, senha: admin)
