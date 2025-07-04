#!/bin/bash
# Script para execução otimizada do pipeline com configurações melhoradas

# Configurar nível de log para reduzir verbosidade
# Filtragem agressiva de warnings, principalmente relacionados a fontes
export PYTHONWARNINGS="ignore::RuntimeWarning,ignore::UserWarning,ignore::DeprecationWarning"
export LOGLEVEL="INFO"
# Variáveis específicas para desativar warnings do Matplotlib
export MPLBACKEND="Agg"  # Usar o backend não interativo (evita warnings de UI)

echo "===== Iniciando Pipeline de Análise Otimizado ====="
echo "Data de execução: $(date)"

# Limpar logs anteriores
echo "" > pipeline.log

# Executar o pipeline com configuração atualizada e filtragem avançada de warnings
# Filtramos todos os warnings contendo "findfont" ou "serif" e outros warnings comuns
python run_pipeline.py --config config/pipeline_config_sfi2.yaml 2> >(grep -v "findfont\|serif" | tee -a pipeline.log) | tee -a pipeline.log

# Verificar se o pipeline foi executado com sucesso
if [ $? -eq 0 ]; then
  echo "===== Pipeline executado com sucesso ====="
  echo "Verificando resultados:"
  
  # Verificar os arquivos de saída críticos
  output_dir="./outputs/sfi2-paper-analysis"
  
  echo -n "Relatório de análise multi-round: "
  if [ -f "${output_dir}/multi_round_analysis/multi_round_analysis/multi_round_analysis_report.md" ]; then
    echo "✅ OK"
  else
    echo "❌ Não encontrado"
  fi
  
  echo -n "Gráficos de correlação: "
  if [ -d "${output_dir}/correlation_graphs" ]; then
    echo "✅ OK ($(ls ${output_dir}/correlation_graphs | grep .png | wc -l) arquivos)"
  else
    echo "❌ Diretório não encontrado"
  fi
  
  echo -n "Relatórios de insights: "
  if [ -d "${output_dir}/insights" ]; then
    echo "✅ OK"
  else
    echo "❌ Diretório não encontrado"
  fi
  
  echo "===== Resumo da execução ====="
  echo "Tempo total: $(grep "Pipeline execution finished" pipeline.log | tail -n 1)"
  echo "Últimas linhas do log:"
  tail -n 10 pipeline.log
  
else
  echo "===== Pipeline falhou ====="
  echo "Verifique o arquivo pipeline.log para detalhes"
  echo "Últimas linhas de erro:"
  tail -n 20 pipeline.log
fi
