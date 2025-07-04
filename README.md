# Multi-Tenant Time Series Analysis System

🎉 **Status**: Pipeline totalmente funcional end-to-end (Atualizado: 03/07/2025) - Implementação completa de análise de robustez, meta-visualizações, aceleração GPU e sistema avançado de cache

This system provides a **complete, production-ready pipeline** for multi-tenant time series analysis of Kubernetes metrics, with advanced focus on identifying "noisy neighbors" and causality analysis between tenants.

## ✅ Pipeline Status - All Stages Operational

The pipeline consists of **10 fully functional stages** that execute without errors:

1.  ✅ **Data Ingestion**: Loads raw data from experiments (supports Parquet and experiment_folder)
2.  ✅ **Data Validation**: Validates data quality and generates validation reports
3.  ✅ **Data Export**: Saves consolidated data in efficient formats
4.  ✅ **Descriptive Analysis**: Calculates statistics and generates time-relative visualizations
5.  ✅ **Correlation Analysis**: Cross-correlation analysis (CCF) with 61+ plots generated
6.  ✅ **Causality Analysis**: Granger tests and Transfer Entropy implementation
7.  ✅ **Phase Comparison**: Compares baseline/attack/recovery phases
8.  ✅ **Report Generation**: Consolidated Markdown reports with insights
9.  ✅ **Insight Aggregation**: High-level insights with tenant rankings
10. ✅ **Multi-Round Analysis**: Análise de consistência e robustez entre rodadas

## 🔍 Novos Recursos (03/07/2025)

1. **Análise de Robustez Completa**:
   - ✅ Análise leave-one-out para testar a estabilidade dos resultados
   - ✅ Análise de sensibilidade com limiares alfa
   - ✅ Sistema de pontuação de robustez para confiança nos resultados
   - ✅ Integração com o pipeline de visualização

2. **Meta-Visualizações Aprimoradas para Múltiplas Rodadas**:
   - ✅ Gráficos de floresta estilo meta-análise para tamanhos de efeito
   - ✅ Mapas de calor de tamanhos de efeito aprimorados com indicadores de confiabilidade
   - ✅ Gráficos de barras de erro aprimorados com intervalos de confiança
   - ✅ Gráficos de dispersão 3D para análise de efeito multivariado

3. **Redes de Correlação Avançadas**:
   - ✅ Detecção de comunidades para agrupamento visual de tenants relacionados
   - ✅ Filtragem inteligente para visualizações com grande volume de dados
   - ✅ Opções personalizáveis de estética e layout
   - ✅ Destaque para correlações mais significativas

4. **Sistema de Cache Inteligente**:
   - ✅ Evita reprocessamento de análises computacionalmente intensivas
   - ✅ Rastreamento de dependências para invalidação automática
   - ✅ Gestão automática de ciclo de vida do cache
   - ✅ Estatísticas de economia de tempo e recursos

5. **Aceleração GPU para Grandes Volumes de Dados**:
   - ✅ Suporte para CuPy, PyTorch e TensorFlow como backends
   - ✅ Aceleração para cálculos de correlação em grandes matrizes
   - ✅ Cálculo de tamanho de efeito otimizado para GPU
   - ✅ Fallback automático para CPU quando GPU não disponível
   - ✅ Configuração flexível via YAML

## Project Structure

-   **src/**: Project source code (**All modules functional**)
    -   **pipeline.py**: Main pipeline orchestration system (**✅ Working**)
    -   **analysis_multi_round.py**: Multi-round analysis (**✅ Fixed 24/06/2025**)
    -   **data_ingestion.py**: Data ingestion with experiment_folder support (**✅ Working**)
    -   **analysis_*.py**: Complete suite of analysis modules (**✅ All Working**)
    -   **gpu_acceleration.py**: GPU acceleration for large datasets (**✅ NEW 03/07/2025**)
    -   **smart_cache.py**: Intelligent caching system (**✅ Working**)
    -   **visualization/**: Professional visualizations with relative time (**✅ Working**)
-   **config/**: Configuration files (**✅ Tested configurations available**)
-   **outputs/**: Analysis results (**✅ Complete outputs generated**)
    -   **plots/**: Professional visualizations
    -   **reports/**: Detailed Markdown reports  
    -   **insights/**: Aggregated insights and rankings
    -   **multi_round_analysis/**: Cross-round analysis results (**✅ Working**)

## Quick Start

```bash
# Clone repository  
git clone https://github.com/pserpaschiavo/k8s-noisy-analysis.git
cd k8s-noisy-analysis

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
# .\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For GPU acceleration (optional)
# Uncomment and install the appropriate backend in requirements.txt:
# - CuPy (CUDA NumPy): Recommended for best NumPy compatibility
# - PyTorch: Alternative for tensor operations
# - TensorFlow: Alternative for tensor operations
```

## Basic Usage

To run the complete pipeline with a specific configuration:

```bash
# Using the standard execution method:
python3 run_pipeline.py --config config/pipeline_config.yaml

# OR using the optimized execution script (recommended):
./run_optimized_pipeline.sh

# Test GPU acceleration (if GPU is available):
python3 test_gpu_acceleration.py
```

The `run_optimized_pipeline.sh` script provides:
- Enhanced warning suppression
- Output verification
- Summary of execution results

### Command-Line Options

The pipeline execution is controlled by the `run_pipeline.py` script, which accepts the following arguments:

-   `--config`: Path to the YAML configuration file (required).
-   `--data-root`: Overrides the data root directory specified in the config file.
-   `--output-dir`: Overrides the output directory specified in the config file.
-   `--selected-metrics`: A list of specific metrics to analyze, overriding the config file.
-   `--selected-tenants`: A list of specific tenants to analyze, overriding the config file.
-   `--selected-rounds`: A list of specific rounds to analyze, overriding the config file.

For more details, run:
```bash
python3 run_pipeline.py --help
```

## Visualization Configuration (New in 07/2025)

The pipeline includes an enhanced visualization system with configurable settings:

```yaml
# Example from config/pipeline_config_sfi2.yaml
visualization:
  plot_quality: "high"  # high, medium, low - affects DPI and size
  fonts:
    family: "sans-serif"  # Universally available font family
    size: 12              # Base font size
  correlation_graph:
    threshold: 0.3        # Minimum correlation to display
  heatmap:
    cmap: "coolwarm"      # Color map for heatmaps
```

For detailed visualization configuration options, see:
- [Visualization Configuration Guide](./docs/visualization_config_guide.md)
- [Work Plan 07/02/2025](./work-plan-20250702.md) for the latest improvements
