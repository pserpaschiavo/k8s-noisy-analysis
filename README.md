# Multi-Tenant Time Series Analysis System

🎉 **Status**: Pipeline totalmente funcional end-to-end (Atualizado: 24/06/2025)

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
10. ✅ **Multi-Round Analysis**: **FIXED 24/06/2025** - Cross-round consistency analysis

## 🚀 Recent Major Achievement

**Multi-Round Analysis Stage Fixed**: The critical `NotImplementedError` has been resolved. The stage now successfully generates:
- Consolidated boxplots for all metrics
- CV heatmaps by tenant and metric  
- Multi-round analysis reports
- Round consistency data
- Tenant stability scores

## Project Structure

-   **src/**: Project source code (**All modules functional**)
    -   **pipeline.py**: Main pipeline orchestration system (**✅ Working**)
    -   **analysis_multi_round.py**: Multi-round analysis (**✅ Fixed 24/06/2025**)
    -   **data_ingestion.py**: Data ingestion with experiment_folder support (**✅ Working**)
    -   **analysis_*.py**: Complete suite of analysis modules (**✅ All Working**)
    -   **visualization/**: Professional visualizations with relative time (**✅ Working**)
-   **config/**: Configuration files (**✅ Tested configurations available**)
-   **outputs/**: Analysis results (**✅ Complete outputs generated**)
    -   **plots/**: Professional visualizations
    -   **reports/**: Detailed Markdown reports  
    -   **insights/**: Aggregated insights and rankings
    -   **multi_round_analysis/**: Cross-round analysis results (**✅ NEW**)

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
```

## Basic Usage

To run the complete pipeline with a specific configuration:

```bash
python3 run_pipeline.py --config config/pipeline_config.yaml
```

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
