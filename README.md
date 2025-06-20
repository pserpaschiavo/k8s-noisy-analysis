# Multi-Tenant Time Series Analysis System

This system provides a structured pipeline for multi-tenant time series analysis of Kubernetes metrics, with a focus on identifying "noisy neighbors" and causality analysis between tenants.

## Pipeline Structure

The pipeline is organized into sequential, easily extensible stages:

1.  **Data Ingestion**: Loads raw data from experiments.
2.  **DataFrame Export**: Saves consolidated data in an efficient format.
3.  **Descriptive Analysis**: Calculates basic statistics and generates visualizations.
4.  **Correlation Analysis**: Examines relationships between metrics of different tenants.
5.  **Causality Analysis**: Investigates causal relationships using Granger and Transfer Entropy.
6.  **Phase Comparison**: Compares metrics between different experimental phases.
7.  **Report Generation**: Consolidates results into a report identifying tenants with the greatest impact.
8.  **Insight Aggregation**: Generates high-level insights from the analysis results.

## Project Structure

-   **src/**: Project source code
    -   **pipeline.py**: Main pipeline orchestration system.
    -   **data_ingestion.py**: Module for data ingestion and consolidation.
    -   **data_export.py**: Module for exporting and loading DataFrames.
    -   **data_segment.py**: Utilities for data segmentation and transformation.
    -   **analysis_descriptive.py**: Descriptive analyses and visualizations.
    -   **analysis_correlation.py**: Correlation and covariance analyses.
    -   **analysis_causality.py**: Causality analyses (Granger and Transfer Entropy).
    -   **analysis_anomaly.py**: Anomaly detection in time series.
    -   **analysis_sliding_window.py**: Sliding window analyses.
    -   **analysis_phase_comparison.py**: Comparative analyses between experimental phases.
    -   **report_generation.py**: Generation of consolidated reports.
    -   **insight_aggregation.py**: Aggregation of insights from analysis results.
    -   **utils.py**: Utility functions.
-   **config/**: Configuration files
    -   **pipeline_config.yaml**: Main pipeline configuration.
-   **data/processed/**: Processed data in efficient formats (Parquet).
-   **outputs/**: Analysis results
    -   **plots/**: Generated visualizations.
    -   **reports/**: Consolidated reports in Markdown format.
-   **docs/**: Project documentation.

## Installation

```bash
# Clone repository
git clone https://github.com/pserpaschiavo/k8s-noisy-analysis.git
cd k8s-noisy-analysis

# Set up virtual environment (recommended)
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
