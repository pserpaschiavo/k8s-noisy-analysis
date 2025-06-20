# Adaptation Plan for New Experimental Phase Structure

## Summary of the New Approach

The new approach implemented in the `refactor/new-approach` branch modifies the experiment structure, changing from 3 phases (Baseline, Attack, Recovery) to 7 distinct phases:

1. **Baseline Phase**: All tenants operating normally without noise.
2. **CPU Noise Phase**: Application of CPU-specific stress.
3. **Memory Noise Phase**: Application of memory-specific stress.
4. **Network Noise Phase**: Application of network-specific stress.
5. **Disk Noise Phase**: Application of disk (I/O) specific stress.
6. **Combined Noise Phase**: Application of stress on all resources simultaneously.
7. **Recovery Phase**: Removal of all noise to observe workload recovery.

This modification impacted several components of the analysis pipeline, requiring adaptations in multiple modules.

## Final Status: Completed

All planned adaptations have been successfully implemented. The pipeline now fully supports the 7-phase structure, and all code, documentation, and outputs are in English.

## Completed Tasks

### 1. Data Ingestion Adaptations

- [x] Modify `data_ingestion.py` to recognize the 7 experimental phases.
- [x] Adapt directory naming conventions for the new phases.
- [x] Implement structure validation for different experiment formats.

### 2. Phase Analysis Adaptations

- [x] Update `analysis_phase_comparison.py` to compare all 7 phases.
- [x] Enhance comparative phase visualization to display all phases efficiently.
- [x] Implement new comparison metrics specific to each phase type (CPU, memory, etc.) in reporting.
- [x] Create specific analyses for each type of affected resource within the generated reports and insights.

### 3. Visualization Adaptations

- [x] Modify color palettes in `visualization_config.py` to accommodate 7 phases.
- [x] Adapt graph layouts to include more categories.
- [x] Add specific visualizations for comparing different noise types (e.g., grouped bar charts).
- [x] Create a unified markdown report that serves as a dashboard with an overview of all noise types.

### 4. Report Generation and Insight Aggregation

- [x] Modify `report_generation.py` to include specific analyses for each noise type.
- [x] Adapt report templates to accommodate the additional information and visualizations.
- [x] Implement comparative impact analyses between different noise types.
- [x] Update `insight_aggregation.py` to generate insights based on the new 7-phase structure and per-noise-type sensitivity.

### 5. Codebase and Documentation Translation

- [x] Translate all code comments, docstrings, log messages, and user-facing strings from Portuguese to English.
- [x] Translate all documentation files in `docs/` to English and rename files accordingly.
- [x] Update the main `README.md` to reflect the current project state in English.

### 6. Pipeline and Project Structure Refinement

- [x] Refactor and fix the main pipeline orchestration in `src/pipeline.py`.
- [x] Update the entry point script `run_pipeline.py`.
- [x] Clean up the project by removing obsolete scripts, backups, and temporary files.

### 7. Academic Publication Standardization

- [x] Standardize titles, legends, and labels in English for academic publications.
- [x] Establish a consistent color scheme for each tenant and phase.
- [x] Standardize units of measurement across all visualizations.
- [x] Create graph templates with academic publication quality.
- [x] Implement a system for exporting high-resolution images (300+ DPI).
- [ ] Add support for exporting in vector formats (SVG, EPS) - *Future work*.

## Phase 4: Debugging and Validation of Analysis Pipeline

### Issue: Analysis plots are not being generated

**Observation:** During the execution of the analysis pipeline (`run_pipeline.py`), it was noted that several key visualizations—including causality graphs, correlation heatmaps, cross-correlation plots, and phase comparison charts—were not being generated or displayed, despite the pipeline completing without critical errors.

**Investigation and Root Causes:**

1.  **Silent Failures in Plotting Functions:** The initial analysis of the pipeline's output log (`pipeline.log`) revealed numerous warnings of `Insufficient data` for various combinations of metrics, experimental phases, and rounds.
    - The core analysis modules (`src/analysis_correlation.py`, `src/analysis_causality.py`, `src/analysis_phase_comparison.py`) are designed to fail gracefully. They contain checks to validate if the input DataFrame for a given analysis contains sufficient data points (e.g., data from at least two tenants for correlation/causality).
    - If the data is insufficient, the function logs a warning and returns `None` or an empty object, preventing the corresponding plotting function from being called. This defensive programming approach prevents the pipeline from crashing but also hides the plotting issue from a cursory glance.

2.  **Data Ingestion and Filtering:** The root cause of the "insufficient data" issue lies in the data ingestion and filtering stages.
    - **Missing/Empty Data Files:** For some metrics defined in the configuration (`config/pipeline_config_sfi2.yaml`), the corresponding `.csv` files are either missing from the experimental data folders (`exp_data/`) or are empty for certain phases.
    - **Metric Naming Mismatches:** Inconsistencies between the `selected_metrics` in the configuration file and the actual filenames in the data directories can lead to data not being loaded for those metrics. For instance, a metric named `network_usage` was configured but lacked corresponding data files, causing downstream analysis to fail for that metric.
    - **Strict Phase Naming:** The `src/data_ingestion.py` module uses a hardcoded list of phase names (`PHASE_ORDER`). Any deviation in the directory names within `exp_data/` would cause that phase's data to be ignored entirely.

3.  **Visualization Configuration:**
    - **Display Names:** The `src/visualization_config.py` file, which maps metric names to human-readable labels for plots, was missing entries for some metrics being processed (e.g., `memory_usage`, `disk_usage`). While this does not prevent plot generation, it can lead to inconsistent or uninformative labels.
    - **Interactive Display:** Some plotting functions, like `plot_anomalies` in `src/analysis_anomaly.py`, were saving the figures to a file but not displaying them interactively (missing `plt.show()`), making it seem like they were not being generated at all during a manual run.

**Corrective Actions and Solutions:**

1.  **Data Integrity and Configuration Validation:**
    - **Action:** Implement a pre-flight check at the beginning of the pipeline to validate that the data for all `selected_metrics` and `selected_tenants` in the configuration exists and is not empty.
    - **Action:** The pipeline should produce a clear report or raise a more visible warning if required data is missing, rather than relying on scattered log messages.
    - **Solution:** The `network_usage` metric was removed from `config/pipeline_config_sfi2.yaml` to align the configuration with the available data, immediately resolving a subset of the "insufficient data" warnings.

2.  **Improved Logging and Error Reporting:**
    - **Action:** Enhance the logging in the analysis stages. Instead of just "Insufficient data," the logs should specify the metric, phase, and round, and state *why* the data is insufficient (e.g., "only 1 tenant found," "DataFrame is empty after filtering").

3.  **Code Adjustments for Robustness:**
    - **Action:** Modify the plotting functions to handle `None` or empty data inputs more gracefully, ensuring they don't attempt to render a plot if no data was passed from the analysis stage.
    - **Solution:** The `plot_anomalies` function was updated to include `plt.show()` for interactive display during debugging sessions. The `visualization_config.py` was updated to include display names for `memory_usage` and `disk_usage`, ensuring label consistency.

**Action Plan to Ensure Plot Generation:**

1.  **Implement Pre-Analysis Data Validation (High Priority):**
    - **Task:** Create a new function in `src/utils.py` or `src/pipeline.py` called `validate_data_availability`.
    - **Logic:** This function will iterate through the `selected_metrics` and `selected_tenants` from the pipeline configuration. For each combination, it will check for the existence and integrity of the corresponding data files for all experimental phases.
    - **Output:** It will generate a summary report at the start of the pipeline run, clearly listing which metrics or tenants have missing or empty data. The pipeline should proceed but with a clear and prominent warning.

2.  **Refine Analysis Functions (Medium Priority):**
    - **Task:** Review and refactor the main analysis functions (`run_correlation_analysis`, `run_causality_analysis`, `run_phase_comparison_analysis`, `run_cross_correlation_analysis`).
    - **Logic:** Ensure that the return value is consistent (e.g., always return a dictionary with results or a specific status like `SUCCESS`, `SKIPPED`).
    - **Logging:** Improve log messages to be more specific, as detailed in the "Investigation" section.

3.  **Centralize and Standardize Plotting (Medium Priority):**
    - **Task:** Create a centralized plotting module (e.g., `src/visualization/plots.py`) to consolidate plotting logic.
    - **Logic:** Each function in this module will take a standardized data object (e.g., a dictionary containing the DataFrame, title, labels) and be responsible for generating, saving, and (if configured) displaying the plot. This will reduce code duplication and make it easier to manage plotting behavior globally.
    - **Benefit:** This makes it straightforward to add a global "show plots" flag in the main configuration for debugging purposes.

4.  **Systematic Data Verification (Low Priority - Manual Task):**
    - **Task:** Manually review the contents of the `exp_data/sfi2-paper/` directory to confirm which metrics are available for each of the 7 phases.
    - **Action:** Update the default `pipeline_config.yaml` to reflect a known-good configuration that is guaranteed to produce all plots, which can serve as a baseline for future experiments.
