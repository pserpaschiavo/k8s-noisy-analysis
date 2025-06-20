# Work Plan Update - July/2025

## Current Project Status

- ✅ **Completed**: Main project structure implemented, data ingestion (including direct Parquet loading support), segmentation, persistence, basic descriptive, correlation, and causality analysis components, insight aggregation, multi-round analysis, direct ingestion of parquet files with relative and absolute path resolution. Bug fixes in the pipeline, including issues with dict comparisons in the Granger test, obsolete use of `Series.fillna` in the causality module, and problems in the insight aggregation stage. Implementation of `experiment_folder` support to specify specific experiments within `data_root`.
- 🔄 **In progress**: Refinement of the Causality module with Transfer Entropy, complete unit tests, sliding window analysis, missing/incomplete visualizations.
- ❌ **Pending**: Comparative reports between experimental phases, full integration of all components.

## Recent Implementations (June/2025)

### 1. Full Support for the `experiment_folder` Parameter

- ✅ **Implemented in the pipeline core**: Integrated into `DataIngestionStage` to allow specifying specific experiments.
- ✅ **Helper functions**: Added `get_experiment_folder` and `get_experiment_dir` in `src/parse_config.py`.
- ✅ **Path duplication prevention**: Robust logic to avoid problems with path concatenation.
- ✅ **Flag propagation**: Implemented a system to track when `experiment_folder` is applied.
- ✅ **Default configuration**: Added `DEFAULT_EXPERIMENT_FOLDER` constant in `src/config.py`.

### 2. Support Scripts and Documentation

- ✅ **Convenience scripts**: Created `run_pipeline_with_experiment.py` and `run_pipeline_3_rounds.py`.
- ✅ **Comprehensive documentation**: Created `docs/experiment_folder_guide.md` and `docs/README_experiment_folder.md`.
- ✅ **Tests**: Developed `test_experiment_folder.py`, `debug_experiment_folder.py`, and `src/test_experiment_folder_parameter.py`.
- ✅ **Test configuration**: Created `config/pipeline_config_3rounds.yaml`.
- ✅ **Automation**: `make_scripts_executable.sh` script to ensure execution permissions.

## Priorities for the Next Cycle (July-August/2025)

### High Priority

1.  **Unified Pipeline Consolidation**:
    -   ❌ Unify the different pipeline implementations (`pipeline.py`, `pipeline_new.py`, `pipeline_with_sliding_window.py`) into a modular, plugin-based architecture.
    -   ❌ Create a central configuration system to control which analysis modules are activated.
    -   ❌ Implement a dependency mechanism between pipeline stages to ensure correct execution.

2.  **Visualization Correction and Verification**:
    -   ❌ Confirm that all implemented visualizations are being generated correctly.
    -   ❌ Investigate why only covariance visualizations are being generated (correlation is missing).
    -   ❌ Integrate sliding window plots into the main pipeline.

3.  **Full Integration of the `experiment_folder` Parameter**:
    -   ❌ Extend support to all scripts and analysis modes (including sliding windows and multi-round analysis).
    -   ❌ Implement validation to ensure the experiment directory exists before execution.
    -   ❌ Add detailed logging about the selected experiment.

### Medium Priority

1.  **Improved Unit Tests**:
    -   ❌ Expand test coverage for extreme and edge cases.
    -   ❌ Implement integration tests for the complete pipeline.
    -   ❌ Create an automated testing environment for continuous validation.

2.  **Technical Documentation**:
    -   ❌ Create detailed documentation on the pipeline architecture and data flow.
    -   ❌ Document all available configuration parameters.
    -   ❌ Develop step-by-step guides for common analyses.

3.  **Performance Optimization**:
    -   ❌ Implement a caching system for intermediate results.
    -   ❌ Add parallelization for independent analyses.
    -   ❌ Optimize memory usage for large datasets.

### Low Priority

1.  **Improved User Interface**:
    -   ❌ Develop a more user-friendly command-line interface.
    -   ❌ Consider implementing a simple web interface for visualizing results.
    -   ❌ Create an alerting system for detected anomalies.

2.  **Expansion of Analytical Functionalities**:
    -   ❌ Add new causality analysis methods.
    -   ❌ Implement more sophisticated anomaly detection.
    -   ❌ Develop time-series forecasting capabilities.

3.  **Interoperability with Other Systems**:
    -   ❌ Create exporters for common formats (JSON, CSV, etc.).
    -   ❌ Develop APIs for integration with other systems.
    -   ❌ Implement mechanisms to import data from various sources.

## Technical Roadmap

### Phase 1: Consolidation and Stabilization (July/2025)
1.  Week 1-2: Pipeline unification and visualization correction
2.  Week 3-4: Extensive testing and technical documentation

### Phase 2: Optimization and Extensibility (August/2025)
1.  Week 1-2: Implementation of caching system and parallelization
2.  Week 3-4: Development of the plugin-based architecture

### Phase 3: User Experience and Advanced Features (September/2025)
1.  Week 1-2: User interface improvement
2.  Week 3-4: Addition of advanced analytical functionalities

## Architectural Considerations

To move towards a more modular and extensible architecture, we recommend:

1.  **Plugin-Based Architecture**:
    -   Define a clear interface for analysis modules
    -   Create a plugin registration and discovery system
