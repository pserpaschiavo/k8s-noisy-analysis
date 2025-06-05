# Using the `experiment_folder` Parameter

## Overview

The `experiment_folder` parameter allows you to specify a specific experiment folder within the data root directory when running the data analysis pipeline. This feature is particularly useful when you have multiple experiment datasets organized in different subdirectories.

## Key Benefits

- **Organization**: Keep multiple experiments neatly organized in separate folders
- **Flexibility**: Easily switch between different experiments without changing the entire data path
- **Compatibility**: Works with both the standard pipeline and special analysis modes (sliding window, multi-round)

## Basic Usage

1. **Add to your configuration file**:
   ```yaml
   data_root: /path/to/data
   experiment_folder: my-experiment-name
   ```

2. **Run the pipeline with the wrapper**:
   ```bash
   ./run_pipeline_with_experiment.py --config config/your_config.yaml
   ```

For more detailed information, see the [Experiment Folder Guide](experiment_folder_guide.md).
