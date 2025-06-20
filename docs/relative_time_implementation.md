# Implementation of Relative Time in Visualizations

## Overview

This implementation modifies all pipeline visualization functions to display time in seconds from the beginning of each phase (relative time), instead of absolute timestamps. This change significantly improves the readability and interpretability of the graphs, making it easier to compare different phases and experiments.

Previously, some visualizations alternated between seconds and minutes depending on the duration of the data. Now, all visualizations consistently use seconds as the unit for the X-axis, regardless of the duration of the phase or experiment.

## Modified Files

1.  `analysis_descriptive.py`:
    -   `plot_metric_timeseries_multi_tenant` function
    -   `plot_anomalies` function
    -   `plot_metric_timeseries_multi_tenant_all_phases` function

2.  `analysis_anomaly.py`:
    -   `plot_anomalies` function

3.  `analysis_sliding_window.py`:
    -   `plot_sliding_window_correlation` function
    -   `plot_sliding_window_causality` function

## Main Modifications

### Relative Time Calculation
In each visualization function, we implemented the calculation of relative time:

```python
# Calculate the start timestamp of the phase
phase_start = subset['timestamp'].min()

# Convert timestamps to relative seconds
elapsed = (group['timestamp'] - phase_start).dt.total_seconds()
```

### Data Type Handling
We added explicit conversions to ensure compatibility with Matplotlib:

```python
# Secure conversion to compatible numeric types
x_plot_data = np.array(elapsed_times, dtype=float)
y_plot_data = pd.to_numeric(values).to_numpy(dtype=float, na_value=np.nan)
```

### Label Standardization
We standardized all X-axis labels to use seconds in all visualizations:

```python
# Always use seconds for consistency
time_unit = 1  # Always use seconds
x_label = "Seconds since phase start"

# Apply to the X-axis in all visualizations
ax.set_xlabel(x_label)
```

We also updated all time units in sliding windows to use seconds:

```python
# Before
window_size='5min'
step_size='1min'

# After
window_size='300s'  # 5min converted to seconds
step_size='60s'     # 1min converted to seconds
```

## Benefits

1.  **Better Interpretability**: The graphs now show how much time has passed since the beginning of the phase, making it easier to understand the sequence and duration of events.

2.  **Easier Comparison**: It is possible to directly compare different phases and experiments, as they all start from "time zero".

3.  **Cleaner Visualization**: Eliminates the complexity of reading full timestamps in the axis labels.

4.  **Quantitative Analysis**: Facilitates the precise measurement of time between events or anomalies within the same visualization.

## Validation

The implementation was validated by running the complete pipeline with the 3-round configuration:

```bash
python -m src.run_unified_pipeline --config config/pipeline_config_3rounds.yaml
```

The results show that all 798 visualizations were generated correctly with the relative time format.

## Next Steps

1.  **Configuration Option**: Add a configuration parameter to allow switching between relative and absolute time.

2.  **Additional Annotations**: Consider adding markers or annotations at key points in the visualizations.

3.  **Interactive Zoom**: Explore options for interactive visualizations that allow zooming in on specific time regions.
