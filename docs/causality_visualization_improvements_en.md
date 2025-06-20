# Improvements in Causality Visualization

## Problems Fixed

1.  **Nodes hidden behind edges**: In the original visualizations, nodes were sometimes hidden behind edges, making them difficult to see. This happened because Matplotlib did not properly respect the drawing order (z-order) of the elements.

2.  **Lack of consolidated multi-metric visualization**: There was no standard way to visualize the causal relationship between different metrics in a single graph.

## Implemented Improvements

### 1. Enhanced Visualization of Causality Graphs

-   **Controlled z-order**: Implemented z-order control to ensure that nodes always appear in front of edges.
-   **Improved readability**: We increased the size of the nodes, improved color contrast, and adjusted the size and positioning of the labels.
-   **Enhanced aesthetics**: Smoother edges, consistent colors, and a more balanced layout.

### 2. Consolidated Multi-Metric Graph

-   **Multi-metric visualization**: We created a function that allows visualizing the causal relationship of multiple metrics in a single graph, using different colors for each metric.
-   **Explanatory legends**: Added legends that explain the meaning of the colors and threshold values.
-   **Automatic matrix type detection**: Intelligent detection of whether the matrix represents p-values (Granger) or Transfer Entropy (TE) values.

### 3. Pipeline Integration

-   The new visualizations have been integrated into the existing pipeline, maintaining compatibility with legacy code.
-   Added functions to automatically generate consolidated visualizations for each combination of experiment, phase, and round.
-   The original visualizations are kept, and the improved ones are generated in specific directories.

## How to Use

### Improved Individual Visualizations

The improved visualizations are automatically generated in:
`outputs/plots/causality/improved/`

### Consolidated Multi-Metric Visualizations

The consolidated visualizations are generated in:
`outputs/plots/causality/consolidated/`

### Manual Usage

To generate visualizations manually, use the functions:

```python
from src.improved_causality_graph import plot_improved_causality_graph, plot_consolidated_causality_graph

# For an individual graph
plot_improved_causality_graph(
    causality_matrix,
    output_path,
    threshold=0.05,
    directed=True,
    metric="cpu_usage"
)

# For a consolidated multi-metric graph
plot_consolidated_causality_graph(
    {
        "cpu_usage": cpu_matrix,
        "memory_usage": memory_matrix,
        "disk_io": disk_matrix
    },
    output_path,
    threshold=0.05,
    directed=True,
    phase="1 - Baseline",
    round_id="round-1"
)
```

## Interpreting the Visualizations

### Individual Visualization

-   **Nodes**: Represent tenants in the system.
-   **Edges**: Indicate a causal relationship between tenants.
-   **Edge thickness**: Represents the strength of the causal relationship.
-   **Edge direction**: Indicates the direction of causality (from cause to effect).

### Consolidated Visualization

-   **Edge colors**: Each color represents a different metric.
-   **Edge thickness**: Represents the strength of the causal relationship.
-   **Shared nodes**: The same tenant can have causal relationships in different metrics.

## Examples

Examples of generated visualizations can be found in the directories mentioned above. To generate test examples, run:

```bash
python test_causality_visualizations.py
```
