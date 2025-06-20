# Visualization of Impact in a Multi-Tenant Kubernetes Environment
*Supplemental Material for Seminar*

This document presents visualizations and examples to illustrate how the different components of the impact score work in practice, complementing the main educational material.

## Visual Examples of Causality

### Example 1: Causality Detection via Granger's Test

Considering two time series of CPU utilization by different tenants:

```
Tenant A (possible cause): [60%, 75%, 85%, 78%, 65%, 55%, 70%, 80%]
                               ↓    ↓    ↓    ↓    ↓    ↓    ↓
Tenant B (possibly affected):   [40%, 50%, 65%, 70%, 72%, 60%, 55%, 65%]
```

When applying the Granger causality test:

1.  **Restricted Model**: Prediction of B based only on previous values of B
    -   Mean squared error: 85.2

2.  **Unrestricted Model**: Prediction of B based on previous values of B and A
    -   Mean squared error: 45.7

3.  **F-Test Result**: p-value = 0.03 (significant at the 5% level)

4.  **Interpretation**: With 97% confidence (1 - p-value), we can say that the activity of Tenant A causally influences Tenant B.

### Example 2: Transfer Entropy in Action

Let's consider two tenants with memory usage patterns over time:

```
Tenant C (discretized): [1, 2, 4, 6, 7, 6, 5, 4, 3, 5, 7]
Tenant D (discretized): [2, 2, 3, 5, 6, 7, 6, 5, 4, 4, 6]
```

Transfer Entropy calculates:

1.  **Uncertainty about D using only D**: H(D_future | D_past) = 1.2 bits
2.  **Uncertainty about D using C and D**: H(D_future | D_past, C_past) = 0.7 bits
3.  **TE(C→D)** = 1.2 - 0.7 = 0.5 bits

4.  **TE(D→C)** = 0.2 bits (calculated inversely)

5.  **Interpretation**: Since TE(C→D) > TE(D→C), there is a greater flow of information from C to D than vice-versa, suggesting that C has a greater causal influence on D.

## Correlation Visualization

### Example: Correlation Heatmap between Tenants

Correlation matrix for CPU utilization among 4 tenants:

```
       | Tenant A | Tenant B | Tenant C | Tenant D |
-------|----------|----------|----------|----------|
Tenant A|   1.00   |   0.32   |   0.78   |  -0.15   |
Tenant B|   0.32   |   1.00   |   0.23   |   0.65   |
Tenant C|   0.78   |   0.23   |   1.00   |  -0.08   |
Tenant D|  -0.15   |   0.65   |  -0.08   |   1.00   |
```

**Heatmap Interpretation:**
-   **Strong positive correlation** (A-C: 0.78): When A increases, C tends to increase
-   **Moderate positive correlation** (B-D: 0.65): When B increases, D tends to increase
-   **Weak negative correlation** (A-D: -0.15): Slight opposite trend
-   **Diagonal correlation = 1.00**: Self-correlation (always 1)

### Visual Cross-Correlation Function (CCF)

Cross-correlation between Tenants E and F over time:

```
Lag   CCF Value
-5     0.12
-4     0.18
-3     0.25
-2     0.42
-1     0.56
 0     0.65
+1     0.82  ← Maximum (Tenant E influences Tenant F with lag +1)
+2     0.67
+3     0.40
+4     0.22
+5     0.10
```

**Interpretation:**
-   The highest value (0.82) occurs at lag +1
-   This suggests that changes in Tenant E precede similar changes in Tenant F by 1 period
-   The contemporary correlation (lag 0) is also strong (0.65)

## Example of Cross-Phase Variation

### Behavior of a Tenant Across Three Phases

Average latency data (ms) for Tenant G:
