# Impact Calculations in a Multi-Tenant Kubernetes Environment

This document details the algorithms and formulas used to calculate the different impact scores in the multi-tenant time-series analysis system. These metrics are fundamental for identifying and classifying "noisy tenants" and for understanding the influence relationships between different workloads in a Kubernetes environment.

## 1. Overview of Composite Scores

The system uses a weighted combination of three main dimensions to calculate the total impact of a tenant on the environment:

```
Noisy Score = (Causal Impact × 0.5) + (Correlation Strength × 0.3) + (Cross-Phase Variation × 0.2)
```

Where:
-   **Causal Impact (50%)**: Assesses how much a tenant causes changes in others.
-   **Correlation Strength (30%)**: Measures the strength of linear relationships between tenants.
-   **Cross-Phase Variation (20%)**: Quantifies behavioral changes during experimental phases.

## 2. Causal Impact Calculation

The causal impact is calculated by combining two complementary causality metrics: Granger Causality and Transfer Entropy.

### 2.1 Granger Causality

Granger causality tests whether the past values of a time series X help predict the future values of a time series Y, beyond what can be predicted using only the past values of Y.

**Calculation procedure:**

1.  For each pair of tenants (A, B) and each relevant metric:

    a.  Extract the time series of the two tenants.
    b.  Check for stationarity (ADF test) and apply differencing if necessary.
    c.  Compare two models:
        -   Restricted model: Y(t) = f(Y(t-1), Y(t-2), ..., Y(t-p))
        -   Unrestricted model: Y(t) = f(Y(t-1), ..., Y(t-p), X(t-1), ..., X(t-p))
    d.  Calculate the p-value of the F-test for model comparison.
    e.  Convert to an impact score: `impact = 1 - p_value`.
    f.  Store the lowest p-value found considering different lags.

**Implementation in the code:**
```python
# Test Granger causality for each lag
from statsmodels.tsa.stattools import grangercausalitytests
test_results = grangercausalitytests(data, maxlag=maxlag, verbose=False)

# Get the lowest p-value among all tested lags
p_values = [test_results[lag][0]['ssr_chi2test'][1] for lag in range(1, maxlag+1)]
min_p_value = min(p_values) if p_values else np.nan
```

### 2.2 Transfer Entropy (TE)

Transfer Entropy measures the amount of information that flows from one time series to another, quantifying the reduction in uncertainty about the future values of a series when we know the past values of another series.

**Calculation procedure:**

1.  For each pair of tenants (A, B) and each metric:

    a.  Extract the time-aligned series.
    b.  Discretize the values of the continuous series into bins (default: 8 bins).
    c.  Calculate the Transfer Entropy:
        ```
        TE(X→Y) = H(Yt+1|Yt) - H(Yt+1|Yt,Xt)
        ```
        where H represents the conditional entropy.
    d.  Higher TE values indicate greater information transfer (greater causality).

**Implementation in the code:**
```python
def _transfer_entropy(target_series, source_series, bins=8, k=1):
    # Discretize the data
    source_disc = np.digitize(source_series,
                            np.linspace(np.min(source_series),
                                        np.max(source_series), bins))
    target_disc = np.digitize(target_series,
                            np.linspace(np.min(target_series),
                                        np.max(target_series), bins))

    # Calculate TE using pyinform
    from pyinform.transferentropy import transfer_entropy
    te_value = transfer_entropy(source_disc, target_disc, k=k)
    return te_value
```

### 2.3 Combination of Causality Metrics

The final causal impact score is calculated as a weighted combination:

```python
# In the final metrics calculation
tenant_metrics[source]['causality_impact_score'] += np.mean(causal_values)  # For Granger
tenant_metrics[source]['causality_impact_score'] += np.mean(te_values) * 5  # Higher weight for TE
```

Transfer Entropy receives a higher weight (5x) because it is a more robust measure that captures non-linear relationships.

## 3. Correlation Strength Calculation

Correlation strength measures the intensity of linear relationships between pairs of tenants, regardless of the direction of influence.

**Calculation procedure:**
