# Reference Formulas and Algorithms for Impact Calculation

This document serves as a technical reference with the main formulas and procedures used to calculate the different components of the impact score in multi-tenant Kubernetes environments.

## Granger Causality

### Step-by-Step Procedure

1.  **Stationarity Test (ADF)**:
    ```
    H0: The time series contains a unit root (non-stationary)
    H1: The time series is stationary
    ```
    If p-value > 0.05, we apply differencing to make the series stationary.

2.  **Causality Test**:

    Comparison between:

    **Restricted Model**:
    ```
    Y(t) = α₀ + α₁Y(t-1) + ... + αₚY(t-p) + ε(t)
    ```

    **Unrestricted Model**:
    ```
    Y(t) = α₀ + α₁Y(t-1) + ... + αₚY(t-p) + β₁X(t-1) + ... + βₚX(t-p) + ε(t)
    ```

3.  **F-Test**:
    ```
    F = ((RSS_r - RSS_ur)/m) / (RSS_ur/(n-k))
    ```
    where:
    -   RSS_r: Sum of squared residuals of the restricted model
    -   RSS_ur: Sum of squared residuals of the unrestricted model
    -   m: Number of restrictions (additional parameters in the unrestricted model)
    -   n: Number of observations
    -   k: Total number of parameters in the unrestricted model

4.  **Conversion to Causality Score**:
    ```
    score_granger = 1 - p_value
    ```

5.  **Implementation with Statsmodels**:
    ```python
    from statsmodels.tsa.stattools import grangercausalitytests
    result = grangercausalitytests(data, maxlag=5, verbose=False)
    p_values = [result[lag][0]['ssr_chi2test'][1] for lag in range(1, 6)]
    min_p_value = min(p_values) if p_values else np.nan
    score_granger = 1 - min_p_value
    ```

## Transfer Entropy

### Mathematical Formula

The Transfer Entropy from X to Y is defined as:

```
TE(X→Y) = H(Yₜ₊₁|Yₜ) - H(Yₜ₊₁|Yₜ,Xₜ)
```

Where:
-   H(Yₜ₊₁|Yₜ) is the conditional entropy of Yₜ₊₁ given Yₜ
-   H(Yₜ₊₁|Yₜ,Xₜ) is the conditional entropy of Yₜ₊₁ given Yₜ and Xₜ

### Conditional Entropy Calculation

```
H(Y|X) = -∑∑ p(x,y) log(p(y|x))
```

### Procedure for TE Calculation

1.  **Discretization**:
    ```python
    def discretize(series, bins=8):
        return np.digitize(series,
                           np.linspace(np.min(series),
                                       np.max(series), bins))
    ```

2.  **TE Calculation**:
    ```python
    def transfer_entropy(source, target, k=1):
        from pyinform.transferentropy import transfer_entropy
        source_disc = discretize(source)
        target_disc = discretize(target)
        te = transfer_entropy(source_disc, target_disc, k=k)
        return te
    ```

3.  **Conversion to Score**:
    ```python
    # We apply a higher weight to TE for capturing non-linear relationships
    score_te = te_value * 5
    ```
