# Impact Detection in Multi-Tenant Kubernetes Environments
*Educational Material for Seminar*

## Presentation Objectives

This material explains in an educational way how our system calculates the impact between different tenants in a shared Kubernetes environment. We use three main components to build a score that identifies "noisy" tenants—those that most affect the performance of others.

## The Composite Score Formula

Our system uses a weighted equation to combine different impact indicators:

```
Noisy Score = (Causal Impact × 0.5) + (Correlation Strength × 0.3) + (Cross-Phase Variation × 0.2)
```

## 1. Causal Impact (50% of the score)

The causal impact analyzes cause-and-effect relationships between tenants, answering the question: "Does the behavior of tenant A **cause** changes in the behavior of tenant B?"

### Granger Causality

This test checks if the history of tenant A helps to predict the future behavior of tenant B, beyond what B can predict based only on its own history.

**Visual illustration of the concept:**
```
Model 1 (Restricted):   B(t) = f(B(t-1), B(t-2), ...) + error1
Model 2 (Unrestricted): B(t) = f(B(t-1), B(t-2), ..., A(t-1), A(t-2), ...) + error2
```

If error2 is significantly smaller than error1, then A "causes" B in the Granger sense.

**How we calculate it:**
1.  We transform the data to be stationary (ADF test).
2.  We test different lags.
3.  We calculate the p-value of the F-test.
4.  We convert it to a score: `impact = 1 - p_value`.

### Transfer Entropy (TE)

While Granger assumes linear relationships, TE can capture non-linear relationships between tenants.

**What it means:** It quantifies the information transferred from tenant A to tenant B over time.

**Conceptual formula:**
```
TE(A→B) = Uncertainty_about_B_knowing_only_B - Uncertainty_about_B_knowing_A_and_B
```

**Why we give more weight to TE:**
-   It captures non-linear relationships.
-   It is more robust to noise.
-   In the code, we multiply it by 5 to give more weight to the TE results.

## 2. Correlation Strength (30% of the score)

Correlation measures the degree of linear association between tenants, answering the question: "When tenant A changes, does tenant B also change in a similar way?"

**What we measure:**
-   Traditional correlation (at the same point in time).
-   Cross-correlation (with time lags).

### Traditional Correlation

**How we calculate it:**
1.  For each pair of tenants, we calculate the Pearson correlation coefficient.
2.  We use the absolute value (ignoring the sign).
3.  We store the significant values (usually |r| > 0.2).

### Cross-Correlation Function (CCF)

The CCF extends the analysis to include time lags, allowing the detection of patterns where one tenant influences another with a delay.

**Visual example:**

```
Maximum Correlation at Lag +3:
Tenant A: [a1, a2, a3, a4, a5, a6, a7, ...]
                     ↓   ↓   ↓   ↓
Tenant B: [b1, b2, b3, b4, b5, b6, b7, ...]
                 ↑   ↑   ↑   ↑
```

This pattern suggests that changes in Tenant A precede similar changes in Tenant B by 3 periods.

## 3. Cross-Phase Variation (20% of the score)

This component measures a tenant's sensitivity to environmental changes across different experimental phases (baseline, attack, recovery).

**What we calculate:**
```
Variation(%) = ((attack_value - baseline_value) / baseline_value) * 100
```

**Interpretation:**
-   High values (positive or negative): Tenant is very sensitive to environmental changes.
-   Values close to zero: Tenant is stable, little affected by environmental changes.

**Why it is important:** Tenants with high variation are often victims of "noisy neighbors" or are the cause of the problem themselves.

## Practical Example: Discovering the Noisy Tenant
