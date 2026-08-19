# Phase 5: Discrete First-Order Latent-State Baseline (DG-HMM) Report

**Project**: EHR-Neural-SDE  
**Date**: August 18, 2026  
**Model Type**: Discrete First-Order Latent-State Baseline  
**Source Code**: `src/models/dg_hmm.py`  
**Evaluation Script**: `src/evaluation/evaluate_dg_hmm.py`  
**Saved Artifacts**: `outputs/dg_hmm_baseline/model.json`, `outputs/dg_hmm_baseline/model.pkl`  

---

## IMPORTANT RESEARCH LABEL & WARNING

> [!IMPORTANT]
> **Baseline Classification**: This model is explicitly classified and labeled as a **Discrete First-Order Latent-State Baseline**.
> - It is **NOT** a continuous-time model.
> - It does **NOT** use time gaps ($\Delta T$) or continuous time parameterization ($T$).
> - It does **NOT** implement Neural SDEs, continuous drift/diffusion, or neural differential equations.
> - Its purpose is to establish an empirical discrete baseline against which our continuous-time Neural SDE will be benchmarked in subsequent phases.

---

## A. Baseline Architecture

The baseline model parameterizes longitudinal observation sequences as a discrete-time, first-order Markov process with $K=8$ latent states and diagonal Gaussian emissions:

$$\text{Initial Distribution: } \pi[k] = P(z_1 = k) \quad \text{for } k \in \{0, \dots, 7\}$$

$$\text{Transition Matrix: } A[i, j] = P(z_t = j \mid z_{t-1} = i) \quad \text{for } i, j \in \{0, \dots, 7\}$$

$$\text{Emission Distribution: } P(x_{t} \mid z_t = k) = \prod_{d=0}^{4} \mathcal{N}\left(x_{t, d} ; \mu_{k, d}, \sigma_{k, d}^2\right)$$

- **Latent States ($K$)**: 8
- **Feature Dimension ($D$)**: 5 (Heart Rate, Respiratory Rate, SpO2, Systolic BP, Diastolic BP)
- **Covariance Structure**: Independent diagonal Gaussian per state

---

## B. State Representation & Cluster Partitioning

Each observation timestamp $t$ in a trajectory is mapped to a discrete latent health state $z_t \in \{0, \dots, 7\}$. Hard-assignment clustering partitions the continuous observation space into 8 distinct physiological risk profiles.

---

## C. Clustering Method

To learn latent states unsupervised without feeding `NaN`s into clustering algorithms, observations are transformed into a **10-dimensional missingness-aware representation**:

$$V_{t} = \big[ X_{\text{filled}, t} \,,\, M_{t} \big] \in \mathbb{R}^{10}$$

where:
- $X_{\text{filled}, t}$ contains the 5 normalized vital sign features with missing values set to `0.0` (the training mean).
- $M_{t} \in \{0, 1\}^5$ is the binary clinical observation mask.

Deterministic $K$-Means clustering (`scikit-learn`, $K=8$, `random_state=42`, `n_init=10`) was fitted on all 13,205 10D vectors from the 98 training trajectories.

---

## D. Missing-Value Handling

1. **For State Learning**: Concatenating mask $M$ allows $K$-Means to distinguish $X=0, M=1$ (observed at the training mean) from $X=0, M=0$ (unobserved feature).
2. **For Parameter Estimation**: When estimating emission parameters $(\mu_{k, d}, \sigma_{k, d})$, missing values ($M=0$) are **completely excluded**. Means and standard deviations are computed strictly using cells where $M[t, d] == 1$ and $z_t == k$.

---

## E. Transition Matrix & Initial Distribution Estimation

State sequence predictions $Z = z_1 \to z_2 \to \dots \to z_N$ were generated for all training trajectories.

1. **Initial Distribution ($\pi$)**:
   $$\pi[k] = \frac{\text{count}(z_1 = k) + \epsilon}{\sum_{k'=0}^{7} \big(\text{count}(z_1 = k') + \epsilon\big)}$$
2. **Transition Matrix ($A$)**:
   $$A[i, j] = \frac{\text{count}(i \to j) + \epsilon}{\sum_{j'=0}^{7} \big(\text{count}(i \to j') + \epsilon\big)}$$

Additive Laplace smoothing ($\epsilon = 10^{-3}$) was applied to ensure all probabilities remain non-zero and rows sum to $1.0$.

---

## F. Emission Parameter Estimation

For each state $k \in \{0, \dots, 7\}$ and feature $d \in \{0, \dots, 4\}$:
- $\mu_{k, d} = \text{mean}\big(\{ X_{t, d} \mid z_t = k \text{ and } M_{t, d} = 1 \}\big)$
- $\sigma_{k, d} = \max\left( \text{std}\big(\{ X_{t, d} \mid z_t = k \text{ and } M_{t, d} = 1 \}\big) \,,\, \sigma_{\text{min}} \right)$

where $\sigma_{\text{min}} = 10^{-2}$ is a minimum standard deviation floor to prevent zero-variance collapse.

---

## G. Generation Procedure (`generate_trajectory(n_steps)`)

To generate a synthetic baseline trajectory of length $N$:
1. Sample initial state $z_0 \sim \text{Categorical}(\pi)$.
2. For step $t = 1 \dots N-1$:
   - Sample $z_t \sim \text{Categorical}(A[z_{t-1}])$.
3. For step $t = 0 \dots N-1$ and feature $d = 0 \dots 4$:
   - Sample continuous value $x_{t, d} \sim \mathcal{N}(\mu_{z_t, d}, \sigma_{z_t, d}^2)$.

Returns continuous observation array $X_{\text{syn}} \in \mathbb{R}^{N \times 5}$ and state sequence $Z_{\text{syn}} \in \mathbb{Z}^N$.

---

## H. Limitations of the Baseline

1. **First-Order Markov Assumption**: Ignores long-range temporal dependencies and cumulative patient disease trajectories beyond $z_{t-1}$.
2. **Discrete Time Steps**: Completely ignores physical time deltas ($\Delta T$). A step of 1 minute is treated identically to a step of 24 hours.
3. **Independent Diagonal Emissions**: Assumes vital signs are conditionally independent given state $k$.
4. **Hard-Assignment Clustering**: Uses $K$-Means clustering rather than full expectation-maximization (EM / Baum-Welch) soft-state posterior inference.

---

## I. Empirical Results & Evaluation

### 1. State Occupancy Distribution ($K=8$)
Latent state occupancy across 13,205 training observation timestamps:

| State ($k$) | Occupancy Count | Occupancy % | Description |
| :---: | :---: | :---: | :--- |
| **0** | 1,809 | 13.70% | Mildly Elevated Vitals |
| **1** | 1,376 | 10.42% | Stable Low Respiratory Rate |
| **2** | 2,576 | 19.51% | High Occupancy Baseline Vitals |
| **3** | 746 | 5.65% | Hypotensive / High-Risk State |
| **4** | 2,427 | 18.38% | High Occupancy Routine Vitals |
| **5** | 1,100 | 8.33% | Tachycardic / Elevated HR State |
| **6** | 1,730 | 13.10% | Hypertensive State |
| **7** | 1,441 | 10.91% | Moderate Variance State |

### 2. Transition Matrix Entropy
- **Average Transition Entropy**: **`2.3512 bits`** (out of maximum possible $3.00$ bits for $K=8$).
- *Interpretation*: The learned transition matrix exhibits structured transition preferences rather than uniform random transitions.

### 3. Feature Distribution Comparison (Normalized Space)

| Feature Name | Real Train Mean | Synthetic Baseline Mean | Real Train Std | Synthetic Baseline Std |
| :--- | :---: | :---: | :---: | :---: |
| **Heart Rate** | `-0.0000` | `0.1160` | `0.9999` | `0.9299` |
| **Respiratory Rate** | `-0.0000` | `0.0755` | `0.9999` | `0.9094` |
| **SpO2** | `0.0000` | `0.0710` | `0.9999` | `0.8963` |
| **Systolic BP** | `0.0000` | `-0.0491` | `0.9999` | `0.9379` |
| **Diastolic BP** | `0.0000` | `-0.0180` | `0.9999` | `0.9152` |

---

## J. Reproducibility & Saved Artifacts

- **Random Seed**: `42`
- **Model Saved**: `outputs/dg_hmm_baseline/model.json` and `outputs/dg_hmm_baseline/model.pkl`
- **Unit Test Suite**: Passed all 11 unit tests in `tests/test_dg_hmm.py` (`Ran 11 tests in 2.095s, OK`).

---

*Phase 5 Baseline Report for EHR-Neural-SDE.*
