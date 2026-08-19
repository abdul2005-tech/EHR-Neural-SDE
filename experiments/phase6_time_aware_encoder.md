# Phase 6: Time-Aware Encoder & Latent Projection Module Report

**Project**: EHR-Neural-SDE  
**Date**: August 18, 2026  
**Source Files**:
- `src/models/time_aware_encoder.py` (`TimeAwareEncoder`, `TimeFeatureEncoder`)
- `src/models/latent_projection.py` (`LatentProjection`)
- `tests/test_time_aware_encoder.py` (Unit test suite)

---

## 1. Executive Summary

In Phase 6, we implemented and unit-tested the first neural component of our proposed Neural SDE architecture: the **Time-Aware Encoder** (`TimeAwareEncoder`) and **Latent Projection** (`LatentProjection`).

The Time-Aware Encoder processes point-wise irregularly sampled longitudinal EHR observations tuple $(X_t, M_t, T_t, \Delta T_t)$ and projects them into a continuous point-wise hidden representation $h_t \in \mathbb{R}^{64}$. The Latent Projection module then maps $h_t$ into initial latent state representations $z_t \in \mathbb{R}^{32}$, which will later serve as the continuous state variable governed by the Neural SDE latent drift and diffusion functions.

---

## A. Why Time Information is Explicitly Represented

In longitudinal Electronic Health Records, clinical observations are recorded at irregular time intervals. Physical health status evolves continuously over calendar time $T$, while elapsed time since the previous visit $\Delta T = T_t - T_{t-1}$ indicates observation density and urgency.

1. **Relative Time ($T_t$)**: Anchors patient trajectory relative to ICU admission ($t_{\text{rel}} = 0.0$ hours), capturing circadian rhythms and disease progression over the course of hospitalization.
2. **Time Delta ($\Delta T_t$)**: Captures sampling frequency dynamics. A small $\Delta T$ (e.g. 5 minutes) signifies high-frequency acute monitoring, whereas a large $\Delta T$ (e.g. 12 hours) indicates stable, routine care.

---

## B. Why Observation Masks ($M_t$) are Included

In Phase 4, missing physiological features were normalized numerically to $0.0$ (the training mean). Without an explicit observation mask $M_t \in \{0, 1\}^5$, a neural network cannot distinguish:
- $X_{t, d} = 0.0$ with $M_{t, d} = 1 \implies$ Feature $d$ was measured and equals the population average.
- $X_{t, d} = 0.0$ with $M_{t, d} = 0 \implies$ Feature $d$ was unobserved / missing.

Including $M_t$ as a direct input to the encoder enables missingness-aware point representations and prevents informative missingness bias.

---

## C. Encoder Architecture (`TimeAwareEncoder`)

```
                  ┌──────────────────────┐
                  │ T_t , DeltaT_t [N,2] │
                  └──────────┬───────────┘
                             │
                             ▼
                 ┌────────────────────────┐
                 │  TimeFeatureEncoder    │
                 │  Linear(2 -> 16)       │
                 │  ReLU()                │
                 │  Linear(16 -> 16)      │
                 └──────────┬─────────────┘
                            │
                            ▼
                  time_embedding_t [N, 16]
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
   X_t [N, 5]              M_t [N, 5]      time_embedding_t [N, 16]
    │                       │                       │
    └───────────────────────┼───────────────────────┘
                            │
                            ▼
              Concat [X_t, M_t, time_embed_t] [N, 26]
                            │
                            ▼
                    Encoder MLP
                    Linear(26 -> 64)
                    ReLU()
                    Linear(64 -> 64)
                            │
                            ▼
                        h_t [N, 64]
```

---

## D. Time Feature Representation (`TimeFeatureEncoder`)

Because relative time $T_t$ (0 – 500 hours) and time delta $\Delta T_t$ (0.01 – 48 hours) operate on distinct numerical scales, `TimeFeatureEncoder` maps `[T_t, DeltaT_t]` into a 16-dimensional dense embedding space:

$$\text{TimeInput}_t = \big[ T_t \,,\, \Delta T_t \big] \in \mathbb{R}^2$$

$$\text{TimeEmbedding}_t = \text{Linear}_{16 \to 16}\Big( \text{ReLU}\big( \text{Linear}_{2 \to 16}(\text{TimeInput}_t) \big) \Big) \in \mathbb{R}^{16}$$

---

## E. Hidden Dimension ($h_t \in \mathbb{R}^{64}$)

The concatenated input $\big[ X_t \,,\, M_t \,,\, \text{TimeEmbedding}_t \big] \in \mathbb{R}^{26}$ is processed by a 2-layer MLP to produce continuous hidden point representations $h_t \in \mathbb{R}^{64}$.

---

## F. Latent Dimension ($z_t \in \mathbb{R}^{32}$)

The `LatentProjection` module maps $h_t \in \mathbb{R}^{64}$ into a lower-dimensional latent space $z_t \in \mathbb{R}^{32}$:

$$z_t = \text{Linear}_{64 \to 32}\Big( \text{ReLU}\big( \text{Linear}_{64 \to 64}(h_t) \big) \Big) \in \mathbb{R}^{32}$$

This 32-dimensional continuous latent state $z_t$ will serve as the initial state $z_0 = z(0)$ for continuous SDE trajectory solving in Phase 7.

---

## G. Unit Test Results (`tests/test_time_aware_encoder.py`)

All **11 unit tests** passed cleanly:

| ID | Test Description | Status |
| :---: | :--- | :---: |
| **1** | Input $X$ shape `[N, 5]` accepted | **PASSED** |
| **2** | Input $M$ shape `[N, 5]` accepted | **PASSED** |
| **3** | Input $T$ shape `[N]` accepted | **PASSED** |
| **4** | Input $\Delta T$ shape `[N]` accepted | **PASSED** |
| **5** | Output $h$ shape is `[N, 64]` | **PASSED** |
| **6** | Latent output $z$ shape is `[N, 32]` | **PASSED** |
| **7** | Different $\Delta T$ produces distinct $h$ when $X, M, T$ identical | **PASSED** |
| **8** | Different $T$ produces distinct $h$ when $X, M, \Delta T$ identical | **PASSED** |
| **9** | Missingness mask $M$ explicitly alters encoder output $h$ | **PASSED** |
| **10** | Backpropagation produces finite, valid gradients for all parameters | **PASSED** |
| **11** | Zero NaNs or infinities produced for valid input tensors | **PASSED** |

---

## H. Example Tensor Shapes

### Single Unbatched Trajectory ($N=10$)
- Input $X$: `torch.Size([10, 5])`
- Input $M$: `torch.Size([10, 5])`
- Input $T$: `torch.Size([10])`
- Input $\Delta T$: `torch.Size([10])`
- **Output $h$**: `torch.Size([10, 64])`
- **Output $z$**: `torch.Size([10, 32])`

### Batched Trajectories ($B=4, N=15$)
- Input $X$: `torch.Size([4, 15, 5])`
- Input $M$: `torch.Size([4, 15, 5])`
- Input $T$: `torch.Size([4, 15])`
- Input $\Delta T$: `torch.Size([4, 15])`
- **Output $h$**: `torch.Size([4, 15, 64])`
- **Output $z$**: `torch.Size([4, 15, 32])`

---

*Phase 6 Encoder Report for EHR-Neural-SDE.*
