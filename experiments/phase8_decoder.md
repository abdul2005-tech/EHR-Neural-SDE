# Phase 8: Observation Decoder & Continuous-Time Forward-Pass Integration

## Executive Summary

Phase 8 completes the continuous-time latent generative pipeline of the **EHR-Neural-SDE** framework by introducing the observation decoder \(\text{EHRDecoder}: \mathbb{R}^{32} \to \mathbb{R}^5\), masked reconstruction loss functions, feature scaling utilities, unit tests, and a full forward-pass integration test on real patient trajectories from MIMIC-IV.

---

## A. Decoder Architecture

The `EHRDecoder` is a point-wise Multi-Layer Perceptron (MLP) mapping latent SDE states \(z(t) \in \mathbb{R}^{32}\) back into the 5-dimensional physiological observation space \(\hat{X}(t) \in \mathbb{R}^5\):

\[
z(t) \xrightarrow{\text{Linear}(32 \to 64)} \text{ReLU} \xrightarrow{\text{Linear}(64 \to 64)} \text{ReLU} \xrightarrow{\text{Linear}(64 \to 5)} \hat{X}(t)
\]

### Key Design Choices:
1. **Point-wise MLP Decoding**: Operates independently on each latent state along the sequence dimension without autoregressive feedback, RNNs, LSTMs, or Transformers.
2. **Exclusion of Mask & Time Inputs**: The decoder receives only latent vector \(z(t)\). Temporal dynamics and missingness patterns influence predictions indirectly through the `TimeAwareEncoder` and continuous `NeuralSDE` trajectory.

---

## B. Input / Output Dimensions

The 5 physiological variables predicted by the decoder correspond to:
- `0`: Heart Rate (bpm)
- `1`: Respiratory Rate (insp/min)
- `2`: SpO2 (%)
- `3`: Systolic Blood Pressure (mmHg)
- `4`: Diastolic Blood Pressure (mmHg)

| Tensor Name | Shape | Description |
| :--- | :--- | :--- |
| `z` (Input) | `[B, N, 32]` or `[N, 32]` | Latent trajectory states evolved via `NeuralSDE` |
| `X_hat` (Output) | `[B, N, 5]` or `[N, 5]` | Predicted normalized physiological values |

---

## C. Why Decoder Operates in Normalized Space

The decoder operates strictly in **normalized feature space** (zero mean, unit variance defined in `data/processed/scaler.json`):
1. **Gradient Stability**: Prevents clinical variables with large magnitudes (e.g., Systolic BP \(\approx 113.0 \text{ mmHg}\)) from dominating loss gradients over variables with smaller numeric ranges (e.g., Respiratory Rate \(\approx 19.7 \text{ insp/min}\)).
2. **Decoupled Architecture**: Inverse scaling is maintained as a separate modular utility (`src/data/scaler_utils.py`) to prevent coupling loss functions to raw clinical units.

---

## D. Masked Reconstruction Loss

Loss is evaluated strictly over valid observed physiological entries using binary observation mask \(M \in \{0, 1\}^{B \times N \times 5}\):

### 1. Masked Mean Squared Error (MSE)
\[
\mathcal{L}_{\text{MSE}}(\hat{X}, X, M) = \frac{\sum_{i,j,k} M_{i,j,k} \left(\hat{X}_{i,j,k} - X_{i,j,k}\right)^2}{\sum_{i,j,k} M_{i,j,k} + \epsilon}
\]

### 2. Masked Mean Absolute Error (MAE)
\[
\mathcal{L}_{\text{MAE}}(\hat{X}, X, M) = \frac{\sum_{i,j,k} M_{i,j,k} \left|\hat{X}_{i,j,k} - X_{i,j,k}\right|}{\sum_{i,j,k} M_{i,j,k} + \epsilon}
\]

---

## E. Missing Observation Handling

- Missing or unobserved vital signs are designated by \(M_{i,j,k} = 0\).
- Element-wise multiplication by \(M\) zeroes out prediction errors at unobserved locations.
- If an entire sequence or batch contains zero valid observations (\(\sum M = 0\)), loss functions return `0.0` safely without zero-division or NaNs.

---

## F. Initial-State Handling

The initial latent state \(z_0\) is constructed from the patient's first clinical observation:

\[
\left( X[:, 0, :], M[:, 0, :], T[:, 0], \Delta T[:, 0] \right) \xrightarrow{\text{TimeAwareEncoder}} h_0 \xrightarrow{\text{LatentProjection}} z_0 = z(T_0)
\]

If the first clinical event occurs at \(T_0 > 0\) hours after ICU admission (e.g., \(T_0 = 0.9378\text{h}\) in MIMIC-IV sample), \(z_0\) represents physical state \(z(T_0)\). The `NeuralSDE` integrates forward continuously from \(T_0 \to T_1 \to \dots \to T_{N-1}\).

---

## G. Full Forward-Pass Architecture

```
EHR Data: (X, M, T, DeltaT)
    │
    ├─► First Observation: (X_0, M_0, T_0, DeltaT_0)
    │       │
    │       ▼
    │   TimeAwareEncoder [26 -> 64]
    │       │
    │       ▼
    │   LatentProjection [64 -> 32]
    │       │
    │       ▼
    │   Initial Latent State z_0 = z(T_0) [B, 32]
    │
    ▼
NeuralSDE.integrate(z0, T) ──► Continuous Trajectory z(t) [B, N, 32]
                                      │
                                      ▼
                                 EHRDecoder [32 -> 64 -> 64 -> 5]
                                      │
                                      ▼
                               Predictions X_hat [B, N, 5]
```

---

## H. Tensor Shapes Across Pipeline

| Component | Input Shape | Output Shape | Description |
| :--- | :--- | :--- | :--- |
| `TimeAwareEncoder` | `X_0`: `[B, 5]`, `M_0`: `[B, 5]`, `T_0`: `[B]`, `DeltaT_0`: `[B]` | `h_0`: `[B, 64]` | Point-wise observation encoding |
| `LatentProjection` | `h_0`: `[B, 64]` | `z_0`: `[B, 32]` | Initial latent state projection |
| `NeuralSDE.integrate` | `z_0`: `[B, 32]`, `T`: `[B, N]` | `z(t)`: `[B, N, 32]` | Discretized continuous integration trajectory |
| `EHRDecoder` | `z(t)`: `[B, N, 32]` | `X_hat`: `[B, N, 5]` | Decoded physiological predictions |
| `masked_mse_loss` | `X_hat`: `[B, N, 5]`, `X`: `[B, N, 5]`, `M`: `[B, N, 5]` | `loss`: `Scalar` | Masked MSE loss |

---

## I. Unit-Test Results

The test suite in [`tests/test_decoder.py`](file:///c:/Users/Abdullah%20Mutahir/OneDrive/Documents/EHR-Neural-SDE/tests/test_decoder.py) and [`tests/test_full_forward_pass.py`](file:///c:/Users/Abdullah%20Mutahir/OneDrive/Documents/EHR-Neural-SDE/tests/test_full_forward_pass.py) passed all test cases:

```
test_01_decoder_accepts_bn32_input (tests.test_decoder.TestEHRDecoder) ... ok
test_02_output_shape_is_bn5 (tests.test_decoder.TestEHRDecoder) ... ok
test_03_output_contains_no_nans (tests.test_decoder.TestEHRDecoder) ... ok
test_04_backpropagation_finite_gradients (tests.test_decoder.TestEHRDecoder) ... ok
test_05_masked_mse_ignores_missing_values (tests.test_decoder.TestEHRDecoder) ... ok
test_06_masked_mae_ignores_missing_values (tests.test_decoder.TestEHRDecoder) ... ok
test_07_changing_z_changes_decoder_output (tests.test_decoder.TestEHRDecoder) ... ok
test_08_decoder_works_for_different_sequence_lengths (tests.test_decoder.TestEHRDecoder) ... ok
test_09_all_zero_mask_handled_safely (tests.test_decoder.TestEHRDecoder) ... ok
test_01_end_to_end_forward_pass_on_real_trajectory (tests.test_full_forward_pass.TestFullForwardPass) ... ok
test_02_initial_state_alignment_for_delayed_first_observation (tests.test_full_forward_pass.TestFullForwardPass) ... ok

----------------------------------------------------------------------
Ran 11 tests in 1.515s

OK
```

---

## J. Untrained Forward-Pass Metrics (Sanity Check)

Forward-pass results on a sample MIMIC-IV trajectory (`stay_id = 30000625`, sequence length \(N = 364\), first observation \(T_0 = 0.9378\text{h}\)):

- **Masked MSE (Untrained)**: `94,760,296,448.0`
- **Masked MAE (Untrained)**: `73,091.31`

> [!WARNING]
> **SANITY-CHECK METRICS ONLY**: These loss values are calculated on randomly initialized model weights before any training has occurred. They serve exclusively as architectural validation that tensor flow and autograd computation graphs function without errors or NaNs.
