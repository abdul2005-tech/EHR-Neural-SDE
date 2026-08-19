# Phase 7: Neural SDE Continuous-Time Latent Dynamics & Numerical Solver

## Executive Summary

Phase 7 establishes the continuous-time stochastic latent dynamics of the EHR-Neural-SDE framework. The initial latent state \(z_0 \in \mathbb{R}^{32}\) produced by the `TimeAwareEncoder` is continuously evolved across irregular patient observation timestamps through a learned Neural Stochastic Differential Equation (Neural SDE) integrated using an adaptive Euler-Maruyama solver.

---

## A. Neural SDE Equation

The continuous latent trajectory \(z(t) \in \mathbb{R}^{32}\) evolves according to the Stochastic Differential Equation:

\[
dz_t = f_\theta(z_t, t) \, dt + g_\theta(z_t, t) \, dW_t
\]

where:
- \(z_t \in \mathbb{R}^{32}\) is the continuous latent state vector at physical time \(t\) (in hours).
- \(f_\theta(z_t, t): \mathbb{R}^{32} \times \mathbb{R} \to \mathbb{R}^{32}\) is the learned deterministic drift function.
- \(g_\theta(z_t, t): \mathbb{R}^{32} \times \mathbb{R} \to \mathbb{R}^{32}_{>0}\) is the learned diagonal diffusion function.
- \(W_t \in \mathbb{R}^{32}\) represents a standard 32-dimensional Wiener process (Brownian motion) with independent components.

---

## B. Drift Function

The drift function \(f_\theta(z, t)\) models the predictable, deterministic trend of patient health evolution over continuous time:

- **Input**: Concatenation of latent state \(z \in \mathbb{R}^{32}\) and physical time \(t \in \mathbb{R}^1\) resulting in \([z, t] \in \mathbb{R}^{33}\).
- **Architecture**: A 3-layer Multi-Layer Perceptron (MLP):
  \[
  \text{Input } (33) \xrightarrow{\text{Linear}} \text{Hidden } (64) \xrightarrow{\text{ReLU}} \text{Hidden } (64) \xrightarrow{\text{ReLU}} \text{Output } (32)
  \]
- **Role**: Computes the directional velocity of health trajectory shifts (e.g., organ degradation or gradual physiological recovery).

---

## C. Diffusion Function

The diffusion function \(g_\theta(z, t)\) models the magnitude of stochastic variability and unobserved intra-patient fluctuations:

- **Input**: Concatenation \([z, t] \in \mathbb{R}^{33}\).
- **Architecture**: A 3-layer MLP matching the drift network depth:
  \[
  \text{Input } (33) \xrightarrow{\text{Linear}} \text{Hidden } (64) \xrightarrow{\text{ReLU}} \text{Hidden } (64) \xrightarrow{\text{ReLU}} \text{Output } (32) \xrightarrow{\text{Softplus} + \epsilon} \mathbb{R}^{32}_{>0}
  \]
- **Positive Parameterization**: Output is passed through `softplus(out) + eps` (\(\epsilon = 10^{-5}\)) ensuring strictly non-negative, non-zero diffusion magnitudes to maintain stochastic variance without numerical underflow.

---

## D. Diagonal Diffusion Assumption

To maintain computational efficiency and prevent parameter explosion, a **diagonal diffusion matrix** \(\text{diag}(g_\theta(z, t))\) is used rather than a full \(32 \times 32\) covariance matrix:
- **Dimensionality**: \(g_\theta(z, t)\) outputs a vector of shape `[B, 32]`.
- **Independence**: Latent dimensions experience independent stochastic Brownian kicks weighted by component-specific magnitudes.
- **Scalability**: Reduces parameter count from \(\mathcal{O}(D^2)\) to \(\mathcal{O}(D)\) while allowing dimension-specific noise amplitude modeling.

---

## E. Euler-Maruyama Method

The Euler-Maruyama integrator discretizes continuous-time SDE integration over sub-intervals \(\Delta t\):

\[
z_{k+1} = z_k + f_\theta(z_k, t_k) \, \Delta t + g_\theta(z_k, t_k) \, \sqrt{\Delta t} \odot \epsilon_k
\]

where:
- \(\epsilon_k \sim \mathcal{N}(0, I_{32})\) is i.i.d. standard Gaussian noise sampled at substep \(k\).
- \(\odot\) denotes element-wise Hadamard multiplication.

---

## F. Brownian Noise

Brownian motion \(W_t\) satisfies non-overlapping increment independence:

\[
W_{t + \Delta t} - W_t \sim \mathcal{N}(0, \Delta t \cdot I_{32})
\]

The random vector \(\epsilon_k \sim \mathcal{N}(0, I_{32})\) is sampled independently at every substep. An optional `torch.Generator` parameter is exposed to support seed-controlled deterministic sampling during testing and reproducible evaluation.

---

## G. \(\sqrt{\Delta t}\) Scaling

The appearance of \(\sqrt{\Delta t}\) on the stochastic term arises directly from the variance scaling of Brownian motion:

\[
\text{Var}(W_{t + \Delta t} - W_t) = \Delta t \implies \sigma(W_{t + \Delta t} - W_t) = \sqrt{\Delta t}
\]

While deterministic drift accumulates linearly with time (\(\mathcal{O}(\Delta t)\)), stochastic uncertainty accumulates with the square root of time (\(\mathcal{O}(\sqrt{\Delta t})\)). Omitting \(\sqrt{\Delta t}\) leads to incorrect noise variance scaling across different solver step sizes.

---

## H. Irregular Time Handling

Clinical Electronic Health Records feature highly irregular inter-observation intervals (e.g., timestamps `[0.0, 0.05, 0.687, 1.687, 5.2, 20.0]` hours).

The solver:
1. Accepts raw observation timestamp vectors `times` of shape `[N]` or `[B, N]`.
2. Computes the exact elapsed time \(\Delta t = t_{i} - t_{i-1}\) between consecutive observations.
3. Does **NOT** project timestamps onto a uniform hourly grid or distort physical timing.

---

## I. Maximum Internal Step Size

The dataset exhibits observation gaps as large as **43.8 hours**. Taking a single Euler-Maruyama step over \(\Delta t = 43.8\text{h}\) causes severe integration truncation error and numerical instability.

To solve this, `NeuralSDE` incorporates an internal solver parameter:

\[
\text{max\_step\_size} = 0.1 \text{ hours}
\]

For any interval \(\Delta t\):
\[
n_{\text{steps}} = \left\lceil \frac{\Delta t}{\text{max\_step\_size}} \right\rceil, \quad \delta t = \frac{\Delta t}{n_{\text{steps}}}
\]
The solver takes \(n_{\text{steps}}\) internal substeps of size \(\delta t \le 0.1\text{h}\), ensuring bounded local truncation error while returning exact latent representations at the requested observation times.

---

## J. Numerical Stability

Numerical stability is guaranteed through three primary safeguards:
1. **Dynamic Interval Sub-division**: Limits step sizes to \(\le 0.1\text{h}\), preventing explosive linear extrapolation over multi-day gaps.
2. **Softplus + Epsilon Parameterization**: \(\text{softplus}(\cdot) + 10^{-5}\) prevents non-positive or zero diffusion values that could cause gradient collapse or numerical underflow.
3. **Clamped \(\Delta t\)**: Clamping \(\Delta t \ge 10^{-8}\) avoids division-by-zero or negative roots during square-root evaluation.

---

## K. Tensor Shapes

| Network / Operation | Input Shape | Output Shape | Description |
| :--- | :--- | :--- | :--- |
| `DriftNetwork` | `z`: `[B, 32]`, `t`: `[B, 1]` | `drift`: `[B, 32]` | Deterministic velocity \(f_\theta(z,t)\) |
| `DiffusionNetwork` | `z`: `[B, 32]`, `t`: `[B, 1]` | `diffusion`: `[B, 32]` | Non-negative noise magnitude \(g_\theta(z,t)\) |
| `euler_maruyama_step` | `z`: `[B, 32]`, `t`: `[B, 1]`, `dt`: `float` / `[B, 1]` | `z_next`: `[B, 32]` | Single discretized stochastic update |
| `integrate` | `z0`: `[B, 32]`, `times`: `[N]` or `[B, N]` | `z_trajectory`: `[B, N, 32]` | Integrated latent state sequence at observation timestamps |

---

## L. Unit Test Results

The test suite in [`tests/test_neural_sde.py`](file:///c:/Users/Abdullah%20Mutahir/OneDrive/Documents/EHR-Neural-SDE/tests/test_neural_sde.py) verifies all 14 Phase 7 requirements:

```
test_01_drift_output_shape (__main__.TestNeuralSDE) ... ok
test_02_diffusion_output_shape (__main__.TestNeuralSDE) ... ok
test_03_diffusion_outputs_non_negative (__main__.TestNeuralSDE) ... ok
test_04_euler_maruyama_step_shape (__main__.TestNeuralSDE) ... ok
test_05_integration_output_shape (__main__.TestNeuralSDE) ... ok
test_06_first_integrated_state_equals_z0 (__main__.TestNeuralSDE) ... ok
test_07_increasing_dt_changes_drift_contribution (__main__.TestNeuralSDE) ... ok
test_08_deterministic_integration_when_diffusion_disabled (__main__.TestNeuralSDE) ... ok
test_09_stochastic_integration_with_nonzero_diffusion (__main__.TestNeuralSDE) ... ok
test_10_backpropagation_finite_gradients (__main__.TestNeuralSDE) ... ok
test_11_no_nans_or_infinities (__main__.TestNeuralSDE) ... ok
test_12_irregular_time_arrays_accepted (__main__.TestNeuralSDE) ... ok
test_13_large_time_gaps_subdivided (__main__.TestNeuralSDE) ... ok
test_14_exact_requested_observation_times_returned (__main__.TestNeuralSDE) ... ok

----------------------------------------------------------------------
Ran 14 tests in 0.667s

OK
```
