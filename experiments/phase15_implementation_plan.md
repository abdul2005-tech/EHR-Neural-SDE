# Phase 15: State-Dependent and Physiologically Constrained Multivariate OU Residual Dynamics

## Executive Summary & Research Motivation

In **Phase 14**, we introduced a continuous-time multivariate Ornstein-Uhlenbeck (OU) observation residual process with cross-feature covariance $\Sigma_{\text{empirical}}$. The Phase 14 Champion ($S=0.35$) achieved **100% physiological plausibility**, strong cross-feature correlation realism ($0.0468$ correlation error), and realistic multi-future diversity ($7.7470$).

However, the primary remaining limitation of Phase 14 is:
- **Step Volatility Underestimation**: Generated step volatility $\text{std}(\Delta X) = 3.0118$, whereas real ICU data exhibits $\text{std}(\Delta X) = 8.4308$.
- **Plausibility Trade-off under Global Scaling**: Increasing the global noise scale $S \to 0.75\text{--}1.00$ elevates volatility to $\approx 6.75\text{--}8.99$, but causes physiological range violations, dropping plausibility to $97\text{--}98\%$.

### Phase 15 Core Hypothesis
Instead of applying a single static, global scalar $S$ across all vital signs and clinical states, we propose **State-Dependent and Physiologically Constrained Scaling**:
$$X_{\text{synth}, d}(t) = \mu_{\text{phys}, d}(t) + S_{\text{eff}, d}(t) \cdot r_{\text{OU}, d}(t)$$

where the effective residual amplitude $S_{\text{eff}, d}(t)$ adapts dynamically to:
1. **Feature-Specific Baseline Variability**: $S_{\text{feature}, d}$ calibrated per vital sign (e.g. HR vs SpO2).
2. **Physiological Boundary Proximity**: Smooth attenuation $f(m_d(t))$ near sanity bounds to prevent range violations without artificial hard clipping.
3. **Model Prediction Uncertainty**: Modulation $g(\sigma_{\text{decoder}, d}(t))$ based on latent state uncertainty from the Phase 11 decoder.

---

## Controlled Candidates (15A to 15F)

- **15A**: Phase 14 Champion Frozen Control (Full empirical covariance, global scale $S=0.35$).
- **15B**: Feature-Specific Constant Scales $S = [S_{\text{HR}}, S_{\text{RR}}, S_{\text{SpO2}}, S_{\text{SBP}}, S_{\text{DBP}}]$ (no state dependence).
- **15C**: Distance-to-Boundary Adaptive Scaling ($S_{\text{base}, d} \cdot f(m_d(t))$).
- **15D**: Variance-Aware Adaptive Scaling ($S_{\text{base}, d} \cdot g(\sigma_{\text{decoder}, d}(t))$).
- **15E**: Combined Feature-Specific + Boundary-Aware + Variance-Aware Scaling ($S_{\text{feature}, d} \cdot f(m_d(t)) \cdot g(\sigma_{\text{decoder}, d}(t))$).
- **15F (Optional)**: Bounded Learned State-Dependent Scale Network (only if 15B--15E do not meet criteria).

---

## Planned Code Components

1. **`src/models/state_dependent_residual.py`**: Implementation of `StateDependentMultivariateTemporalResidualModel`.
2. **`tests/test_state_dependent_residual.py`**: Unit tests covering 17 critical mathematical and structural requirements.
3. **`src/evaluation/fit_phase15_scales.py`**: Calibration script estimating feature-specific scale baselines on TRAIN data only.
4. **`src/evaluation/evaluate_phase15.py`**: Evaluation suite supporting quick validation, full validation, and test evaluation modes.

---

## Verification Protocol

1. Run unit test suite: `pytest tests/test_state_dependent_residual.py -v`.
2. Fit feature scales on TRAIN: `python -m src.evaluation.fit_phase15_scales`.
3. Quick validation screening: `python -m src.evaluation.evaluate_phase15 --mode quick`.
4. Full validation suite: `python -m src.evaluation.evaluate_phase15 --mode full_val`.
5. Single-pass TEST evaluation on validation champion: `python -m src.evaluation.evaluate_phase15 --mode test`.
