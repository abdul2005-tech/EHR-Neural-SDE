"""
Phase 15 Evaluation & Controlled Ablation Suite for State-Dependent Residual Dynamics.

Evaluates continuous-time state-dependent multivariate OU residual process candidates:
- 15A: Phase 14 Champion Frozen Control (Full empirical covariance, global scale S=0.35)
- 15B: Feature-Specific Constant Scales S = [S_HR, S_RR, S_SpO2, S_SBP, S_DBP]
- 15C_tanh: Distance-to-Boundary Adaptive Scaling (Tanh attenuation)
- 15C_sigmoid: Distance-to-Boundary Adaptive Scaling (Sigmoid attenuation)
- 15C_exp: Distance-to-Boundary Adaptive Scaling (Smooth exponential attenuation)
- 15C_rational: Distance-to-Boundary Adaptive Scaling (Rational attenuation, tau=0.5)
- 15C_rational_tau1.0: Distance-to-Boundary Adaptive Scaling (Rational attenuation, tau=1.0)
- 15C_rational_scaled0.9: Distance-to-Boundary Adaptive Scaling (Rational attenuation, tau=1.0, scale=0.90)
- 15D: Variance-Aware Adaptive Scaling (g(sigma_decoder))
- 15E: Combined Feature-Specific + Boundary-Aware + Variance-Aware Scaling

Modes:
- quick: Fast validation screening on 5 stays x 5 samples
- full_val: Full validation suite on all validation stays x 20 samples
- test: Final evaluation on TEST set ONCE using frozen validation champion
"""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.state_dependent_residual import (
    StateDependentMultivariateTemporalResidualModel,
    SANITY_RANGES,
    DEFAULT_MEAN,
    DEFAULT_STD,
)
from src.evaluation.fit_phase15_scales import calibrate_feature_scales_from_train


def load_dataset_trajectories(
    data_path: str,
    scaler_path: str = "data/processed/scaler.json",
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[str]]:
    """Loads dataset trajectories and denormalized flat feature arrays."""
    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)

    flat_dict = {name: [] for name in feature_names}
    for traj in trajectories:
        X_phys = denormalize(traj["X"], mean, std)
        M = traj["M"]
        for i, name in enumerate(feature_names):
            obs_mask = M[:, i] == 1.0
            flat_dict[name].extend(X_phys[obs_mask, i])

    flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in flat_dict.items()}
    return trajectories, flat_arr, feature_names


def precompute_latent_trajectories(
    model: EHRNeuralSDE,
    trajectories: List[Dict[str, Any]],
    n_samples: int = 20,
    base_seed: int = 42,
) -> Dict[Tuple[int, Any], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Pre-computes and caches continuous mean mu(t), decoder uncertainty sigma(t),
    and latent trajectories z(t) to ensure exact reproducibility across candidates.
    """
    device = next(model.parameters()).device
    cached = {}

    with torch.no_grad():
        for idx, traj in enumerate(trajectories):
            stay_id = traj["stay_id"]
            X_real = traj["X"]
            T_real = traj["T"]
            DeltaT_real = traj["DeltaT"]
            M_real = traj["M"]

            X_0 = torch.tensor(X_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            M_0 = torch.tensor(M_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            T_0 = torch.tensor(T_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
            DeltaT_0 = torch.tensor(DeltaT_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
            T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0).to(device)

            h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
            z_0 = model.latent_projection(h_0)

            # Deterministic continuous ODE mean trajectory
            z_traj_det = model.sde.integrate(z0=z_0, times=T_target, enable_noise=False)
            mu_det, sigma_det = model.decoder(z_traj_det)
            cached[(stay_id, "det")] = (
                mu_det.squeeze(0).cpu().numpy(),
                sigma_det.squeeze(0).cpu().numpy(),
                z_traj_det.squeeze(0).cpu().numpy(),
            )

            # Stochastic SDE futures
            for s_idx in range(n_samples):
                seed_val = base_seed + idx * 50 + s_idx * 1000
                gen_s = torch.Generator(device=device).manual_seed(seed_val)
                z_traj_stoch = model.sde.integrate(
                    z0=z_0, times=T_target, generator=gen_s, enable_noise=True
                )
                mu_s, sigma_s = model.decoder(z_traj_stoch)
                cached[(stay_id, s_idx)] = (
                    mu_s.squeeze(0).cpu().numpy(),
                    sigma_s.squeeze(0).cpu().numpy(),
                    z_traj_stoch.squeeze(0).cpu().numpy(),
                )

            if (idx + 1) % 5 == 0 or (idx + 1) == len(trajectories):
                print(f"  Pre-computed {idx + 1}/{len(trajectories)} stays...", flush=True)

    return cached


def generate_phase15_futures_cached(
    trajectories: List[Dict[str, Any]],
    cached_predictions: Dict[Tuple[int, Any], Tuple[np.ndarray, np.ndarray, np.ndarray]],
    residual_model: StateDependentMultivariateTemporalResidualModel,
    scaler_path: str = "data/processed/scaler.json",
    n_samples: int = 5,
    base_seed: int = 42,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    """Generates synthetic physical futures and effective scale trajectories."""
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    synth_by_stay = {}
    scale_by_stay = {}
    device = residual_model.feature_scales.device

    for idx, traj in enumerate(trajectories):
        stay_id = traj["stay_id"]
        T_real = traj["T"]
        T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0).to(device)

        synth_list = []
        scale_list = []

        for s_idx in range(n_samples):
            mu_np, sigma_np, z_np = cached_predictions[(stay_id, s_idx)]
            mu_t = torch.tensor(mu_np, dtype=torch.float32).unsqueeze(0).to(device)
            sigma_t = torch.tensor(sigma_np, dtype=torch.float32).unsqueeze(0).to(device)
            z_t = torch.tensor(z_np, dtype=torch.float32).unsqueeze(0).to(device)

            seed_val = base_seed + idx * 50 + s_idx * 1000 + 777
            gen_res = torch.Generator(device=device).manual_seed(seed_val)

            X_synth_norm, S_eff = residual_model.apply_residual(
                mu_norm=mu_t,
                T=T_target,
                sigma_decoder=sigma_t,
                z_latent=z_t,
                generator=gen_res,
            )

            X_synth_phys = denormalize(X_synth_norm.squeeze(0).cpu().numpy(), mean, std)
            S_eff_np = S_eff.squeeze(0).cpu().numpy()

            synth_list.append(X_synth_phys)
            scale_list.append(S_eff_np)

        synth_by_stay[stay_id] = synth_list
        scale_by_stay[stay_id] = scale_list

    return synth_by_stay, scale_by_stay


def compute_comprehensive_metrics(
    real_trajectories: List[Dict[str, Any]],
    real_flat: Dict[str, np.ndarray],
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
    real_corr_matrix: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Computes distribution, correlation, temporal, diversity, and plausibility metrics."""
    D = len(feature_names)

    # 1. Flatten all synthetic observations
    synth_flat = {name: [] for name in feature_names}
    for stay_id, futures in synth_by_stay.items():
        for X_s in futures:
            for i, name in enumerate(feature_names):
                synth_flat[name].extend(X_s[:, i])

    synth_flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in synth_flat.items()}

    # Distribution metrics (Wasserstein & KS)
    wasserstein_list = []
    ks_list = []

    for name in feature_names:
        r_v = real_flat[name]
        s_v = synth_flat_arr[name]
        w_d = wasserstein_distance(r_v, s_v)
        ks_stat, _ = ks_2samp(r_v, s_v)
        wasserstein_list.append(w_d)
        ks_list.append(ks_stat)

    avg_wasserstein = float(np.mean(wasserstein_list))
    avg_ks = float(np.mean(ks_list))

    # 2. Physiological Plausibility %
    total_obs = 0
    in_range_obs = 0
    for name in feature_names:
        low_b, high_b = SANITY_RANGES[name]
        vals = synth_flat_arr[name]
        total_obs += len(vals)
        in_range_obs += np.sum((vals >= low_b) & (vals <= high_b))

    plausibility_pct = (in_range_obs / max(total_obs, 1)) * 100.0

    # 3. Cross-Feature Correlation Error
    if real_corr_matrix is None:
        real_corr_matrix = compute_empirical_correlation(real_trajectories, feature_names)
    synth_corr = compute_synthetic_correlation(synth_by_stay, feature_names)
    corr_error = float(np.linalg.norm(real_corr_matrix - synth_corr, ord="fro"))

    # 4. Temporal Volatility std(delta X) & Autocorrelations (Lag-1, Lag-5)
    real_deltas = []
    synth_deltas = []
    synth_ac1_list = []
    synth_ac5_list = []

    for traj in real_trajectories:
        stay_id = traj["stay_id"]
        T_real = traj["T"]
        M_real = traj["M"]
        X_real_phys = denormalize(traj["X"], np.array(DEFAULT_MEAN), np.array(DEFAULT_STD))

        # Real deltas
        for s in range(len(T_real) - 1):
            if T_real[s + 1] - T_real[s] > 1e-4:
                obs_m = (M_real[s] == 1.0) & (M_real[s + 1] == 1.0)
                if np.any(obs_m):
                    real_deltas.extend((X_real_phys[s + 1] - X_real_phys[s])[obs_m])

        # Synthetic deltas and autocorrelations
        if stay_id in synth_by_stay:
            for X_s in synth_by_stay[stay_id]:
                for s in range(len(T_real) - 1):
                    if T_real[s + 1] - T_real[s] > 1e-4:
                        synth_deltas.extend(X_s[s + 1] - X_s[s])

                # Autocorrelation over trajectory features
                for f_idx in range(D):
                    ts = X_s[:, f_idx]
                    if len(ts) >= 6:
                        mean_ts = np.mean(ts)
                        var_ts = np.var(ts)
                        if var_ts > 1e-6:
                            ac1 = np.mean((ts[:-1] - mean_ts) * (ts[1:] - mean_ts)) / var_ts
                            ac5 = np.mean((ts[:-5] - mean_ts) * (ts[5:] - mean_ts)) / var_ts
                            synth_ac1_list.append(ac1)
                            synth_ac5_list.append(ac5)

    real_step_vol = float(np.std(real_deltas)) if len(real_deltas) > 0 else 0.0
    synth_step_vol = float(np.std(synth_deltas)) if len(synth_deltas) > 0 else 0.0
    lag1_ac = float(np.mean(synth_ac1_list)) if len(synth_ac1_list) > 0 else 0.0
    lag5_ac = float(np.mean(synth_ac5_list)) if len(synth_ac5_list) > 0 else 0.0

    # 5. Multi-Future Diversity RMSE
    diversity_rmse_list = []
    for stay_id, futures in synth_by_stay.items():
        K = len(futures)
        if K >= 2:
            for i in range(K):
                for j in range(i + 1, K):
                    rmse = float(np.sqrt(np.mean((futures[i] - futures[j]) ** 2)))
                    diversity_rmse_list.append(rmse)

    avg_diversity = float(np.mean(diversity_rmse_list)) if len(diversity_rmse_list) > 0 else 0.0

    return {
        "plausibility_pct": plausibility_pct,
        "wasserstein": avg_wasserstein,
        "ks_statistic": avg_ks,
        "correlation_error": corr_error,
        "real_step_volatility": real_step_vol,
        "step_volatility": synth_step_vol,
        "volatility_ratio": synth_step_vol / max(real_step_vol, 1e-4),
        "lag1_ac": lag1_ac,
        "lag5_ac": lag5_ac,
        "diversity_rmse": avg_diversity,
        "wasserstein_by_feature": dict(zip(feature_names, wasserstein_list)),
        "ks_by_feature": dict(zip(feature_names, ks_list)),
    }


def compute_empirical_correlation(
    trajectories: List[Dict[str, Any]],
    feature_names: List[str],
) -> np.ndarray:
    """Computes 5x5 empirical correlation matrix on co-observed real features."""
    D = len(feature_names)
    cov = np.zeros((D, D), dtype=np.float64)
    counts = np.zeros((D, D), dtype=np.int64)

    mean = np.array(DEFAULT_MEAN)
    std = np.array(DEFAULT_STD)

    for traj in trajectories:
        X_phys = denormalize(traj["X"], mean, std)
        M = traj["M"]
        for s in range(len(X_phys)):
            obs_mask = M[s] == 1.0
            idx = np.where(obs_mask)[0]
            for i in idx:
                for j in idx:
                    cov[i, j] += X_phys[s, i] * X_phys[s, j]
                    counts[i, j] += 1

    counts_safe = np.maximum(counts, 1)
    cov = cov / counts_safe
    std_vec = np.sqrt(np.maximum(np.diag(cov), 1e-8))
    corr = cov / (std_vec[:, np.newaxis] * std_vec[np.newaxis, :])
    return np.clip(corr, -1.0, 1.0)


def compute_synthetic_correlation(
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
) -> np.ndarray:
    """Computes 5x5 correlation matrix across all synthetic observations."""
    D = len(feature_names)
    flat_matrix = []
    for futures in synth_by_stay.values():
        for X_s in futures:
            flat_matrix.append(X_s)

    all_obs = np.concatenate(flat_matrix, axis=0)  # [Total_steps, 5]
    corr = np.corrcoef(all_obs, rowvar=False)
    return np.clip(corr, -1.0, 1.0)


def compute_featurewise_and_boundary_analysis(
    real_flat: Dict[str, np.ndarray],
    synth_by_stay: Dict[int, List[np.ndarray]],
    scale_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Computes detailed per-feature metrics and boundary proximity stats.
    """
    featurewise_rows = []
    boundary_rows = []

    # Flatten synthetic and scale observations by feature
    synth_flat = {name: [] for name in feature_names}
    scale_flat = {name: [] for name in feature_names}
    synth_deltas_feat = {name: [] for name in feature_names}

    for stay_id, futures in synth_by_stay.items():
        scales = scale_by_stay[stay_id]
        for X_s, S_s in zip(futures, scales):
            for i, name in enumerate(feature_names):
                synth_flat[name].extend(X_s[:, i])
                scale_flat[name].extend(S_s[:, i])

                # Consecutive step deltas
                for s in range(len(X_s) - 1):
                    synth_deltas_feat[name].append(X_s[s + 1, i] - X_s[s, i])

    for f_idx, f_name in enumerate(feature_names):
        low_b, high_b = SANITY_RANGES[f_name]
        r_vals = real_flat[f_name]
        s_vals = np.array(synth_flat[f_name], dtype=np.float64)
        sc_vals = np.array(scale_flat[f_name], dtype=np.float64)
        s_deltas = np.array(synth_deltas_feat[f_name], dtype=np.float64)

        w_dist = wasserstein_distance(r_vals, s_vals)
        ks_stat, _ = ks_2samp(r_vals, s_vals)

        s_vol = float(np.std(s_deltas))
        in_mask = (s_vals >= low_b) & (s_vals <= high_b)
        plaus_pct = (np.sum(in_mask) / max(len(s_vals), 1)) * 100.0

        # Boundary proximity: % within 5% of boundary edge
        span = high_b - low_b
        near_lower_mask = (s_vals >= low_b) & (s_vals <= low_b + 0.05 * span)
        near_upper_mask = (s_vals <= high_b) & (s_vals >= high_b - 0.05 * span)
        pct_near_lower = (np.sum(near_lower_mask) / max(len(s_vals), 1)) * 100.0
        pct_near_upper = (np.sum(near_upper_mask) / max(len(s_vals), 1)) * 100.0
        pct_violations = 100.0 - plaus_pct

        featurewise_rows.append({
            "feature": f_name,
            "synthetic_std_dX": s_vol,
            "wasserstein": w_dist,
            "ks_statistic": ks_stat,
            "plausibility_pct": plaus_pct,
            "mean_effective_scale": float(np.mean(sc_vals)),
            "min_effective_scale": float(np.min(sc_vals)),
            "max_effective_scale": float(np.max(sc_vals)),
        })

        boundary_rows.append({
            "feature": f_name,
            "sanity_lower_bound": low_b,
            "sanity_upper_bound": high_b,
            "synth_min": float(np.min(s_vals)),
            "synth_max": float(np.max(s_vals)),
            "pct_near_lower_bound": pct_near_lower,
            "pct_near_upper_bound": pct_near_upper,
            "pct_boundary_violations": pct_violations,
        })

    return featurewise_rows, boundary_rows


def evaluate_phase15_suite(
    mode: str = "quick",
    data_dir: str = "data/processed/datasets",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
    device: str = "cpu",
) -> None:
    """Executes Phase 15 evaluation suite."""
    print(f"\n=======================================================", flush=True)
    print(f"       PHASE 15 EVALUATION SUITE [Mode: {mode.upper()}]", flush=True)
    print(f"=======================================================\n", flush=True)

    # 1. Load calibrated TRAIN scales
    train_scales_path = Path(experiments_dir) / "phase15_train_scales.json"
    if not train_scales_path.exists():
        print("Calibrated train scales not found. Running calibrate_feature_scales_from_train()...", flush=True)
        train_scale_data = calibrate_feature_scales_from_train(
            train_path=f"{data_dir}/train.pkl",
            checkpoint_path=checkpoint_path,
            scaler_path=scaler_path,
            output_json_path=str(train_scales_path),
            device=device,
        )
    else:
        with open(train_scales_path, "r") as f:
            train_scale_data = json.load(f)

    calibrated_scales = train_scale_data["calibrated_scales_safe"]
    lambda_rates = train_scale_data["lambda_rates"]
    empirical_cov = np.array(train_scale_data["empirical_covariance"])

    # Determine dataset path and sampling parameters based on mode
    if mode == "quick":
        eval_path = f"{data_dir}/val.pkl"
        n_stays = 5
        n_samples = 5
        csv_filename = "phase15_quick_validation.csv"
    elif mode == "full_val":
        eval_path = f"{data_dir}/val.pkl"
        n_stays = None  # All val stays
        n_samples = 20
        csv_filename = "phase15_full_validation.csv"
    elif mode == "test":
        eval_path = f"{data_dir}/test.pkl"
        n_stays = None  # All test stays
        n_samples = 20
        csv_filename = "phase15_test_evaluation.csv"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    eval_trajectories, real_flat, feature_names = load_dataset_trajectories(
        data_path=eval_path, scaler_path=scaler_path
    )
    if n_stays is not None:
        eval_trajectories = eval_trajectories[:n_stays]

    print(f"Loaded {len(eval_trajectories)} stays from {eval_path} ({n_samples} futures per stay).", flush=True)

    # Pre-compute empirical real correlation matrix once
    real_corr_matrix = compute_empirical_correlation(eval_trajectories, feature_names)

    # Load frozen Phase 11 model
    ckpt_p = Path(checkpoint_path)
    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    checkpoint = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Precompute latent SDE predictions to ensure exact identical predictions for all candidates
    print("Pre-computing continuous latent SDE predictions...", flush=True)
    cached_predictions = precompute_latent_trajectories(
        model=model, trajectories=eval_trajectories, n_samples=n_samples
    )

    # 2. Define controlled ablation candidates (15A to 15E)
    candidates = {}

    # 15A: Phase 14 Champion Frozen Control (Scalar S=0.35, no state attenuation)
    candidates["15A_Phase14_Control"] = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        lambda_val=lambda_rates,
        cov_matrix=empirical_cov,
        feature_scales=[1.0] * 5,
        global_scale=0.35,
        attenuation_mode="none",
    ).to(device)

    # 15B: Feature-Specific Constant Scales (Calibrated scales, no state attenuation)
    candidates["15B_Feature_Specific_Const"] = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        lambda_val=lambda_rates,
        cov_matrix=empirical_cov,
        feature_scales=calibrated_scales,
        global_scale=1.0,
        attenuation_mode="none",
    ).to(device)

    # 15C Attenuation Candidates (Distance-to-Boundary Adaptive Scaling)
    for attn_mode in ["tanh", "sigmoid", "exp", "rational"]:
        candidates[f"15C_Boundary_Adaptive_{attn_mode}"] = StateDependentMultivariateTemporalResidualModel(
            output_dim=5,
            lambda_val=lambda_rates,
            cov_matrix=empirical_cov,
            feature_scales=calibrated_scales,
            global_scale=1.0,
            attenuation_mode=attn_mode,
            attenuation_k=2.0,
            attenuation_tau=0.5,
        ).to(device)

    # 15C Tuned Rational Attenuation (tau=1.0)
    candidates["15C_Rational_tau1.0"] = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        lambda_val=lambda_rates,
        cov_matrix=empirical_cov,
        feature_scales=calibrated_scales,
        global_scale=1.0,
        attenuation_mode="rational",
        attenuation_tau=1.0,
    ).to(device)

    # 15C Tuned Rational Attenuation (tau=1.0, global_scale=0.90)
    candidates["15C_Rational_scaled0.9"] = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        lambda_val=lambda_rates,
        cov_matrix=empirical_cov,
        feature_scales=[s * 0.90 for s in calibrated_scales],
        global_scale=1.0,
        attenuation_mode="rational",
        attenuation_tau=1.0,
    ).to(device)

    # 15D: Variance-Aware Adaptive Scaling (g(sigma_decoder))
    candidates["15D_Variance_Aware"] = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        lambda_val=lambda_rates,
        cov_matrix=empirical_cov,
        feature_scales=calibrated_scales,
        global_scale=1.0,
        attenuation_mode="none",
        use_uncertainty_modulation=True,
    ).to(device)

    # 15E: Combined Feature-Specific + Boundary-Aware + Variance-Aware Scaling
    candidates["15E_Combined_Adaptive"] = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        lambda_val=lambda_rates,
        cov_matrix=empirical_cov,
        feature_scales=[s * 0.90 for s in calibrated_scales],
        global_scale=1.0,
        attenuation_mode="rational",
        attenuation_tau=1.0,
        use_uncertainty_modulation=True,
    ).to(device)

    # 3. Evaluate each candidate
    results_list = []
    featurewise_all = {}
    boundary_all = {}

    print("\n" + "=" * 110, flush=True)
    print(f"{'Candidate':<32} | {'Plaus %':<9} | {'Wasserstein':<11} | {'KS':<7} | {'CorrErr':<9} | {'std(dX)':<8} | {'Lag1 AC':<8} | {'Diversity':<9}", flush=True)
    print("=" * 110, flush=True)

    for cand_name, cand_model in candidates.items():
        print(f"Evaluating {cand_name}...", flush=True)
        synth_by_stay, scale_by_stay = generate_phase15_futures_cached(
            trajectories=eval_trajectories,
            cached_predictions=cached_predictions,
            residual_model=cand_model,
            scaler_path=scaler_path,
            n_samples=n_samples,
        )

        metrics = compute_comprehensive_metrics(
            real_trajectories=eval_trajectories,
            real_flat=real_flat,
            synth_by_stay=synth_by_stay,
            feature_names=feature_names,
            real_corr_matrix=real_corr_matrix,
        )

        metrics["candidate"] = cand_name
        results_list.append(metrics)

        # Compute per-feature and boundary details
        f_rows, b_rows = compute_featurewise_and_boundary_analysis(
            real_flat=real_flat,
            synth_by_stay=synth_by_stay,
            scale_by_stay=scale_by_stay,
            feature_names=feature_names,
        )
        featurewise_all[cand_name] = f_rows
        boundary_all[cand_name] = b_rows

        print(
            f"{cand_name:<32} | {metrics['plausibility_pct']:9.2f}% | "
            f"{metrics['wasserstein']:11.4f} | {metrics['ks_statistic']:7.4f} | "
            f"{metrics['correlation_error']:9.4f} | {metrics['step_volatility']:8.4f} | "
            f"{metrics['lag1_ac']:8.4f} | {metrics['diversity_rmse']:9.4f}",
            flush=True,
        )

    print("=" * 110 + "\n", flush=True)

    # 4. Save CSV results
    csv_path = Path(experiments_dir) / csv_filename
    fieldnames = [
        "candidate",
        "plausibility_pct",
        "wasserstein",
        "ks_statistic",
        "correlation_error",
        "real_step_volatility",
        "step_volatility",
        "volatility_ratio",
        "lag1_ac",
        "lag5_ac",
        "diversity_rmse",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for res in results_list:
            row = {k: res[k] for k in fieldnames}
            writer.writerow(row)

    print(f"Saved evaluation results to: {csv_path}", flush=True)

    # Select champion if full_val mode
    if mode in ["quick", "full_val"]:
        select_phase15_champion(results_list, featurewise_all, boundary_all, experiments_dir)


def select_phase15_champion(
    results_list: List[Dict[str, Any]],
    featurewise_all: Dict[str, List[Dict[str, Any]]],
    boundary_all: Dict[str, List[Dict[str, Any]]],
    experiments_dir: str = "experiments",
) -> Dict[str, Any]:
    """
    Performs multi-objective composite model selection on VALIDATION data.
    Rule: Require Plausibility >= 99.0%, maximize step volatility towards real ICU data (8.43),
    while minimizing Wasserstein and correlation error.
    """
    valid_candidates = [r for r in results_list if r["plausibility_pct"] >= 99.0]
    if not valid_candidates:
        max_plaus = max(r["plausibility_pct"] for r in results_list)
        valid_candidates = [r for r in results_list if r["plausibility_pct"] >= max_plaus - 0.5]

    # Composite Score: Volatility_Ratio * 0.4 + (1 - Wasserstein/10) * 0.3 + (1 - CorrelationError) * 0.3
    best_cand = None
    best_score = -1e9

    for cand in valid_candidates:
        vol_score = min(cand["volatility_ratio"], 1.0)
        wass_score = max(0.0, 1.0 - cand["wasserstein"] / 10.0)
        corr_score = max(0.0, 1.0 - cand["correlation_error"])
        score = 0.4 * vol_score + 0.3 * wass_score + 0.3 * corr_score
        cand["composite_score"] = score

        if score > best_score:
            best_score = score
            best_cand = cand

    champion_name = best_cand["candidate"]
    print(f"\n=======================================================", flush=True)
    print(f" PHASE 15 VALIDATION CHAMPION: {champion_name}", flush=True)
    print(f" Composite Score: {best_score:.4f}", flush=True)
    print(f" Plausibility: {best_cand['plausibility_pct']:.2f}% | Volatility: {best_cand['step_volatility']:.4f} | Lag-1 AC: {best_cand['lag1_ac']:.4f}", flush=True)
    print(f"=======================================================\n", flush=True)

    # Save featurewise and boundary analysis for Champion
    f_csv_path = Path(experiments_dir) / "phase15_featurewise_metrics.csv"
    b_csv_path = Path(experiments_dir) / "phase15_boundary_analysis.csv"

    with open(f_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(featurewise_all[champion_name][0].keys()))
        writer.writeheader()
        writer.writerows(featurewise_all[champion_name])

    with open(b_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(boundary_all[champion_name][0].keys()))
        writer.writeheader()
        writer.writerows(boundary_all[champion_name])

    # Save outputs/checkpoints/phase15_config.json
    cfg = {
        "champion_candidate": champion_name,
        "validation_metrics": best_cand,
        "phase11_checkpoint": "outputs/checkpoints/phase11_probabilistic.pt",
        "phase15_scales_config": "experiments/phase15_train_scales.json",
    }
    cfg_path = Path("outputs/checkpoints/phase15_config.json")
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)

    return best_cand


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 15 Evaluation & Ablation Suite")
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "full_val", "test"],
        help="Evaluation mode (quick screening, full validation, or test evaluation)",
    )
    args = parser.parse_args()
    evaluate_phase15_suite(mode=args.mode)
