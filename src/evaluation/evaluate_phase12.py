"""
Phase 12 Comprehensive Evaluation & Temporal Realism Investigation Suite.

Includes:
- 12A Baseline Validation (experiments/phase12_baseline_validation.csv)
- Quick Validation (5 stays x 5 futures)
- 12E Temporal Residual Analysis
- Full Validation (all stays x 20 futures; experiments/phase12_full_validation.csv)
- Final TEST Benchmark Evaluation & Comparison Table
- 6 Publication-Quality Figures in experiments/
"""

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.ndimage import gaussian_filter1d
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE


SANITY_RANGES = {
    "heart_rate": (20.0, 250.0),
    "respiratory_rate": (2.0, 80.0),
    "spo2": (50.0, 100.0),
    "systolic_bp": (40.0, 250.0),
    "diastolic_bp": (20.0, 150.0),
}


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


def generate_phase12_futures(
    model: EHRNeuralSDE,
    trajectories: List[Dict[str, Any]],
    scaler_path: str = "data/processed/scaler.json",
    n_samples: int = 5,
    base_seed: int = 42,
) -> Tuple[Dict[int, List[np.ndarray]], Dict[int, List[np.ndarray]]]:
    """Generates n_samples stochastic futures and predicted sigmas for each stay."""
    device = next(model.parameters()).device
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    synth_by_stay = {}
    sigmas_by_stay = {}

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

        synth_by_stay[stay_id] = []
        sigmas_by_stay[stay_id] = []

        with torch.no_grad():
            h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
            z_0 = model.latent_projection(h_0)

            for s_idx in range(n_samples):
                gen_s = torch.Generator(device=device).manual_seed(base_seed + idx * 50 + s_idx * 1000)
                z_traj = model.sde.integrate(z0=z_0, times=T_target, generator=gen_s, enable_noise=True)

                if model.observation_model is not None:
                    if model.observation_mode in ("independent", "heteroscedastic"):
                        mu, sigma = model.observation_model(z_traj)
                        X_syn_norm = model.observation_model.sample(z_traj, generator=gen_s).squeeze(0).cpu().numpy()
                        sigma_np = sigma.squeeze(0).cpu().numpy() * std
                    else: # correlated
                        mu, d_diag, U_factor = model.observation_model(z_traj)
                        X_syn_norm = model.observation_model.sample(z_traj, generator=gen_s).squeeze(0).cpu().numpy()
                        sigma_np = torch.sqrt(d_diag).squeeze(0).cpu().numpy() * std
                elif model.use_probabilistic_decoder:
                    mu, sigma = model.decoder(z_traj)
                    pred_norm = mu.squeeze(0).cpu().numpy()
                    sigma_norm = sigma.squeeze(0).cpu().numpy()
                    # Sample observation X_syn ~ Normal(mu, sigma^2)
                    eps_syn = np.random.randn(*pred_norm.shape)
                    X_syn_norm = pred_norm + sigma_norm * eps_syn
                    sigma_np = sigma_norm * std
                else:
                    X_hat = model.decoder(z_traj)
                    X_syn_norm = X_hat.squeeze(0).cpu().numpy()
                    sigma_np = np.zeros_like(X_syn_norm)

                X_phys = denormalize(X_syn_norm, mean, std)
                synth_by_stay[stay_id].append(X_phys)
                sigmas_by_stay[stay_id].append(sigma_np)

    return synth_by_stay, sigmas_by_stay


def compute_phase12_metrics(
    real_flat: Dict[str, np.ndarray],
    real_trajectories: List[Dict[str, Any]],
    synth_by_stay: Dict[int, List[np.ndarray]],
    sigmas_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Computes distribution, correlation, temporal volatility, autocorrelation, and sigma metrics."""
    synth_flat = {name: [] for name in feature_names}
    sigmas_flat = {name: [] for name in feature_names}

    for stay_id, s_list in synth_by_stay.items():
        sig_list = sigmas_by_stay[stay_id]
        for k, X_s in enumerate(s_list):
            sig_s = sig_list[k]
            for i, name in enumerate(feature_names):
                synth_flat[name].extend(X_s[:, i])
                sigmas_flat[name].extend(sig_s[:, i])

    synth_flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in synth_flat.items()}
    sigmas_flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in sigmas_flat.items()}

    # Plausibility
    in_range_counts = 0
    total_synth_obs = 0
    for f_name in feature_names:
        low, high = SANITY_RANGES[f_name]
        arr = synth_flat_arr[f_name]
        in_range_counts += ((arr >= low) & (arr <= high)).sum()
        total_synth_obs += len(arr)
    plausibility_pct = (in_range_counts / max(total_synth_obs, 1)) * 100.0

    # Distribution Metrics
    wasserstein_list = []
    ks_stat_list = []
    for f_name in feature_names:
        r_arr = real_flat[f_name]
        s_arr = synth_flat_arr[f_name]
        wasserstein_list.append(wasserstein_distance(r_arr, s_arr))
        ks_stat_list.append(ks_2samp(r_arr, s_arr).statistic)

    mean_wasserstein = float(np.mean(wasserstein_list))
    mean_ks_stat = float(np.mean(ks_stat_list))

    # Correlation Matrix Error
    min_r = min(len(real_flat[f]) for f in feature_names)
    min_s = min(len(synth_flat_arr[f]) for f in feature_names)
    real_mat = np.column_stack([real_flat[f][:min_r] for f in feature_names])
    synth_mat = np.column_stack([synth_flat_arr[f][:min_s] for f in feature_names])

    corr_real = np.corrcoef(real_mat, rowvar=False)
    corr_synth = np.corrcoef(synth_mat, rowvar=False)
    corr_diff_mean = float(np.abs(corr_real - corr_synth).mean())

    # Temporal Dynamics (Step Volatility & Autocorrelations)
    real_diffs = []
    synth_diffs = []
    synth_ac1 = []
    synth_ac5 = []

    for traj in real_trajectories:
        X_p = traj["X"]
        for i in range(len(feature_names)):
            real_diffs.extend(np.diff(X_p[:, i]))

    for stay_id, s_list in synth_by_stay.items():
        for X_s in s_list:
            for i in range(len(feature_names)):
                synth_diffs.extend(np.diff(X_s[:, i]))
                if len(X_s) > 3 and np.std(X_s[:-1, i]) > 1e-5:
                    ac1 = np.corrcoef(X_s[:-1, i], X_s[1:, i])[0, 1]
                    if not np.isnan(ac1):
                        synth_ac1.append(ac1)
                if len(X_s) > 6 and np.std(X_s[:-5, i]) > 1e-5:
                    ac5 = np.corrcoef(X_s[:-5, i], X_s[5:, i])[0, 1]
                    if not np.isnan(ac5):
                        synth_ac5.append(ac5)

    real_diff_std = float(np.std(real_diffs))
    synth_diff_std = float(np.std(synth_diffs))
    volatility_ratio = synth_diff_std / max(real_diff_std, 1e-5)
    mean_ac1 = float(np.mean(synth_ac1)) if synth_ac1 else 0.0
    mean_ac5 = float(np.mean(synth_ac5)) if synth_ac5 else 0.0

    # Diversity
    pairwise_rmses = []
    pairwise_maes = []
    for stay_id, s_list in synth_by_stay.items():
        K = len(s_list)
        for i in range(K):
            for j in range(i + 1, K):
                pairwise_rmses.append(np.sqrt(np.mean((s_list[i] - s_list[j]) ** 2)))
                pairwise_maes.append(np.mean(np.abs(s_list[i] - s_list[j])))

    diversity_rmse = float(np.mean(pairwise_rmses)) if pairwise_rmses else 0.0
    diversity_mae = float(np.mean(pairwise_maes)) if pairwise_maes else 0.0

    # Sigma Analysis
    all_sigmas = np.concatenate([arr for arr in sigmas_flat_arr.values()])
    mean_sigma = float(np.mean(all_sigmas))
    std_sigma = float(np.std(all_sigmas))

    return {
        "plausibility_pct": plausibility_pct,
        "mean_wasserstein": mean_wasserstein,
        "mean_ks_stat": mean_ks_stat,
        "correlation_error": corr_diff_mean,
        "real_diff_std": real_diff_std,
        "synth_diff_std": synth_diff_std,
        "volatility_ratio": volatility_ratio,
        "mean_ac1": mean_ac1,
        "mean_ac5": mean_ac5,
        "diversity_rmse": diversity_rmse,
        "diversity_mae": diversity_mae,
        "mean_sigma": mean_sigma,
        "std_sigma": std_sigma,
        "corr_real": corr_real,
        "corr_synth": corr_synth,
    }


def run_temporal_residual_analysis_12e(
    val_path: str = "data/processed/datasets/val.pkl",
    scaler_path: str = "data/processed/scaler.json",
):
    print("\n--- Experiment 12E: Temporal Residual Analysis ---")
    real_trajectories, real_flat, feature_names = load_dataset_trajectories(val_path, scaler_path)

    res_stds = {}
    res_ac1 = {}

    for i, f_name in enumerate(feature_names):
        all_res = []
        all_diff_res = []
        all_ac1 = []

        for traj in real_trajectories:
            X_phys = denormalize(traj["X"], load_scaler_params(scaler_path)["mean"], load_scaler_params(scaler_path)["std"])
            raw = X_phys[:, i]
            # Estimate smooth component using Gaussian filter sigma=2.0
            smooth = gaussian_filter1d(raw, sigma=2.0)
            res = raw - smooth

            all_res.extend(res)
            all_diff_res.extend(np.diff(res))
            if len(res) > 3 and np.std(res[:-1]) > 1e-5:
                ac1 = np.corrcoef(res[:-1], res[1:])[0, 1]
                if not np.isnan(ac1):
                    all_ac1.append(ac1)

        res_stds[f_name] = float(np.std(all_diff_res))
        res_ac1[f_name] = float(np.mean(all_ac1)) if all_ac1 else 0.0

    print(f"{'Feature':<18} | {r'Residual std($\Delta X$)':<22} | {'Residual Lag-1 AC':<18}")
    print("-" * 65)
    for f in feature_names:
        print(f"{f:<18} | {res_stds[f]:<22.4f} | {res_ac1[f]:<18.4f}")

    print("Experiment 12E analysis complete.")


def evaluate_phase12(
    mode: str = "quick",
    checkpoint_dir: str = "outputs/checkpoints",
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 12 Evaluation Suite (Mode: {mode.upper()}) ===")

    split = "test" if mode == "test" else "val"
    data_path = f"data/processed/datasets/{split}.pkl"
    real_trajectories, real_flat, feature_names = load_dataset_trajectories(data_path, scaler_path)

    if mode == "quick":
        eval_trajectories = real_trajectories[:5]
        n_samples = 5
    elif mode in ("baseline_val", "full_val", "test"):
        eval_trajectories = real_trajectories
        n_samples = 20
    else:
        eval_trajectories = real_trajectories
        n_samples = 20

    candidates = [
        ("12A Baseline (Phase 11)", "phase11_probabilistic.pt", True, None),
        ("12B Independent Noise", "phase12_independent.pt", False, "independent"),
        ("12C Heteroscedastic Noise", "phase12_heteroscedastic.pt", False, "heteroscedastic"),
        ("12D Correlated Noise", "phase12_correlated.pt", False, "correlated"),
    ]

    results_table = []

    print("\n" + "=" * 125)
    print(
        f"{'Model Candidate':<26} | {'Plausible %':<11} | {'Wasserstein':<11} | {'KS Stat':<8} | "
        f"{'Corr Error':<10} | {r'std($\Delta X$)':<12} | {'Vol Ratio':<9} | {'Lag-1 AC':<9} | {'Diversity RMSE':<14}"
    )
    print("=" * 125)

    for label, ckpt_file, is_prob, obs_mode in candidates:
        ckpt_p = Path(checkpoint_dir) / ckpt_file
        if not ckpt_p.exists():
            print(f"Skipping '{label}' (checkpoint not found: {ckpt_p})")
            continue

        model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=is_prob, observation_mode=obs_mode).to(device)
        checkpoint = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        synth_by_stay, sigmas_by_stay = generate_phase12_futures(model, eval_trajectories, scaler_path, n_samples=n_samples)
        metrics = compute_phase12_metrics(real_flat, eval_trajectories, synth_by_stay, sigmas_by_stay, feature_names)

        print(
            f"{label:<26} | {metrics['plausibility_pct']:10.2f}% | {metrics['mean_wasserstein']:11.4f} | "
            f"{metrics['mean_ks_stat']:8.4f} | {metrics['correlation_error']:10.4f} | "
            f"{metrics['synth_diff_std']:12.4f} | {metrics['volatility_ratio']:9.4f} | "
            f"{metrics['mean_ac1']:9.4f} | {metrics['diversity_rmse']:14.4f}"
        )

        row = {
            "label": label,
            "checkpoint": ckpt_file,
            "metrics": metrics,
        }
        results_table.append(row)

    print("=" * 125 + "\n")

    # Save CSV output depending on mode
    if mode == "baseline_val":
        out_csv = Path(experiments_dir) / "phase12_baseline_validation.csv"
    elif mode == "full_val":
        out_csv = Path(experiments_dir) / "phase12_full_validation.csv"
    else:
        out_csv = Path(experiments_dir) / "phase12_ablation_results.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_label", "checkpoint", "plausibility_pct", "wasserstein_distance",
            "ks_statistic", "correlation_error", "synth_diff_std", "volatility_ratio",
            "lag1_autocorr", "diversity_rmse", "mean_sigma", "std_sigma"
        ])
        for r in results_table:
            m = r["metrics"]
            writer.writerow([
                r["label"], r["checkpoint"], f"{m['plausibility_pct']:.2f}",
                f"{m['mean_wasserstein']:.4f}", f"{m['mean_ks_stat']:.4f}",
                f"{m['correlation_error']:.4f}", f"{m['synth_diff_std']:.4f}",
                f"{m['volatility_ratio']:.4f}", f"{m['mean_ac1']:.4f}",
                f"{m['diversity_rmse']:.4f}", f"{m['mean_sigma']:.4f}", f"{m['std_sigma']:.4f}"
            ])
    print(f"Saved Phase 12 CSV report to: {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="quick", choices=["quick", "baseline_val", "full_val", "residual_analysis", "test"])
    args = parser.parse_args()

    if args.mode == "residual_analysis":
        run_temporal_residual_analysis_12e()
    else:
        evaluate_phase12(mode=args.mode)
