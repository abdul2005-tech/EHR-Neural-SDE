"""
Phase 13 Evaluation & Controlled Ablation Suite.

Evaluates continuous-time Ornstein-Uhlenbeck (OU) residual process candidates
against Phase 11 baseline, Phase 11 probabilistic, Phase 12 independent control,
and calibrated OU residual scales.

Outputs:
- experiments/phase13_ou_validation_results.csv
- Detailed metrics including plausibility %, Wasserstein, KS stat, correlation error,
  step volatility, volatility ratio, lag-1 AC, lag-5 AC, diversity RMSE/MAE, etc.
"""

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.temporal_residual import TemporalResidualModel
from src.evaluation.fit_phase13_residuals import estimate_ou_parameters_from_train


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


def precompute_latent_trajectories(
    model: EHRNeuralSDE,
    trajectories: List[Dict[str, Any]],
    n_samples: int = 20,
    base_seed: int = 42,
) -> Dict[Tuple[int, Any], Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Pre-computes and caches latent SDE trajectories and decoder outputs."""
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

            # 1. Deterministic ODE integration for continuous mean
            z_traj_det = model.sde.integrate(z0=z_0, times=T_target, enable_noise=False)
            mu_det, _ = model.decoder(z_traj_det)
            cached[(stay_id, "det")] = (mu_det.squeeze(0).cpu().numpy(), None)

            # 2. Stochastic SDE futures
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
                )

    return cached


def generate_phase13_futures_cached(
    trajectories: List[Dict[str, Any]],
    cached_predictions: Dict[Tuple[int, Any], Tuple[np.ndarray, Optional[np.ndarray]]],
    candidate_type: str,
    residual_model: Optional[TemporalResidualModel] = None,
    scale: float = 1.0,
    scaler_path: str = "data/processed/scaler.json",
    n_samples: int = 5,
    base_seed: int = 42,
) -> Dict[int, List[np.ndarray]]:
    """Generates synthetic futures using pre-cached latent SDE predictions."""
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    synth_by_stay = {}
    device = residual_model.lambda_rate.device if residual_model is not None else torch.device("cpu")

    for idx, traj in enumerate(trajectories):
        stay_id = traj["stay_id"]
        T_real = traj["T"]
        T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0).to(device)

        synth_by_stay[stay_id] = []

        if candidate_type == "13A0_continuous_mean":
            mu_np, _ = cached_predictions[(stay_id, "det")]
            X_phys = denormalize(mu_np, mean, std)
            for _ in range(n_samples):
                synth_by_stay[stay_id].append(X_phys.copy())
        else:
            for s_idx in range(n_samples):
                seed_val = base_seed + idx * 50 + s_idx * 1000
                gen_s = torch.Generator(device=device).manual_seed(seed_val)

                mu_np, sigma_np = cached_predictions[(stay_id, s_idx)]

                if candidate_type == "13A1_probabilistic":
                    np.random.seed(seed_val)
                    eps_syn = np.random.randn(*mu_np.shape)
                    X_syn_norm = mu_np + sigma_np * eps_syn

                elif candidate_type == "13B_independent":
                    np.random.seed(seed_val)
                    eps_syn = np.random.randn(*mu_np.shape)
                    if residual_model is not None:
                        sigma_stat_np = (
                            residual_model.sigma_stat.cpu().numpy() * scale
                        )
                    else:
                        sigma_stat_np = np.ones(5, dtype=np.float32)
                    X_syn_norm = mu_np + sigma_stat_np * eps_syn

                elif candidate_type in ("13C_ou_full", "13D_ou_scaled"):
                    if residual_model is None:
                        raise ValueError("residual_model required for OU candidates")
                    r_tensor = residual_model(
                        T_target, generator=gen_s, scale=scale
                    )  # [1, N, 5]
                    r_np = r_tensor.squeeze(0).cpu().numpy()
                    X_syn_norm = mu_np + r_np
                else:
                    raise ValueError(f"Unknown candidate_type: {candidate_type}")

                X_phys = denormalize(X_syn_norm, mean, std)
                synth_by_stay[stay_id].append(X_phys)

    return synth_by_stay


def compute_phase13_metrics(
    real_flat: Dict[str, np.ndarray],
    real_trajectories: List[Dict[str, Any]],
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Computes comprehensive 14-metric benchmark evaluation dictionary."""
    D = len(feature_names)
    synth_flat = {name: [] for name in feature_names}

    for stay_id, s_list in synth_by_stay.items():
        for X_s in s_list:
            for i, name in enumerate(feature_names):
                synth_flat[name].extend(X_s[:, i])

    synth_flat_arr = {
        name: np.array(vals, dtype=np.float64) for name, vals in synth_flat.items()
    }

    # 1. Plausibility
    in_range_counts = 0
    total_synth_obs = 0
    for f_name in feature_names:
        low, high = SANITY_RANGES[f_name]
        arr = synth_flat_arr[f_name]
        in_range_counts += ((arr >= low) & (arr <= high)).sum()
        total_synth_obs += len(arr)
    plausibility_pct = (in_range_counts / max(total_synth_obs, 1)) * 100.0

    # 2. Wasserstein & 3. KS Statistic
    wasserstein_list = []
    ks_stat_list = []
    feature_stds_synth = {}
    feature_stds_real = {}

    for f_name in feature_names:
        r_arr = real_flat[f_name]
        s_arr = synth_flat_arr[f_name]
        wasserstein_list.append(wasserstein_distance(r_arr, s_arr))
        ks_stat_list.append(ks_2samp(r_arr, s_arr).statistic)
        feature_stds_real[f_name] = float(np.std(r_arr))
        feature_stds_synth[f_name] = float(np.std(s_arr))

    mean_wasserstein = float(np.mean(wasserstein_list))
    mean_ks_stat = float(np.mean(ks_stat_list))

    # 4. Cross-Feature Correlation Matrix Error
    min_r = min(len(real_flat[f]) for f in feature_names)
    min_s = min(len(synth_flat_arr[f]) for f in feature_names)
    real_mat = np.column_stack([real_flat[f][:min_r] for f in feature_names])
    synth_mat = np.column_stack([synth_flat_arr[f][:min_s] for f in feature_names])

    corr_real = np.corrcoef(real_mat, rowvar=False)
    corr_synth = np.corrcoef(synth_mat, rowvar=False)
    corr_diff_mean = float(np.abs(corr_real - corr_synth).mean())

    # 5. Volatility & Autocorrelations
    real_diffs = []
    synth_diffs = []
    synth_ac1 = []
    synth_ac5 = []
    feature_volatilities = {f: [] for f in feature_names}
    feature_ac1s = {f: [] for f in feature_names}

    for traj in real_trajectories:
        X_p = traj["X"]
        for i in range(D):
            real_diffs.extend(np.diff(X_p[:, i]))

    for stay_id, s_list in synth_by_stay.items():
        for X_s in s_list:
            for i, f_name in enumerate(feature_names):
                diffs = np.diff(X_s[:, i])
                synth_diffs.extend(diffs)
                feature_volatilities[f_name].extend(diffs)

                if len(X_s) > 3 and np.std(X_s[:-1, i]) > 1e-5:
                    ac1 = np.corrcoef(X_s[:-1, i], X_s[1:, i])[0, 1]
                    if not np.isnan(ac1):
                        synth_ac1.append(ac1)
                        feature_ac1s[f_name].append(ac1)
                if len(X_s) > 6 and np.std(X_s[:-5, i]) > 1e-5:
                    ac5 = np.corrcoef(X_s[:-5, i], X_s[5:, i])[0, 1]
                    if not np.isnan(ac5):
                        synth_ac5.append(ac5)

    real_diff_std = float(np.std(real_diffs))
    synth_diff_std = float(np.std(synth_diffs))
    volatility_ratio = synth_diff_std / max(real_diff_std, 1e-5)
    mean_ac1 = float(np.mean(synth_ac1)) if synth_ac1 else 0.0
    mean_ac5 = float(np.mean(synth_ac5)) if synth_ac5 else 0.0

    # 6. Multi-future Diversity (RMSE & MAE)
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
        "feature_stds_real": feature_stds_real,
        "feature_stds_synth": feature_stds_synth,
        "feature_volatilities": {f: float(np.std(v)) for f, v in feature_volatilities.items()},
        "feature_ac1s": {f: float(np.mean(v)) if v else 0.0 for f, v in feature_ac1s.items()},
    }


def evaluate_phase13(
    mode: str = "quick",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 13 Evaluation Suite (Mode: {mode.upper()}) ===", flush=True)

    # Select split based on mode (test split ONLY used in 'test' mode)
    split = "test" if mode == "test" else "val"
    data_path = f"data/processed/datasets/{split}.pkl"
    real_trajectories, real_flat, feature_names = load_dataset_trajectories(data_path, scaler_path)

    if mode == "quick":
        eval_trajectories = real_trajectories[:5]
        n_samples = 5
    else:  # full_val or test
        eval_trajectories = real_trajectories
        n_samples = 20

    # Fit OU residual parameters on TRAIN data only
    print("\nFitting OU residual parameters on TRAIN dataset...", flush=True)
    lambdas, sigmas, _ = estimate_ou_parameters_from_train(device=device)

    # Initialize TemporalResidualModel with fitted parameters
    ou_model = TemporalResidualModel(
        output_dim=len(feature_names),
        lambda_val=lambdas,
        sigma_val=sigmas,
    ).to(device)

    # Load frozen Phase 11 champion model
    ckpt_p = Path(checkpoint_path)
    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    checkpoint = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Pre-compute latent SDE trajectories for evaluation stays once to maximize speed
    print(f"Pre-computing latent trajectories for {len(eval_trajectories)} stays x {n_samples} samples...", flush=True)
    cached_predictions = precompute_latent_trajectories(
        model=model, trajectories=eval_trajectories, n_samples=n_samples
    )

    # Candidates to evaluate
    if mode in ("quick", "full_val"):
        candidates = [
            ("13A0 Continuous Mean Baseline", "13A0_continuous_mean", 1.0),
            ("13A1 Existing Probabilistic", "13A1_probabilistic", 1.0),
            ("13B Independent Noise Control", "13B_independent", 1.0),
            ("13C Full OU Residual (scale=1.0)", "13C_ou_full", 1.0),
            ("13D Calibrated OU (scale=0.10)", "13D_ou_scaled", 0.10),
            ("13D Calibrated OU (scale=0.25)", "13D_ou_scaled", 0.25),
            ("13D Calibrated OU (scale=0.50)", "13D_ou_scaled", 0.50),
            ("13D Calibrated OU (scale=0.75)", "13D_ou_scaled", 0.75),
        ]
    else:  # test mode
        # Selected candidate (scale=0.25 or best candidate determined on validation)
        candidates = [
            ("13A0 Continuous Mean Baseline", "13A0_continuous_mean", 1.0),
            ("13A1 Existing Probabilistic", "13A1_probabilistic", 1.0),
            ("13B Independent Noise Control", "13B_independent", 1.0),
            ("13D Selected Calibrated OU (scale=0.25)", "13D_ou_scaled", 0.25),
        ]

    results_table = []

    print("\n" + "=" * 130, flush=True)
    print(
        f"{'Candidate Label':<38} | {'Plausible %':<11} | {'Wasserstein':<11} | {'KS Stat':<8} | "
        f"{'Corr Error':<10} | {r'std($\Delta X$)':<12} | {'Vol Ratio':<9} | {'Lag-1 AC':<9} | {'Diversity RMSE':<14}",
        flush=True,
    )
    print("=" * 130, flush=True)

    for label, cand_type, scale in candidates:
        print(f"Evaluating candidate: {label}...", flush=True)
        synth_by_stay = generate_phase13_futures_cached(
            trajectories=eval_trajectories,
            cached_predictions=cached_predictions,
            candidate_type=cand_type,
            residual_model=ou_model,
            scale=scale,
            scaler_path=scaler_path,
            n_samples=n_samples,
        )

        metrics = compute_phase13_metrics(
            real_flat, eval_trajectories, synth_by_stay, feature_names
        )

        print(
            f"{label:<38} | {metrics['plausibility_pct']:10.2f}% | {metrics['mean_wasserstein']:11.4f} | "
            f"{metrics['mean_ks_stat']:8.4f} | {metrics['correlation_error']:10.4f} | "
            f"{metrics['synth_diff_std']:12.4f} | {metrics['volatility_ratio']:9.4f} | "
            f"{metrics['mean_ac1']:9.4f} | {metrics['diversity_rmse']:14.4f}",
            flush=True,
        )

        results_table.append(
            {
                "label": label,
                "cand_type": cand_type,
                "scale": scale,
                "metrics": metrics,
            }
        )

    print("=" * 130 + "\n", flush=True)

    # Save CSV output
    out_csv = Path(experiments_dir) / "phase13_ou_validation_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_label",
                "candidate_type",
                "scale",
                "plausibility_pct",
                "wasserstein_distance",
                "ks_statistic",
                "correlation_error",
                "synth_diff_std",
                "volatility_ratio",
                "lag1_autocorr",
                "lag5_autocorr",
                "diversity_rmse",
                "diversity_mae",
            ]
        )
        for r in results_table:
            m = r["metrics"]
            writer.writerow(
                [
                    r["label"],
                    r["cand_type"],
                    r["scale"],
                    f"{m['plausibility_pct']:.2f}",
                    f"{m['mean_wasserstein']:.4f}",
                    f"{m['mean_ks_stat']:.4f}",
                    f"{m['correlation_error']:.4f}",
                    f"{m['synth_diff_std']:.4f}",
                    f"{m['volatility_ratio']:.4f}",
                    f"{m['mean_ac1']:.4f}",
                    f"{m['mean_ac5']:.4f}",
                    f"{m['diversity_rmse']:.4f}",
                    f"{m['diversity_mae']:.4f}",
                ]
            )

    print(f"Saved Phase 13 CSV report to: {out_csv}", flush=True)
    return results_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="quick", choices=["quick", "full_val", "test"]
    )
    args = parser.parse_args()

    evaluate_phase13(mode=args.mode)
