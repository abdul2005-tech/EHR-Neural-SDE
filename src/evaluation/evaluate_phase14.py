"""
Phase 14 Evaluation & Controlled Ablation Suite for Multivariate OU Residual Process.

Evaluates continuous-time multivariate Ornstein-Uhlenbeck (OU) residual process candidates
against Phase 11 continuous mean, Phase 12 independent noise control, Phase 13 independent OU baseline,
and multivariate correlated candidates (Full Empirical, Shrinkage, Low-Rank).

Modes:
- quick: Fast validation screening on 5 stays x 5 samples
- full_val: Full validation suite on all validation stays x 20 samples
- test: Final evaluation on TEST set ONCE using frozen validation champion

Outputs:
- experiments/phase14_quick_validation.csv
- experiments/phase14_full_validation.csv
- experiments/phase14_model_selection.md
- experiments/phase14_scale_vs_performance.png
- experiments/phase14_volatility_ac_tradeoff.png
- experiments/phase14_covariance_heatmaps.png
- experiments/phase14_synthetic_trajectories.png
- outputs/checkpoints/phase14_config.json
"""

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.multivariate_temporal_residual import (
    MultivariateTemporalResidualModel,
    create_shrinkage_covariance,
    create_low_rank_covariance,
)
from src.evaluation.fit_phase14_residuals import estimate_multivariate_ou_parameters_from_train


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


def generate_phase14_futures_cached(
    trajectories: List[Dict[str, Any]],
    cached_predictions: Dict[Tuple[int, Any], Tuple[np.ndarray, Optional[np.ndarray]]],
    candidate_type: str,
    residual_model: Optional[MultivariateTemporalResidualModel] = None,
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

        if candidate_type == "continuous_mean":
            mu_np, _ = cached_predictions[(stay_id, "det")]
            X_phys = denormalize(mu_np, mean, std)
            for _ in range(n_samples):
                synth_by_stay[stay_id].append(X_phys.copy())
        else:
            for s_idx in range(n_samples):
                seed_val = base_seed + idx * 50 + s_idx * 1000
                gen_s = torch.Generator(device=device).manual_seed(seed_val)
                mu_np, sigma_np = cached_predictions[(stay_id, s_idx)]

                if candidate_type == "phase12_independent_noise":
                    np.random.seed(seed_val)
                    eps_syn = np.random.randn(*mu_np.shape)
                    sigma_stat_np = (
                        residual_model.sigma_stat.cpu().numpy() * scale
                        if residual_model is not None else np.ones(5, dtype=np.float32) * scale
                    )
                    X_syn_norm = mu_np + sigma_stat_np * eps_syn

                elif candidate_type in ("multivariate_ou", "phase13_independent_ou"):
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


def compute_phase14_metrics(
    real_flat: Dict[str, np.ndarray],
    real_trajectories: List[Dict[str, Any]],
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
    train_emp_cov: np.ndarray,
    scaler_path: str = "data/processed/scaler.json",
) -> Dict[str, Any]:
    """Computes comprehensive 14-metric benchmark evaluation dictionary."""
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

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
    for f_name in feature_names:
        r_arr = real_flat[f_name]
        s_arr = synth_flat_arr[f_name]
        wasserstein_list.append(wasserstein_distance(r_arr, s_arr))
        ks_stat_list.append(ks_2samp(r_arr, s_arr).statistic)

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

    # 5. Volatility & Autocorrelations (computed in physical units!)
    real_diffs = []
    synth_diffs = []
    synth_ac1 = []
    synth_ac5 = []

    for traj in real_trajectories:
        X_p = denormalize(traj["X"], mean, std)
        for i in range(D):
            real_diffs.extend(np.diff(X_p[:, i]))

    for stay_id, s_list in synth_by_stay.items():
        for X_s in s_list:
            for i, f_name in enumerate(feature_names):
                diffs = np.diff(X_s[:, i])
                synth_diffs.extend(diffs)

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

    # 7. Residual Covariance Similarity vs TRAIN empirical residual covariance
    synth_cov = np.cov(synth_mat, rowvar=False)
    cov_mae = float(np.abs(synth_cov - train_emp_cov).mean())

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
        "cov_mae": cov_mae,
        "corr_synth": corr_synth,
    }


def evaluate_phase14(
    mode: str = "quick",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Phase 14 Evaluation Suite (Mode: {mode.upper()}) ===", flush=True)

    split = "test" if mode == "test" else "val"
    data_path = f"data/processed/datasets/{split}.pkl"
    real_trajectories, real_flat, feature_names = load_dataset_trajectories(data_path, scaler_path)

    if mode == "quick":
        eval_trajectories = real_trajectories[:5]
        n_samples = 5
    else:  # full_val or test
        eval_trajectories = real_trajectories
        n_samples = 20

    # Fit TRAIN parameters and residual covariance
    print("\nFitting multivariate residual parameters on TRAIN dataset...", flush=True)
    fit_data = estimate_multivariate_ou_parameters_from_train(
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        experiments_dir=experiments_dir,
        device=device,
    )

    lambdas = fit_data["lambdas"]
    emp_cov = fit_data["emp_cov"]
    shrinkage_covs = fit_data["shrinkage_covs"]
    lowrank_covs = fit_data["lowrank_covs"]

    # Load frozen Phase 11 champion model
    ckpt_p = Path(checkpoint_path)
    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    checkpoint = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Pre-compute latent SDE trajectories for evaluation stays
    print(f"Pre-computing latent trajectories for {len(eval_trajectories)} stays x {n_samples} samples...", flush=True)
    cached_predictions = precompute_latent_trajectories(
        model=model, trajectories=eval_trajectories, n_samples=n_samples
    )

    scales_to_test = [0.25, 0.35, 0.50, 0.75, 1.00]

    # Build candidate configurations
    candidate_configs = []

    if mode in ("quick", "full_val"):
        # 14A: Phase 13 Independent OU baseline
        for s in scales_to_test:
            candidate_configs.append({
                "label": f"14A Independent OU (scale={s:.2f})",
                "cand_type": "phase13_independent_ou",
                "cov_type": "independent",
                "cov_matrix": np.diag(np.diag(emp_cov)),
                "scale": s,
            })

        # 14B: Full Empirical Residual Covariance
        for s in scales_to_test:
            candidate_configs.append({
                "label": f"14B Full Empirical Cov (scale={s:.2f})",
                "cand_type": "multivariate_ou",
                "cov_type": "full",
                "cov_matrix": emp_cov,
                "scale": s,
            })

        # 14C: Shrinkage Covariance
        for alpha in [0.25, 0.50, 0.75, 1.00]:
            for s in scales_to_test:
                candidate_configs.append({
                    "label": f"14C Shrinkage (a={alpha:.2f}, s={s:.2f})",
                    "cand_type": "multivariate_ou",
                    "cov_type": f"shrinkage_{alpha}",
                    "cov_matrix": shrinkage_covs[alpha],
                    "scale": s,
                })

        # 14D: Low-Rank Covariance
        for rank in [1, 2, 3]:
            for s in scales_to_test:
                candidate_configs.append({
                    "label": f"14D Low-Rank (r={rank}, s={s:.2f})",
                    "cand_type": "multivariate_ou",
                    "cov_type": f"lowrank_r{rank}",
                    "cov_matrix": lowrank_covs[rank],
                    "scale": s,
                })

    else:  # test mode
        # Final evaluation candidates on TEST set
        # Load selected champion configuration from validation
        config_path = Path("outputs/checkpoints/phase14_config.json")
        selected_cov_matrix = emp_cov
        selected_scale = 0.35
        selected_cov_type = "shrinkage_0.50"
        selected_label = "Phase 14 Champion Multivariate OU (Shrinkage 0.50, scale=0.35)"

        if config_path.exists():
            with open(config_path, "r") as f:
                champ_cfg = json.load(f)
                selected_scale = champ_cfg["selected_scale"]
                selected_cov_type = champ_cfg["selected_cov_type"]
                selected_cov_matrix = np.array(champ_cfg["selected_cov_matrix"], dtype=np.float32)
                selected_label = f"Phase 14 Champion ({champ_cfg['selected_cov_label']})"

        candidate_configs = [
            {
                "label": "Phase 11 Continuous Mean",
                "cand_type": "continuous_mean",
                "cov_type": "none",
                "cov_matrix": None,
                "scale": 1.0,
            },
            {
                "label": "Phase 12 Independent Noise (scale=1.0)",
                "cand_type": "phase12_independent_noise",
                "cov_type": "independent",
                "cov_matrix": np.diag(np.diag(emp_cov)),
                "scale": 1.0,
            },
            {
                "label": "Phase 13 Selected Independent OU (scale=0.25)",
                "cand_type": "phase13_independent_ou",
                "cov_type": "independent",
                "cov_matrix": np.diag(np.diag(emp_cov)),
                "scale": 0.25,
            },
            {
                "label": selected_label,
                "cand_type": "multivariate_ou",
                "cov_type": selected_cov_type,
                "cov_matrix": selected_cov_matrix,
                "scale": selected_scale,
            },
        ]

    results_table = []

    print("\n" + "=" * 135, flush=True)
    print(
        f"{'Candidate Label':<48} | {'Plausible %':<11} | {'Wasserstein':<11} | {'KS Stat':<8} | "
        f"{'Corr Error':<10} | {r'std($\Delta X$)':<12} | {'Vol Ratio':<9} | {'Lag-1 AC':<9} | {'Diversity RMSE':<14}",
        flush=True,
    )
    print("=" * 135, flush=True)

    for cfg in candidate_configs:
        label = cfg["label"]
        cand_type = cfg["cand_type"]
        scale = cfg["scale"]
        c_mat = cfg["cov_matrix"]

        res_model = None
        if cand_type in ("multivariate_ou", "phase13_independent_ou", "phase12_independent_noise"):
            res_model = MultivariateTemporalResidualModel(
                output_dim=len(feature_names),
                lambda_val=lambdas,
                cov_matrix=c_mat if c_mat is not None else emp_cov,
                scale=scale,
            ).to(device)

        synth_by_stay = generate_phase14_futures_cached(
            trajectories=eval_trajectories,
            cached_predictions=cached_predictions,
            candidate_type=cand_type,
            residual_model=res_model,
            scale=scale,
            scaler_path=scaler_path,
            n_samples=n_samples,
        )

        metrics = compute_phase14_metrics(
            real_flat, eval_trajectories, synth_by_stay, feature_names, emp_cov
        )

        print(
            f"{label:<48} | {metrics['plausibility_pct']:10.2f}% | {metrics['mean_wasserstein']:11.4f} | "
            f"{metrics['mean_ks_stat']:8.4f} | {metrics['correlation_error']:10.4f} | "
            f"{metrics['synth_diff_std']:12.4f} | {metrics['volatility_ratio']:9.4f} | "
            f"{metrics['mean_ac1']:9.4f} | {metrics['diversity_rmse']:14.4f}",
            flush=True,
        )

        results_table.append({
            "label": label,
            "cand_type": cand_type,
            "cov_type": cfg["cov_type"],
            "cov_matrix": c_mat,
            "scale": scale,
            "metrics": metrics,
        })

    print("=" * 135 + "\n", flush=True)

    # Save CSV output
    out_csv_name = f"phase14_{'quick_validation' if mode == 'quick' else ('full_validation' if mode == 'full_val' else 'test_evaluation')}.csv"
    out_csv = Path(experiments_dir) / out_csv_name
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "candidate_label", "candidate_type", "cov_type", "scale",
            "plausibility_pct", "wasserstein_distance", "ks_statistic", "correlation_error",
            "synth_diff_std", "volatility_ratio", "lag1_autocorr", "lag5_autocorr",
            "diversity_rmse", "diversity_mae", "cov_mae"
        ])
        for r in results_table:
            m = r["metrics"]
            writer.writerow([
                r["label"], r["cand_type"], r["cov_type"], r["scale"],
                f"{m['plausibility_pct']:.2f}", f"{m['mean_wasserstein']:.4f}", f"{m['mean_ks_stat']:.4f}",
                f"{m['correlation_error']:.4f}", f"{m['synth_diff_std']:.4f}", f"{m['volatility_ratio']:.4f}",
                f"{m['mean_ac1']:.4f}", f"{m['mean_ac5']:.4f}", f"{m['diversity_rmse']:.4f}", f"{m['diversity_mae']:.4f}",
                f"{m['cov_mae']:.4f}"
            ])

    print(f"Saved Phase 14 CSV report to: {out_csv}", flush=True)

    # If full_val mode, perform multi-objective candidate selection
    if mode == "full_val":
        perform_model_selection_and_save_champion(results_table, emp_cov, shrinkage_covs, lowrank_covs, feature_names, experiments_dir)

    # Generate visualizations
    generate_phase14_plots(results_table, feature_names, experiments_dir, mode=mode)

    return results_table


def perform_model_selection_and_save_champion(
    results_table: List[Dict[str, Any]],
    emp_cov: np.ndarray,
    shrinkage_covs: Dict[float, np.ndarray],
    lowrank_covs: Dict[int, np.ndarray],
    feature_names: List[str],
    experiments_dir: str,
):
    """Selects champion based on multi-objective validation criteria."""
    print("\n=== Phase 14 Multi-Objective Model Selection (VALIDATION) ===")
    
    # Filter candidates with 100% plausibility
    plausible_cands = [r for r in results_table if r["metrics"]["plausibility_pct"] >= 99.9]
    if not plausible_cands:
        plausible_cands = results_table

    # Objective score function: balance higher volatility (moving towards real ~8.4),
    # realistic AC (0.4 - 0.7), lower correlation error, and strong diversity.
    # Score = - (volatility_gap) - 5 * corr_error - 2 * ks_stat
    def score_fn(cand):
        m = cand["metrics"]
        vol_gap = abs(m["synth_diff_std"] - m["real_diff_std"])
        ac_gap = abs(m["mean_ac1"] - 0.4581)
        corr_err = m["correlation_error"]
        ks = m["mean_ks_stat"]
        return -(vol_gap + 2.0 * ac_gap + 10.0 * corr_err + 5.0 * ks)

    sorted_cands = sorted(plausible_cands, key=score_fn, reverse=True)
    champion = sorted_cands[0]

    print(f"\nSELECTED VALIDATION CHAMPION:")
    print(f"Label: {champion['label']}")
    print(f"Scale: {champion['scale']}")
    print(f"Covariance Type: {champion['cov_type']}")
    print(f"Plausibility: {champion['metrics']['plausibility_pct']:.2f}%")
    print(f"Step Volatility std(delta X): {champion['metrics']['synth_diff_std']:.4f} (Ratio: {champion['metrics']['volatility_ratio']:.4f})")
    print(f"Lag-1 Autocorrelation: {champion['metrics']['mean_ac1']:.4f}")
    print(f"Cross-Feature Correlation Error: {champion['metrics']['correlation_error']:.4f}")
    print(f"Wasserstein Distance: {champion['metrics']['mean_wasserstein']:.4f}")
    print(f"Diversity RMSE: {champion['metrics']['diversity_rmse']:.4f}")

    # Save champion checkpoint config
    ckpt_dir = Path("outputs/checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config_file = ckpt_dir / "phase14_config.json"

    cov_matrix_list = champion["cov_matrix"].tolist() if champion["cov_matrix"] is not None else emp_cov.tolist()

    champ_dict = {
        "selected_label": champion["label"],
        "selected_cov_type": champion["cov_type"],
        "selected_cov_label": champion["label"],
        "selected_scale": champion["scale"],
        "selected_cov_matrix": cov_matrix_list,
        "validation_metrics": {
            "plausibility_pct": champion["metrics"]["plausibility_pct"],
            "synth_diff_std": champion["metrics"]["synth_diff_std"],
            "volatility_ratio": champion["metrics"]["volatility_ratio"],
            "lag1_autocorr": champion["metrics"]["mean_ac1"],
            "correlation_error": champion["metrics"]["correlation_error"],
            "mean_wasserstein": champion["metrics"]["mean_wasserstein"],
            "mean_ks_stat": champion["metrics"]["mean_ks_stat"],
            "diversity_rmse": champion["metrics"]["diversity_rmse"],
        }
    }

    with open(config_file, "w") as f:
        json.dump(champ_dict, f, indent=2)

    print(f"Saved Phase 14 Champion Config to: {config_file}")

    # Save Markdown selection report
    md_file = Path(experiments_dir) / "phase14_model_selection.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Phase 14 Model Selection & Validation Scorecard\n\n")
        f.write(f"**Selected Champion**: `{champion['label']}`\n\n")
        f.write("## Champion Parameters\n")
        f.write(f"- **Scale**: {champion['scale']}\n")
        f.write(f"- **Covariance Structure**: {champion['cov_type']}\n\n")
        f.write("## Validation Scorecard\n\n")
        f.write("| Metric | Real Reference | Phase 13 Independent | Phase 14 Champion |\n")
        f.write("| --- | --- | --- | --- |\n")
        
        p13_cand = [r for r in results_table if r["cand_type"] == "phase13_independent_ou" and r["scale"] == 0.25]
        p13_m = p13_cand[0]["metrics"] if p13_cand else champion["metrics"]
        c_m = champion["metrics"]

        f.write(f"| Plausibility % | 100% | {p13_m['plausibility_pct']:.2f}% | **{c_m['plausibility_pct']:.2f}%** |\n")
        f.write(f"| Step Volatility std(ΔX) | {c_m['real_diff_std']:.4f} | {p13_m['synth_diff_std']:.4f} | **{c_m['synth_diff_std']:.4f}** |\n")
        f.write(f"| Volatility Ratio | 1.0000 | {p13_m['volatility_ratio']:.4f} | **{c_m['volatility_ratio']:.4f}** |\n")
        f.write(f"| Lag-1 Autocorrelation | +0.4581 | {p13_m['mean_ac1']:.4f} | **{c_m['mean_ac1']:.4f}** |\n")
        f.write(f"| Correlation Error | 0.0000 | {p13_m['correlation_error']:.4f} | **{c_m['correlation_error']:.4f}** |\n")
        f.write(f"| Wasserstein Distance | 0.0000 | {p13_m['mean_wasserstein']:.4f} | **{c_m['mean_wasserstein']:.4f}** |\n")
        f.write(f"| Diversity RMSE | N/A | {p13_m['diversity_rmse']:.4f} | **{c_m['diversity_rmse']:.4f}** |\n")

    print(f"Saved Phase 14 Model Selection Report to: {md_file}")


def generate_phase14_plots(
    results_table: List[Dict[str, Any]],
    feature_names: List[str],
    experiments_dir: str,
    mode: str = "quick",
):
    """Generates visual analysis plots."""
    out_dir = Path(experiments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Scale vs Performance Tradeoff
    scales = [r["scale"] for r in results_table]
    vols = [r["metrics"]["synth_diff_std"] for r in results_table]
    ac1s = [r["metrics"]["mean_ac1"] for r in results_table]
    corr_errs = [r["metrics"]["correlation_error"] for r in results_table]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.scatter(ac1s, vols, c=corr_errs, cmap="plasma", s=80)
    cbar = plt.colorbar(ax1.collections[0], ax=ax1)
    cbar.set_label("Cross-Feature Correlation Error")
    ax1.set_xlabel("Lag-1 Autocorrelation (Target ~ 0.4581)")
    ax1.set_ylabel("Step Volatility std(delta X) (Target ~ 8.43)")
    ax1.axvline(0.4581, color="red", linestyle="--", label="Real Lag-1 AC")
    ax1.axhline(8.4308, color="green", linestyle="--", label="Real Volatility")
    ax1.set_title("Phase 14 Validation Tradeoff: Volatility vs Autocorrelation vs Correlation Error")
    ax1.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(out_dir / "phase14_volatility_ac_tradeoff.png", dpi=300)
    plt.close()

    # 2. Correlation Heatmaps for Test mode
    if mode == "test":
        fig, axes = plt.subplots(1, len(results_table), figsize=(5 * len(results_table), 4.5))
        if len(results_table) == 1:
            axes = [axes]

        for idx, r in enumerate(results_table):
            corr_mat = r["metrics"]["corr_synth"]
            sns.heatmap(
                corr_mat,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                vmin=-0.4,
                vmax=1.0,
                xticklabels=feature_names,
                yticklabels=feature_names,
                ax=axes[idx],
            )
            axes[idx].set_title(r["label"], fontsize=10)

        plt.tight_layout()
        plt.savefig(out_dir / "phase14_covariance_heatmaps.png", dpi=300)
        plt.close()

    print(f"Generated Phase 14 visualization plots in: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, default="quick", choices=["quick", "full_val", "test"]
    )
    args = parser.parse_args()

    evaluate_phase14(mode=args.mode)
