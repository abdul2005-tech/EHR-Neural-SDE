"""
Phase 14 Observation Residual Parameter & Covariance Estimation Module.

Fits continuous-time multivariate Ornstein-Uhlenbeck (OU) residual parameters and
cross-feature residual covariance matrices using TRAIN data ONLY against frozen
Phase 11 champion predictions.

Pipeline:
1. Load train.pkl and Phase 11 champion model (outputs/checkpoints/phase11_probabilistic.pt).
2. For each training stay, extract initial state z0 and integrate mean z(t) -> mu_norm(t).
3. Compute observed normalized residual: r(t) = X_real_norm(t) - mu_norm(t).
4. Estimate 5x5 empirical residual covariance matrix Sigma_empirical and correlation matrix R_empirical.
5. Compute shrinkage covariance matrices (alpha = 0.25, 0.50, 0.75, 1.00).
6. Compute low-rank covariance matrices (rank = 1, 2, 3).
7. Save experiments/phase14_train_residual_covariance.csv and heatmap plot experiments/phase14_train_residual_covariance.png.
"""

import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.data.scaler_utils import load_scaler_params
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.multivariate_temporal_residual import (
    create_shrinkage_covariance,
    create_low_rank_covariance,
)
from src.evaluation.fit_phase13_residuals import estimate_ou_parameters_from_train


def estimate_multivariate_ou_parameters_from_train(
    train_path: str = "data/processed/datasets/train.pkl",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Estimates feature-wise lambda and multivariate residual covariance matrices
    from TRAIN dataset only.

    Returns:
        Dict containing fitted lambdas, empirical residual covariance/correlation matrices,
        shrinkage matrices, low-rank matrices, and feature names.
    """
    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    D = len(feature_names)

    # 1. Get feature-wise lambda rates from Phase 13 TRAIN fit
    lambdas_fit, sigmas_fit, _ = estimate_ou_parameters_from_train(
        train_path=train_path,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        device=device,
    )

    with open(train_path, "rb") as f:
        train_trajectories = pickle.load(f)

    # Load frozen Phase 11 model
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Phase 11 champion checkpoint not found at: {ckpt_p}")

    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    checkpoint = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_raw_obs = []  # raw normalized X observations where observed
    all_residuals = []  # residuals r(t) = X_real - mu where observed

    with torch.no_grad():
        for traj in train_trajectories:
            X_real = traj["X"]  # normalized observations [N, 5]
            T_real = traj["T"]  # relative timestamps [N]
            DeltaT_real = traj["DeltaT"]
            M_real = traj["M"]  # observation mask [N, 5]

            X_0 = torch.tensor(X_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            M_0 = torch.tensor(M_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            T_0 = torch.tensor(T_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
            DeltaT_0 = torch.tensor(DeltaT_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
            T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0).to(device)

            h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
            z_0 = model.latent_projection(h_0)

            # Deterministic ODE integration (enable_noise=False) for smooth mean mu(t)
            z_traj = model.sde.integrate(z0=z_0, times=T_target, enable_noise=False)
            mu_norm, _ = model.decoder(z_traj)
            mu_np = mu_norm.squeeze(0).cpu().numpy()  # [N, 5]

            # Residual r(t) = X_real(t) - mu(t)
            res = X_real - mu_np  # [N, 5]

            # Collect co-observed time steps for cross-feature covariance estimation
            for step_idx in range(len(T_real)):
                obs_mask = M_real[step_idx] == 1.0
                if np.sum(obs_mask) >= 2:  # at least 2 co-observed vital signs at this timestep
                    all_raw_obs.append((X_real[step_idx], obs_mask))
                    all_residuals.append((res[step_idx], obs_mask))

    # Compute empirical covariance matrix on co-observed residual pairs
    emp_cov = np.zeros((D, D), dtype=np.float64)
    obs_counts = np.zeros((D, D), dtype=np.int64)

    for r_vec, mask in all_residuals:
        idx = np.where(mask)[0]
        for i in idx:
            for j in idx:
                emp_cov[i, j] += r_vec[i] * r_vec[j]
                obs_counts[i, j] += 1

    obs_counts_safe = np.maximum(obs_counts, 1)
    emp_cov = emp_cov / obs_counts_safe
    emp_cov = 0.5 * (emp_cov + emp_cov.T)  # Force exact symmetry

    # Compute empirical correlation matrix R_empirical
    std_vec = np.sqrt(np.maximum(np.diag(emp_cov), 1e-8))
    emp_corr = emp_cov / (std_vec[:, np.newaxis] * std_vec[np.newaxis, :])
    emp_corr = np.clip(emp_corr, -1.0, 1.0)

    # Compute raw observation correlation matrix for comparison
    raw_cov = np.zeros((D, D), dtype=np.float64)
    raw_counts = np.zeros((D, D), dtype=np.int64)
    for x_vec, mask in all_raw_obs:
        idx = np.where(mask)[0]
        for i in idx:
            for j in idx:
                raw_cov[i, j] += x_vec[i] * x_vec[j]
                raw_counts[i, j] += 1
    raw_cov = raw_cov / np.maximum(raw_counts, 1)
    raw_std = np.sqrt(np.maximum(np.diag(raw_cov), 1e-8))
    raw_corr = np.clip(raw_cov / (raw_std[:, np.newaxis] * raw_std[np.newaxis, :]), -1.0, 1.0)

    # 2. Derive shrinkage candidates
    shrinkage_covs = {
        alpha: create_shrinkage_covariance(emp_cov, alpha)
        for alpha in [0.25, 0.50, 0.75, 1.00]
    }

    # 3. Derive low-rank candidates
    lowrank_covs = {
        rank: create_low_rank_covariance(emp_cov, rank)
        for rank in [1, 2, 3]
    }

    # 4. Save CSV report
    out_dir = Path(experiments_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "phase14_train_residual_covariance.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_i", "feature_j", "residual_covariance", "residual_correlation", "raw_obs_correlation"])
        for i in range(D):
            for j in range(D):
                writer.writerow([
                    feature_names[i],
                    feature_names[j],
                    f"{emp_cov[i, j]:.6f}",
                    f"{emp_corr[i, j]:.6f}",
                    f"{raw_corr[i, j]:.6f}",
                ])

    print(f"Saved Phase 14 TRAIN residual covariance CSV to: {out_csv}")

    # 5. Generate visual heatmap plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(
        raw_corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=axes[0],
    )
    axes[0].set_title("Raw Observation Correlation (TRAIN)")

    sns.heatmap(
        emp_corr,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.5,
        vmax=1.0,
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=axes[1],
    )
    axes[1].set_title("Phase 11 Residual Correlation (TRAIN)")

    corr_diff = emp_corr - raw_corr
    sns.heatmap(
        corr_diff,
        annot=True,
        fmt=".3f",
        cmap="vlag",
        vmin=-0.5,
        vmax=0.5,
        xticklabels=feature_names,
        yticklabels=feature_names,
        ax=axes[2],
    )
    axes[2].set_title("Difference (Residual - Raw)")

    plt.tight_layout()
    out_png = out_dir / "phase14_train_residual_covariance.png"
    plt.savefig(out_png, dpi=300)
    plt.close()
    print(f"Saved Phase 14 TRAIN residual correlation heatmap to: {out_png}")

    return {
        "lambdas": lambdas_fit,
        "sigmas": sigmas_fit,
        "emp_cov": emp_cov,
        "emp_corr": emp_corr,
        "raw_corr": raw_corr,
        "shrinkage_covs": shrinkage_covs,
        "lowrank_covs": lowrank_covs,
        "feature_names": feature_names,
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Phase 14 TRAIN Multivariate Residual Parameter & Covariance Estimation ===")
    results = estimate_multivariate_ou_parameters_from_train(device=device)

    f_names = results["feature_names"]
    lambdas = results["lambdas"]
    emp_cov = results["emp_cov"]
    emp_corr = results["emp_corr"]

    print(f"\n{'Feature':<18} | {'Fitted Lambda (1/hr)':<20} | {'Residual Std (norm)':<20}")
    print("-" * 65)
    for i, f in enumerate(f_names):
        print(f"{f:<18} | {lambdas[i]:<20.4f} | {np.sqrt(emp_cov[i, i]):<20.4f}")

    print("\nPhase 11 Empirical Residual Correlation Matrix (TRAIN):")
    print(f"{'':<18} " + " ".join([f"{f[:6]:>8}" for f in f_names]))
    for i, f in enumerate(f_names):
        row_str = " ".join([f"{emp_corr[i, j]:>8.4f}" for j in range(len(f_names))])
        print(f"{f:<18} {row_str}")
