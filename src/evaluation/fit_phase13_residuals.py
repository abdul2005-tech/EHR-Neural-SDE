"""
Phase 13 Observation Residual Parameter Estimation Module.

Fits continuous-time Ornstein-Uhlenbeck (OU) residual parameters (lambda, sigma_stat)
using TRAIN data ONLY against frozen Phase 11 champion predictions.

Pipeline:
1. Load train.pkl and Phase 11 champion model (outputs/checkpoints/phase11_probabilistic.pt).
2. For each training stay, extract initial state z0 and integrate mean z(t) -> mu_norm(t).
3. Compute observed normalized residual: r(t) = X_real_norm(t) - mu_norm(t).
4. Estimate feature-wise stationary standard deviation sigma_stat_d = std(r_d).
5. Fit feature-wise continuous-time mean-reversion rate lambda_d > 0 using scipy optimization
   on irregular time gap pairs (t_i, t_j) with Delta t = t_j - t_i > 0.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from scipy.optimize import minimize_scalar
import torch

from src.data.scaler_utils import load_scaler_params
from src.models.ehr_neural_sde import EHRNeuralSDE


def estimate_ou_parameters_from_train(
    train_path: str = "data/processed/datasets/train.pkl",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Estimates feature-wise lambda and sigma_stat from TRAIN dataset only.

    Returns:
        lambda_fit: Array of shape (5,) containing mean-reversion rates (1/hours).
        sigma_fit: Array of shape (5,) containing normalized stationary std.
        feature_names: List of feature names.
    """
    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    D = len(feature_names)

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

    residuals_by_feature = [[] for _ in range(D)]
    pair_gaps_by_feature = [[] for _ in range(D)]
    pair_products_by_feature = [[] for _ in range(D)]

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

            for d in range(D):
                obs_indices = np.where(M_real[:, d] == 1.0)[0]
                if len(obs_indices) == 0:
                    continue

                res_obs = res[obs_indices, d]
                t_obs = T_real[obs_indices]
                residuals_by_feature[d].extend(res_obs)

                # Collect consecutive observation pairs for gap/autocorrelation analysis
                for i in range(len(obs_indices) - 1):
                    dt = t_obs[i + 1] - t_obs[i]
                    if 1e-3 <= dt <= 48.0:  # focus on meaningful gap range <= 48h
                        pair_gaps_by_feature[d].append(dt)
                        pair_products_by_feature[d].append((res_obs[i], res_obs[i + 1]))

    lambda_fit = np.zeros(D, dtype=np.float32)
    sigma_fit = np.zeros(D, dtype=np.float32)

    for d in range(D):
        res_arr = np.array(residuals_by_feature[d], dtype=np.float64)
        sigma_d = np.std(res_arr) if len(res_arr) > 0 else 1.0
        sigma_fit[d] = float(sigma_d)

        gaps = np.array(pair_gaps_by_feature[d], dtype=np.float64)
        products = np.array(pair_products_by_feature[d], dtype=np.float64)

        if len(gaps) > 10 and sigma_d > 1e-4:
            # Theoretical autocorrelation rho(dt) = exp(-lambda * dt)
            # Empirical product normalized by var = r_i * r_{i+1} / sigma^2
            norm_products = products[:, 0] * products[:, 1] / (sigma_d**2 + 1e-8)

            def loss_fn(l_val):
                l_val = max(l_val, 1e-4)
                theo_rho = np.exp(-l_val * gaps)
                return np.mean((norm_products - theo_rho) ** 2)

            res_opt = minimize_scalar(loss_fn, bounds=(1e-3, 20.0), method="bounded")
            lambda_fit[d] = float(res_opt.x)
        else:
            lambda_fit[d] = 1.0

    return lambda_fit, sigma_fit, feature_names


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=== Phase 13 TRAIN Residual Parameter Estimation ===")
    lambdas, sigmas, f_names = estimate_ou_parameters_from_train(device=device)

    print(f"\n{'Feature':<18} | {'Fitted Lambda (1/hr)':<20} | {'Fitted Sigma (norm)':<20}")
    print("-" * 65)
    for i, f in enumerate(f_names):
        print(f"{f:<18} | {lambdas[i]:<20.4f} | {sigmas[i]:<20.4f}")
