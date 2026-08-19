"""
Phase 15 Feature-Specific Residual Scale Calibration Module.

Estimates feature-wise baseline residual scales S_feature = [S_HR, S_RR, S_SpO2, S_SBP, S_DBP]
using TRAIN dataset ONLY against frozen Phase 11 champion predictions and Phase 14 OU dynamics.

Protocol:
1. Load train.pkl and Phase 11 champion model (outputs/checkpoints/phase11_probabilistic.pt).
2. Compute feature-wise real step volatility std(delta X_real) on TRAIN data.
3. Compute feature-wise Phase 14 Champion residual step volatility std(delta X_Phase14) on TRAIN data.
4. Calculate volatility adjustment ratios:
       ratio_d = std(delta X_real_d) / std(delta X_Phase14_d)
       S_calibrated_d = S_Phase14 * ratio_d
5. Save results to experiments/phase15_train_scales.json.
"""

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.multivariate_temporal_residual import MultivariateTemporalResidualModel
from src.evaluation.fit_phase14_residuals import estimate_multivariate_ou_parameters_from_train


def calibrate_feature_scales_from_train(
    train_path: str = "data/processed/datasets/train.pkl",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    output_json_path: str = "experiments/phase15_train_scales.json",
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Fits feature-specific baseline scales on TRAIN data ONLY.

    Returns:
        Dict containing calibrated feature scales, real train volatilities,
        Phase 14 baseline volatilities, and target scale vectors.
    """
    print("=== Phase 15: Calibrating Feature-Specific Scales on TRAIN Set ===", flush=True)

    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)
    D = len(feature_names)

    # 1. Estimate Phase 14 multivariate OU parameters from TRAIN
    ou_params = estimate_multivariate_ou_parameters_from_train(
        train_path=train_path,
        checkpoint_path=checkpoint_path,
        scaler_path=scaler_path,
        device=device,
    )
    lambda_rates = ou_params["lambdas"]
    cov_matrix = ou_params["emp_cov"]

    # Initialize base Phase 14 residual model with scale=0.35
    phase14_residual_model = MultivariateTemporalResidualModel(
        output_dim=D,
        lambda_val=lambda_rates,
        cov_matrix=cov_matrix,
        scale=0.35,
    ).to(device)

    # Load frozen Phase 11 model
    ckpt_p = Path(checkpoint_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"Phase 11 champion checkpoint not found at: {ckpt_p}")

    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    checkpoint = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(train_path, "rb") as f:
        train_trajectories = pickle.load(f)

    print(f"Loaded {len(train_trajectories)} TRAIN stays. Extracting step volatilities...", flush=True)

    # 2. Collect step differences for Real TRAIN data and Phase 14 Synthetic trajectories
    real_delta_by_feature = {name: [] for name in feature_names}
    phase14_delta_by_feature = {name: [] for name in feature_names}

    with torch.no_grad():
        for traj_idx, traj in enumerate(train_trajectories):
            X_real_norm = traj["X"]  # [N, 5]
            T_real = traj["T"]      # [N]
            DeltaT_real = traj["DeltaT"]
            M_real = traj["M"]      # [N, 5]

            # Denormalize real observations to physical units
            X_real_phys = denormalize(X_real_norm, mean, std)

            # Compute real step differences where consecutive Timesteps are observed
            for step_idx in range(len(T_real) - 1):
                dt = T_real[step_idx + 1] - T_real[step_idx]
                if dt > 1e-4:
                    both_obs = (M_real[step_idx] == 1.0) & (M_real[step_idx + 1] == 1.0)
                    delta_x = X_real_phys[step_idx + 1] - X_real_phys[step_idx]
                    for f_idx, f_name in enumerate(feature_names):
                        if both_obs[f_idx]:
                            real_delta_by_feature[f_name].append(delta_x[f_idx])

            # Generate Phase 14 continuous mean + OU residual
            X_0 = torch.tensor(X_real_norm[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            M_0 = torch.tensor(M_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            T_0 = torch.tensor(T_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
            DeltaT_0 = torch.tensor(DeltaT_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
            T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0).to(device)

            h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
            z_0 = model.latent_projection(h_0)

            z_traj = model.sde.integrate(z0=z_0, times=T_target, enable_noise=False)
            mu_norm, _ = model.decoder(z_traj)

            # Sample Phase 14 residual with seed
            gen_s = torch.Generator(device=device).manual_seed(42 + traj_idx)
            r_ou_norm = phase14_residual_model.sample_residual(T=T_target, generator=gen_s)

            X_synth_norm = mu_norm + r_ou_norm
            X_synth_phys = denormalize(X_synth_norm.squeeze(0).cpu().numpy(), mean, std)

            # Compute Phase 14 synthetic step differences
            for step_idx in range(len(T_real) - 1):
                dt = T_real[step_idx + 1] - T_real[step_idx]
                if dt > 1e-4:
                    delta_synth = X_synth_phys[step_idx + 1] - X_synth_phys[step_idx]
                    for f_idx, f_name in enumerate(feature_names):
                        phase14_delta_by_feature[f_name].append(delta_synth[f_idx])

            if (traj_idx + 1) % 100 == 0:
                print(f"  Processed {traj_idx + 1}/{len(train_trajectories)} TRAIN stays...", flush=True)

    # 3. Calculate feature-wise volatilities and calibration ratios
    real_vol_dict = {}
    phase14_vol_dict = {}
    ratio_dict = {}
    calibrated_scales = []

    print("\n" + "=" * 80, flush=True)
    print(f"{'Feature':<18} | {'Real Train std(dX)':<18} | {'Phase 14 std(dX)':<18} | {'Ratio (Real/P14)':<18} | {'Calibrated S':<12}", flush=True)
    print("=" * 80, flush=True)

    for f_idx, f_name in enumerate(feature_names):
        r_vol = float(np.std(real_delta_by_feature[f_name])) if len(real_delta_by_feature[f_name]) > 0 else 1.0
        p14_vol = float(np.std(phase14_delta_by_feature[f_name])) if len(phase14_delta_by_feature[f_name]) > 0 else 1.0
        ratio = r_vol / max(p14_vol, 1e-4)

        # Calibrated baseline scale: S_base = 0.35 * ratio
        s_cal = 0.35 * ratio

        real_vol_dict[f_name] = r_vol
        phase14_vol_dict[f_name] = p14_vol
        ratio_dict[f_name] = ratio
        calibrated_scales.append(float(s_cal))

        print(
            f"{f_name:<18} | {r_vol:18.4f} | {p14_vol:18.4f} | {ratio:18.4f} | {s_cal:12.4f}",
            flush=True,
        )

    print("=" * 80 + "\n", flush=True)

    # Safe conservative target scale vector (clamped to max 1.5 to prevent initial overshoot before boundary attenuation)
    safe_calibrated_scales = [float(np.clip(s, 0.1, 1.5)) for s in calibrated_scales]

    result = {
        "feature_names": feature_names,
        "real_train_step_volatility": real_vol_dict,
        "phase14_train_step_volatility": phase14_vol_dict,
        "volatility_ratios": ratio_dict,
        "calibrated_scales_raw": calibrated_scales,
        "calibrated_scales_safe": safe_calibrated_scales,
        "global_phase14_scale": 0.35,
        "lambda_rates": lambda_rates.tolist() if isinstance(lambda_rates, np.ndarray) else lambda_rates,
        "empirical_covariance": cov_matrix.tolist() if isinstance(cov_matrix, np.ndarray) else cov_matrix,
    }

    out_p = Path(output_json_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Saved Phase 15 TRAIN scale calibration to: {out_p}", flush=True)
    return result


if __name__ == "__main__":
    calibrate_feature_scales_from_train()
