"""
Evaluation Script for Trained EHR-Neural-SDE Model.

Evaluates the best checkpoint (outputs/checkpoints/best_model.pt) across:
1. Training split (data/processed/datasets/train.pkl)
2. Validation split (data/processed/datasets/val.pkl)
3. Testing split (data/processed/datasets/test.pkl)

Computes:
- Overall Masked MSE
- Overall Masked MAE
- Feature-wise Masked MSE (Heart Rate, Resp Rate, SpO2, Sys BP, Dia BP)
- Feature-wise Masked MAE (Heart Rate, Resp Rate, SpO2, Sys BP, Dia BP)
- Mean Diffusion Magnitude
"""

import json
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.trajectory_dataset import ICUTrajectoryDataset
from src.data.collate import collate_icu_trajectories
from src.data.scaler_utils import load_scaler_params
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.losses import masked_mse_loss, masked_mae_loss


def evaluate_split(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    feature_names: list,
    seed: int = 4200,
) -> Dict[str, Any]:
    """
    Evaluates EHRNeuralSDE on a dataset split and returns metrics.

    Args:
        model: Trained EHRNeuralSDE model
        dataloader: DataLoader for the dataset split
        device: Torch computation device
        feature_names: List of physiological feature names
        seed: Random seed for reproducible evaluation noise

    Returns:
        Dictionary containing overall and feature-wise MSE and MAE, plus diffusion magnitude.
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_diff_mag = 0.0
    num_batches = len(dataloader)

    num_features = len(feature_names)
    feature_sq_err = torch.zeros(num_features, device=device)
    feature_abs_err = torch.zeros(num_features, device=device)
    feature_obs_cnt = torch.zeros(num_features, device=device)

    gen_eval = torch.Generator(device=device).manual_seed(seed)

    with torch.no_grad():
        for batch in dataloader:
            X = batch["X_padded"].to(device)
            T = batch["T_padded"].to(device)
            DeltaT = batch["DeltaT_padded"].to(device)
            M = batch["M_padded"].to(device)
            sequence_lengths = batch["sequence_lengths"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            effective_mask = M * padding_mask.unsqueeze(-1)

            X_hat, z_trajectory = model(
                X=X,
                M=M,
                T=T,
                DeltaT=DeltaT,
                sequence_lengths=sequence_lengths,
                padding_mask=padding_mask,
                generator=gen_eval,
                enable_noise=True,
            )

            mse = masked_mse_loss(X_hat, X, effective_mask)
            mae = masked_mae_loss(X_hat, X, effective_mask)
            diff_mag = model.get_mean_diffusion_magnitude(z_trajectory, T)

            total_mse += mse.item()
            total_mae += mae.item()
            total_diff_mag += diff_mag

            # Feature-wise statistics accumulator
            sq_err = (X_hat - X) ** 2 * effective_mask
            abs_err = torch.abs(X_hat - X) * effective_mask

            # Sum over batch and sequence dimensions (dim=(0, 1))
            feature_sq_err += sq_err.sum(dim=(0, 1))
            feature_abs_err += abs_err.sum(dim=(0, 1))
            feature_obs_cnt += effective_mask.sum(dim=(0, 1))

    feature_mse = (feature_sq_err / torch.clamp(feature_obs_cnt, min=1.0)).cpu().numpy()
    feature_mae = (feature_abs_err / torch.clamp(feature_obs_cnt, min=1.0)).cpu().numpy()

    feature_mse_dict = {name: float(val) for name, val in zip(feature_names, feature_mse)}
    feature_mae_dict = {name: float(val) for name, val in zip(feature_names, feature_mae)}

    return {
        "masked_mse": total_mse / num_batches,
        "masked_mae": total_mae / num_batches,
        "feature_wise_mse": feature_mse_dict,
        "feature_wise_mae": feature_mae_dict,
        "mean_diffusion_magnitude": total_diff_mag / num_batches,
    }


def run_evaluation(
    checkpoint_path: str = "outputs/checkpoints/best_model.pt",
    scaler_path: str = "data/processed/scaler.json",
    train_path: str = "data/processed/datasets/train.pkl",
    val_path: str = "data/processed/datasets/val.pkl",
    test_path: str = "data/processed/datasets/test.pkl",
    batch_size: int = 16,
) -> Dict[str, Any]:
    """
    Main evaluation pipeline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Evaluating Trained EHR-Neural-SDE Model ===")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load scaler info for feature names
    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]

    # Instantiate and load model
    model = EHRNeuralSDE(max_step_size=0.25).to(device)
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint saved at Epoch {checkpoint.get('epoch', 'N/A')}")

    results = {}

    splits = [
        ("train", train_path),
        ("val", val_path),
        ("test", test_path),
    ]

    for name, path in splits:
        if not Path(path).exists():
            print(f"Skipping {name} split (file not found: {path})")
            continue

        dataset = ICUTrajectoryDataset(path)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_icu_trajectories,
        )

        metrics = evaluate_split(
            model=model,
            dataloader=loader,
            device=device,
            feature_names=feature_names,
            seed=4200,
        )
        results[name] = metrics

        print(f"\n--- {name.upper()} SPLIT RESULTS ---")
        print(f"Masked MSE : {metrics['masked_mse']:.6f}")
        print(f"Masked MAE : {metrics['masked_mae']:.6f}")
        print(f"Diffusion  : {metrics['mean_diffusion_magnitude']:.6f}")
        print("Feature-wise Masked MSE:")
        for feat, val in metrics["feature_wise_mse"].items():
            print(f"  - {feat:18s}: {val:.6f}")
        print("Feature-wise Masked MAE:")
        for feat, val in metrics["feature_wise_mae"].items():
            print(f"  - {feat:18s}: {val:.6f}")

    return results


if __name__ == "__main__":
    run_evaluation()
