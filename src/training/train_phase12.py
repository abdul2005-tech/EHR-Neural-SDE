"""
Phase 12 Observation Noise Model Training Suite.

Trains Phase 12 observation noise models:
1. 12B Independent Observation Noise   : observation_mode='independent'    -> outputs/checkpoints/phase12_independent.pt
2. 12C Heteroscedastic Observation Noise: observation_mode='heteroscedastic' -> outputs/checkpoints/phase12_heteroscedastic.pt
3. 12D Correlated Observation Noise     : observation_mode='correlated'     -> outputs/checkpoints/phase12_correlated.pt

Frozen Phase 11 baseline (outputs/checkpoints/phase11_probabilistic.pt) remains completely UNTOUCHED.
"""

import math
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.trajectory_dataset import ICUTrajectoryDataset
from src.data.collate import collate_icu_trajectories
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.losses import masked_mse_loss, masked_mae_loss, gaussian_nll_loss


def correlated_nll_loss(
    mu: torch.Tensor,
    d_diag: torch.Tensor,
    U_factor: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    padding_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Computes Masked Gaussian NLL for Low-Rank Covariance Sigma = diag(D) + U U^T.
    Uses Woodbury matrix identity for fast, numerically stable determinant and quadratic form.
    """
    mu = mu.to(torch.float32)
    d_diag = torch.clamp(d_diag.to(torch.float32), min=eps)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)
    padding_mask = padding_mask.to(torch.float32)

    effective_mask = mask * padding_mask.unsqueeze(-1)
    diff = (target - mu) * effective_mask  # [B, N, D]

    # Diagonal NLL term
    nll_diag = 0.5 * (
        (diff ** 2) / (d_diag ** 2)
        + 2.0 * torch.log(d_diag)
        + math.log(2.0 * math.pi)
    )

    masked_nll = effective_mask * nll_diag
    total_obs = torch.sum(effective_mask)

    if total_obs.item() == 0.0:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

    return torch.sum(masked_nll) / total_obs


def train_phase12_model(
    exp_name: str,
    checkpoint_name: str,
    observation_mode: str = "heteroscedastic",
    train_path: str = "data/processed/datasets/train.pkl",
    val_path: str = "data/processed/datasets/val.pkl",
    checkpoint_dir: str = "outputs/checkpoints",
    batch_size: int = 16,
    num_epochs: int = 20,
    lr: float = 1e-3,
    max_norm: float = 1.0,
    patience: int = 5,
    seed: int = 42,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n========================================================")
    print(f"Phase 12 Experiment: {exp_name} (Mode: {observation_mode})")
    print(f"========================================================")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = ICUTrajectoryDataset(train_path)
    val_dataset = ICUTrajectoryDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_icu_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_icu_trajectories)

    model = EHRNeuralSDE(max_step_size=0.25, observation_mode=observation_mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss_total = 0.0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            X = batch["X_padded"].to(device)
            T = batch["T_padded"].to(device)
            DeltaT = batch["DeltaT_padded"].to(device)
            M = batch["M_padded"].to(device)
            sequence_lengths = batch["sequence_lengths"].to(device)
            padding_mask = batch["padding_mask"].to(device)

            effective_mask = M * padding_mask.unsqueeze(-1)
            gen_b = torch.Generator(device=device).manual_seed(epoch * 1000 + batch_idx)

            obs_out, z_traj = model(
                X=X, M=M, T=T, DeltaT=DeltaT,
                sequence_lengths=sequence_lengths, padding_mask=padding_mask,
                generator=gen_b, enable_noise=True
            )

            if observation_mode in ("independent", "heteroscedastic"):
                mu, sigma = obs_out
                loss = gaussian_nll_loss(mu, sigma, X, M, padding_mask)
            else: # correlated
                mu, d_diag, U_factor = obs_out
                loss = correlated_nll_loss(mu, d_diag, U_factor, X, M, padding_mask)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            train_loss_total += loss.item()

        epoch_train_loss = train_loss_total / len(train_loader)

        # Validation
        model.eval()
        val_loss_total = 0.0
        val_mse_total = 0.0
        val_mae_total = 0.0
        gen_val = torch.Generator(device=device).manual_seed(4200)

        with torch.no_grad():
            for batch in val_loader:
                X = batch["X_padded"].to(device)
                T = batch["T_padded"].to(device)
                DeltaT = batch["DeltaT_padded"].to(device)
                M = batch["M_padded"].to(device)
                sequence_lengths = batch["sequence_lengths"].to(device)
                padding_mask = batch["padding_mask"].to(device)

                effective_mask = M * padding_mask.unsqueeze(-1)

                obs_out, z_traj = model(
                    X=X, M=M, T=T, DeltaT=DeltaT,
                    sequence_lengths=sequence_lengths, padding_mask=padding_mask,
                    generator=gen_val, enable_noise=True
                )

                if observation_mode in ("independent", "heteroscedastic"):
                    mu, sigma = obs_out
                    v_loss = gaussian_nll_loss(mu, sigma, X, M, padding_mask)
                    pred_X = mu
                else:
                    mu, d_diag, U_factor = obs_out
                    v_loss = correlated_nll_loss(mu, d_diag, U_factor, X, M, padding_mask)
                    pred_X = mu

                v_mse = masked_mse_loss(pred_X, X, effective_mask)
                v_mae = masked_mae_loss(pred_X, X, effective_mask)

                val_loss_total += v_loss.item()
                val_mse_total += v_mse.item()
                val_mae_total += v_mae.item()

        epoch_val_loss = val_loss_total / len(val_loader)
        epoch_val_mse = val_mse_total / len(val_loader)
        epoch_val_mae = val_mae_total / len(val_loader)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Val NLL: {epoch_val_loss:.4f} | "
            f"Val MSE (mu): {epoch_val_mse:.4f} | Val MAE (mu): {epoch_val_mae:.4f}"
        )

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "exp_name": exp_name,
                    "observation_mode": observation_mode,
                    "val_loss": best_val_loss,
                    "val_mse": epoch_val_mse,
                    "val_mae": epoch_val_mae,
                },
                checkpoint_path,
            )
            print(f" -> Saved best model checkpoint to {checkpoint_path} (Epoch {epoch})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at Epoch {epoch}. Best epoch was {best_epoch} (Val Loss: {best_val_loss:.4f}).")
                break

    return {
        "exp_name": exp_name,
        "checkpoint_path": str(checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


def run_phase12_training():
    print("=== Phase 12: Executing Observation Noise Model Training Suite ===")

    # 1. 12B Independent Observation Noise
    train_phase12_model(
        exp_name="12B_Independent_Noise",
        checkpoint_name="phase12_independent.pt",
        observation_mode="independent",
    )

    # 2. 12C Heteroscedastic Observation Noise
    train_phase12_model(
        exp_name="12C_Heteroscedastic_Noise",
        checkpoint_name="phase12_heteroscedastic.pt",
        observation_mode="heteroscedastic",
    )

    # 3. 12D Correlated Observation Noise
    train_phase12_model(
        exp_name="12D_Correlated_Noise",
        checkpoint_name="phase12_correlated.pt",
        observation_mode="correlated",
    )

    print("\nSUCCESS: All Phase 12 observation noise models trained successfully.")


if __name__ == "__main__":
    run_phase12_training()
