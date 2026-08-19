"""
Ablation Training Suite for EHR-Neural-SDE (Phase 11 Experiments 11A - 11G).

Trains candidate ablation models:
1. 11A Baseline Control        : L = L_MSE                              -> outputs/checkpoints/phase11_baseline.pt
2. 11B Temporal Difference Loss: L = L_MSE + lambda_delta * L_delta    -> outputs/checkpoints/phase11_temporal.pt (lambda_delta in {0.01, 0.1, 0.5})
3. 11C Temporal Rate Loss      : L = L_MSE + 0.1*L_delta + 0.01*L_rate -> outputs/checkpoints/phase11_rate.pt
4. 11F Probabilistic Decoder   : L = L_NLL                              -> outputs/checkpoints/phase11_probabilistic.pt
5. 11G Combined Model          : L = 1.0*L_MSE + 0.1*L_delta + 0.1*L_NLL -> outputs/checkpoints/phase11_combined.pt

Strictly uses VALIDATION set for model selection and hyperparameter tuning. TEST set remains untouched!
"""

import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.trajectory_dataset import ICUTrajectoryDataset
from src.data.collate import collate_icu_trajectories
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.losses import (
    masked_mse_loss,
    masked_mae_loss,
    temporal_difference_loss,
    temporal_rate_loss,
    gaussian_nll_loss,
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_ablation_model(
    exp_name: str,
    checkpoint_name: str,
    use_probabilistic: bool = False,
    lambda_mse: float = 1.0,
    lambda_delta: float = 0.0,
    lambda_rate: float = 0.0,
    lambda_nll: float = 0.0,
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
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========================================================")
    print(f"Running Experiment: {exp_name}")
    print(f"Config: prob={use_probabilistic}, mse={lambda_mse}, delta={lambda_delta}, rate={lambda_rate}, nll={lambda_nll}")
    print(f"========================================================")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    train_dataset = ICUTrajectoryDataset(train_path)
    val_dataset = ICUTrajectoryDataset(val_path)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_icu_trajectories)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_icu_trajectories)

    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=use_probabilistic).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = Path(checkpoint_dir) / checkpoint_name

    history = {"train_loss": [], "val_loss": [], "val_mse": [], "val_mae": []}

    for epoch in range(1, num_epochs + 1):
        # 1. Train Epoch
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
            gen_batch = torch.Generator(device=device).manual_seed(epoch * 1000 + batch_idx)

            if use_probabilistic:
                mu, z_traj, sigma = model(
                    X=X, M=M, T=T, DeltaT=DeltaT,
                    sequence_lengths=sequence_lengths, padding_mask=padding_mask,
                    generator=gen_batch, enable_noise=True
                )
                pred_X = mu
            else:
                X_hat, z_traj = model(
                    X=X, M=M, T=T, DeltaT=DeltaT,
                    sequence_lengths=sequence_lengths, padding_mask=padding_mask,
                    generator=gen_batch, enable_noise=True
                )
                pred_X = X_hat

            # Compute combined loss components
            loss = torch.tensor(0.0, device=device)

            if lambda_mse > 0:
                l_mse = masked_mse_loss(pred_X, X, effective_mask)
                loss = loss + lambda_mse * l_mse

            if lambda_delta > 0:
                l_delta = temporal_difference_loss(pred_X, X, M, padding_mask)
                loss = loss + lambda_delta * l_delta

            if lambda_rate > 0:
                l_rate = temporal_rate_loss(pred_X, X, DeltaT, M, padding_mask)
                loss = loss + lambda_rate * l_rate

            if lambda_nll > 0 and use_probabilistic:
                l_nll = gaussian_nll_loss(mu, sigma, X, M, padding_mask)
                loss = loss + lambda_nll * l_nll

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            optimizer.step()

            train_loss_total += loss.item()

        epoch_train_loss = train_loss_total / len(train_loader)

        # 2. Evaluate Validation Epoch
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

                if use_probabilistic:
                    mu, z_traj, sigma = model(
                        X=X, M=M, T=T, DeltaT=DeltaT,
                        sequence_lengths=sequence_lengths, padding_mask=padding_mask,
                        generator=gen_val, enable_noise=True
                    )
                    pred_X = mu
                else:
                    X_hat, z_traj = model(
                        X=X, M=M, T=T, DeltaT=DeltaT,
                        sequence_lengths=sequence_lengths, padding_mask=padding_mask,
                        generator=gen_val, enable_noise=True
                    )
                    pred_X = X_hat

                v_mse = masked_mse_loss(pred_X, X, effective_mask)
                v_mae = masked_mae_loss(pred_X, X, effective_mask)

                v_loss = torch.tensor(0.0, device=device)
                if lambda_mse > 0:
                    v_loss += lambda_mse * v_mse
                if lambda_delta > 0:
                    v_loss += lambda_delta * temporal_difference_loss(pred_X, X, M, padding_mask)
                if lambda_rate > 0:
                    v_loss += lambda_rate * temporal_rate_loss(pred_X, X, DeltaT, M, padding_mask)
                if lambda_nll > 0 and use_probabilistic:
                    v_loss += lambda_nll * gaussian_nll_loss(mu, sigma, X, M, padding_mask)

                val_loss_total += v_loss.item()
                val_mse_total += v_mse.item()
                val_mae_total += v_mae.item()

        epoch_val_loss = val_loss_total / len(val_loader)
        epoch_val_mse = val_mse_total / len(val_loader)
        epoch_val_mae = val_mae_total / len(val_loader)

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["val_mse"].append(epoch_val_mse)
        history["val_mae"].append(epoch_val_mae)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | "
            f"Val MSE: {epoch_val_mse:.4f} | Val MAE: {epoch_val_mae:.4f}"
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
                    "use_probabilistic": use_probabilistic,
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
        "history": history,
    }


def run_all_ablation_experiments():
    print("=== Phase 11: Executing Controlled Ablation Experiments ===")

    # 1. 11A Baseline Control
    train_ablation_model(
        exp_name="11A_Baseline",
        checkpoint_name="phase11_baseline.pt",
        use_probabilistic=False,
        lambda_mse=1.0, lambda_delta=0.0, lambda_rate=0.0, lambda_nll=0.0,
    )

    # 2. 11B Temporal Difference Loss Tuning (\lambda_\delta \in {0.01, 0.1, 0.5})
    best_delta_loss = float("inf")
    best_delta_lambda = 0.1

    for l_d in [0.01, 0.1, 0.5]:
        res_d = train_ablation_model(
            exp_name=f"11B_Temporal_delta_{l_d}",
            checkpoint_name=f"phase11_temporal_delta_{l_d}.pt",
            use_probabilistic=False,
            lambda_mse=1.0, lambda_delta=l_d, lambda_rate=0.0, lambda_nll=0.0,
        )
        if res_d["best_val_loss"] < best_delta_loss:
            best_delta_loss = res_d["best_val_loss"]
            best_delta_lambda = l_d

    print(f"\n[Validation Tuning] Selected best lambda_delta = {best_delta_lambda}")

    # Copy best temporal model to phase11_temporal.pt
    best_temp_ckpt = Path("outputs/checkpoints") / f"phase11_temporal_delta_{best_delta_lambda}.pt"
    target_temp_ckpt = Path("outputs/checkpoints/phase11_temporal.pt")
    if best_temp_ckpt.exists():
        import shutil
        shutil.copy(best_temp_ckpt, target_temp_ckpt)
        print(f"Copied best temporal checkpoint to: {target_temp_ckpt}")

    # 3. 11C Temporal Rate-of-Change Loss
    train_ablation_model(
        exp_name="11C_Rate_Loss",
        checkpoint_name="phase11_rate.pt",
        use_probabilistic=False,
        lambda_mse=1.0, lambda_delta=best_delta_lambda, lambda_rate=0.01, lambda_nll=0.0,
    )

    # 4. 11F Probabilistic Decoder (NLL)
    train_ablation_model(
        exp_name="11F_Probabilistic_NLL",
        checkpoint_name="phase11_probabilistic.pt",
        use_probabilistic=True,
        lambda_mse=0.0, lambda_delta=0.0, lambda_rate=0.0, lambda_nll=1.0,
    )

    # 5. 11G Combined Model
    train_ablation_model(
        exp_name="11G_Combined",
        checkpoint_name="phase11_combined.pt",
        use_probabilistic=True,
        lambda_mse=1.0, lambda_delta=0.1, lambda_rate=0.0, lambda_nll=0.1,
    )

    print("\nSUCCESS: All Phase 11 ablation models trained successfully.")


if __name__ == "__main__":
    run_all_ablation_experiments()
