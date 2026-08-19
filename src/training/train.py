"""
Full Training Script for EHR-Neural-SDE Model.

Implements Phase 9 continuous-time conditional trajectory generation training:
- Dataset loading (train.pkl, val.pkl) via ICUTrajectoryDataset and collate_icu_trajectories.
- Adam optimizer (initial lr = 1e-3), gradient clipping (max_norm = 1.0), 20 epochs.
- Early stopping with patience = 5 epochs based on validation masked MSE.
- Saves best checkpoint to outputs/checkpoints/best_model.pt.
- Saves training curves to experiments/phase9_training_curves.png.
- Saves training logs to experiments/phase9_training_log.csv.
- Strict numerical stability checks with detailed error reporting on NaNs/Infs.
"""

import csv
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.trajectory_dataset import ICUTrajectoryDataset
from src.data.collate import collate_icu_trajectories
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.losses import masked_mse_loss, masked_mae_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    Trains EHRNeuralSDE for one epoch.

    Returns:
        Tuple of (epoch_mse, epoch_mae, epoch_grad_norm, epoch_diff_mag)
    """
    model.train()
    total_mse = 0.0
    total_mae = 0.0
    total_grad_norm = 0.0
    total_diff_mag = 0.0
    num_batches = len(dataloader)

    for batch_idx, batch in enumerate(dataloader):
        optimizer.zero_grad()

        X = batch["X_padded"].to(device)
        T = batch["T_padded"].to(device)
        DeltaT = batch["DeltaT_padded"].to(device)
        M = batch["M_padded"].to(device)
        sequence_lengths = batch["sequence_lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        stay_ids = batch["stay_ids"].tolist()

        effective_mask = M * padding_mask.unsqueeze(-1)

        # Stochastic noise with seed derived from epoch and batch
        gen_batch = torch.Generator(device=device).manual_seed(epoch * 1000 + batch_idx)

        X_hat, z_trajectory = model(
            X=X,
            M=M,
            T=T,
            DeltaT=DeltaT,
            sequence_lengths=sequence_lengths,
            padding_mask=padding_mask,
            generator=gen_batch,
            enable_noise=True,
        )

        # Numerical stability checks
        if torch.isnan(X_hat).any() or torch.isinf(X_hat).any():
            raise RuntimeError(
                f"NaN or Inf encountered in X_hat at Epoch {epoch}, Batch {batch_idx}! Stay IDs: {stay_ids}"
            )

        loss = masked_mse_loss(X_hat, X, effective_mask)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            raise RuntimeError(
                f"NaN or Inf loss encountered at Epoch {epoch}, Batch {batch_idx}! Stay IDs: {stay_ids}"
            )

        mae = masked_mae_loss(X_hat, X, effective_mask)

        loss.backward()

        # Check parameter gradients
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    raise RuntimeError(
                        f"NaN/Inf gradient in parameter {name} at Epoch {epoch}, Batch {batch_idx}! Stay IDs: {stay_ids}"
                    )

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        diff_mag = model.get_mean_diffusion_magnitude(z_trajectory, T)

        total_mse += loss.item()
        total_mae += mae.item()
        total_grad_norm += float(grad_norm)
        total_diff_mag += diff_mag

    return (
        total_mse / num_batches,
        total_mae / num_batches,
        total_grad_norm / num_batches,
        total_diff_mag / num_batches,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    seed: int = 4200,
) -> Tuple[float, float, float]:
    """
    Evaluates EHRNeuralSDE on validation dataset with controlled stochastic noise.

    Returns:
        Tuple of (val_mse, val_mae, val_diff_mag)
    """
    model.eval()
    total_mse = 0.0
    total_mae = 0.0
    total_diff_mag = 0.0
    num_batches = len(dataloader)

    gen_val = torch.Generator(device=device).manual_seed(seed)

    for batch_idx, batch in enumerate(dataloader):
        X = batch["X_padded"].to(device)
        T = batch["T_padded"].to(device)
        DeltaT = batch["DeltaT_padded"].to(device)
        M = batch["M_padded"].to(device)
        sequence_lengths = batch["sequence_lengths"].to(device)
        padding_mask = batch["padding_mask"].to(device)
        stay_ids = batch["stay_ids"].tolist()

        effective_mask = M * padding_mask.unsqueeze(-1)

        X_hat, z_trajectory = model(
            X=X,
            M=M,
            T=T,
            DeltaT=DeltaT,
            sequence_lengths=sequence_lengths,
            padding_mask=padding_mask,
            generator=gen_val,
            enable_noise=True,
        )

        if torch.isnan(X_hat).any() or torch.isinf(X_hat).any():
            raise RuntimeError(f"NaN or Inf encountered in val X_hat at Batch {batch_idx}! Stay IDs: {stay_ids}")

        loss = masked_mse_loss(X_hat, X, effective_mask)
        mae = masked_mae_loss(X_hat, X, effective_mask)
        diff_mag = model.get_mean_diffusion_magnitude(z_trajectory, T)

        total_mse += loss.item()
        total_mae += mae.item()
        total_diff_mag += diff_mag

    return total_mse / num_batches, total_mae / num_batches, total_diff_mag / num_batches


def train_pipeline(
    train_path: str = "data/processed/datasets/train.pkl",
    val_path: str = "data/processed/datasets/val.pkl",
    checkpoint_dir: str = "outputs/checkpoints",
    experiments_dir: str = "experiments",
    batch_size: int = 16,
    num_epochs: int = 20,
    lr: float = 1e-3,
    max_norm: float = 1.0,
    patience: int = 5,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Starting EHR-Neural-SDE Training Pipeline ===")
    print(f"Device: {device}")
    print(f"Hyperparameters: epochs={num_epochs}, lr={lr}, batch_size={batch_size}, max_norm={max_norm}, patience={patience}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(experiments_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    train_dataset = ICUTrajectoryDataset(train_path)
    val_dataset = ICUTrajectoryDataset(val_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_icu_trajectories,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_icu_trajectories,
    )

    print(f"Loaded {len(train_dataset)} training samples ({len(train_loader)} batches)")
    print(f"Loaded {len(val_dataset)} validation samples ({len(val_loader)} batches)")

    # 2. Instantiate Model & Optimizer
    model = EHRNeuralSDE(max_step_size=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 3. Training Loop State Tracking
    best_val_mse = float("inf")
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = Path(checkpoint_dir) / "best_model.pt"

    history = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "train_mae": [],
        "val_mae": [],
        "grad_norm": [],
        "diffusion_mean": [],
        "learning_rate": [],
    }

    # 4. Execute Epochs
    for epoch in range(1, num_epochs + 1):
        train_mse, train_mae, grad_norm, train_diff_mag = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            max_norm=max_norm,
        )

        val_mse, val_mae, val_diff_mag = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            seed=4200,
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(epoch)
        history["train_mse"].append(train_mse)
        history["val_mse"].append(val_mse)
        history["train_mae"].append(train_mae)
        history["val_mae"].append(val_mae)
        history["grad_norm"].append(grad_norm)
        history["diffusion_mean"].append(train_diff_mag)
        history["learning_rate"].append(current_lr)

        print(
            f"Epoch {epoch:02d}/{num_epochs:02d} | "
            f"Train MSE: {train_mse:.4f} | Val MSE: {val_mse:.4f} | "
            f"Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f} | "
            f"GradNorm: {grad_norm:.4f} | DiffMag: {train_diff_mag:.4f}"
        )

        # Checkpoint & Early Stopping
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "train_mse": train_mse,
                    "train_mae": train_mae,
                },
                checkpoint_path,
            )
            print(f" -> Best validation MSE improved to {val_mse:.4f}. Model saved to {checkpoint_path}")
        else:
            patience_counter += 1
            print(f" -> No improvement in validation MSE for {patience_counter}/{patience} epochs.")
            if patience_counter >= patience:
                print(f"Early stopping triggered at Epoch {epoch}. Best epoch was Epoch {best_epoch} (Val MSE: {best_val_mse:.4f}).")
                break

    # 5. Save Training Logs CSV
    log_csv_path = Path(experiments_dir) / "phase9_training_log.csv"
    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_mse", "val_mse", "train_mae", "val_mae", "grad_norm", "diffusion_mean", "learning_rate"])
        for i in range(len(history["epoch"])):
            writer.writerow([
                history["epoch"][i],
                f"{history['train_mse'][i]:.6f}",
                f"{history['val_mse'][i]:.6f}",
                f"{history['train_mae'][i]:.6f}",
                f"{history['val_mae'][i]:.6f}",
                f"{history['grad_norm'][i]:.6f}",
                f"{history['diffusion_mean'][i]:.6f}",
                f"{history['learning_rate'][i]:.6e}",
            ])
    print(f"Saved training log CSV to: {log_csv_path}")

    # 6. Plot Training Curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    epochs_range = history["epoch"]

    # Panel 1: Training MSE
    axes[0, 0].plot(epochs_range, history["train_mse"], "b-o", label="Train Masked MSE")
    axes[0, 0].set_title("Training Masked MSE vs Epoch")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("MSE Loss")
    axes[0, 0].grid(True, linestyle="--", alpha=0.6)
    axes[0, 0].legend()

    # Panel 2: Validation MSE
    axes[0, 1].plot(epochs_range, history["val_mse"], "r-s", label="Val Masked MSE")
    axes[0, 1].axvline(best_epoch, color="g", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[0, 1].set_title("Validation Masked MSE vs Epoch")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("MSE Loss")
    axes[0, 1].grid(True, linestyle="--", alpha=0.6)
    axes[0, 1].legend()

    # Panel 3: Training MAE
    axes[1, 0].plot(epochs_range, history["train_mae"], "b--^", label="Train Masked MAE")
    axes[1, 0].set_title("Training Masked MAE vs Epoch")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("MAE Metric")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)
    axes[1, 0].legend()

    # Panel 4: Validation MAE
    axes[1, 1].plot(epochs_range, history["val_mae"], "r--d", label="Val Masked MAE")
    axes[1, 1].axvline(best_epoch, color="g", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[1, 1].set_title("Validation Masked MAE vs Epoch")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("MAE Metric")
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)
    axes[1, 1].legend()

    plt.tight_layout()
    curves_plot_path = Path(experiments_dir) / "phase9_training_curves.png"
    plt.savefig(curves_plot_path, dpi=300)
    plt.close()
    print(f"Saved training curves plot to: {curves_plot_path}")

    return history, best_val_mse


if __name__ == "__main__":
    train_pipeline()
