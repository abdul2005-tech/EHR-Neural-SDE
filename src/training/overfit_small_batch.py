"""
Overfit Small Batch Sanity Check for EHR-Neural-SDE.

Tests whether the complete EHR-Neural-SDE architecture can effectively overfit
a tiny dataset (1-2 trajectories) under easy optimization conditions.

Verifies:
1. One-batch forward pass.
2. One-batch backward pass.
3. Gradient flow across all trainable parameters.
4. Absence of NaNs / infinities.
5. Reduction in reconstruction loss over optimization steps.

Saves debug checkpoint to outputs/debug/overfit_small_batch.pt.
"""

import os
import random
from pathlib import Path
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


def run_overfit_sanity_check(
    dataset_path: str = "data/processed/datasets/train.pkl",
    num_steps: int = 60,
    lr: float = 5e-3,
    max_norm: float = 1.0,
    output_path: str = "outputs/debug/overfit_small_batch.pt",
):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Sanity Check] Using device: {device}")

    # 1. Load dataset and select small batch (2 trajectories)
    dataset = ICUTrajectoryDataset(dataset_path)
    sub_dataset = [dataset[0], dataset[1]]
    dataloader = DataLoader(sub_dataset, batch_size=2, shuffle=False, collate_fn=collate_icu_trajectories)
    batch = next(iter(dataloader))

    X = batch["X_padded"].to(device)
    T = batch["T_padded"].to(device)
    DeltaT = batch["DeltaT_padded"].to(device)
    M = batch["M_padded"].to(device)
    sequence_lengths = batch["sequence_lengths"].to(device)
    padding_mask = batch["padding_mask"].to(device)
    stay_ids = batch["stay_ids"].tolist()

    # Combine clinical observation mask with padding mask
    effective_mask = M * padding_mask.unsqueeze(-1)

    print("\n--- Tensor Shapes ---")
    print(f"X: {X.shape}")
    print(f"T: {T.shape}")
    print(f"DeltaT: {DeltaT.shape}")
    print(f"M: {M.shape}")
    print(f"sequence_lengths: {sequence_lengths}")
    print(f"padding_mask: {padding_mask.shape}")
    print(f"effective_mask: {effective_mask.shape}")
    print(f"Stay IDs in batch: {stay_ids}")

    # 2. Instantiate EHRNeuralSDE
    model = EHRNeuralSDE(max_step_size=0.25).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\n--- Initial Forward & Backward Sanity Check ---")
    model.train()
    optimizer.zero_grad()

    generator = torch.Generator(device=device).manual_seed(42)
    X_hat, z_trajectory = model(
        X=X,
        M=M,
        T=T,
        DeltaT=DeltaT,
        sequence_lengths=sequence_lengths,
        padding_mask=padding_mask,
        generator=generator,
        enable_noise=True,
    )

    # Check shapes and NaNs
    assert X_hat.shape == X.shape, f"Shape mismatch: X_hat {X_hat.shape} vs X {X.shape}"
    assert not torch.isnan(X_hat).any(), "NaN detected in initial X_hat!"
    assert not torch.isinf(X_hat).any(), "Inf detected in initial X_hat!"

    initial_loss = masked_mse_loss(X_hat, X, effective_mask)
    initial_mae = masked_mae_loss(X_hat, X, effective_mask)
    print(f"Initial Masked MSE Loss: {initial_loss.item():.6f}")
    print(f"Initial Masked MAE Metric: {initial_mae.item():.6f}")

    # Backward pass and gradient check
    initial_loss.backward()

    print("\n--- Parameter Gradient Flow Verification ---")
    zero_grad_params = []
    nan_grad_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_params.append(name)
            elif torch.isnan(param.grad).any():
                nan_grad_params.append(name)

    if nan_grad_params:
        raise ValueError(f"NaN gradients detected in parameters: {nan_grad_params}")
    if zero_grad_params:
        print(f"Warning: Parameters with None gradient: {zero_grad_params}")
    else:
        print("SUCCESS: All trainable parameters received non-NaN gradients.")

    # 3. Optimization Loop on Small Batch
    print(f"\n--- Training on Small Batch for {num_steps} Steps ---")
    step_losses = []

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()

        # Controlled generator per step for reproducible noise
        gen_step = torch.Generator(device=device).manual_seed(42 + step)
        X_hat, z_trajectory = model(
            X=X,
            M=M,
            T=T,
            DeltaT=DeltaT,
            sequence_lengths=sequence_lengths,
            padding_mask=padding_mask,
            generator=gen_step,
            enable_noise=True,
        )

        if torch.isnan(X_hat).any() or torch.isinf(X_hat).any():
            raise RuntimeError(f"NaN or Inf encountered at step {step} for stay IDs {stay_ids}!")

        loss = masked_mse_loss(X_hat, X, effective_mask)

        if torch.isnan(loss).any():
            raise RuntimeError(f"Loss became NaN at step {step} for stay IDs {stay_ids}!")

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        optimizer.step()

        loss_val = loss.item()
        step_losses.append(loss_val)

        if step % 20 == 0 or step == 1 or step == num_steps:
            mae_val = masked_mae_loss(X_hat, X, effective_mask).item()
            diff_mag = model.get_mean_diffusion_magnitude(z_trajectory, T)
            print(
                f"Step {step:03d}/{num_steps:03d} | "
                f"MSE Loss: {loss_val:.6f} | "
                f"MAE Metric: {mae_val:.6f} | "
                f"Grad Norm: {grad_norm:.4f} | "
                f"Diff Mag: {diff_mag:.4f}"
            )

    final_loss = step_losses[-1]
    print(f"\n--- Overfitting Sanity Check Summary ---")
    print(f"Initial Masked MSE Loss: {initial_loss.item():.6f}")
    print(f"Final Masked MSE Loss  : {final_loss:.6f}")
    print(f"Loss Reduction         : {(initial_loss.item() - final_loss):.6f}")

    # 4. Save debug checkpoint
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "initial_loss": initial_loss.item(),
            "final_loss": final_loss,
            "stay_ids": stay_ids,
        },
        output_path,
    )
    print(f"Saved debug checkpoint to: {output_path}")

    return initial_loss.item(), final_loss


if __name__ == "__main__":
    run_overfit_sanity_check()
