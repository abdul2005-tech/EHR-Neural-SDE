"""
Masked Loss Functions for EHR Observation Reconstruction and Evaluation.

Computes Mean Squared Error (MSE) and Mean Absolute Error (MAE) exclusively over valid,
observed clinical features indicated by binary mask M (M=1: observed, M=0: missing/padded).
"""

import torch


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes Masked Mean Squared Error (MSE) Loss:
        L_MSE = sum(M * (pred - target)^2) / sum(M)

    Only observed physiological values (M=1) contribute to loss calculation. Missing values (M=0)
    are strictly ignored. Handles all-zero mask cases safely.

    Args:
        pred: Predicted physiological tensor of shape (..., D)
        target: Target physiological tensor of shape (..., D)
        mask: Binary observation mask tensor of shape (..., D)
        eps: Small epsilon constant to prevent division by zero when sum(M) == 0

    Returns:
        Scalar masked MSE loss tensor.
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)

    diff_sq = (pred - target) ** 2
    masked_sq_err = mask * diff_sq

    total_obs = torch.sum(mask)

    if total_obs.item() == 0.0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype, requires_grad=pred.requires_grad)

    return torch.sum(masked_sq_err) / total_obs


def masked_mae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes Masked Mean Absolute Error (MAE):
        L_MAE = sum(M * |pred - target|) / sum(M)

    Only observed physiological values (M=1) contribute to metric calculation. Missing values (M=0)
    are strictly ignored. Handles all-zero mask cases safely.

    Args:
        pred: Predicted physiological tensor of shape (..., D)
        target: Target physiological tensor of shape (..., D)
        mask: Binary observation mask tensor of shape (..., D)
        eps: Small epsilon constant to prevent division by zero when sum(M) == 0

    Returns:
        Scalar masked MAE metric tensor.
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)

    abs_diff = torch.abs(pred - target)
    masked_abs_err = mask * abs_diff

    total_obs = torch.sum(mask)

    if total_obs.item() == 0.0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return torch.sum(masked_abs_err) / total_obs


def temporal_difference_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    padding_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes Masked Temporal Difference Loss:
        L_delta = sum(M_pair * ( (pred_t - pred_{t-1}) - (target_t - target_{t-1}) )^2) / sum(M_pair)

    Requires valid observations at both timestep t and t-1, excluding padding.
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)
    padding_mask = padding_mask.to(torch.float32)

    delta_target = target[:, 1:, :] - target[:, :-1, :]
    delta_pred = pred[:, 1:, :] - pred[:, :-1, :]

    # Mask valid observation pairs at both t and t-1, along with sequence padding validity
    mask_pair = (
        mask[:, 1:, :]
        * mask[:, :-1, :]
        * padding_mask[:, 1:].unsqueeze(-1)
        * padding_mask[:, :-1].unsqueeze(-1)
    )

    diff_sq = (delta_pred - delta_target) ** 2
    masked_sq_err = mask_pair * diff_sq

    total_pairs = torch.sum(mask_pair)
    if total_pairs.item() == 0.0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return torch.sum(masked_sq_err) / total_pairs


def temporal_rate_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta_t: torch.Tensor,
    mask: torch.Tensor,
    padding_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Computes Masked Rate-of-Change Loss over irregular continuous-time intervals:
        rate_target = (target_t - target_{t-1}) / max(DeltaT_t, eps)
        rate_pred   = (pred_t - pred_{t-1}) / max(DeltaT_t, eps)
    """
    pred = pred.to(torch.float32)
    target = target.to(torch.float32)
    delta_t = delta_t.to(torch.float32)
    mask = mask.to(torch.float32)
    padding_mask = padding_mask.to(torch.float32)

    dt_clamped = torch.clamp(delta_t[:, 1:], min=eps).unsqueeze(-1)

    rate_target = (target[:, 1:, :] - target[:, :-1, :]) / dt_clamped
    rate_pred = (pred[:, 1:, :] - pred[:, :-1, :]) / dt_clamped

    mask_pair = (
        mask[:, 1:, :]
        * mask[:, :-1, :]
        * padding_mask[:, 1:].unsqueeze(-1)
        * padding_mask[:, :-1].unsqueeze(-1)
    )

    rate_sq_err = mask_pair * ((rate_pred - rate_target) ** 2)
    total_pairs = torch.sum(mask_pair)

    if total_pairs.item() == 0.0:
        return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

    return torch.sum(rate_sq_err) / total_pairs


def gaussian_nll_loss(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    padding_mask: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """
    Computes Masked Gaussian Negative Log-Likelihood (NLL) Loss:
        NLL = 0.5 * [ ((x - mu)^2 / sigma^2) + 2 * log(sigma) + log(2*pi) ]
    """
    import math

    mu = mu.to(torch.float32)
    sigma = torch.clamp(sigma.to(torch.float32), min=eps)
    target = target.to(torch.float32)
    mask = mask.to(torch.float32)
    padding_mask = padding_mask.to(torch.float32)

    effective_mask = mask * padding_mask.unsqueeze(-1)

    nll = 0.5 * (
        ((target - mu) ** 2) / (sigma ** 2)
        + 2.0 * torch.log(sigma)
        + math.log(2.0 * math.pi)
    )

    masked_nll = effective_mask * nll
    total_obs = torch.sum(effective_mask)

    if total_obs.item() == 0.0:
        return torch.tensor(0.0, device=mu.device, dtype=mu.dtype)

    return torch.sum(masked_nll) / total_obs

