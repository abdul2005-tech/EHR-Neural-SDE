"""
Loss Functions for Latent Neural SDE Training.

Computes negative log-likelihood / reconstruction loss and latent SDE KL divergence.
"""

import torch


def compute_reconstruction_loss(
    x_true: torch.Tensor,
    x_hat: torch.Tensor,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """
    Compute observation space reconstruction error (e.g., MSE or BCE).

    Args:
        x_true: Ground truth EHR observations (batch_size, seq_len, dim)
        x_hat: Reconstructed/decoded observations (batch_size, seq_len, dim)
        mask: Observation mask (batch_size, seq_len)

    Returns:
        Scalar loss tensor
    """
    raise NotImplementedError("Reconstruction loss stub.")


def compute_sde_kl_divergence(
    posterior_drift: torch.Tensor,
    prior_drift: torch.Tensor,
) -> torch.Tensor:
    """
    Compute KL divergence between posterior Latent SDE drift and prior drift process.

    Returns:
        Scalar KL divergence loss tensor
    """
    raise NotImplementedError("SDE KL divergence loss stub.")
