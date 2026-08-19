"""
Probabilistic EHR Observation Decoder Module (Phase 11 Experiment 11D).

Maps continuous stochastic latent SDE trajectory states z(t) in R^32 to an observation
distribution p(X_t | z_t) = Normal(mu_t, sigma_t).

Architecture:
    z_t (R^32) -> Shared MLP Trunk -> h_dec (R^64)
    h_dec -> Mean Head Linear(64 -> 5) -> mu_t (R^5)
    h_dec -> Variance Head Linear(64 -> 5) -> log_sigma_t (R^5)
    sigma_t = Softplus(log_sigma_t) + eps (eps = 1e-4)

Distinguishes decoder output/observation uncertainty (sigma) from latent SDE dynamic diffusion (g(z,t)).
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticEHRDecoder(nn.Module):
    """
    Probabilistic Point-wise EHR Observation Decoder module.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 5,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.eps = eps

        # Shared feature representation trunk
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Mean output head (mu)
        self.mu_head = nn.Linear(hidden_dim, output_dim)

        # Output variance head (log_sigma)
        self.log_sigma_head = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Small initial weight initialization for numerical stability."""
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.mu_head.bias)

        # Initialize log_sigma head so initial sigma is ~1.0 (softplus(0) = ln(2) ~ 0.693 + eps)
        nn.init.normal_(self.log_sigma_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.log_sigma_head.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes latent state z into mean (mu) and standard deviation (sigma) of observation distribution.

        Args:
            z: Latent state tensor of shape (..., 32)

        Returns:
            Tuple of:
                - mu: Predicted mean physiological values (..., 5)
                - sigma: Predicted output standard deviation (..., 5)
        """
        z = z.to(torch.float32)
        h = self.trunk(z)

        mu = self.mu_head(h)
        log_sigma = self.log_sigma_head(h)

        # Softplus parameterization ensures strictly positive standard deviation >= eps
        sigma = F.softplus(log_sigma) + self.eps

        return mu, sigma
