"""
Observation Model Module for Continuous-Time EHR Trajectories (Phase 12 Experiments 12B - 12D).

Explicitly separates:
    1. Continuous latent physiological dynamics (Neural SDE latent state z(t) in R^32)
    2. Observation-level stochastic variability / measurement noise (ObservationModel)

Modes supported:
    - 'independent' (12B)    : Feature-wise homoscedastic noise scale sigma_d >= 1e-4
    - 'heteroscedastic' (12C): State-dependent heteroscedastic noise sigma_d(z_t) >= 1e-4
    - 'correlated' (12D)     : Low-rank covariance matrix Sigma(z_t) = D(z_t) + U(z_t) U(z_t)^T
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ObservationModel(nn.Module):
    """
    Explicit Observation Noise Model mapping latent state z(t) to observation distribution p(X_t | z_t).
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 5,
        mode: str = "heteroscedastic",
        rank: int = 2,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mode = mode.lower()
        self.rank = rank
        self.eps = eps

        # Shared feature trunk for mean prediction mu(t)
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, output_dim)

        if self.mode == "independent":
            # Homoscedastic learnable log_sigma parameter vector [output_dim]
            self.log_sigma_param = nn.Parameter(torch.zeros(output_dim))
        elif self.mode == "heteroscedastic":
            # State-dependent heteroscedastic log_sigma head
            self.sigma_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        elif self.mode == "correlated":
            # Low-rank covariance components: D(z) diagonal and U(z) [output_dim, rank] matrix
            self.diag_head = nn.Linear(hidden_dim, output_dim)
            self.factor_head = nn.Linear(hidden_dim, output_dim * rank)
        else:
            raise ValueError(f"Unknown ObservationModel mode: {mode}. Must be 'independent', 'heteroscedastic', or 'correlated'.")

        self._init_weights()

    def _init_weights(self):
        """Initializes heads with small weights for stable training."""
        nn.init.normal_(self.mu_head.weight, mean=0.0, std=1e-2)
        nn.init.zeros_(self.mu_head.bias)

        if self.mode == "heteroscedastic":
            nn.init.normal_(self.sigma_head[-1].weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.sigma_head[-1].bias)
        elif self.mode == "correlated":
            nn.init.normal_(self.diag_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.diag_head.bias)
            nn.init.normal_(self.factor_head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(self.factor_head.bias)

    def forward(
        self, z: torch.Tensor
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Forward pass decoding latent state z into observation parameters.

        Args:
            z: Latent state tensor of shape (..., 32)

        Returns:
            - If mode in ('independent', 'heteroscedastic'): (mu, sigma)
            - If mode == 'correlated': (mu, D_diag, U_factor)
        """
        z = z.to(torch.float32)
        h = self.trunk(z)
        mu = self.mu_head(h)

        if self.mode == "independent":
            # Broadcast homoscedastic sigma across batch and sequence dims
            sigma = F.softplus(self.log_sigma_param) + self.eps
            # Expand to match shape of mu
            sigma = sigma.expand_as(mu)
            return mu, sigma

        elif self.mode == "heteroscedastic":
            log_sigma = self.sigma_head(h)
            sigma = F.softplus(log_sigma) + self.eps
            return mu, sigma

        elif self.mode == "correlated":
            log_diag = self.diag_head(h)
            d_diag = F.softplus(log_diag) + self.eps
            U_factor = self.factor_head(h).view(*z.shape[:-1], self.output_dim, self.rank)
            return mu, d_diag, U_factor

    def sample(
        self,
        z: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Samples observation X_syn ~ Normal(mu, Sigma) given latent state z.

        Args:
            z: Latent state tensor of shape (..., 32)
            generator: Optional PyTorch random generator for reproducibility

        Returns:
            X_syn: Sampled synthetic observation tensor of shape (..., 5)
        """
        if self.mode in ("independent", "heteroscedastic"):
            mu, sigma = self.forward(z)
            eps = torch.randn(mu.shape, generator=generator, device=z.device, dtype=z.dtype)
            return mu + sigma * eps
        else: # correlated
            mu, d_diag, U_factor = self.forward(z)
            # Sample independent standard normal noise vectors z_d [..., 5] and z_u [..., rank]
            eps_d = torch.randn(mu.shape, generator=generator, device=z.device, dtype=z.dtype)
            eps_u = torch.randn(*z.shape[:-1], self.rank, generator=generator, device=z.device, dtype=z.dtype)

            # X = mu + sqrt(D) * eps_d + U * eps_u
            noise_diag = torch.sqrt(d_diag) * eps_d
            noise_corr = torch.matmul(U_factor, eps_u.unsqueeze(-1)).squeeze(-1)
            return mu + noise_diag + noise_corr
