"""
Latent Projection Module.

Projects point-wise hidden representation h_t in R^64 into latent state space z_t in R^32.

Architecture:
    h_t (R^64) -> Linear(64 -> 64) -> ReLU() -> Linear(64 -> 32) -> z_t (R^32)
"""

import torch
import torch.nn as nn


class LatentProjection(nn.Module):
    """
    Projects encoder hidden representation h_t into continuous latent state space z_t.
    """

    def __init__(self, hidden_dim: int = 64, latent_dim: int = 32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Encoder hidden state tensor of shape (N, 64) or (B, N, 64)

        Returns:
            Latent state representation z_t of shape (N, 32) or (B, N, 32)
        """
        h = h.to(torch.float32)
        return self.proj(h)
