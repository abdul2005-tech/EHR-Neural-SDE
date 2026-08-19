"""
EHR Observation Decoder Module.

Maps continuous stochastic latent SDE trajectory states z(t) in R^32 back into
physiological observation space x_hat(t) in R^5.

Architecture:
    z_t (R^32) -> Linear(32 -> 64) -> ReLU() -> Linear(64 -> 64) -> ReLU() -> Linear(64 -> 5) -> x_hat_t (R^5)

Design Choices:
- Point-wise MLP decoding operates independently on each latent state along the sequence dimension.
- Operates in NORMALIZED physiological feature space.
- Excludes direct observation mask (M) and time (T, DeltaT) inputs; temporal dynamics and measurement
  patterns influence x_hat through TimeAwareEncoder and NeuralSDE trajectories.
"""

import torch
import torch.nn as nn


class EHRDecoder(nn.Module):
    """
    Point-wise EHR Observation Decoder module.

    Decodes continuous latent SDE state trajectories z(t) back into predicted physiological values x_hat.
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, output_dim: int = 5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodes latent trajectory states z into physiological prediction space x_hat.

        Args:
            z: Continuous latent state tensor of shape (N, 32) or (B, N, 32)

        Returns:
            Decoded physiological values x_hat of shape (N, 5) or (B, N, 5)
        """
        z = z.to(torch.float32)
        return self.net(z)


# Backward compatibility alias
ObservationDecoder = EHRDecoder
