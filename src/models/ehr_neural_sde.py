"""
End-to-End Continuous-Time EHR Neural SDE Pipeline Model.

Integrates all four architectural stages:
    1. TimeAwareEncoder  : Irregular EHR observation tuple (X_0, M_0, T_0, DeltaT_0) -> h_0 (R^64)
    2. LatentProjection  : Encoder representation h_0 -> initial latent state z_0 = z(T_0) (R^32)
    3. NeuralSDE         : Latent state z_0 continuous evolution across irregular times T -> z(t) (R^32)
    4. EHRDecoder        : Continuous latent state trajectory z(t) -> predicted vital signs X_hat (R^5)

Features length-aware sample integration to ensure padded sequence steps do not pollute SDE dynamics.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.time_aware_encoder import TimeAwareEncoder
from src.models.latent_projection import LatentProjection
from src.models.neural_sde import NeuralSDE
from src.models.decoder import EHRDecoder
from src.models.probabilistic_decoder import ProbabilisticEHRDecoder
from src.models.observation_model import ObservationModel


class EHRNeuralSDE(nn.Module):
    """
    Unified EHR Neural Stochastic Differential Equation (Neural SDE) Model.
    """

    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        time_embed_dim: int = 16,
        max_step_size: float = 0.1,
        eps: float = 1e-5,
        use_probabilistic_decoder: bool = False,
        observation_mode: Optional[str] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.time_embed_dim = time_embed_dim
        self.max_step_size = max_step_size
        self.use_probabilistic_decoder = use_probabilistic_decoder
        self.observation_mode = observation_mode

        self.encoder = TimeAwareEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
        )
        self.latent_projection = LatentProjection(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.sde = NeuralSDE(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            max_step_size=max_step_size,
            eps=eps,
        )
        if observation_mode is not None:
            self.observation_model = ObservationModel(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
                mode=observation_mode,
            )
            self.decoder = None
        elif use_probabilistic_decoder:
            self.decoder = ProbabilisticEHRDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
            )
            self.observation_model = None
        else:
            self.decoder = EHRDecoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                output_dim=input_dim,
            )
            self.observation_model = None
        self._init_weights()

    def _init_weights(self):
        """Initializes network weights for stable continuous-time integration."""
        nn.init.normal_(self.sde.drift_net.net[-1].weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.sde.drift_net.net[-1].bias)
        if self.decoder is not None and not self.use_probabilistic_decoder:
            nn.init.normal_(self.decoder.net[-1].weight, mean=0.0, std=1e-2)
            nn.init.zeros_(self.decoder.net[-1].bias)

    def get_initial_state(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        T: torch.Tensor,
        DeltaT: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extracts initial latent state z0 = z(T0) from the FIRST clinical observation tuple.

        Args:
            X: Observation tensor of shape (B, N, 5) or (B, 1, 5)
            M: Observation mask tensor of shape (B, N, 5) or (B, 1, 5)
            T: Relative time tensor of shape (B, N) or (B, 1)
            DeltaT: Elapsed gap tensor of shape (B, N) or (B, 1)

        Returns:
            z0: Initial latent state tensor of shape (B, 32)
        """
        # Ensure 3D shapes for sequence step 0
        X_0 = X[:, 0:1, :]          # [B, 1, 5]
        M_0 = M[:, 0:1, :]          # [B, 1, 5]
        T_0 = T[:, 0:1]             # [B, 1]
        DeltaT_0 = DeltaT[:, 0:1]   # [B, 1]

        h_0 = self.encoder(X_0, M_0, T_0, DeltaT_0)  # [B, 1, 64]
        h_0 = h_0.squeeze(1)                         # [B, 64]
        z_0 = self.latent_projection(h_0)            # [B, 32]
        return z_0

    def forward(
        self,
        X: torch.Tensor,
        M: torch.Tensor,
        T: torch.Tensor,
        DeltaT: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        enable_noise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes complete forward pass:
            Observation (X0, M0, T0, DeltaT0) -> z0 -> NeuralSDE.integrate(z0, T) -> z(t) -> EHRDecoder -> X_hat

        Args:
            X: Observation tensor [B, N, 5]
            M: Observation mask tensor [B, N, 5]
            T: Relative observation timestamps [B, N]
            DeltaT: Time gap tensor [B, N]
            sequence_lengths: Actual unpadded trajectory lengths tensor [B] (int64)
            padding_mask: Binary padding mask tensor [B, N] (1=valid, 0=padded)
            generator: Optional PyTorch random number generator for reproducible sampling
            enable_noise: If False, disables Brownian noise (deterministic ODE integration)

        Returns:
            Tuple of:
                - X_hat: Reconstructed physiological predictions of shape (B, N, 5)
                - z_trajectory: Integrated latent trajectory of shape (B, N, 32)
        """
        # Handle unbatched inputs [N, 5] / [N]
        if X.dim() == 2:
            X = X.unsqueeze(0)
            M = M.unsqueeze(0)
            T = T.unsqueeze(0)
            DeltaT = DeltaT.unsqueeze(0)
            if sequence_lengths is not None and sequence_lengths.dim() == 0:
                sequence_lengths = sequence_lengths.unsqueeze(0)
            if padding_mask is not None and padding_mask.dim() == 1:
                padding_mask = padding_mask.unsqueeze(0)

        B, N, D = X.shape

        # 1. Compute initial latent state z0 = z(T0) for the batch [B, 32]
        z_0 = self.get_initial_state(X, M, T, DeltaT)

        # 2. Continuous SDE integration across actual timestamps per sample to prevent padding pollution
        if sequence_lengths is not None:
            z_traj_list = []
            for i in range(B):
                N_i = int(sequence_lengths[i].item())
                z0_i = z_0[i : i + 1]  # [1, 32]
                T_i = T[i : i + 1, :N_i]  # [1, N_i]
                z_traj_i = self.sde.integrate(
                    z0=z0_i,
                    times=T_i,
                    generator=generator,
                    enable_noise=enable_noise,
                )  # [1, N_i, 32]
                if N_i < N:
                    pad_len = N - N_i
                    z_traj_i = F.pad(z_traj_i, (0, 0, 0, pad_len))  # [1, N, 32]
                z_traj_list.append(z_traj_i)
            z_trajectory = torch.cat(z_traj_list, dim=0)  # [B, N, 32]
        else:
            z_trajectory = self.sde.integrate(
                z0=z_0,
                times=T,
                generator=generator,
                enable_noise=enable_noise,
            )

        # 4. Decode latent state trajectory -> observation distribution space
        if self.observation_model is not None:
            obs_out = self.observation_model(z_trajectory)
            return obs_out, z_trajectory
        elif self.use_probabilistic_decoder:
            mu, sigma = self.decoder(z_trajectory)
            return mu, z_trajectory, sigma
        else:
            X_hat = self.decoder(z_trajectory)
            return X_hat, z_trajectory

    def get_mean_diffusion_magnitude(self, z_trajectory: torch.Tensor, T: torch.Tensor) -> float:
        """
        Computes the mean diffusion magnitude g_theta(z, t) across the trajectory.
        """
        B, N, D = z_trajectory.shape
        t_flat = T.reshape(-1, 1)
        z_flat = z_trajectory.reshape(-1, D)

        with torch.no_grad():
            diff = self.sde.g(t_flat, z_flat)
            return float(diff.mean().item())
