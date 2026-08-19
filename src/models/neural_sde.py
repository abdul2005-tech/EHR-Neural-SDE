"""
Latent Neural Stochastic Differential Equation (Neural SDE) Dynamics & Euler-Maruyama Integrator.

Defines continuous-time stochastic latent dynamics:
    dz(t) = f_theta(z(t), t) dt + g_theta(z(t), t) dW(t)

where:
    f_theta(z, t): Learned drift network governing deterministic trajectory direction
    g_theta(z, t): Learned diffusion network governing stochastic patient-to-patient variation (diagonal)
    dW(t)       : Brownian motion process (scaled by sqrt(dt))

Features:
- Subdivides large time gaps (up to 43.8 hours) into internal substeps <= max_step_size (0.1 hours)
- Preserves exact requested irregular observation timestamps
- Fully differentiable via PyTorch autograd with respect to drift, diffusion, and initial state z0
"""

import math
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DriftNetwork(nn.Module):
    """
    Learned drift function f_theta(z, t).

    Maps [z, t] (R^33) -> drift vector (R^32).
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent state tensor of shape (B, 32)
            t: Time tensor of shape (B, 1) or (B,)

        Returns:
            Drift tensor f(z, t) of shape (B, 32)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t = t.to(z.dtype)
        zt = torch.cat([z, t], dim=-1)
        return self.net(zt)


class DiffusionNetwork(nn.Module):
    """
    Learned diagonal diffusion function g_theta(z, t).

    Maps [z, t] (R^33) -> positive diffusion magnitude vector (R^32).
    """

    def __init__(self, latent_dim: int = 32, hidden_dim: int = 64, eps: float = 1e-5):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.eps = eps

        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent state tensor of shape (B, 32)
            t: Time tensor of shape (B, 1) or (B,)

        Returns:
            Non-negative diagonal diffusion magnitude tensor g(z, t) of shape (B, 32)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        t = t.to(z.dtype)
        zt = torch.cat([z, t], dim=-1)
        out = self.net(zt)
        return torch.sigmoid(out) + self.eps


class NeuralSDE(nn.Module):
    """
    Neural Stochastic Differential Equation model and Euler-Maruyama continuous-time solver.
    """

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        max_step_size: float = 0.1,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.max_step_size = max_step_size
        self.eps = eps

        self.drift_net = DriftNetwork(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.diffusion_net = DiffusionNetwork(latent_dim=latent_dim, hidden_dim=hidden_dim, eps=eps)

    def f(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Drift evaluation f_theta(z, t)."""
        return self.drift_net(z, t)

    def g(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Diffusion evaluation g_theta(z, t)."""
        return self.diffusion_net(z, t)

    def euler_maruyama_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        dt: Union[float, torch.Tensor],
        generator: Optional[torch.Generator] = None,
        enable_noise: bool = True,
    ) -> torch.Tensor:
        """
        Performs a single Euler-Maruyama step:
            z_next = z + f(z, t) * dt + g(z, t) * sqrt(dt) * noise

        Args:
            z: Latent state tensor (B, 32)
            t: Time tensor (B, 1) or (B,)
            dt: Time step size (float or tensor (B, 1))
            generator: Optional torch.Generator for deterministic noise sampling
            enable_noise: If False, sets noise to zero (deterministic drift-only ODE integration)

        Returns:
            Next latent state tensor z_next of shape (B, 32)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        if isinstance(dt, (float, int)):
            dt_tensor = torch.tensor(dt, device=z.device, dtype=z.dtype).expand(z.shape[0], 1)
        else:
            dt_tensor = dt.to(z.dtype)
            if dt_tensor.dim() == 0:
                dt_tensor = dt_tensor.unsqueeze(0).expand(z.shape[0], 1)
            elif dt_tensor.dim() == 1:
                dt_tensor = dt_tensor.unsqueeze(-1)

        drift = self.f(t, z)         # [B, 32]
        diffusion = self.g(t, z)     # [B, 32]

        if enable_noise:
            if generator is not None:
                noise = torch.randn(z.shape, generator=generator, device=z.device, dtype=z.dtype)
            else:
                noise = torch.randn_like(z)
        else:
            noise = torch.zeros_like(z)

        # dz = f(z,t)*dt + g(z,t)*sqrt(dt)*noise
        dt_clamp = torch.clamp(dt_tensor, min=1e-8)
        z_next = z + drift * dt_tensor + diffusion * torch.sqrt(dt_clamp) * noise
        return z_next

    def integrate(
        self,
        z0: torch.Tensor,
        times: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        enable_noise: bool = True,
    ) -> torch.Tensor:
        """
        Integrates the Neural SDE from initial state z0 across requested observation timestamps times.

        Subdivides intervals dt > max_step_size into internal substeps to guarantee numerical stability.

        Args:
            z0: Initial latent state tensor of shape (B, 32) or (32,)
            times: Observation timestamps tensor of shape (N,) or (B, N)
            generator: Optional PyTorch random number generator for reproducible sampling
            enable_noise: If False, disables stochastic Brownian noise (ODE trajectory)

        Returns:
            z_trajectory: Integrated latent trajectory of shape (B, N, 32)
        """
        if z0.dim() == 1:
            z0 = z0.unsqueeze(0)

        B, D = z0.shape

        if times.dim() == 1:
            times_b = times.unsqueeze(0).repeat(B, 1)
        else:
            times_b = times

        B_t, N = times_b.shape
        assert B_t == B, f"Batch dimension mismatch: z0 has B={B}, times has B={B_t}"

        z_steps = [z0]
        z_curr = z0

        for i in range(1, N):
            t_prev = times_b[:, i - 1]  # [B]
            t_curr = times_b[:, i]      # [B]

            dt_full = t_curr - t_prev    # [B]
            max_dt_val = float(torch.max(dt_full).item())

            if max_dt_val <= 0.0:
                z_steps.append(z_curr)
                continue

            # Calculate internal substeps to respect max_step_size
            n_substeps = int(math.ceil(max_dt_val / self.max_step_size))
            n_substeps = max(n_substeps, 1)

            dt_sub = dt_full / n_substeps  # [B]
            t_sub_curr = t_prev.clone()

            for step in range(n_substeps):
                z_curr = self.euler_maruyama_step(
                    z=z_curr,
                    t=t_sub_curr,
                    dt=dt_sub,
                    generator=generator,
                    enable_noise=enable_noise,
                )
                t_sub_curr = t_sub_curr + dt_sub

            z_steps.append(z_curr)

        z_trajectory = torch.stack(z_steps, dim=1)  # [B, N, 32]
        return z_trajectory


# Backward compatibility alias for early interface stubs
LatentNeuralSDE = NeuralSDE

