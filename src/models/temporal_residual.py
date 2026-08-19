"""
Continuous-Time Ornstein-Uhlenbeck (OU) Observation Residual Process Model.

Implements a continuous-time, irregular-interval-aware OU residual process r(t):
    dr(t) = -lambda * r(t) dt + sigma_diff * dW(t)

For irregular timestamps t_0, t_1, ..., t_{N-1} with time gaps Delta t_i = t_{i+1} - t_i:
    r_0 ~ Normal(0, sigma_stat^2)
    alpha_i = exp(-lambda * Delta t_i)
    r_{i+1} = alpha_i * r_i + sigma_stat * sqrt(1 - alpha_i^2) * epsilon_i
    epsilon_i ~ Normal(0, 1)

where:
    - lambda in R^D is feature-wise mean-reversion rate (per hour)
    - sigma_stat in R^D is feature-wise stationary standard deviation
    - Delta t_i is continuous relative time gap in hours
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class TemporalResidualModel(nn.Module):
    """
    Continuous-Time Ornstein-Uhlenbeck (OU) Residual Process for EHR Vital Signs.
    """

    def __init__(
        self,
        output_dim: int = 5,
        lambda_val: Optional[Union[float, torch.Tensor, List[float]]] = None,
        sigma_val: Optional[Union[float, torch.Tensor, List[float]]] = None,
        scale: float = 1.0,
    ):
        """
        Args:
            output_dim: Feature dimension D (default 5 for vital signs).
            lambda_val: Feature-wise mean-reversion rates (lambda > 0).
            sigma_val: Feature-wise stationary standard deviations (sigma_stat > 0).
            scale: Overall residual scaling factor (default 1.0).
        """
        super().__init__()
        self.output_dim = output_dim
        self.scale = float(scale)

        # Parse lambda_val
        if lambda_val is None:
            init_lambda = torch.ones(output_dim, dtype=torch.float32)
        elif isinstance(lambda_val, (int, float)):
            init_lambda = torch.full((output_dim,), float(lambda_val), dtype=torch.float32)
        elif isinstance(lambda_val, (list, np.ndarray)):
            init_lambda = torch.tensor(lambda_val, dtype=torch.float32)
        elif isinstance(lambda_val, torch.Tensor):
            init_lambda = lambda_val.to(torch.float32).clone()
        else:
            raise ValueError(f"Invalid lambda_val type: {type(lambda_val)}")

        # Parse sigma_val
        if sigma_val is None:
            init_sigma = torch.ones(output_dim, dtype=torch.float32)
        elif isinstance(sigma_val, (int, float)):
            init_sigma = torch.full((output_dim,), float(sigma_val), dtype=torch.float32)
        elif isinstance(sigma_val, (list, np.ndarray)):
            init_sigma = torch.tensor(sigma_val, dtype=torch.float32)
        elif isinstance(sigma_val, torch.Tensor):
            init_sigma = sigma_val.to(torch.float32).clone()
        else:
            raise ValueError(f"Invalid sigma_val type: {type(sigma_val)}")

        # Ensure non-negative parameters
        init_lambda = torch.clamp(init_lambda, min=1e-4)
        init_sigma = torch.clamp(init_sigma, min=0.0)

        # Register buffers so they automatically move to device with model
        self.register_buffer("lambda_rate", init_lambda)
        self.register_buffer("sigma_stat", init_sigma)

    def set_parameters(
        self,
        lambda_val: Union[float, torch.Tensor, List[float], np.ndarray],
        sigma_val: Union[float, torch.Tensor, List[float], np.ndarray],
    ) -> None:
        """
        Updates model feature-wise parameters.
        """
        if isinstance(lambda_val, (int, float)):
            l_tensor = torch.full((self.output_dim,), float(lambda_val), dtype=torch.float32)
        elif isinstance(lambda_val, (list, np.ndarray)):
            l_tensor = torch.tensor(lambda_val, dtype=torch.float32)
        else:
            l_tensor = lambda_val.to(torch.float32)

        if isinstance(sigma_val, (int, float)):
            s_tensor = torch.full((self.output_dim,), float(sigma_val), dtype=torch.float32)
        elif isinstance(sigma_val, (list, np.ndarray)):
            s_tensor = torch.tensor(sigma_val, dtype=torch.float32)
        else:
            s_tensor = sigma_val.to(torch.float32)

        self.lambda_rate.copy_(torch.clamp(l_tensor, min=1e-4).to(self.lambda_rate.device))
        self.sigma_stat.copy_(torch.clamp(s_tensor, min=0.0).to(self.sigma_stat.device))



    def sample_residual(
        self,
        T: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generates continuous-time OU residual trajectory r(t) for timestamps T.

        Args:
            T: Relative timestamps tensor of shape (B, N) or (N,) in hours.
            sequence_lengths: Tensor of valid sequence lengths per batch item (B,) (int64).
            generator: Optional PyTorch random number generator for reproducibility.
            scale: Optional float multiplier overriding self.scale.

        Returns:
            r: Sampled residual tensor of shape (B, N, D) or (N, D).
        """
        is_1d = T.dim() == 1
        if is_1d:
            T_batched = T.unsqueeze(0)  # [1, N]
        else:
            T_batched = T  # [B, N]

        B, N = T_batched.shape
        D = self.output_dim
        eff_scale = float(scale) if scale is not None else self.scale
        eff_sigma = self.sigma_stat * eff_scale  # [D]

        r = torch.zeros((B, N, D), device=T.device, dtype=torch.float32)

        # Initial residual at t0 ~ Normal(0, sigma_stat^2)
        eps_0 = torch.randn((B, D), device=T.device, dtype=torch.float32, generator=generator)
        r[:, 0, :] = eff_sigma * eps_0

        # Exact continuous-time discrete transition step by step
        for i in range(1, N):
            dt = T_batched[:, i] - T_batched[:, i - 1]  # [B]
            dt = torch.clamp(dt, min=0.0)  # Numerical safety against non-monotonic times
            dt_exp = dt.unsqueeze(-1)  # [B, 1]

            # Alpha_i = exp(-lambda * dt_i)
            alpha = torch.exp(-self.lambda_rate * dt_exp)  # [B, D]
            decay_var = torch.clamp(1.0 - alpha**2, min=0.0)
            std_inc = torch.sqrt(decay_var)  # [B, D]

            eps_i = torch.randn((B, D), device=T.device, dtype=torch.float32, generator=generator)
            r[:, i, :] = alpha * r[:, i - 1, :] + eff_sigma * std_inc * eps_i

        # Mask padded elements if sequence_lengths is provided
        if sequence_lengths is not None:
            if sequence_lengths.dim() == 0:
                sequence_lengths = sequence_lengths.unsqueeze(0)
            seq_mask = (
                torch.arange(N, device=T.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
            )  # [B, N]
            r = r * seq_mask.unsqueeze(-1).to(r.dtype)

        if is_1d:
            return r.squeeze(0)  # [N, D]
        return r  # [B, N, D]

    def forward(
        self,
        T: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Alias for sample_residual.
        """
        return self.sample_residual(
            T, sequence_lengths=sequence_lengths, generator=generator, scale=scale
        )

    def apply_residual(
        self,
        mu: torch.Tensor,
        T: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Combines smooth continuous physiological mean trajectory mu with OU residual r(t).

        Args:
            mu: Predicted continuous mean trajectory of shape (B, N, D) or (N, D).
            T: Relative timestamps tensor of shape (B, N) or (N,).
            sequence_lengths: Optional sequence lengths tensor (B,).
            generator: Optional PyTorch random generator.
            scale: Optional float residual scale factor.

        Returns:
            X_synth = mu + r(t)
        """
        r = self.sample_residual(
            T, sequence_lengths=sequence_lengths, generator=generator, scale=scale
        )
        return mu + r
