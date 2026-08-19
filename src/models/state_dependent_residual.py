"""
State-Dependent and Physiologically Constrained Multivariate OU Residual Process.

Extends the continuous-time multivariate OU residual model from Phase 14 with:
1. Feature-specific residual scale vectors S = [S_HR, S_RR, S_SpO2, S_SBP, S_DBP].
2. Smooth, differentiable physiological boundary attenuation functions f(m_d(t))
   to attenuate residual variance near clinical sanity bounds without hard clipping.
3. Decoder uncertainty modulation g(sigma_decoder_d(t)).
4. Optional small learned state-dependent scale network S(z, mu, t).

Main equations:
    X_synth,d(t) = mu_phys,d(t) + S_eff,d(t) * r_OU,d(t)
    S_eff,d(t) = S_feature,d * f(distance_to_bounds_d(t)) * g(sigma_decoder_d(t))
"""

from typing import List, Optional, Tuple, Union, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.multivariate_temporal_residual import MultivariateTemporalResidualModel


# Broad physiological sanity bounds in physical units
SANITY_RANGES = {
    "heart_rate": (20.0, 250.0),
    "respiratory_rate": (2.0, 80.0),
    "spo2": (50.0, 100.0),
    "systolic_bp": (40.0, 250.0),
    "diastolic_bp": (20.0, 150.0),
}

# Standard scaler defaults (MIMIC-III benchmark fit)
DEFAULT_MEAN = [91.055371, 19.742635, 96.831034, 113.036638, 62.33106]
DEFAULT_STD = [18.947034, 5.501502, 3.107534, 21.543833, 13.493742]


def compute_normalized_sanity_bounds(
    mean: Union[List[float], np.ndarray, torch.Tensor] = DEFAULT_MEAN,
    std: Union[List[float], np.ndarray, torch.Tensor] = DEFAULT_STD,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts physical sanity bounds into normalized lower and upper bound tensors.

    Returns:
        lower_norm: Tensor of shape (5,)
        upper_norm: Tensor of shape (5,)
    """
    feature_names = ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp"]
    lower_phys = [SANITY_RANGES[f][0] for f in feature_names]
    upper_phys = [SANITY_RANGES[f][1] for f in feature_names]

    mean_t = torch.tensor(mean, dtype=torch.float32, device=device)
    std_t = torch.tensor(std, dtype=torch.float32, device=device)

    lower_norm = (torch.tensor(lower_phys, dtype=torch.float32, device=device) - mean_t) / std_t
    upper_norm = (torch.tensor(upper_phys, dtype=torch.float32, device=device) - mean_t) / std_t

    return lower_norm, upper_norm


class SmoothBoundaryAttenuation(nn.Module):
    """
    Smooth, Differentiable Physiological Boundary Attenuation Function.

    Calculates safety margin m_d(t) = min(mu_d(t) - lower_d, upper_d - mu_d(t))
    and maps it to an attenuation scale in [0, 1].

    Supported modes:
    - 'sigmoid'  : f(m) = 2 / (1 + exp(-k * max(m, 0))) - 1
    - 'tanh'     : f(m) = tanh(k * max(m, 0))
    - 'exp'      : f(m) = 1 - exp(-max(m, 0) / tau)
    - 'rational' : f(m) = max(m, 0) / (max(m, 0) + tau)
    - 'none'     : f(m) = 1.0 (no boundary attenuation)
    """

    def __init__(
        self,
        mode: str = "tanh",
        k: float = 2.0,
        tau: float = 0.5,
        mean: Union[List[float], np.ndarray, torch.Tensor] = DEFAULT_MEAN,
        std: Union[List[float], np.ndarray, torch.Tensor] = DEFAULT_STD,
    ):
        super().__init__()
        self.mode = mode.lower()
        self.k = float(k)
        self.tau = float(tau)

        lower_norm, upper_norm = compute_normalized_sanity_bounds(mean, std)
        self.register_buffer("lower_norm", lower_norm)
        self.register_buffer("upper_norm", upper_norm)

    def compute_margin(self, mu_norm: torch.Tensor) -> torch.Tensor:
        """
        Computes safety margin m_d(t) for normalized predicted mean tensor mu_norm (..., D).

        Returns:
            margin: Safety margin tensor of shape (..., D)
        """
        dist_lower = mu_norm - self.lower_norm
        dist_upper = self.upper_norm - mu_norm
        return torch.minimum(dist_lower, dist_upper)

    def forward(self, mu_norm: torch.Tensor) -> torch.Tensor:
        """
        Computes smooth boundary attenuation factor in [0, 1] for predicted mean mu_norm.

        Args:
            mu_norm: Predicted mean tensor in normalized space, shape (..., D).

        Returns:
            attenuation: Tensor of shape (..., D) with values in [0, 1].
        """
        if self.mode == "none":
            return torch.ones_like(mu_norm)

        margin = self.compute_margin(mu_norm)
        m_pos = torch.clamp(margin, min=0.0)

        if self.mode == "sigmoid":
            attn = 2.0 / (1.0 + torch.exp(-self.k * m_pos)) - 1.0
        elif self.mode == "tanh":
            attn = torch.tanh(self.k * m_pos)
        elif self.mode == "exp":
            attn = 1.0 - torch.exp(-m_pos / max(self.tau, 1e-4))
        elif self.mode == "rational":
            attn = m_pos / (m_pos + max(self.tau, 1e-4))
        else:
            raise ValueError(f"Unknown attenuation mode: {self.mode}")

        # If already outside physical boundary (margin < 0), decay smoothly toward zero
        negative_mask = (margin < 0.0)
        if negative_mask.any():
            # Exponential decay for points outside physical bounds
            decay = torch.exp(margin)  # margin is negative, so decay < 1
            attn = torch.where(negative_mask, decay * 0.01, attn)

        return torch.clamp(attn, min=0.0, max=1.0)


class LearnedScaleNetwork(nn.Module):
    """
    Optional Small Neural Network to Predict Bounded State-Dependent Scales S(z, mu).

    Architecture:
        [z, mu] -> Linear -> ReLU -> Linear -> Sigmoid -> scaled to [S_min, S_max]
    """

    def __init__(
        self,
        latent_dim: int = 32,
        output_dim: int = 5,
        hidden_dim: int = 32,
        s_min: float = 0.1,
        s_max: float = 1.5,
    ):
        super().__init__()
        self.s_min = float(s_min)
        self.s_max = float(s_max)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid(),
        )

        # Initialize network weights small so initial scaling starts near mid-point
        with torch.no_grad():
            nn.init.normal_(self.net[0].weight, std=0.01)
            nn.init.zeros_(self.net[0].bias)
            nn.init.normal_(self.net[2].weight, std=0.01)
            nn.init.zeros_(self.net[2].bias)

    def forward(self, z: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Latent state tensor (..., latent_dim)
            mu: Predicted mean tensor (..., output_dim)

        Returns:
            scale: State-dependent scale tensor (..., output_dim) in [S_min, S_max].
        """
        inputs = torch.cat([z, mu], dim=-1)
        raw_sig = self.net(inputs)
        return self.s_min + (self.s_max - self.s_min) * raw_sig


class StateDependentMultivariateTemporalResidualModel(nn.Module):
    """
    State-Dependent & Physiologically Constrained Multivariate OU Residual Model.

    Combines:
    - Base Multivariate OU process r_OU(t) (Phase 14 full empirical covariance)
    - Feature-specific scales S_feature
    - Smooth boundary attenuation f(m(t))
    - Decoder variance modulation g(sigma(t))
    - Optional learned scale network
    """

    def __init__(
        self,
        output_dim: int = 5,
        lambda_val: Optional[Union[float, torch.Tensor, List[float], np.ndarray]] = None,
        cov_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        feature_scales: Optional[Union[float, List[float], torch.Tensor, np.ndarray]] = None,
        global_scale: float = 1.0,
        attenuation_mode: str = "tanh",
        attenuation_k: float = 2.0,
        attenuation_tau: float = 0.5,
        use_uncertainty_modulation: bool = False,
        uncertainty_gamma: float = 0.5,
        uncertainty_ref: float = 1.0,
        use_learned_scale_net: bool = False,
        mean: Union[List[float], np.ndarray, torch.Tensor] = DEFAULT_MEAN,
        std: Union[List[float], np.ndarray, torch.Tensor] = DEFAULT_STD,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.global_scale = float(global_scale)
        self.use_uncertainty_modulation = use_uncertainty_modulation
        self.uncertainty_gamma = float(uncertainty_gamma)
        self.uncertainty_ref = float(uncertainty_ref)
        self.use_learned_scale_net = use_learned_scale_net

        # Core multivariate OU residual model
        self.base_ou_model = MultivariateTemporalResidualModel(
            output_dim=output_dim,
            lambda_val=lambda_val,
            cov_matrix=cov_matrix,
            scale=1.0,  # Base scale 1.0, effective scale applied dynamically
        )

        # Feature-specific baseline scales
        if feature_scales is None:
            init_f_scales = torch.ones(output_dim, dtype=torch.float32)
        elif isinstance(feature_scales, (int, float)):
            init_f_scales = torch.full((output_dim,), float(feature_scales), dtype=torch.float32)
        elif isinstance(feature_scales, (list, np.ndarray)):
            init_f_scales = torch.tensor(feature_scales, dtype=torch.float32)
        elif isinstance(feature_scales, torch.Tensor):
            init_f_scales = feature_scales.to(torch.float32).clone()
        else:
            raise ValueError(f"Invalid feature_scales type: {type(feature_scales)}")

        self.register_buffer("feature_scales", init_f_scales)

        # Smooth boundary attenuation module
        self.boundary_attenuation = SmoothBoundaryAttenuation(
            mode=attenuation_mode,
            k=attenuation_k,
            tau=attenuation_tau,
            mean=mean,
            std=std,
        )

        # Optional learned scale network
        if use_learned_scale_net:
            self.learned_scale_net = LearnedScaleNetwork(
                latent_dim=32,
                output_dim=output_dim,
                hidden_dim=32,
            )
        else:
            self.learned_scale_net = None

    def set_feature_scales(
        self,
        feature_scales: Union[float, List[float], torch.Tensor, np.ndarray],
    ) -> None:
        """Updates feature-specific scale vector."""
        if isinstance(feature_scales, (int, float)):
            f_tensor = torch.full((self.output_dim,), float(feature_scales), dtype=torch.float32)
        elif isinstance(feature_scales, (list, np.ndarray)):
            f_tensor = torch.tensor(feature_scales, dtype=torch.float32)
        else:
            f_tensor = feature_scales.to(torch.float32)

        self.feature_scales.copy_(f_tensor.to(self.feature_scales.device))

    def compute_effective_scale(
        self,
        mu_norm: torch.Tensor,
        sigma_decoder: Optional[torch.Tensor] = None,
        z_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes state-dependent effective residual scale S_eff(t) of shape (B, N, D) or (N, D).

        Args:
            mu_norm: Predicted mean trajectory in normalized space, shape (B, N, D) or (N, D).
            sigma_decoder: Optional decoder standard deviation, shape (B, N, D) or (N, D).
            z_latent: Optional continuous latent trajectory, shape (B, N, 32) or (N, 32).

        Returns:
            S_eff: Effective residual scale tensor of same shape as mu_norm.
        """
        device = mu_norm.device
        D = self.output_dim

        # 1. Feature-specific baseline scale vector S_feature (broadcastable)
        S_eff = self.global_scale * self.feature_scales.to(device)  # [D]

        # Expand S_eff to match mu_norm shape
        if mu_norm.dim() == 3:
            B, N, _ = mu_norm.shape
            S_eff = S_eff.unsqueeze(0).unsqueeze(0).expand(B, N, D)
        elif mu_norm.dim() == 2:
            N, _ = mu_norm.shape
            S_eff = S_eff.unsqueeze(0).expand(N, D)

        # 2. Apply smooth boundary attenuation f(margin(mu_norm))
        attn_factor = self.boundary_attenuation(mu_norm)  # (..., D)
        S_eff = S_eff * attn_factor

        # 3. Apply decoder variance modulation g(sigma_decoder) if enabled
        if self.use_uncertainty_modulation and sigma_decoder is not None:
            # g(sigma) = clamp((sigma / sigma_ref)^gamma, min=0.2, max=2.0)
            sig_ratio = sigma_decoder / max(self.uncertainty_ref, 1e-4)
            u_factor = torch.clamp(torch.pow(sig_ratio, self.uncertainty_gamma), min=0.2, max=2.0)
            S_eff = S_eff * u_factor

        # 4. Apply optional learned scale network S(z, mu) if enabled
        if self.use_learned_scale_net and self.learned_scale_net is not None and z_latent is not None:
            learned_factor = self.learned_scale_net(z_latent, mu_norm)
            S_eff = S_eff * learned_factor

        return S_eff

    def sample_state_dependent_residual(
        self,
        T: torch.Tensor,
        mu_norm: torch.Tensor,
        sigma_decoder: Optional[torch.Tensor] = None,
        z_latent: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples multivariate continuous-time OU process and applies state-dependent scaling.

        Args:
            T: Relative timestamps tensor of shape (B, N) or (N,) in hours.
            mu_norm: Predicted continuous mean trajectory of shape (B, N, D) or (N, D).
            sigma_decoder: Optional predicted decoder uncertainty (B, N, D) or (N, D).
            z_latent: Optional latent SDE trajectory (B, N, 32) or (N, 32).
            sequence_lengths: Optional valid sequence lengths per batch item (B,).
            generator: Optional PyTorch random number generator.

        Returns:
            Tuple of:
                - r_scaled: State-dependent scaled residual trajectory (B, N, D) or (N, D).
                - S_eff: Effective scale tensor (B, N, D) or (N, D).
        """
        # Sample base unscaled multivariate OU residual trajectory with unit scale (scale=1.0)
        r_unscaled = self.base_ou_model.sample_residual(
            T=T,
            sequence_lengths=sequence_lengths,
            generator=generator,
            scale=1.0,
        )

        # Compute effective scale S_eff(t) dynamically
        S_eff = self.compute_effective_scale(
            mu_norm=mu_norm,
            sigma_decoder=sigma_decoder,
            z_latent=z_latent,
        )

        # Apply state-dependent scaling: r_scaled(t) = S_eff(t) * r_unscaled(t)
        r_scaled = S_eff * r_unscaled

        # Mask padded steps if sequence_lengths is provided
        if sequence_lengths is not None:
            is_1d = T.dim() == 1
            N = T.shape[0] if is_1d else T.shape[1]
            if sequence_lengths.dim() == 0:
                sequence_lengths = sequence_lengths.unsqueeze(0)
            seq_mask = (
                torch.arange(N, device=T.device).unsqueeze(0) < sequence_lengths.unsqueeze(1)
            )  # [B, N]
            if not is_1d:
                r_scaled = r_scaled * seq_mask.unsqueeze(-1).to(r_scaled.dtype)
                S_eff = S_eff * seq_mask.unsqueeze(-1).to(S_eff.dtype)

        return r_scaled, S_eff

    def forward(
        self,
        T: torch.Tensor,
        mu_norm: torch.Tensor,
        sigma_decoder: Optional[torch.Tensor] = None,
        z_latent: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Alias for sample_state_dependent_residual."""
        return self.sample_state_dependent_residual(
            T=T,
            mu_norm=mu_norm,
            sigma_decoder=sigma_decoder,
            z_latent=z_latent,
            sequence_lengths=sequence_lengths,
            generator=generator,
        )

    def apply_residual(
        self,
        mu_norm: torch.Tensor,
        T: torch.Tensor,
        sigma_decoder: Optional[torch.Tensor] = None,
        z_latent: Optional[torch.Tensor] = None,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Combines continuous mean trajectory mu_norm with state-dependent residual r_scaled(t).

        Returns:
            Tuple of:
                - X_synth_norm: Synthetic trajectory in normalized space (mu_norm + r_scaled).
                - S_eff: Effective scale tensor.
        """
        r_scaled, S_eff = self.sample_state_dependent_residual(
            T=T,
            mu_norm=mu_norm,
            sigma_decoder=sigma_decoder,
            z_latent=z_latent,
            sequence_lengths=sequence_lengths,
            generator=generator,
        )
        return mu_norm + r_scaled, S_eff
