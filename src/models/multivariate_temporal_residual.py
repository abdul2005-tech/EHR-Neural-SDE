"""
Multivariate Continuous-Time Ornstein-Uhlenbeck (OU) Observation Residual Process.

Implements a continuous-time, irregular-interval-aware multivariate OU residual process r(t):
    dr(t) = A r(t) dt + L dW(t)

where:
    - r(t) in R^D is the D-dimensional residual vector (default D=5 for vital signs).
    - A = diag(-lambda_1, ..., -lambda_D) is the diagonal mean-reversion rate matrix (lambda_d > 0).
    - Q = L L^T is the cross-feature diffusion covariance matrix.
    - Sigma_residual in R^{D x D} is the stationary residual covariance matrix.

For irregular timestamps t_0, t_1, ..., t_{N-1} with time gaps Delta t_i = t_{i+1} - t_i >= 0:
    r_0 ~ Normal(0, scale^2 * Sigma_residual)
    alpha_i = exp(-lambda * Delta t_i)  # [B, D]
    r_{i+1} = alpha_i * r_i + epsilon_i
    epsilon_i ~ Normal(0, Q_Delta(Delta t_i))

where the exact step transition covariance matrix Q_Delta(Delta t) is:
    (Q_Delta)_{jk} = (Sigma_residual)_{jk} * (1 - exp(-(lambda_j + lambda_k) * Delta t)) * scale^2
                   = (Sigma_residual)_{jk} * (1 - alpha_j * alpha_k) * scale^2
"""

from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn


class MultivariateTemporalResidualModel(nn.Module):
    """
    Multivariate Continuous-Time Ornstein-Uhlenbeck (OU) Residual Process
    with Cross-Feature Residual Covariance Structure.
    """

    def __init__(
        self,
        output_dim: int = 5,
        lambda_val: Optional[Union[float, torch.Tensor, List[float], np.ndarray]] = None,
        sigma_val: Optional[Union[float, torch.Tensor, List[float], np.ndarray]] = None,
        cov_matrix: Optional[Union[torch.Tensor, np.ndarray]] = None,
        scale: float = 1.0,
        jitter: float = 1e-6,
    ):
        """
        Args:
            output_dim: Feature dimension D (default 5 for vital signs).
            lambda_val: Feature-wise mean-reversion rates (lambda > 0).
            sigma_val: Optional feature-wise stationary std dev (used if cov_matrix is None).
            cov_matrix: Stationary residual covariance matrix Sigma_residual of shape (D, D).
            scale: Overall residual scaling factor (default 1.0).
            jitter: Small diagonal float added to covariance matrices for Cholesky stability.
        """
        super().__init__()
        self.output_dim = output_dim
        self.scale = float(scale)
        self.jitter = float(jitter)

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

        # Ensure positive lambda rates
        init_lambda = torch.clamp(init_lambda, min=1e-4)

        # Parse stationary covariance matrix Sigma_residual
        if cov_matrix is not None:
            if isinstance(cov_matrix, np.ndarray):
                init_cov = torch.tensor(cov_matrix, dtype=torch.float32)
            elif isinstance(cov_matrix, torch.Tensor):
                init_cov = cov_matrix.to(torch.float32).clone()
            else:
                raise ValueError(f"Invalid cov_matrix type: {type(cov_matrix)}")
            
            if init_cov.shape != (output_dim, output_dim):
                raise ValueError(f"cov_matrix shape must be ({output_dim}, {output_dim}), got {init_cov.shape}")
        else:
            # Fallback to diagonal covariance based on sigma_val
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
            
            init_sigma = torch.clamp(init_sigma, min=0.0)
            init_cov = torch.diag(init_sigma ** 2)

        # Ensure symmetry
        init_cov = 0.5 * (init_cov + init_cov.T)

        self.register_buffer("lambda_rate", init_lambda)
        self.register_buffer("cov_matrix", init_cov)

    @property
    def sigma_stat(self) -> torch.Tensor:
        """Feature-wise stationary standard deviation sqrt(diag(Sigma))."""
        return torch.sqrt(torch.clamp(torch.diag(self.cov_matrix), min=0.0))

    def set_parameters(
        self,
        lambda_val: Union[float, torch.Tensor, List[float], np.ndarray],
        cov_matrix: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """Updates model mean-reversion rates and covariance matrix."""
        if isinstance(lambda_val, (int, float)):
            l_tensor = torch.full((self.output_dim,), float(lambda_val), dtype=torch.float32)
        elif isinstance(lambda_val, (list, np.ndarray)):
            l_tensor = torch.tensor(lambda_val, dtype=torch.float32)
        else:
            l_tensor = lambda_val.to(torch.float32)

        if isinstance(cov_matrix, np.ndarray):
            c_tensor = torch.tensor(cov_matrix, dtype=torch.float32)
        else:
            c_tensor = cov_matrix.to(torch.float32)

        c_tensor = 0.5 * (c_tensor + c_tensor.T)

        self.lambda_rate.copy_(torch.clamp(l_tensor, min=1e-4).to(self.lambda_rate.device))
        self.cov_matrix.copy_(c_tensor.to(self.cov_matrix.device))

    def _sample_multivariate_normal(
        self,
        covariance: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Samples multivariate normal given covariance matrix of shape (B, D, D) or (D, D).
        Returns samples of shape (B, D) or (D,).
        """
        device = covariance.device
        is_batched = covariance.dim() == 3

        if is_batched:
            B, D, _ = covariance.shape
            # Add diagonal jitter for numerical stability
            eye = torch.eye(D, device=device, dtype=covariance.dtype).unsqueeze(0)
            cov_stable = covariance + self.jitter * eye
            try:
                L = torch.linalg.cholesky(cov_stable)
            except RuntimeError:
                # Fallback to SVD / Eigen decomposition if Cholesky fails due to semi-definiteness
                L = self._stable_cholesky_fallback(cov_stable)
            
            z = torch.randn((B, D, 1), device=device, dtype=covariance.dtype, generator=generator)
            samples = (L @ z).squeeze(-1)  # [B, D]
        else:
            D = covariance.shape[0]
            eye = torch.eye(D, device=device, dtype=covariance.dtype)
            cov_stable = covariance + self.jitter * eye
            try:
                L = torch.linalg.cholesky(cov_stable)
            except RuntimeError:
                L = self._stable_cholesky_fallback(cov_stable.unsqueeze(0)).squeeze(0)
            
            z = torch.randn((D, 1), device=device, dtype=covariance.dtype, generator=generator)
            samples = (L @ z).squeeze(-1)  # [D]

        return samples

    def _stable_cholesky_fallback(self, cov_batch: torch.Tensor) -> torch.Tensor:
        """Fallback matrix square root via eigendecomposition when Cholesky fails."""
        # cov_batch shape: [B, D, D]
        eigenvalues, eigenvectors = torch.linalg.eigh(cov_batch)
        eigenvalues = torch.clamp(eigenvalues, min=1e-8)
        sqrt_eigenvalues = torch.sqrt(eigenvalues)
        # L = V * diag(sqrt(evals))
        L = eigenvectors @ torch.diag_embed(sqrt_eigenvalues)
        return L

    def sample_residual(
        self,
        T: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
        scale: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generates multivariate continuous-time OU residual trajectory r(t) for timestamps T.

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

        r = torch.zeros((B, N, D), device=T.device, dtype=torch.float32)

        # 1. Initial residual at t0 ~ Normal(0, eff_scale^2 * Sigma_residual)
        cov_t0 = self.cov_matrix.unsqueeze(0).expand(B, D, D)  # [B, D, D]
        r[:, 0, :] = eff_scale * self._sample_multivariate_normal(cov_t0, generator=generator)

        # 2. Sequential transition step by step over irregular gaps dt
        for i in range(1, N):
            dt = T_batched[:, i] - T_batched[:, i - 1]  # [B]
            dt = torch.clamp(dt, min=0.0)  # Numerical safety for non-monotonic time
            dt_exp = dt.unsqueeze(-1)  # [B, 1]

            # Mean reversion decay alpha_i = exp(-lambda * dt_i)
            alpha = torch.exp(-self.lambda_rate * dt_exp)  # [B, D]

            # Construct exact step transition covariance Q_Delta(Delta t)
            # (Q_Delta)_{jk} = (Sigma_residual)_{jk} * (1 - alpha_j * alpha_k)
            alpha_outer = alpha.unsqueeze(2) * alpha.unsqueeze(1)  # [B, D, D]
            decay_mat = torch.clamp(1.0 - alpha_outer, min=0.0)  # [B, D, D]
            cov_delta = self.cov_matrix.unsqueeze(0) * decay_mat  # [B, D, D]

            # Sample step transition error
            eps_raw = self._sample_multivariate_normal(cov_delta, generator=generator)  # [B, D]
            # Zero out eps for zero time gap steps (dt == 0)
            dt_mask = (dt > 1e-8).unsqueeze(-1).to(r.dtype)  # [B, 1]
            eps_i = eff_scale * eps_raw * dt_mask

            # Apply state update: r_i = alpha * r_{i-1} + eps_i
            r[:, i, :] = alpha * r[:, i - 1, :] + eps_i

        # Mask padded sequence steps if sequence_lengths is provided
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
        """Alias for sample_residual."""
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
        Combines smooth continuous physiological mean trajectory mu with multivariate OU residual r(t).

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


def create_shrinkage_covariance(
    empirical_cov: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Computes shrinkage covariance matrix:
        Sigma_shrink = alpha * Sigma_empirical + (1 - alpha) * diag(Sigma_empirical)
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    diag_cov = np.diag(np.diag(empirical_cov))
    return alpha * empirical_cov + (1.0 - alpha) * diag_cov


def create_low_rank_covariance(
    empirical_cov: np.ndarray,
    rank: int,
) -> np.ndarray:
    """
    Computes low-rank approximation covariance matrix:
        Sigma_lowrank = D + U U^T
    where D is diagonal noise and U is top-r eigenvectors scaled by sqrt(eigenvalues).
    """
    D_dim = empirical_cov.shape[0]
    rank = int(np.clip(rank, 1, D_dim))

    # Eigendecomposition of symmetric covariance matrix
    evals, evecs = np.linalg.eigh(empirical_cov)
    # Sort in descending order
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Top-rank components
    top_evals = np.maximum(evals[:rank], 0.0)
    top_evecs = evecs[:, :rank]
    U = top_evecs * np.sqrt(top_evals)[np.newaxis, :]  # [D, rank]

    # Low-rank covariance component U @ U^T
    uu_t = U @ U.T

    # Residual diagonal D = max(diag(empirical_cov - U U^T), 1e-4)
    res_diag = np.maximum(np.diag(empirical_cov) - np.diag(uu_t), 1e-4)
    D_mat = np.diag(res_diag)

    return D_mat + uu_t
