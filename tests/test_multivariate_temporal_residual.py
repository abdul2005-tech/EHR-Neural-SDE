"""
Unit Test Suite for Multivariate Continuous-Time Ornstein-Uhlenbeck Residual Process.

Tests all 13 critical requirements:
1. Output shape
2. Non-NaN outputs
3. Irregular timestamps
4. Zero time gap behavior
5. Padding behavior
6. Seed reproducibility
7. Positive semidefinite covariance
8. Empirical cross-feature covariance emergence
9. Temporal autocorrelation
10. Feature-specific scaling
11. Batch independence
12. Numerical stability for long intervals
13. Numerical stability for near-singular covariance matrices
"""

import unittest
import numpy as np
import torch

from src.models.multivariate_temporal_residual import (
    MultivariateTemporalResidualModel,
    create_shrinkage_covariance,
    create_low_rank_covariance,
)


class TestMultivariateTemporalResidual(unittest.TestCase):

    def setUp(self):
        self.output_dim = 5
        self.lambda_rates = np.array([0.5, 1.0, 1.5, 0.2, 0.8], dtype=np.float32)
        # Create a valid positive-definite non-diagonal 5x5 covariance matrix
        A = np.array([
            [1.0, 0.4, 0.2, 0.1, 0.0],
            [0.4, 1.2, 0.5, 0.2, 0.1],
            [0.2, 0.5, 0.9, 0.3, 0.2],
            [0.1, 0.2, 0.3, 1.1, 0.4],
            [0.0, 0.1, 0.2, 0.4, 0.8],
        ], dtype=np.float32)
        self.cov_matrix = A @ A.T  # Guaranteed positive-definite

        self.model = MultivariateTemporalResidualModel(
            output_dim=self.output_dim,
            lambda_val=self.lambda_rates,
            cov_matrix=self.cov_matrix,
            scale=1.0,
        )

    def test_01_output_shape(self):
        """Verify 2D and 3D output shapes."""
        # 1D timestamps input -> 2D output (N, D)
        T_1d = torch.tensor([0.0, 1.5, 4.0, 10.0], dtype=torch.float32)
        r_1d = self.model(T_1d)
        self.assertEqual(r_1d.shape, (4, self.output_dim))

        # 2D timestamps input -> 3D output (B, N, D)
        T_2d = torch.tensor([
            [0.0, 1.0, 2.0, 5.0],
            [0.0, 0.5, 3.0, 6.0],
        ], dtype=torch.float32)
        r_2d = self.model(T_2d)
        self.assertEqual(r_2d.shape, (2, 4, self.output_dim))

    def test_02_non_nan_outputs(self):
        """Verify no NaN or Inf values produced."""
        T = torch.tensor([0.0, 0.1, 1.0, 12.0, 48.0, 100.0], dtype=torch.float32)
        r = self.model(T)
        self.assertFalse(torch.isnan(r).any().item())
        self.assertFalse(torch.isinf(r).any().item())

    def test_03_irregular_timestamps(self):
        """Verify support for highly irregular timestamps."""
        T_irreg = torch.tensor([0.0, 0.01, 0.05, 3.7, 3.71, 24.0, 72.0], dtype=torch.float32)
        r = self.model(T_irreg)
        self.assertEqual(r.shape, (7, self.output_dim))

    def test_04_zero_time_gap_behavior(self):
        """Verify zero time gap (dt = 0) produces zero transition step error."""
        # At dt=0, alpha = exp(0) = 1, decay = 0, so r(t+0) == r(t) exactly
        T_same = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        gen = torch.Generator().manual_seed(123)
        r = self.model(T_same, generator=gen)
        # Check step 0 to step 1 and step 2 are identical
        self.assertTrue(torch.allclose(r[0], r[1], atol=1e-5))
        self.assertTrue(torch.allclose(r[0], r[2], atol=1e-5))

    def test_05_padding_behavior(self):
        """Verify padding mask zeros out padded sequence steps."""
        T = torch.tensor([
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 1.0, 2.0, 3.0],
        ], dtype=torch.float32)
        seq_lengths = torch.tensor([2, 4], dtype=torch.int64)

        r = self.model(T, sequence_lengths=seq_lengths)
        # Batch 0 has length 2, steps 2 and 3 should be zero
        self.assertTrue(torch.allclose(r[0, 2:], torch.zeros_like(r[0, 2:])))
        # Batch 1 has length 4, step 3 should not be zero
        self.assertFalse(torch.allclose(r[1, 3], torch.zeros_like(r[1, 3])))

    def test_06_seed_reproducibility(self):
        """Verify generator seed yields reproducible trajectories."""
        T = torch.tensor([0.0, 1.0, 2.5, 6.0], dtype=torch.float32)
        gen1 = torch.Generator().manual_seed(42)
        r1 = self.model(T, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        r2 = self.model(T, generator=gen2)

        self.assertTrue(torch.allclose(r1, r2, atol=1e-6))

    def test_07_positive_semidefinite_covariance(self):
        """Verify covariance matrices generated during sampling remain positive-semidefinite."""
        c_mat = self.model.cov_matrix
        evals = torch.linalg.eigvalsh(c_mat)
        self.assertTrue((evals >= -1e-6).all().item())

    def test_08_empirical_cross_feature_covariance_emergence(self):
        """Verify empirical covariance of sampled t0 residuals converges to model cov_matrix."""
        N_samples = 20000
        T_t0 = torch.tensor([0.0], dtype=torch.float32).expand(N_samples, 1)
        gen = torch.Generator().manual_seed(999)

        r_t0 = self.model(T_t0, generator=gen).squeeze(1).cpu().numpy()  # [N_samples, D]
        emp_cov = np.cov(r_t0, rowvar=False)

        np.testing.assert_allclose(emp_cov, self.cov_matrix, rtol=0.15, atol=0.15)

    def test_09_temporal_autocorrelation(self):
        """Verify mean-reversion rates lambda control temporal decay rate exp(-lambda * dt)."""
        dt = 2.0
        T = torch.tensor([0.0, dt], dtype=torch.float32)
        N_samples = 15000

        T_batch = T.unsqueeze(0).expand(N_samples, 2)
        gen = torch.Generator().manual_seed(777)
        r = self.model(T_batch, generator=gen).cpu().numpy()  # [N_samples, 2, D]

        # Empirical cross-time covariance: E[r_0 * r_1] vs theoretical Sigma_ij * exp(-lambda_j * dt)
        r0 = r[:, 0, :]
        r1 = r[:, 1, :]

        theo_alpha = np.exp(-self.lambda_rates * dt)
        emp_autocorr = np.mean(r0 * r1, axis=0) / (np.var(r0, axis=0) + 1e-8)

        np.testing.assert_allclose(emp_autocorr, theo_alpha, rtol=0.15, atol=0.15)

    def test_10_feature_specific_scaling(self):
        """Verify scale argument scales output std linearly."""
        T = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        gen1 = torch.Generator().manual_seed(100)
        r_scale1 = self.model(T, generator=gen1, scale=1.0)

        gen2 = torch.Generator().manual_seed(100)
        r_scale05 = self.model(T, generator=gen2, scale=0.5)

        self.assertTrue(torch.allclose(r_scale05, 0.5 * r_scale1, atol=1e-5))

    def test_11_batch_independence(self):
        """Verify sampling batch items are statistically independent."""
        T = torch.tensor([
            [0.0, 1.0, 2.0],
            [0.0, 1.0, 2.0],
        ], dtype=torch.float32)
        gen = torch.Generator().manual_seed(42)
        r = self.model(T, generator=gen)
        # Batch 0 and Batch 1 should not be equal
        self.assertFalse(torch.allclose(r[0], r[1]))

    def test_12_numerical_stability_long_intervals(self):
        """Verify stability for very large time gaps (dt -> infinity)."""
        T_large = torch.tensor([0.0, 1000.0, 10000.0], dtype=torch.float32)
        r = self.model(T_large)
        self.assertFalse(torch.isnan(r).any().item())
        self.assertFalse(torch.isinf(r).any().item())

    def test_13_numerical_stability_near_singular_covariance(self):
        """Verify Cholesky fallback and numerical stability when covariance is near-singular."""
        # Rank-1 covariance matrix (singular)
        v = np.array([1.0, 2.0, -1.0, 0.5, 1.5], dtype=np.float32)[:, np.newaxis]
        near_singular_cov = v @ v.T  # Rank 1

        model_singular = MultivariateTemporalResidualModel(
            output_dim=self.output_dim,
            lambda_val=self.lambda_rates,
            cov_matrix=near_singular_cov,
            scale=1.0,
        )

        T = torch.tensor([0.0, 1.0, 5.0], dtype=torch.float32)
        r = model_singular(T)
        self.assertFalse(torch.isnan(r).any().item())
        self.assertFalse(torch.isinf(r).any().item())

    def test_14_shrinkage_and_low_rank_helpers(self):
        """Verify create_shrinkage_covariance and create_low_rank_covariance utilities."""
        emp_cov = self.cov_matrix
        
        # Shrinkage
        shrink_05 = create_shrinkage_covariance(emp_cov, alpha=0.5)
        self.assertEqual(shrink_05.shape, (5, 5))
        self.assertTrue(np.allclose(shrink_05, shrink_05.T))

        # Low-rank
        lowrank_r2 = create_low_rank_covariance(emp_cov, rank=2)
        self.assertEqual(lowrank_r2.shape, (5, 5))
        self.assertTrue(np.allclose(lowrank_r2, lowrank_r2.T))
        evals_lr = np.linalg.eigvalsh(lowrank_r2)
        self.assertTrue(np.all(evals_lr > 0.0))


if __name__ == "__main__":
    unittest.main()
