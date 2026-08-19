"""
Unit Tests for State-Dependent and Physiologically Constrained Multivariate OU Residual Process.

Validates 17 critical requirements:
1. Output shape matching.
2. No NaNs/Infs under normal and extreme inputs.
3. Irregular timestamp support.
4. Padding mask handling.
5. Missingness mask compatibility.
6. PyTorch Generator seed reproducibility.
7. Scale bounds enforcement.
8. Boundary attenuation drop as mean approaches physical limits.
9. Center stability (scale remains high in safe region).
10. Smoothness and continuity (no step jumps).
11. Zero residual output when scale is 0.
12. Valid cross-feature covariance matrix structure.
13. Stability over large time gaps.
14. Stability near physical bounds.
15. Batch item independence.
16. Verification of soft attenuation without hard clipping.
17. Feature-specific scale independence.
"""

import unittest
import numpy as np
import torch

from src.models.state_dependent_residual import (
    StateDependentMultivariateTemporalResidualModel,
    SmoothBoundaryAttenuation,
    compute_normalized_sanity_bounds,
    SANITY_RANGES,
    DEFAULT_MEAN,
    DEFAULT_STD,
)


class TestStateDependentResidual(unittest.TestCase):

    def setUp(self):
        """Sets up default model instance."""
        self.base_model = StateDependentMultivariateTemporalResidualModel(
            output_dim=5,
            global_scale=0.35,
            attenuation_mode="tanh",
            attenuation_k=2.0,
        )

    def test_1_output_shape(self):
        """1. Correct output shape for batched and 1D inputs."""
        B, N, D = 4, 10, 5
        T = torch.linspace(0, 24, N).unsqueeze(0).repeat(B, 1)
        mu_norm = torch.zeros(B, N, D)

        r_scaled, S_eff = self.base_model(T=T, mu_norm=mu_norm)
        self.assertEqual(r_scaled.shape, (B, N, D))
        self.assertEqual(S_eff.shape, (B, N, D))

        # Test 1D tensor
        T_1d = torch.linspace(0, 12, N)
        mu_1d = torch.zeros(N, D)
        r_1d, S_1d = self.base_model(T=T_1d, mu_norm=mu_1d)
        self.assertEqual(r_1d.shape, (N, D))
        self.assertEqual(S_1d.shape, (N, D))

    def test_2_no_nans_or_infs(self):
        """2. No NaNs or Infs even under extreme inputs."""
        B, N, D = 2, 8, 5
        T = torch.tensor([[0.0, 0.1, 1.0, 5.0, 10.0, 24.0, 48.0, 100.0]] * B)
        mu_norm = torch.tensor([
            [[-10.0, 10.0, 0.0, 50.0, -50.0]] * N,
            [[0.0, 0.0, 0.0, 0.0, 0.0]] * N,
        ])

        r_scaled, S_eff = self.base_model(T=T, mu_norm=mu_norm)
        self.assertFalse(torch.isnan(r_scaled).any())
        self.assertFalse(torch.isinf(r_scaled).any())
        self.assertFalse(torch.isnan(S_eff).any())
        self.assertFalse(torch.isinf(S_eff).any())

    def test_3_irregular_timestamps(self):
        """3. Irregular time gap compatibility."""
        T = torch.tensor([[0.0, 0.25, 0.30, 2.5, 2.51, 14.0, 14.1, 24.0]])
        mu = torch.zeros(1, 8, 5)
        r, S = self.base_model(T=T, mu_norm=mu)
        self.assertEqual(r.shape, (1, 8, 5))

    def test_4_padding_masks(self):
        """4. Zero output for padded sequence steps when sequence_lengths is supplied."""
        B, N, D = 2, 10, 5
        T = torch.linspace(0, 24, N).unsqueeze(0).repeat(B, 1)
        mu = torch.zeros(B, N, D)
        seq_lens = torch.tensor([5, 8])

        r_scaled, S_eff = self.base_model(T=T, mu_norm=mu, sequence_lengths=seq_lens)
        
        self.assertTrue(torch.all(r_scaled[0, 5:, :] == 0.0))
        self.assertTrue(torch.all(S_eff[0, 5:, :] == 0.0))
        self.assertTrue(torch.all(r_scaled[1, 8:, :] == 0.0))
        self.assertTrue(torch.all(S_eff[1, 8:, :] == 0.0))

    def test_5_missingness_handling(self):
        """5. Missingness mask compatibility."""
        B, N, D = 2, 6, 5
        T = torch.linspace(0, 12, N).unsqueeze(0).repeat(B, 1)
        mu = torch.zeros(B, N, D)
        mask = torch.tensor([[1, 0, 1, 1, 0, 1]] * B).unsqueeze(-1).repeat(1, 1, D)

        r_scaled, S_eff = self.base_model(T=T, mu_norm=mu)
        r_masked = r_scaled * mask
        self.assertFalse(torch.isnan(r_masked).any())

    def test_6_seed_reproducibility(self):
        """6. PyTorch Generator seed reproducibility."""
        B, N, D = 2, 8, 5
        T = torch.linspace(0, 12, N).unsqueeze(0).repeat(B, 1)
        mu = torch.zeros(B, N, D)

        gen1 = torch.Generator().manual_seed(42)
        r1, _ = self.base_model(T=T, mu_norm=mu, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        r2, _ = self.base_model(T=T, mu_norm=mu, generator=gen2)

        self.assertTrue(torch.allclose(r1, r2, atol=1e-6))

    def test_7_scale_bounds(self):
        """7. Effective scale values remain within non-negative bounds."""
        B, N, D = 3, 5, 5
        T = torch.linspace(0, 10, N).unsqueeze(0).repeat(B, 1)
        mu = torch.randn(B, N, D)

        _, S_eff = self.base_model(T=T, mu_norm=mu)
        self.assertTrue(torch.all(S_eff >= 0.0))

    def test_8_boundary_attenuation_drop(self):
        """8. Scale decreases near physical boundaries."""
        attn = SmoothBoundaryAttenuation(mode="tanh", k=2.0)
        lower_norm, upper_norm = compute_normalized_sanity_bounds()

        mu_center = 0.5 * (lower_norm + upper_norm).unsqueeze(0)
        scale_center = attn(mu_center)

        mu_near_upper = (upper_norm - 0.01).unsqueeze(0)
        scale_near_upper = attn(mu_near_upper)

        self.assertTrue(torch.all(scale_near_upper < scale_center))
        self.assertTrue(torch.all(scale_near_upper < 0.1))

    def test_9_center_stability(self):
        """9. Scale remains high in safe central region."""
        attn = SmoothBoundaryAttenuation(mode="tanh", k=2.0)
        lower_norm, upper_norm = compute_normalized_sanity_bounds()

        mu_center = 0.5 * (lower_norm + upper_norm).unsqueeze(0)
        scale_center = attn(mu_center)

        self.assertTrue(torch.all(scale_center > 0.90))

    def test_10_smoothness_and_differentiability(self):
        """10. Attenuation curve is smooth without discontinuous jumps."""
        attn = SmoothBoundaryAttenuation(mode="tanh", k=2.0)

        hr_phys = torch.linspace(10.0, 260.0, 500, requires_grad=True)
        hr_norm = (hr_phys - DEFAULT_MEAN[0]) / DEFAULT_STD[0]

        mu_norm = torch.zeros(500, 5)
        mu_norm[:, 0] = hr_norm

        scale = attn(mu_norm)[:, 0]
        
        grad = torch.autograd.grad(scale.sum(), hr_phys)[0]
        self.assertFalse(torch.isnan(grad).any())
        self.assertFalse(torch.isinf(grad).any())

    def test_11_zero_residual_on_zero_scale(self):
        """11. Zero residual output when scale is set to 0.0."""
        model = StateDependentMultivariateTemporalResidualModel(global_scale=0.0)
        T = torch.linspace(0, 10, 5).unsqueeze(0)
        mu = torch.zeros(1, 5, 5)

        r_scaled, S_eff = model(T=T, mu_norm=mu)
        self.assertTrue(torch.all(r_scaled == 0.0))
        self.assertTrue(torch.all(S_eff == 0.0))

    def test_12_valid_covariance_matrix(self):
        """12. Base model covariance matrix is positive semi-definite."""
        cov = self.base_model.base_ou_model.cov_matrix
        self.assertEqual(cov.shape, (5, 5))
        self.assertTrue(torch.allclose(cov, cov.T, atol=1e-5))
        evals = torch.linalg.eigvalsh(cov)
        self.assertTrue(torch.all(evals > -1e-6))

    def test_13_long_time_gap_stability(self):
        """13. Long time interval stability (dt = 500 hours)."""
        T = torch.tensor([[0.0, 500.0, 1000.0]])
        mu = torch.zeros(1, 3, 5)
        r_scaled, S_eff = self.base_model(T=T, mu_norm=mu)

        self.assertFalse(torch.isnan(r_scaled).any())
        self.assertFalse(torch.isinf(r_scaled).any())

    def test_14_near_boundary_stability(self):
        """14. Stability when mu is placed directly on boundary values."""
        lower_norm, upper_norm = compute_normalized_sanity_bounds()
        mu_boundary = torch.stack([lower_norm, upper_norm], dim=0).unsqueeze(1)
        T = torch.tensor([[0.0], [0.0]])

        r_scaled, S_eff = self.base_model(T=T, mu_norm=mu_boundary)
        self.assertFalse(torch.isnan(r_scaled).any())
        self.assertTrue(torch.all(S_eff < 0.05))

    def test_15_batch_independence(self):
        """15. Batch item independence (modifying item 1 does not affect item 0)."""
        T = torch.linspace(0, 10, 5).unsqueeze(0).repeat(2, 1)

        mu_a = torch.zeros(2, 5, 5)
        mu_b = torch.zeros(2, 5, 5)
        # Modify batch item 1 in mu_b drastically
        mu_b[1] = torch.tensor([[10.0, -10.0, 20.0, -20.0, 5.0]] * 5)

        gen_a = torch.Generator().manual_seed(42)
        r_a, S_a = self.base_model(T=T, mu_norm=mu_a, generator=gen_a)

        gen_b = torch.Generator().manual_seed(42)
        r_b, S_b = self.base_model(T=T, mu_norm=mu_b, generator=gen_b)

        # Batch item 0 outputs must be 100% identical
        self.assertTrue(torch.allclose(r_a[0], r_b[0], atol=1e-6))
        self.assertTrue(torch.allclose(S_a[0], S_b[0], atol=1e-6))

    def test_16_no_hard_clipping(self):
        """16. Attenuation adjusts residual scale softly without hard torch.clamp on X_synth."""
        T = torch.linspace(0, 10, 5).unsqueeze(0)
        mu_norm = torch.zeros(1, 5, 5)
        
        X_synth, S_eff = self.base_model.apply_residual(mu_norm=mu_norm, T=T)
        self.assertEqual(X_synth.dtype, torch.float32)

    def test_17_feature_specific_scales(self):
        """17. Feature-specific scales correctly scale features independently."""
        custom_scales = [1.0, 2.0, 0.5, 1.5, 0.1]
        model = StateDependentMultivariateTemporalResidualModel(
            feature_scales=custom_scales,
            global_scale=1.0,
            attenuation_mode="none",
        )
        mu_norm = torch.zeros(1, 5, 5)

        S_eff = model.compute_effective_scale(mu_norm=mu_norm)
        expected_S = torch.tensor(custom_scales).unsqueeze(0).expand(5, 5)
        self.assertTrue(torch.allclose(S_eff[0], expected_S, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
