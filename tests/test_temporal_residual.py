"""
Unit Tests for Continuous-Time Ornstein-Uhlenbeck (OU) Temporal Residual Model.
"""

import unittest
import torch
import numpy as np
from src.models.temporal_residual import TemporalResidualModel


class TestTemporalResidualModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        np.random.seed(42)
        self.output_dim = 5
        self.model = TemporalResidualModel(
            output_dim=self.output_dim,
            lambda_val=[0.5, 1.0, 2.0, 0.8, 1.5],
            sigma_val=[2.0, 1.5, 3.0, 5.0, 4.0],
            scale=1.0,
        )

    def test_output_shape(self):
        """1. Output shape verification for 2D and 1D inputs."""
        B, N, D = 4, 20, 5
        T = torch.sort(torch.rand(B, N) * 24.0, dim=1).values
        r = self.model(T)
        self.assertEqual(r.shape, (B, N, D))

        T_1d = torch.sort(torch.rand(N) * 24.0).values
        r_1d = self.model(T_1d)
        self.assertEqual(r_1d.shape, (N, D))

    def test_no_nans(self):
        """2. Ensures no NaNs or Infs are produced."""
        B, N = 8, 50
        T = torch.sort(torch.rand(B, N) * 48.0, dim=1).values
        r = self.model(T)
        self.assertFalse(torch.isnan(r).any().item())
        self.assertFalse(torch.isinf(r).any().item())

    def test_zero_sigma_deterministic(self):
        """3. Zero sigma produces deterministic zero residual behavior."""
        zero_model = TemporalResidualModel(output_dim=5, sigma_val=0.0)
        T = torch.arange(0.0, 10.0, 0.5).unsqueeze(0)
        r = zero_model(T)
        self.assertTrue(torch.allclose(r, torch.zeros_like(r), atol=1e-6))

    def test_very_small_dt_behavior(self):
        """4. Very small dt results in smooth, highly correlated steps."""
        # dt = 1e-4 hours
        T = torch.arange(0.0, 0.01, 1e-4).unsqueeze(0)  # N=100 steps
        r = self.model(T)  # [1, 100, 5]
        diffs = r[:, 1:, :] - r[:, :-1, :]
        # Step differences for tiny dt should be small relative to overall variance
        self.assertLess(torch.std(diffs).item(), torch.std(r).item() * 0.5)

    def test_large_dt_behavior(self):
        """5. Large dt causes alpha -> 0, approaching stationary i.i.d. Normal(0, sigma^2)."""
        # dt = 1000 hours, lambda = 1.0 -> exp(-1000) ~ 0
        T = torch.tensor([[0.0, 1000.0, 2000.0, 3000.0]])
        large_model = TemporalResidualModel(output_dim=5, lambda_val=1.0, sigma_val=2.0)
        gen = torch.Generator().manual_seed(123)
        r = large_model(T, generator=gen)  # [1, 4, 5]
        self.assertFalse(torch.isnan(r).any())
        self.assertEqual(r.shape, (1, 4, 5))

    def test_irregular_timestamps(self):
        """6. Handles arbitrary irregular timestamps correctly."""
        t_list = [0.0, 0.1, 0.35, 1.2, 5.7, 6.0, 12.4, 24.0]
        T = torch.tensor(t_list).unsqueeze(0)
        r = self.model(T)
        self.assertEqual(r.shape, (1, len(t_list), self.output_dim))
        self.assertFalse(torch.isnan(r).any())

    def test_different_seeds_different_residuals(self):
        """7. Different random seeds generate different residual trajectories."""
        T = torch.arange(0.0, 10.0, 0.5).unsqueeze(0)
        gen1 = torch.Generator().manual_seed(111)
        gen2 = torch.Generator().manual_seed(222)
        r1 = self.model(T, generator=gen1)
        r2 = self.model(T, generator=gen2)
        self.assertFalse(torch.allclose(r1, r2))

    def test_same_seed_reproducible(self):
        """8. Same random seed reproduces exact same residuals."""
        T = torch.arange(0.0, 10.0, 0.5).unsqueeze(0)
        gen1 = torch.Generator().manual_seed(777)
        gen2 = torch.Generator().manual_seed(777)
        r1 = self.model(T, generator=gen1)
        r2 = self.model(T, generator=gen2)
        self.assertTrue(torch.allclose(r1, r2, atol=1e-7))

    def test_padded_sequence_handling(self):
        """9. Padded elements past sequence_lengths are zeroed out."""
        B, N = 2, 10
        T = torch.arange(0.0, 10.0, 1.0).unsqueeze(0).repeat(B, 1)
        seq_lengths = torch.tensor([5, 8], dtype=torch.int64)
        r = self.model(T, sequence_lengths=seq_lengths)
        # Check item 0 past length 5 is zero
        self.assertTrue(torch.allclose(r[0, 5:], torch.zeros_like(r[0, 5:])))
        # Check item 1 past length 8 is zero
        self.assertTrue(torch.allclose(r[1, 8:], torch.zeros_like(r[1, 8:])))
        # Check valid parts are non-zero
        self.assertFalse(torch.allclose(r[0, :5], torch.zeros_like(r[0, :5])))

    def test_batch_handling(self):
        """10. Batch processing produces correct individual sample shapes."""
        B, N = 3, 15
        T = torch.arange(0.0, 15.0, 1.0).unsqueeze(0).repeat(B, 1)
        r = self.model(T)
        self.assertEqual(r.shape, (B, N, self.output_dim))

    def test_feature_wise_parameters(self):
        """11. Distinct feature-wise lambdas and sigmas affect features independently."""
        sigmas = [1.0, 10.0, 0.1, 50.0, 5.0]
        model = TemporalResidualModel(output_dim=5, lambda_val=1.0, sigma_val=sigmas)
        T = torch.arange(0.0, 100.0, 0.5).unsqueeze(0)
        gen = torch.Generator().manual_seed(999)
        r = model(T, generator=gen)  # [1, 200, 5]
        feature_stds = r.squeeze(0).std(dim=0).tolist()
        # Feature 3 (sigma=50) should have much larger std than Feature 2 (sigma=0.1)
        self.assertGreater(feature_stds[3], feature_stds[2] * 10)

    def test_ou_temporal_correlation_positive(self):
        """12. OU temporal correlation is positive for small time gaps dt."""
        T = torch.arange(0.0, 500.0, 0.25).unsqueeze(0)  # dt = 0.25h
        model = TemporalResidualModel(output_dim=5, lambda_val=0.2, sigma_val=2.0)
        gen = torch.Generator().manual_seed(42)
        r = model(T, generator=gen).squeeze(0).numpy()  # [2000, 5]
        
        for d in range(5):
            ac1 = np.corrcoef(r[:-1, d], r[1:, d])[0, 1]
            # exp(-0.2 * 0.25) ~ 0.95 -> lag-1 AC should be high and positive (> 0.8)
            self.assertGreater(ac1, 0.7)

    def test_residual_variance_with_lambda(self):
        """13. Larger lambda causes faster decay and lower autocorrelation."""
        T = torch.arange(0.0, 500.0, 0.5).unsqueeze(0)
        model_fast = TemporalResidualModel(output_dim=5, lambda_val=5.0, sigma_val=2.0)
        model_slow = TemporalResidualModel(output_dim=5, lambda_val=0.1, sigma_val=2.0)
        gen1 = torch.Generator().manual_seed(100)
        gen2 = torch.Generator().manual_seed(100)

        r_fast = model_fast(T, generator=gen1).squeeze(0).numpy()
        r_slow = model_slow(T, generator=gen2).squeeze(0).numpy()

        ac_fast = np.corrcoef(r_fast[:-1, 0], r_fast[1:, 0])[0, 1]
        ac_slow = np.corrcoef(r_slow[:-1, 0], r_slow[1:, 0])[0, 1]

        self.assertGreater(ac_slow, ac_fast)


if __name__ == "__main__":
    unittest.main()
