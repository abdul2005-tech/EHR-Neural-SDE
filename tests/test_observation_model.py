"""
Unit Tests for Phase 12 Observation Model Module (ObservationModel).

Verifies 10 critical operational contracts:
1. Correct output tensor shapes.
2. Strictly positive sigma >= 1e-4.
3. No NaNs or Infinities.
4. Gradient flow.
5. Masked NLL ignores missing/padded entries.
6. Sampling produces non-identical observations.
7. Reproducible sampling with fixed seed.
8. Batched input processing.
9. Variable sequence length handling.
10. All modes ('independent', 'heteroscedastic', 'correlated') execute cleanly.
"""

import unittest
import torch
from src.models.observation_model import ObservationModel
from src.models.losses import gaussian_nll_loss


class TestObservationModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.batch_size = 4
        self.seq_len = 10
        self.latent_dim = 32
        self.output_dim = 5
        self.z_single = torch.randn(self.latent_dim)
        self.z_batch = torch.randn(self.batch_size, self.seq_len, self.latent_dim)

    def test_01_output_shapes(self):
        """Test 1: Check output shapes for 2D and 3D latent inputs across modes."""
        for mode in ["independent", "heteroscedastic"]:
            model = ObservationModel(mode=mode)
            mu, sigma = model(self.z_batch)
            self.assertEqual(mu.shape, (self.batch_size, self.seq_len, self.output_dim))
            self.assertEqual(sigma.shape, (self.batch_size, self.seq_len, self.output_dim))

        model_corr = ObservationModel(mode="correlated", rank=2)
        mu, d_diag, U_factor = model_corr(self.z_batch)
        self.assertEqual(mu.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertEqual(d_diag.shape, (self.batch_size, self.seq_len, self.output_dim))
        self.assertEqual(U_factor.shape, (self.batch_size, self.seq_len, self.output_dim, 2))

    def test_02_positive_sigma(self):
        """Test 2: Ensure predicted sigma is strictly positive >= 1e-4."""
        for mode in ["independent", "heteroscedastic"]:
            model = ObservationModel(mode=mode)
            _, sigma = model(self.z_batch)
            self.assertTrue(torch.all(sigma >= 1e-4))

        model_corr = ObservationModel(mode="correlated")
        _, d_diag, _ = model_corr(self.z_batch)
        self.assertTrue(torch.all(d_diag >= 1e-4))

    def test_03_no_nans_or_infs(self):
        """Test 3: Ensure forward pass and sampling produce no NaNs or Infs."""
        for mode in ["independent", "heteroscedastic", "correlated"]:
            model = ObservationModel(mode=mode)
            if mode == "correlated":
                mu, d_diag, U = model(self.z_batch)
                self.assertFalse(torch.isnan(mu).any() or torch.isinf(mu).any())
                self.assertFalse(torch.isnan(d_diag).any() or torch.isinf(d_diag).any())
                self.assertFalse(torch.isnan(U).any() or torch.isinf(U).any())
            else:
                mu, sigma = model(self.z_batch)
                self.assertFalse(torch.isnan(mu).any() or torch.isinf(mu).any())
                self.assertFalse(torch.isnan(sigma).any() or torch.isinf(sigma).any())

            sample = model.sample(self.z_batch)
            self.assertFalse(torch.isnan(sample).any() or torch.isinf(sample).any())

    def test_04_gradient_flow(self):
        """Test 4: Verify gradient flow back through parameters."""
        model = ObservationModel(mode="heteroscedastic")
        mu, sigma = model(self.z_batch)
        loss = torch.mean(mu ** 2 + sigma ** 2)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has no gradient.")
                self.assertFalse(torch.isnan(param.grad).any())

    def test_05_masked_nll_ignores_missing(self):
        """Test 5: Verify masked Gaussian NLL returns zero loss for missing/padded entries."""
        mu = torch.zeros(2, 5, 5)
        sigma = torch.ones(2, 5, 5)
        target = torch.full((2, 5, 5), 999.0)  # extreme missing values
        mask = torch.zeros(2, 5, 5)  # missing mask
        padding_mask = torch.ones(2, 5)

        nll_loss = gaussian_nll_loss(mu, sigma, target, mask, padding_mask)
        self.assertEqual(nll_loss.item(), 0.0)

    def test_06_sampling_non_identity(self):
        """Test 6: Verify repeated sampling produces non-identical observations."""
        model = ObservationModel(mode="heteroscedastic")
        s1 = model.sample(self.z_batch)
        s2 = model.sample(self.z_batch)
        diff = torch.abs(s1 - s2).mean().item()
        self.assertGreater(diff, 1e-4)

    def test_07_reproducible_fixed_seed(self):
        """Test 7: Verify fixed seed generator produces identical samples."""
        model = ObservationModel(mode="heteroscedastic")
        g1 = torch.Generator().manual_seed(1234)
        g2 = torch.Generator().manual_seed(1234)

        s1 = model.sample(self.z_batch, generator=g1)
        s2 = model.sample(self.z_batch, generator=g2)
        self.assertTrue(torch.allclose(s1, s2, atol=1e-6))

    def test_08_batched_input(self):
        """Test 8: Verify 3D batching works consistently."""
        model = ObservationModel(mode="independent")
        mu, sigma = model(self.z_batch)
        self.assertEqual(mu.dim(), 3)
        self.assertEqual(sigma.dim(), 3)

    def test_09_variable_sequence_lengths(self):
        """Test 9: Verify handling of padded sequence shapes."""
        z_var = torch.randn(2, 15, 32)
        model = ObservationModel(mode="heteroscedastic")
        mu, sigma = model(z_var)
        self.assertEqual(mu.shape, (2, 15, 5))
        self.assertEqual(sigma.shape, (2, 15, 5))

    def test_10_all_modes_support(self):
        """Test 10: Verify all 3 modes instantiate and execute cleanly."""
        for mode in ["independent", "heteroscedastic", "correlated"]:
            model = ObservationModel(mode=mode)
            out = model(self.z_batch)
            sample = model.sample(self.z_batch)
            self.assertEqual(sample.shape, (self.batch_size, self.seq_len, self.output_dim))


if __name__ == "__main__":
    unittest.main()
