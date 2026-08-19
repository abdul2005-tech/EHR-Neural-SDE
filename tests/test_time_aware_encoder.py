"""
Unit tests for Time-Aware Encoder and Latent Projection modules.

Verifies 11 Phase 6 requirements:
1. Input X shape [N, 5] works.
2. Input M shape [N, 5] works.
3. Input T shape [N] works.
4. Input DeltaT shape [N] works.
5. Output h shape is [N, 64].
6. Latent output z shape is [N, 32].
7. Different DeltaT values produce different encoder representations.
8. Different T values produce different encoder representations.
9. Missingness mask M affects encoder output.
10. Backpropagation works and gradients are finite.
11. No NaNs or infinities are produced for normal valid inputs.
"""

import unittest
import torch
from src.models.time_aware_encoder import TimeAwareEncoder, TimeFeatureEncoder
from src.models.latent_projection import LatentProjection


class TestTimeAwareEncoder(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.N = 10
        self.input_dim = 5
        self.hidden_dim = 64
        self.latent_dim = 32

        self.encoder = TimeAwareEncoder(input_dim=self.input_dim, hidden_dim=self.hidden_dim)
        self.projection = LatentProjection(hidden_dim=self.hidden_dim, latent_dim=self.latent_dim)

        self.X = torch.randn(self.N, self.input_dim)
        self.M = torch.ones(self.N, self.input_dim)
        self.T = torch.linspace(0.0, 24.0, self.N)
        self.DeltaT = torch.tensor([0.0] + [2.4] * (self.N - 1))

    def test_1_to_6_shapes(self):
        """Verify input shapes and output dimensions for h [N, 64] and z [N, 32]."""
        # Test 1, 2, 3, 4: input shapes
        self.assertEqual(self.X.shape, (self.N, 5))
        self.assertEqual(self.M.shape, (self.N, 5))
        self.assertEqual(self.T.shape, (self.N,))
        self.assertEqual(self.DeltaT.shape, (self.N,))

        # Test 5: Output h shape is [N, 64]
        h = self.encoder(self.X, self.M, self.T, self.DeltaT)
        self.assertEqual(h.shape, (self.N, 64))

        # Test 6: Latent output z shape is [N, 32]
        z = self.projection(h)
        self.assertEqual(z.shape, (self.N, 32))

    def test_7_different_delta_t_produces_different_h(self):
        """Verify different DeltaT values produce different representations when X and M are identical."""
        X_const = torch.zeros(self.N, self.input_dim)
        M_const = torch.ones(self.N, self.input_dim)
        T_const = torch.ones(self.N) * 5.0

        DeltaT_1 = torch.ones(self.N) * 0.5
        DeltaT_2 = torch.ones(self.N) * 10.0

        h1 = self.encoder(X_const, M_const, T_const, DeltaT_1)
        h2 = self.encoder(X_const, M_const, T_const, DeltaT_2)

        self.assertFalse(torch.allclose(h1, h2, atol=1e-4))

    def test_8_different_T_produces_different_h(self):
        """Verify different T values produce different representations."""
        X_const = torch.zeros(self.N, self.input_dim)
        M_const = torch.ones(self.N, self.input_dim)
        DeltaT_const = torch.ones(self.N) * 1.0

        T_1 = torch.ones(self.N) * 1.0
        T_2 = torch.ones(self.N) * 50.0

        h1 = self.encoder(X_const, M_const, T_1, DeltaT_const)
        h2 = self.encoder(X_const, M_const, T_2, DeltaT_const)

        self.assertFalse(torch.allclose(h1, h2, atol=1e-4))

    def test_9_mask_M_affects_h(self):
        """Verify that observation mask M explicitly affects encoder output."""
        X_const = torch.zeros(self.N, self.input_dim)
        T_const = torch.ones(self.N) * 2.0
        DeltaT_const = torch.ones(self.N) * 1.0

        M_1 = torch.ones(self.N, self.input_dim)  # All observed
        M_2 = torch.zeros(self.N, self.input_dim) # All missing

        h1 = self.encoder(X_const, M_1, T_const, DeltaT_const)
        h2 = self.encoder(X_const, M_2, T_const, DeltaT_const)

        self.assertFalse(torch.allclose(h1, h2, atol=1e-4))

    def test_10_backpropagation_and_gradients(self):
        """Verify backpropagation works and gradients are finite."""
        h = self.encoder(self.X, self.M, self.T, self.DeltaT)
        z = self.projection(h)
        loss = z.sum()
        loss.backward()

        for name, param in self.encoder.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
            self.assertFalse(torch.isinf(param.grad).any())

        for name, param in self.projection.named_parameters():
            self.assertIsNotNone(param.grad)
            self.assertFalse(torch.isnan(param.grad).any())
            self.assertFalse(torch.isinf(param.grad).any())

    def test_11_no_nans_or_infs_for_normal_inputs(self):
        """Verify output tensor contains no NaNs or infinities."""
        h = self.encoder(self.X, self.M, self.T, self.DeltaT)
        z = self.projection(h)

        self.assertFalse(torch.isnan(h).any())
        self.assertFalse(torch.isinf(h).any())
        self.assertFalse(torch.isnan(z).any())
        self.assertFalse(torch.isinf(z).any())

    def test_batched_input_support(self):
        """Verify that encoder and projection support batched tensors [B, N, ...]."""
        B, N = 4, 15
        X_b = torch.randn(B, N, self.input_dim)
        M_b = torch.ones(B, N, self.input_dim)
        T_b = torch.linspace(0.0, 24.0, N).unsqueeze(0).repeat(B, 1)
        DeltaT_b = torch.ones(B, N)

        h_b = self.encoder(X_b, M_b, T_b, DeltaT_b)
        z_b = self.projection(h_b)

        self.assertEqual(h_b.shape, (B, N, 64))
        self.assertEqual(z_b.shape, (B, N, 32))


if __name__ == "__main__":
    unittest.main()
