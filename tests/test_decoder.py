"""
Unit tests for EHR Decoder Module and Masked Loss Functions.

Verifies 9 Phase 8 decoder requirements:
1. Decoder accepts [B, N, 32] input tensors.
2. Output shape is [B, N, 5].
3. Output contains no NaNs.
4. Backpropagation produces finite gradients for decoder parameters.
5. masked_mse_loss ignores missing values (M=0).
6. masked_mae_loss ignores missing values (M=0).
7. Changing z changes decoder output.
8. Decoder works for different sequence lengths (N=1, N=10, N=100).
9. All-zero mask is handled safely without division by zero or NaNs.
"""

import unittest
import torch
from src.models.decoder import EHRDecoder
from src.models.losses import masked_mse_loss, masked_mae_loss


class TestEHRDecoder(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B = 4
        self.N = 12
        self.latent_dim = 32
        self.output_dim = 5

        self.decoder = EHRDecoder(latent_dim=self.latent_dim, output_dim=self.output_dim)
        self.z = torch.randn(self.B, self.N, self.latent_dim)

    def test_01_decoder_accepts_bn32_input(self):
        """1. Verify EHRDecoder accepts input tensor of shape [B, N, 32]."""
        out = self.decoder(self.z)
        self.assertIsNotNone(out)

    def test_02_output_shape_is_bn5(self):
        """2. Verify output shape is [B, N, 5]."""
        out = self.decoder(self.z)
        self.assertEqual(out.shape, (self.B, self.N, self.output_dim))

    def test_03_output_contains_no_nans(self):
        """3. Verify decoder output contains no NaNs or infinities."""
        out = self.decoder(self.z)
        self.assertFalse(torch.isnan(out).any())
        self.assertFalse(torch.isinf(out).any())

    def test_04_backpropagation_finite_gradients(self):
        """4. Verify backpropagation produces finite gradients for decoder parameters."""
        z_req = self.z.clone().detach().requires_grad_(True)
        out = self.decoder(z_req)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(z_req.grad)
        self.assertFalse(torch.isnan(z_req.grad).any())

        for name, p in self.decoder.named_parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.isnan(p.grad).any())

    def test_05_masked_mse_ignores_missing_values(self):
        """5. Verify masked_mse_loss ignores unobserved entries (M=0)."""
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 10.0], [3.0, 10.0]])  # Large errors at index 1
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])      # Mask out index 1 completely

        loss = masked_mse_loss(pred, target, mask)
        # Only index 0 entries contribute: (1-1)^2 + (3-3)^2 = 0.0
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

    def test_06_masked_mae_ignores_missing_values(self):
        """6. Verify masked_mae_loss ignores unobserved entries (M=0)."""
        pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target = torch.tensor([[1.0, 100.0], [3.0, 100.0]])  # Extreme errors at index 1
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])         # Mask out index 1 completely

        mae = masked_mae_loss(pred, target, mask)
        self.assertAlmostEqual(mae.item(), 0.0, places=6)

    def test_07_changing_z_changes_decoder_output(self):
        """7. Verify changing input z changes decoder output."""
        z1 = torch.randn(self.B, self.N, self.latent_dim)
        z2 = z1 + 5.0

        out1 = self.decoder(z1)
        out2 = self.decoder(z2)

        self.assertFalse(torch.allclose(out1, out2, atol=1e-3))

    def test_08_decoder_works_for_different_sequence_lengths(self):
        """8. Verify decoder works for different sequence lengths (N=1, N=10, N=100)."""
        for n_steps in [1, 10, 100]:
            z_seq = torch.randn(2, n_steps, self.latent_dim)
            out_seq = self.decoder(z_seq)
            self.assertEqual(out_seq.shape, (2, n_steps, self.output_dim))
            self.assertFalse(torch.isnan(out_seq).any())

    def test_09_all_zero_mask_handled_safely(self):
        """9. Verify all-zero mask (sum(M)==0) returns 0.0 without division-by-zero or NaNs."""
        pred = torch.randn(4, 10, 5)
        target = torch.randn(4, 10, 5)
        mask_zero = torch.zeros(4, 10, 5)

        loss_mse = masked_mse_loss(pred, target, mask_zero)
        loss_mae = masked_mae_loss(pred, target, mask_zero)

        self.assertFalse(torch.isnan(loss_mse).any())
        self.assertFalse(torch.isnan(loss_mae).any())
        self.assertEqual(loss_mse.item(), 0.0)
        self.assertEqual(loss_mae.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
