"""
Unit tests for EHRNeuralSDE Pipeline Model.
"""

import unittest
import torch
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.losses import masked_mse_loss


class TestEHRNeuralSDE(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B = 4
        self.N = 10
        self.D = 5
        self.latent_dim = 32

        self.model = EHRNeuralSDE(
            input_dim=self.D,
            hidden_dim=64,
            latent_dim=self.latent_dim,
            max_step_size=0.1,
        )

        self.X = torch.randn(self.B, self.N, self.D)
        self.M = torch.ones(self.B, self.N, self.D)
        self.T = torch.linspace(0.0, 5.0, self.N).unsqueeze(0).repeat(self.B, 1)
        self.DeltaT = torch.full((self.B, self.N), 0.5)

    def test_01_forward_pass_shapes(self):
        """Verify EHRNeuralSDE forward pass produces X_hat [B, N, 5] and z_traj [B, N, 32]."""
        X_hat, z_traj = self.model(self.X, self.M, self.T, self.DeltaT, enable_noise=False)
        self.assertEqual(X_hat.shape, (self.B, self.N, self.D))
        self.assertEqual(z_traj.shape, (self.B, self.N, self.latent_dim))
        self.assertFalse(torch.isnan(X_hat).any())
        self.assertFalse(torch.isnan(z_traj).any())

    def test_02_get_initial_state(self):
        """Verify get_initial_state extracts z0 [B, 32] from step 0."""
        z0 = self.model.get_initial_state(self.X, self.M, self.T, self.DeltaT)
        self.assertEqual(z0.shape, (self.B, self.latent_dim))
        self.assertFalse(torch.isnan(z0).any())

    def test_03_padded_timestamps_safety(self):
        """Verify length-aware sequence_lengths clamping prevents NaNs for padded steps."""
        seq_lengths = torch.tensor([10, 5, 8, 3], dtype=torch.int64)
        padding_mask = torch.zeros(self.B, self.N)
        for i, l in enumerate(seq_lengths):
            padding_mask[i, :l] = 1.0

        X_hat, z_traj = self.model(
            self.X,
            self.M,
            self.T,
            self.DeltaT,
            sequence_lengths=seq_lengths,
            padding_mask=padding_mask,
            enable_noise=False,
        )

        self.assertEqual(X_hat.shape, (self.B, self.N, self.D))
        self.assertFalse(torch.isnan(X_hat).any())

    def test_04_full_differentiability(self):
        """Verify backpropagation through EHRNeuralSDE produces finite gradients across all submodules."""
        X_req = self.X.clone().detach().requires_grad_(True)
        X_hat, z_traj = self.model(X_req, self.M, self.T, self.DeltaT, enable_noise=False)

        loss = masked_mse_loss(X_hat, self.X, self.M)
        loss.backward()

        self.assertIsNotNone(X_req.grad)
        self.assertFalse(torch.isnan(X_req.grad).any())

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Param {name} has None grad")
                self.assertFalse(torch.isnan(param.grad).any(), f"Param {name} has NaN grad")


if __name__ == "__main__":
    unittest.main()
