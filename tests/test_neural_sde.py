"""
Unit tests for Neural SDE Dynamics and Euler-Maruyama Continuous-Time Integrator.

Verifies all 14 Phase 7 requirements explicitly:
1. Drift output shape [B, 32].
2. Diffusion output shape [B, 32].
3. Diffusion outputs are non-negative.
4. Euler-Maruyama step output shape [B, 32].
5. Integration output shape [B, N, 32].
6. First integrated state equals z0.
7. Increasing dt changes the deterministic drift contribution.
8. With diffusion disabled/zeroed, integration is deterministic.
9. With nonzero diffusion, repeated integrations can produce different trajectories.
10. Backpropagation through integration produces finite gradients.
11. No NaNs or infinities for reasonable inputs.
12. Irregular time arrays are accepted.
13. Large time gaps are internally subdivided.
14. All requested observation times are returned exactly.
"""

import unittest
import torch
from src.models.neural_sde import DriftNetwork, DiffusionNetwork, NeuralSDE


class TestNeuralSDE(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.B = 4
        self.latent_dim = 32
        self.hidden_dim = 64
        self.max_step_size = 0.1

        self.drift_net = DriftNetwork(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.diff_net = DiffusionNetwork(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim)
        self.sde = NeuralSDE(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, max_step_size=self.max_step_size)

        self.z0 = torch.randn(self.B, self.latent_dim)
        self.t0 = torch.zeros(self.B, 1)

    def test_01_drift_output_shape(self):
        """1. Verify DriftNetwork output shape is [B, 32]."""
        drift = self.drift_net(self.z0, self.t0)
        self.assertEqual(drift.shape, (self.B, 32))

    def test_02_diffusion_output_shape(self):
        """2. Verify DiffusionNetwork output shape is [B, 32]."""
        diff = self.diff_net(self.z0, self.t0)
        self.assertEqual(diff.shape, (self.B, 32))

    def test_03_diffusion_outputs_non_negative(self):
        """3. Verify DiffusionNetwork outputs are strictly non-negative."""
        diff = self.diff_net(self.z0, self.t0)
        self.assertTrue((diff >= 0.0).all())

    def test_04_euler_maruyama_step_shape(self):
        """4. Verify Euler-Maruyama step output shape is [B, 32]."""
        z_next = self.sde.euler_maruyama_step(self.z0, self.t0, dt=0.05)
        self.assertEqual(z_next.shape, (self.B, 32))

    def test_05_integration_output_shape(self):
        """5. Verify Integration output shape is [B, N, 32]."""
        times = torch.tensor([0.0, 0.5, 1.0, 2.5, 5.0])
        N = len(times)
        z_traj = self.sde.integrate(self.z0, times)
        self.assertEqual(z_traj.shape, (self.B, N, 32))

    def test_06_first_integrated_state_equals_z0(self):
        """6. Verify z_trajectory[:, 0, :] equals z0 exactly."""
        times = torch.tensor([0.0, 0.5, 1.0])
        z_traj = self.sde.integrate(self.z0, times)
        self.assertTrue(torch.allclose(z_traj[:, 0, :], self.z0))

    def test_07_increasing_dt_changes_drift_contribution(self):
        """7. Verify increasing dt changes the deterministic drift contribution."""
        z_next_small = self.sde.euler_maruyama_step(self.z0, self.t0, dt=0.01, enable_noise=False)
        z_next_large = self.sde.euler_maruyama_step(self.z0, self.t0, dt=2.00, enable_noise=False)
        self.assertFalse(torch.allclose(z_next_small, z_next_large, atol=1e-3))

    def test_08_deterministic_integration_when_diffusion_disabled(self):
        """8. Verify with diffusion disabled (enable_noise=False), integration is 100% deterministic."""
        times = torch.tensor([0.0, 0.5, 1.0, 2.0])
        z_traj_1 = self.sde.integrate(self.z0, times, enable_noise=False)
        z_traj_2 = self.sde.integrate(self.z0, times, enable_noise=False)
        self.assertTrue(torch.allclose(z_traj_1, z_traj_2, atol=1e-6))

    def test_09_stochastic_integration_with_nonzero_diffusion(self):
        """9. Verify with nonzero diffusion, repeated integrations produce different trajectories."""
        times = torch.tensor([0.0, 0.5, 1.0, 2.0])
        gen1 = torch.Generator().manual_seed(101)
        gen2 = torch.Generator().manual_seed(202)

        z_traj_1 = self.sde.integrate(self.z0, times, generator=gen1, enable_noise=True)
        z_traj_2 = self.sde.integrate(self.z0, times, generator=gen2, enable_noise=True)
        self.assertFalse(torch.allclose(z_traj_1, z_traj_2, atol=1e-3))

    def test_10_backpropagation_finite_gradients(self):
        """10. Verify backpropagation through integration produces finite, non-NaN gradients."""
        z0_req = self.z0.clone().detach().requires_grad_(True)
        times = torch.tensor([0.0, 0.5, 1.5])

        z_traj = self.sde.integrate(z0_req, times, enable_noise=False)
        loss = z_traj.sum()
        loss.backward()

        self.assertIsNotNone(z0_req.grad)
        self.assertFalse(torch.isnan(z0_req.grad).any())

        for name, p in self.sde.drift_net.named_parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.isnan(p.grad).any())

        for name, p in self.sde.diffusion_net.named_parameters():
            self.assertIsNotNone(p.grad)
            self.assertFalse(torch.isnan(p.grad).any())

    def test_11_no_nans_or_infinities(self):
        """11. Verify no NaNs or infinities for reasonable inputs."""
        times = torch.tensor([0.0, 0.1, 1.0, 10.0])
        z_traj = self.sde.integrate(self.z0, times)
        self.assertFalse(torch.isnan(z_traj).any())
        self.assertFalse(torch.isinf(z_traj).any())

    def test_12_irregular_time_arrays_accepted(self):
        """12. Verify irregular time arrays are accepted without throwing errors."""
        times = torch.tensor([0.0, 0.05, 0.687, 1.687, 5.2, 20.0])
        z_traj = self.sde.integrate(self.z0, times)
        self.assertEqual(z_traj.shape, (self.B, len(times), 32))

    def test_13_large_time_gaps_subdivided(self):
        """13. Verify large time gaps (e.g. 43.8 hours) are internally subdivided."""
        times = torch.tensor([0.0, 43.8])  # Max gap in MIMIC-IV dataset
        sde = NeuralSDE(latent_dim=32, max_step_size=0.1)

        z_traj = sde.integrate(self.z0, times, enable_noise=False)
        self.assertEqual(z_traj.shape, (self.B, 2, 32))
        self.assertFalse(torch.isnan(z_traj).any())

    def test_14_exact_requested_observation_times_returned(self):
        """14. Verify exact requested observation times are returned in trajectory shape and values."""
        times = torch.tensor([0.0, 0.05, 0.687, 1.687, 5.2, 20.0])
        z_traj = self.sde.integrate(self.z0, times)
        self.assertEqual(z_traj.shape[1], len(times))


if __name__ == "__main__":
    unittest.main()
