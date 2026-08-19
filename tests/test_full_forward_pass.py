"""
Full Forward-Pass Architecture Integration Test.

Executes an end-to-end forward pass connecting all continuous-time EHR-Neural-SDE components:
    EHR (X, M, T, DeltaT)
        ↓ (First observation: X[:,0,:], M[:,0,:], T[:,0], DeltaT[:,0])
    TimeAwareEncoder -> h_0 [B, 64]
        ↓
    LatentProjection -> z_0 [B, 32]  (Note: z_0 corresponds to z(T[0]))
        ↓
    NeuralSDE.integrate(z0, T) -> z(t) [B, N, 32]
        ↓
    EHRDecoder -> X_hat [B, N, 5]

Verifies tensor shape compatibility, absence of NaNs/Infs, initial state timestamp alignment,
and calculates untrained baseline masked MSE/MAE sanity-check metrics.
"""

import unittest
from pathlib import Path
import torch
from src.data.trajectory_dataset import ICUTrajectoryDataset
from src.models.time_aware_encoder import TimeAwareEncoder
from src.models.latent_projection import LatentProjection
from src.models.neural_sde import NeuralSDE
from src.models.decoder import EHRDecoder
from src.models.losses import masked_mse_loss, masked_mae_loss


class TestFullForwardPass(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.dataset_path = Path("data/processed/datasets/train.pkl")
        self.assertTrue(self.dataset_path.exists(), f"Train dataset missing at {self.dataset_path}")

        # Instantiate all core architectural modules
        self.encoder = TimeAwareEncoder(input_dim=5, hidden_dim=64, time_embed_dim=16)
        self.proj = LatentProjection(hidden_dim=64, latent_dim=32)
        self.sde = NeuralSDE(latent_dim=32, hidden_dim=64, max_step_size=0.1)
        self.decoder = EHRDecoder(latent_dim=32, hidden_dim=64, output_dim=5)

    def test_01_end_to_end_forward_pass_on_real_trajectory(self):
        """Executes full forward pass on a real MIMIC-IV ICU trajectory."""
        dataset = ICUTrajectoryDataset(self.dataset_path)
        sample = dataset[0]  # Load first real trajectory

        X = sample["X"].unsqueeze(0)        # [1, N, 5]
        M = sample["M"].unsqueeze(0)        # [1, N, 5]
        T = sample["T"].unsqueeze(0)        # [1, N]
        DeltaT = sample["DeltaT"].unsqueeze(0)  # [1, N]

        B, N, D = X.shape
        self.assertGreater(N, 0, "Trajectory sequence length must be > 0")

        # 1. Encode first observation tuple to initialize z0 = z(T[0])
        X_0 = X[:, 0:1, :]          # [1, 1, 5]
        M_0 = M[:, 0:1, :]          # [1, 1, 5]
        T_0 = T[:, 0:1]             # [1, 1]
        DeltaT_0 = DeltaT[:, 0:1]   # [1, 1]

        h_0 = self.encoder(X_0, M_0, T_0, DeltaT_0)  # [1, 1, 64]
        h_0 = h_0.squeeze(1)                         # [1, 64]

        z_0 = self.proj(h_0)                         # [1, 32]
        self.assertEqual(z_0.shape, (B, 32))

        # 2. Integrate continuous SDE trajectory across requested timestamps T
        z_traj = self.sde.integrate(z_0, T, enable_noise=False)  # [1, N, 32]
        self.assertEqual(z_traj.shape, (B, N, 32))
        self.assertTrue(torch.allclose(z_traj[:, 0, :], z_0))

        # 3. Decode continuous latent trajectory z(t) -> predicted observations X_hat
        X_hat = self.decoder(z_traj)                 # [1, N, 5]

        # 4. Verify shape, numerical stability, and metrics
        self.assertEqual(X_hat.shape, X.shape)
        self.assertFalse(torch.isnan(X_hat).any(), "X_hat contains NaNs")
        self.assertFalse(torch.isinf(X_hat).any(), "X_hat contains Infs")

        mse = masked_mse_loss(X_hat, X, M)
        mae = masked_mae_loss(X_hat, X, M)

        self.assertFalse(torch.isnan(mse).any())
        self.assertFalse(torch.isnan(mae).any())
        self.assertGreaterEqual(mse.item(), 0.0)
        self.assertGreaterEqual(mae.item(), 0.0)

        # Print sanity check metrics (Untrained network reference)
        print(f"\n[Sanity Check - Untrained Forward Pass]")
        print(f"Trajectory length N = {N}")
        print(f"First timestamp T[0] = {T[0, 0].item():.4f} hours")
        print(f"Masked MSE (Untrained): {mse.item():.4f}")
        print(f"Masked MAE (Untrained): {mae.item():.4f}")

    def test_02_initial_state_alignment_for_delayed_first_observation(self):
        """Verifies z0 aligns with z(T[0]) when T[0] > 0."""
        dataset = ICUTrajectoryDataset(self.dataset_path)

        # Find sample where T[0] > 0 or simulate delayed first observation
        sample = dataset[0]
        T_delayed = sample["T"].unsqueeze(0) + 1.5  # First observation at 1.5h
        X = sample["X"].unsqueeze(0)
        M = sample["M"].unsqueeze(0)
        DeltaT = sample["DeltaT"].unsqueeze(0)

        X_0 = X[:, 0:1, :]
        M_0 = M[:, 0:1, :]
        T_0 = T_delayed[:, 0:1]
        DeltaT_0 = DeltaT[:, 0:1]

        h_0 = self.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
        z_0 = self.proj(h_0)

        z_traj = self.sde.integrate(z_0, T_delayed, enable_noise=False)

        self.assertTrue(torch.allclose(z_traj[:, 0, :], z_0))
        expected_first_t = sample["T"][0].item() + 1.5
        self.assertAlmostEqual(T_delayed[0, 0].item(), expected_first_t, places=4)


if __name__ == "__main__":
    unittest.main()
