"""
Conditional Synthetic ICU Trajectory Generator for EHR-Neural-SDE.

Loads trained EHRNeuralSDE model checkpoint (outputs/checkpoints/best_model.pt) and scaler (data/processed/scaler.json).
Generates continuous stochastic future vital sign trajectories conditioned on initial real clinical observations.

Output directory structure:
outputs/generated/test_stay_<stay_id>/sample_001.pkl ... sample_020.pkl
"""

import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE


class ConditionalTrajectoryGenerator:
    """
    Continuous-Time Conditional Trajectory Generator.
    """

    def __init__(
        self,
        checkpoint_path: str = "outputs/checkpoints/best_model.pt",
        scaler_path: str = "data/processed/scaler.json",
        device: Optional[torch.device] = None,
        max_step_size: float = 0.25,
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.checkpoint_path = checkpoint_path
        self.scaler_path = scaler_path

        # 1. Load scaler parameters
        self.scaler_data = load_scaler_params(scaler_path)
        self.feature_names = self.scaler_data["feature_names"]
        self.mean = np.array(self.scaler_data["mean"], dtype=np.float32)
        self.std = np.array(self.scaler_data["std"], dtype=np.float32)

        # 2. Instantiate and load model
        self.model = EHRNeuralSDE(max_step_size=max_step_size).to(self.device)
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"[ConditionalTrajectoryGenerator] Loaded model checkpoint from Epoch {checkpoint.get('epoch', 'N/A')}")

    @torch.no_grad()
    def generate_single_trajectory(
        self,
        X_0: torch.Tensor,
        M_0: torch.Tensor,
        T_0: torch.Tensor,
        DeltaT_0: torch.Tensor,
        T_target: torch.Tensor,
        generator: Optional[torch.Generator] = None,
        enable_noise: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a single continuous synthetic trajectory conditioned on initial observation.

        Args:
            X_0: Initial observation tensor [1, 1, 5] (normalized)
            M_0: Initial observation mask tensor [1, 1, 5]
            T_0: Initial timestamp tensor [1, 1]
            DeltaT_0: Initial time gap tensor [1, 1]
            T_target: Requested observation timestamps tensor [1, N]
            generator: Optional PyTorch random number generator
            enable_noise: If True, uses stochastic SDE Brownian motion

        Returns:
            Tuple of:
                - X_synthetic_physical: Synthetic vital sign array [N, 5] in physical units
                - z_trajectory: Integrated latent trajectory array [N, 32]
        """
        X_0 = X_0.to(self.device)
        M_0 = M_0.to(self.device)
        T_0 = T_0.to(self.device)
        DeltaT_0 = DeltaT_0.to(self.device)
        T_target = T_target.to(self.device)

        # 1. Compute initial latent state z0 = z(T0) from first observation
        h_0 = self.model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)  # [1, 64]
        z_0 = self.model.latent_projection(h_0)                      # [1, 32]

        # 2. Continuous SDE Integration across requested timestamps
        z_traj = self.model.sde.integrate(
            z0=z_0,
            times=T_target,
            generator=generator,
            enable_noise=enable_noise,
        )  # [1, N, 32]

        # 3. Decode latent state to normalized physiological space
        X_hat_norm = self.model.decoder(z_traj).squeeze(0).cpu().numpy()  # [N, 5]

        # 4. Denormalize to physical clinical units
        X_synthetic_physical = denormalize(X_hat_norm, self.mean, self.std)

        return X_synthetic_physical, z_traj.squeeze(0).cpu().numpy()

    def generate_multiple_futures(
        self,
        X_0: torch.Tensor,
        M_0: torch.Tensor,
        T_0: torch.Tensor,
        DeltaT_0: torch.Tensor,
        T_target: torch.Tensor,
        n_samples: int = 20,
        base_seed: int = 42,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Generates n_samples independent stochastic future trajectories for the SAME initial observation.

        Args:
            X_0: Initial observation tensor [1, 1, 5]
            M_0: Initial observation mask tensor [1, 1, 5]
            T_0: Initial timestamp tensor [1, 1]
            DeltaT_0: Initial time gap tensor [1, 1]
            T_target: Target timestamps tensor [1, N]
            n_samples: Number of stochastic trajectory samples (default=20)
            base_seed: Base seed for distinct per-sample generators

        Returns:
            List of tuples: [(X_synthetic_physical [N, 5], sample_seed), ...]
        """
        samples = []
        for i in range(n_samples):
            sample_seed = base_seed + i * 1000 + 1337
            gen_i = torch.Generator(device=self.device).manual_seed(sample_seed)

            X_synth, _ = self.generate_single_trajectory(
                X_0=X_0,
                M_0=M_0,
                T_0=T_0,
                DeltaT_0=DeltaT_0,
                T_target=T_target,
                generator=gen_i,
                enable_noise=True,
            )
            samples.append((X_synth, sample_seed))

        return samples


def run_generation_pipeline(
    test_path: str = "data/processed/datasets/test.pkl",
    output_dir: str = "outputs/generated",
    n_samples: int = 20,
    base_seed: int = 42,
):
    print("=== Phase 10: Generating Synthetic ICU Trajectories ===")

    generator = ConditionalTrajectoryGenerator()
    out_base = Path(output_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    with open(test_path, "rb") as f:
        test_trajectories = pickle.load(f)

    print(f"Loaded {len(test_trajectories)} real test trajectories from: {test_path}")

    all_generated_count = 0
    sanity_failures = []

    for idx, traj in enumerate(test_trajectories):
        stay_id = traj["stay_id"]
        subject_id = traj["subject_id"]
        X_real = traj["X"]          # [N, 5] (normalized)
        T_real = traj["T"]          # [N]
        DeltaT_real = traj["DeltaT"]  # [N]
        M_real = traj["M"]          # [N, 5]

        N = len(T_real)

        # Prepare 3D tensors for initial observation step 0
        X_0 = torch.tensor(X_real[0:1, :], dtype=torch.float32).unsqueeze(0)        # [1, 1, 5]
        M_0 = torch.tensor(M_real[0:1, :], dtype=torch.float32).unsqueeze(0)        # [1, 1, 5]
        T_0 = torch.tensor(T_real[0:1], dtype=torch.float32).unsqueeze(0)           # [1, 1]
        DeltaT_0 = torch.tensor(DeltaT_real[0:1], dtype=torch.float32).unsqueeze(0) # [1, 1]
        T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0)          # [1, N]

        # Generate 20 stochastic samples
        samples = generator.generate_multiple_futures(
            X_0=X_0,
            M_0=M_0,
            T_0=T_0,
            DeltaT_0=DeltaT_0,
            T_target=T_target,
            n_samples=n_samples,
            base_seed=base_seed + idx * 50,
        )

        stay_dir = out_base / f"test_stay_{stay_id}"
        stay_dir.mkdir(parents=True, exist_ok=True)

        # Verify non-identity across samples for this stay
        sample_diffs = []
        for s1 in range(len(samples)):
            for s2 in range(s1 + 1, len(samples)):
                diff = np.abs(samples[s1][0] - samples[s2][0]).mean()
                sample_diffs.append(diff)
        avg_pairwise_diff = np.mean(sample_diffs) if sample_diffs else 0.0

        for s_idx, (X_synth, sample_seed) in enumerate(samples, start=1):
            # Sanity checks on generated array
            if np.isnan(X_synth).any():
                sanity_failures.append(f"NaN detected in stay {stay_id} sample {s_idx}")
            if np.isinf(X_synth).any():
                sanity_failures.append(f"Inf detected in stay {stay_id} sample {s_idx}")
            if X_synth.shape != (N, 5):
                sanity_failures.append(f"Incorrect shape {X_synth.shape} in stay {stay_id} sample {s_idx}")

            sample_dict = {
                "subject_id": subject_id,
                "conditioning_stay_id": stay_id,
                "T": T_real,
                "X_synthetic": X_synth,
                "feature_names": generator.feature_names,
                "generation_seed": sample_seed,
            }

            sample_file = stay_dir / f"sample_{s_idx:03d}.pkl"
            with open(sample_file, "wb") as f_out:
                pickle.dump(sample_dict, f_out)

            all_generated_count += 1

        print(
            f"Stay {stay_id:8d} (N={N:3d}) -> Generated {n_samples} samples | "
            f"Avg Pairwise Sample Diff: {avg_pairwise_diff:.4f}"
        )

    if sanity_failures:
        raise RuntimeError(f"Generation Sanity Check Failures: {sanity_failures}")

    print(f"\nSUCCESS: Generated {all_generated_count} synthetic trajectories across {len(test_trajectories)} stays.")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    run_generation_pipeline()
