"""
Data-Driven Temporal Profile Generator for Continuous-Time Synthetic EHR Trajectories.

Learns empirical distributions for:
1. Sequence length / number of observation time points N
2. Trajectory duration D (hours)
3. Positive inter-observation gaps dt = T[i+1] - T[i]
4. Initial patient vital sign conditions (X_0, M_0, T_0, DeltaT_0)

Strictly trained on data/processed/datasets/train.pkl ONLY to prevent data leakage.
"""

import json
import os
import pickle, math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import torch


class TemporalProfileGenerator:
    """
    Empirical Data-Driven Sampler for Synthetic Patient Time Grids.
    """

    def __init__(self, train_path: str = "data/processed/datasets/train.pkl"):
        self.train_path = train_path
        self.is_fitted = False
        
        # Empirical parameters learned from train.pkl ONLY
        self.seq_lengths: List[int] = []
        self.durations: List[float] = []
        self.time_gaps: List[float] = []
        self.initial_conditions: List[Dict[str, Any]] = []

    def fit(self, trajectories: Optional[List[Dict[str, Any]]] = None) -> "TemporalProfileGenerator":
        """
        Fits empirical distributions from training trajectories ONLY.
        """
        if trajectories is None:
            if not os.path.exists(self.train_path):
                raise FileNotFoundError(f"Training dataset not found at: {self.train_path}")
            with open(self.train_path, "rb") as f:
                trajectories = pickle.load(f)

        self.seq_lengths = []
        self.durations = []
        self.time_gaps = []
        self.initial_conditions = []

        for traj in trajectories:
            T = np.array(traj["T"], dtype=np.float32)
            X = np.array(traj["X"], dtype=np.float32)
            M = np.array(traj["M"], dtype=np.float32)
            
            N = len(T)
            if N == 0:
                continue

            self.seq_lengths.append(int(N))
            duration = float(T[-1] - T[0]) if N > 1 else 0.0
            self.durations.append(max(0.0, duration))

            if N > 1:
                gaps = np.diff(T)
                valid_gaps = gaps[gaps > 0.0]
                if len(valid_gaps) > 0:
                    self.time_gaps.extend(valid_gaps.tolist())

            # Store initial condition
            delta_t_0 = float(traj["DeltaT"][0]) if "DeltaT" in traj and len(traj["DeltaT"]) > 0 else 0.0
            self.initial_conditions.append({
                "X_0": X[0].tolist(),
                "M_0": M[0].tolist(),
                "T_0": float(T[0]),
                "DeltaT_0": delta_t_0,
            })

        if not self.time_gaps:
            self.time_gaps = [0.5, 1.0, 2.0]  # Fallback default gaps in hours

        self.is_fitted = True
        return self

    def save_profile(self, save_path: str = "outputs/checkpoints/fitted_temporal_profile.json") -> None:
        """
        Saves fitted empirical parameters to JSON.
        """
        if not self.is_fitted:
            raise RuntimeError("Cannot save profile before calling fit()")

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        profile_data = {
            "seq_lengths": self.seq_lengths,
            "durations": self.durations,
            "time_gaps": self.time_gaps,
            "initial_conditions": self.initial_conditions,
            "num_train_samples": len(self.seq_lengths),
        }
        with open(save_path, "w") as f:
            json.dump(profile_data, f, indent=2)

    def load_profile(self, load_path: str = "outputs/checkpoints/fitted_temporal_profile.json") -> "TemporalProfileGenerator":
        """
        Loads fitted empirical parameters from JSON.
        """
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Fitted temporal profile JSON not found at: {load_path}")

        with open(load_path, "r") as f:
            data = json.load(f)

        self.seq_lengths = data["seq_lengths"]
        self.durations = data["durations"]
        self.time_gaps = data["time_gaps"]
        self.initial_conditions = data["initial_conditions"]
        self.is_fitted = True
        return self

    def generate_profile(
        self,
        seed: Optional[int] = None,
        min_steps: int = 10,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates a realistic, data-driven, irregular synthetic temporal grid T_synth.

        Returns:
            Dict containing:
                - 'T': 1D NumPy array of shape (N,) with monotonically increasing hours
                - 'num_time_points': N (int)
                - 'duration': D (float hours)
                - 'initial_condition': Dict of (X_0, M_0, T_0, DeltaT_0)
        """
        if not self.is_fitted:
            self.fit()

        rng = np.random.RandomState(seed)

        # 1. Sample target sequence length N from empirical distribution
        sampled_N = int(rng.choice(self.seq_lengths))
        sampled_N = max(sampled_N, min_steps)
        if max_steps is not None:
            sampled_N = min(sampled_N, max_steps)

        # 2. Sample initial condition (X_0, M_0, T_0, DeltaT_0)
        init_cond_idx = rng.randint(0, len(self.initial_conditions))
        init_cond = self.initial_conditions[init_cond_idx]

        # 3. Sample time gaps dt to construct monotonically increasing irregular time grid
        t_start = init_cond["T_0"]
        sampled_gaps = rng.choice(self.time_gaps, size=sampled_N - 1, replace=True)

        # Add small jitter (+/- 10%) for additional variability while keeping gaps positive
        jitter = rng.uniform(0.9, 1.1, size=sampled_N - 1)
        sampled_gaps = np.maximum(sampled_gaps * jitter, 0.0167)  # At least ~1 minute gap

        T_synth = np.zeros(sampled_N, dtype=np.float32)
        T_synth[0] = t_start
        T_synth[1:] = t_start + np.cumsum(sampled_gaps)

        duration = float(T_synth[-1] - T_synth[0])

        return {
            "T": T_synth,
            "num_time_points": sampled_N,
            "duration": duration,
            "initial_condition": init_cond,
        }

    def sample_initial_condition(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Samples an initial patient condition (X_0, M_0, T_0, DeltaT_0) from training set.
        """
        if not self.is_fitted:
            self.fit()
        rng = np.random.RandomState(seed)
        idx = rng.randint(0, len(self.initial_conditions))
        return self.initial_conditions[idx]
