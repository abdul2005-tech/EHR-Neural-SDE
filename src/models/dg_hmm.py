"""
Discrete First-Order Latent-State Generative Baseline (DG-HMM).

This baseline models longitudinal EHR observation sequences as a discrete-time,
first-order Markov chain over K latent health states with diagonal Gaussian emission distributions.

LIMITATION WARNING:
This is a discrete first-order baseline. It does NOT model continuous-time dynamics,
does NOT use time deltas (DeltaT), and does NOT compute continuous SDE trajectories.
It serves as a benchmark for comparison against our future continuous-time Neural SDE.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.cluster import KMeans


class DGHMMBaseline:
    """
    Discrete First-Order Latent-State Baseline Model.

    Attributes:
        K (int): Number of discrete latent states (default: 8)
        feature_dim (int): Dimensionality of observations (default: 5)
        random_state (int): Random seed for clustering and sampling
        initial_probs (np.ndarray): State initial distribution pi [K]
        transition_matrix (np.ndarray): Transition matrix A [K, K]
        means (np.ndarray): Emission mean per state and feature [K, 5]
        stds (np.ndarray): Emission std per state and feature [K, 5]
    """

    def __init__(self, K: int = 8, feature_dim: int = 5, random_state: int = 42):
        self.K = K
        self.feature_dim = feature_dim
        self.random_state = random_state

        self.initial_probs: Optional[np.ndarray] = None
        self.transition_matrix: Optional[np.ndarray] = None
        self.means: Optional[np.ndarray] = None
        self.stds: Optional[np.ndarray] = None
        self.kmeans: Optional[KMeans] = None
        self.is_fitted = False

    def _prepare_clustering_inputs(self, trajectories: List[Dict[str, Any]]) -> np.ndarray:
        """
        Concatenates normalized features X (missing filled with 0.0) and observation mask M
        into a 10-dimensional representation [X_filled, M] for missing-aware clustering.
        """
        vectors = []
        for traj in trajectories:
            X = traj["X"]
            M = traj["M"]
            if len(X) > 0:
                X_filled = np.nan_to_num(X, nan=0.0)
                # Concatenate 5 normalized values + 5 mask values = 10D
                V = np.hstack([X_filled, M])
                vectors.append(V)

        return np.vstack(vectors)

    def fit(self, trajectories: List[Dict[str, Any]], smoothing: float = 1e-3, min_std: float = 1e-2) -> "DGHMMBaseline":
        """
        Fits discrete latent states using KMeans clustering over [X_filled, M] 10D vectors,
        then estimates transition matrix A, initial distribution pi, and emission parameters (mu, sigma).

        Args:
            trajectories: List of trajectory dictionaries (from train.pkl)
            smoothing: Additive smoothing constant for transition and initial probabilities
            min_std: Minimum standard deviation floor for Gaussian emissions
        """
        V_train = self._prepare_clustering_inputs(trajectories)

        # 1. Clustering / Hard-Assignment
        self.kmeans = KMeans(
            n_clusters=self.K,
            random_state=self.random_state,
            n_init=10,
        )
        self.kmeans.fit(V_train)

        # 2. Predict State Sequences for Training Trajectories
        initial_counts = np.zeros(self.K, dtype=np.float64)
        transition_counts = np.zeros((self.K, self.K), dtype=np.float64)

        # Accumulators for emission parameters
        state_obs_values = {k: [[] for _ in range(self.feature_dim)] for k in range(self.K)}

        for traj in trajectories:
            X = traj["X"]
            M = traj["M"]
            if len(X) == 0:
                continue

            X_filled = np.nan_to_num(X, nan=0.0)
            V = np.hstack([X_filled, M])
            z_seq = self.kmeans.predict(V)

            # Count initial state
            z0 = z_seq[0]
            initial_counts[z0] += 1.0

            # Count transitions
            for t in range(len(z_seq) - 1):
                z_curr = z_seq[t]
                z_next = z_seq[t + 1]
                transition_counts[z_curr, z_next] += 1.0

            # Collect observed emission values for state k
            for t in range(len(z_seq)):
                zk = z_seq[t]
                for d in range(self.feature_dim):
                    if M[t, d] == 1:
                        state_obs_values[zk][d].append(X[t, d])

        # 3. Initial Distribution pi
        self.initial_probs = (initial_counts + smoothing) / np.sum(initial_counts + smoothing)

        # 4. Transition Matrix A
        self.transition_matrix = np.zeros((self.K, self.K), dtype=np.float64)
        for i in range(self.K):
            row_sum = np.sum(transition_counts[i, :] + smoothing)
            self.transition_matrix[i, :] = (transition_counts[i, :] + smoothing) / row_sum

        # 5. Emission Parameters (means, stds)
        self.means = np.zeros((self.K, self.feature_dim), dtype=np.float64)
        self.stds = np.zeros((self.K, self.feature_dim), dtype=np.float64)

        # Global fallbacks in case a state has < 2 observed samples for a feature
        global_obs = [[] for _ in range(self.feature_dim)]
        for k in range(self.K):
            for d in range(self.feature_dim):
                global_obs[d].extend(state_obs_values[k][d])

        global_means = [np.mean(global_obs[d]) if len(global_obs[d]) > 0 else 0.0 for d in range(self.feature_dim)]
        global_stds = [max(np.std(global_obs[d]), min_std) if len(global_obs[d]) > 1 else 1.0 for d in range(self.feature_dim)]

        for k in range(self.K):
            for d in range(self.feature_dim):
                vals = state_obs_values[k][d]
                if len(vals) >= 2:
                    m = np.mean(vals)
                    s = np.std(vals)
                    self.means[k, d] = m
                    self.stds[k, d] = max(s, min_std)
                else:
                    self.means[k, d] = global_means[d]
                    self.stds[k, d] = global_stds[d]

        self.is_fitted = True
        return self

    def predict_states(self, trajectory: Dict[str, Any]) -> np.ndarray:
        """
        Predicts discrete latent state sequence Z [N] for a single trajectory.
        """
        if not self.is_fitted or self.kmeans is None:
            raise RuntimeError("Model must be fitted before calling predict_states.")

        X = trajectory["X"]
        M = trajectory["M"]
        if len(X) == 0:
            return np.array([], dtype=int)

        X_filled = np.nan_to_num(X, nan=0.0)
        V = np.hstack([X_filled, M])
        return self.kmeans.predict(V)

    def sample(self, n_steps: int, rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Alias for generate_trajectory(n_steps).
        """
        return self.generate_trajectory(n_steps, rng=rng)

    def generate_trajectory(self, n_steps: int, rng: Optional[np.random.RandomState] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a synthetic discrete-time trajectory of length n_steps.

        Returns:
            X_synthetic (np.ndarray): Shape [n_steps, 5] continuous emission samples
            Z_states (np.ndarray): Shape [n_steps] discrete latent state sequence
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before generating trajectories.")

        if rng is None:
            rng = np.random.RandomState(self.random_state)

        Z_states = np.zeros(n_steps, dtype=int)
        X_synthetic = np.zeros((n_steps, self.feature_dim), dtype=np.float32)

        # 1. Sample initial state z0
        z_curr = rng.choice(self.K, p=self.initial_probs)
        Z_states[0] = z_curr

        # 2. Transition through Markov chain
        for t in range(1, n_steps):
            z_next = rng.choice(self.K, p=self.transition_matrix[z_curr])
            Z_states[t] = z_next
            z_curr = z_next

        # 3. Sample Gaussian emissions
        for t in range(n_steps):
            zk = Z_states[t]
            mu = self.means[zk]
            sigma = self.stds[zk]
            X_synthetic[t, :] = rng.normal(loc=mu, scale=sigma)

        return X_synthetic, Z_states

    def save(self, path: Union[str, Path]) -> None:
        """
        Saves model parameters and fit metadata to specified directory or JSON file.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        json_path = path / "model.json"
        pickle_path = path / "model.pkl"

        data_json = {
            "K": self.K,
            "feature_dim": self.feature_dim,
            "random_state": self.random_state,
            "is_fitted": self.is_fitted,
            "initial_probs": self.initial_probs.tolist() if self.initial_probs is not None else None,
            "transition_matrix": self.transition_matrix.tolist() if self.transition_matrix is not None else None,
            "means": self.means.tolist() if self.means is not None else None,
            "stds": self.stds.tolist() if self.stds is not None else None,
        }

        with open(json_path, "w") as f:
            json.dump(data_json, f, indent=2)

        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

        print(f"Model saved to: {json_path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DGHMMBaseline":
        """
        Loads fitted model instance from directory or pickle file.
        """
        path = Path(path)
        pickle_path = path / "model.pkl" if path.is_dir() else path
        with open(pickle_path, "rb") as f:
            model = pickle.load(f)
        return model
