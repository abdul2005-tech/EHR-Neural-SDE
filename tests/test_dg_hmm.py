"""
Unit tests for Discrete First-Order HMM Baseline (DGHMMBaseline).

Verifies all 9 requirements specified in Phase 5:
1. initial_probs sums to 1
2. every transition row sums to 1
3. probabilities are non-negative
4. means have shape [K, 5]
5. stds have shape [K, 5]
6. generated state sequence has requested length
7. generated X has shape [N, 5]
8. generated values contain no NaNs
9. save/load produces equivalent model parameters
"""

import tempfile
import unittest
from pathlib import Path
import numpy as np
from src.models.dg_hmm import DGHMMBaseline


class TestDGHMMBaseline(unittest.TestCase):

    def setUp(self):
        """Create mock trajectories for unit testing."""
        self.mock_trajectories = [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "stay_id": 100,
                "intime": "2110-01-01 00:00:00",
                "outtime": "2110-01-02 00:00:00",
                "los_hours": 24.0,
                "X": np.array([
                    [0.1, 0.2, -0.3, 0.4, -0.5],
                    [-0.2, 0.3, 0.1, 0.0, 0.2],
                    [0.5, -0.1, 0.4, 0.2, -0.1]
                ], dtype=np.float32),
                "T": np.array([0.0, 1.0, 2.0], dtype=np.float64),
                "DeltaT": np.array([0.0, 1.0, 1.0], dtype=np.float64),
                "M": np.array([
                    [1, 1, 1, 1, 1],
                    [1, 1, 1, 0, 1],
                    [1, 0, 1, 1, 1]
                ], dtype=np.uint8),
                "bp_modality": np.array(["arterial", "arterial", "non_invasive"]),
            },
            {
                "subject_id": 2,
                "hadm_id": 11,
                "stay_id": 101,
                "intime": "2110-01-01 00:00:00",
                "outtime": "2110-01-02 12:00:00",
                "los_hours": 36.0,
                "X": np.array([
                    [-0.4, 0.5, 0.2, -0.1, 0.0],
                    [0.3, -0.2, 0.1, 0.5, -0.3]
                ], dtype=np.float32),
                "T": np.array([0.5, 1.5], dtype=np.float64),
                "DeltaT": np.array([0.0, 1.0], dtype=np.float64),
                "M": np.array([
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1]
                ], dtype=np.uint8),
                "bp_modality": np.array(["non_invasive", "non_invasive"]),
            },
        ]
        self.K = 4
        self.feature_dim = 5
        self.model = DGHMMBaseline(K=self.K, feature_dim=self.feature_dim, random_state=42)
        self.model.fit(self.mock_trajectories)

    def test_1_initial_probs_sum_to_one(self):
        """Verify that initial_probs sums to 1.0."""
        self.assertIsNotNone(self.model.initial_probs)
        self.assertTrue(np.isclose(np.sum(self.model.initial_probs), 1.0, atol=1e-5))

    def test_2_transition_rows_sum_to_one(self):
        """Verify that every row of transition_matrix sums to 1.0."""
        self.assertIsNotNone(self.model.transition_matrix)
        row_sums = np.sum(self.model.transition_matrix, axis=1)
        self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))

    def test_3_probabilities_non_negative(self):
        """Verify that initial_probs and transition_matrix are non-negative."""
        self.assertTrue((self.model.initial_probs >= 0.0).all())
        self.assertTrue((self.model.transition_matrix >= 0.0).all())

    def test_4_means_shape(self):
        """Verify that emission means has shape [K, 5]."""
        self.assertEqual(self.model.means.shape, (self.K, self.feature_dim))

    def test_5_stds_shape(self):
        """Verify that emission stds has shape [K, 5]."""
        self.assertEqual(self.model.stds.shape, (self.K, self.feature_dim))

    def test_6_7_8_trajectory_generation(self):
        """Verify trajectory sampling lengths, shapes, and absence of NaNs."""
        n_steps = 15
        X_syn, Z_syn = self.model.generate_trajectory(n_steps)

        # Check 6: generated state sequence length
        self.assertEqual(len(Z_syn), n_steps)

        # Check 7: generated X shape [N, 5]
        self.assertEqual(X_syn.shape, (n_steps, self.feature_dim))

        # Check 8: no NaNs in generated values
        self.assertFalse(np.isnan(X_syn).any())

    def test_9_save_load_reproducibility(self):
        """Verify that save/load produces identical model parameters."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = Path(tmp_dir) / "model_test"
            self.model.save(save_path)

            loaded_model = DGHMMBaseline.load(save_path)
            self.assertEqual(loaded_model.K, self.model.K)
            self.assertEqual(loaded_model.feature_dim, self.model.feature_dim)
            self.assertTrue(np.allclose(loaded_model.initial_probs, self.model.initial_probs))
            self.assertTrue(np.allclose(loaded_model.transition_matrix, self.model.transition_matrix))
            self.assertTrue(np.allclose(loaded_model.means, self.model.means))
            self.assertTrue(np.allclose(loaded_model.stds, self.model.stds))


if __name__ == "__main__":
    unittest.main()
