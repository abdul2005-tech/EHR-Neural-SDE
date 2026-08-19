"""
Unit tests for data sequence builder, PyTorch Dataset, and custom collate function.
"""

import unittest
import numpy as np
import torch
from src.data.sequence_builder import compute_time_gaps
from src.data.trajectory_dataset import ICUTrajectoryDataset
from src.data.collate import collate_icu_trajectories


class TestDataProcessing(unittest.TestCase):

    def test_compute_time_gaps(self):
        """Test that time gaps delta_t = t_i - t_{i-1} are calculated correctly."""
        timestamps = np.array([0.0, 1.5, 4.0, 10.0], dtype=np.float32)
        expected_gaps = np.array([0.0, 1.5, 2.5, 6.0], dtype=np.float32)

        gaps = compute_time_gaps(timestamps)
        np.testing.assert_allclose(gaps, expected_gaps, rtol=1e-5)

    def test_compute_time_gaps_empty(self):
        """Test empty timestamp array behavior."""
        timestamps = np.array([], dtype=np.float32)
        gaps = compute_time_gaps(timestamps)
        self.assertEqual(len(gaps), 0)

    def test_icu_trajectory_dataset_and_collate(self):
        """Test dataset sample loading and batch padding collation."""
        sample_trajs = [
            {
                "subject_id": 100,
                "hadm_id": 1000,
                "stay_id": 1,
                "intime": "2110-01-01 00:00:00",
                "outtime": "2110-01-02 00:00:00",
                "los_hours": 24.0,
                "X": np.array([[0.5, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.2, 0.0, 0.0]], dtype=np.float32),
                "T": np.array([0.0, 1.0], dtype=np.float64),
                "DeltaT": np.array([0.0, 1.0], dtype=np.float64),
                "M": np.array([[1, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=np.uint8),
                "bp_modality": np.array(["none", "none"]),
            },
            {
                "subject_id": 101,
                "hadm_id": 1001,
                "stay_id": 2,
                "intime": "2110-01-01 00:00:00",
                "outtime": "2110-01-02 12:00:00",
                "los_hours": 36.0,
                "X": np.array([[0.1, 0.2, 0.3, 0.4, 0.5]], dtype=np.float32),
                "T": np.array([0.5], dtype=np.float64),
                "DeltaT": np.array([0.0], dtype=np.float64),
                "M": np.array([[1, 1, 1, 1, 1]], dtype=np.uint8),
                "bp_modality": np.array(["arterial"]),
            },
        ]

        ds = ICUTrajectoryDataset(sample_trajs)
        self.assertEqual(len(ds), 2)
        sample0 = ds[0]
        self.assertEqual(sample0["X"].shape, (2, 5))
        self.assertEqual(sample0["subject_id"], 100)

        # Collate batch
        batch = collate_icu_trajectories([ds[0], ds[1]])
        self.assertEqual(batch["X_padded"].shape, (2, 2, 5))
        self.assertEqual(batch["T_padded"].shape, (2, 2))
        self.assertEqual(batch["padding_mask"].shape, (2, 2))
        self.assertEqual(list(batch["sequence_lengths"].numpy()), [2, 1])


if __name__ == "__main__":
    unittest.main()
