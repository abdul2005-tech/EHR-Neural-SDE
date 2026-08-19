"""
PyTorch Dataset for Continuous-Time ICU Trajectories.

Loads processed trajectory dictionaries and formats individual ICU trajectories as PyTorch Tensors.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Union
import torch
from torch.utils.data import Dataset


class ICUTrajectoryDataset(Dataset):
    """
    PyTorch Dataset wrapper for individual continuous-time ICU trajectories.

    Returns for each trajectory sample:
        - "X": Tensor of shape (N, 5) - Normalized observation values (0.0 for missing)
        - "T": Tensor of shape (N,) - Relative timestamps in hours
        - "DeltaT": Tensor of shape (N,) - Inter-observation time gaps in hours
        - "M": Tensor of shape (N, 5) - Binary clinical observation mask (1=observed, 0=missing)
        - "subject_id": int - Patient identifier
        - "hadm_id": int - Hospital admission identifier
        - "stay_id": int - ICU stay identifier
        - "los_hours": float - ICU length of stay duration
        - "bp_modality": np.ndarray / list - BP modality string array per timestamp
    """

    def __init__(self, data_or_path: Union[List[Dict[str, Any]], str, Path]):
        if isinstance(data_or_path, (str, Path)):
            with open(data_or_path, "rb") as f:
                self.trajectories = pickle.load(f)
        else:
            self.trajectories = data_or_path

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        traj = self.trajectories[idx]

        return {
            "X": torch.tensor(traj["X"], dtype=torch.float32),
            "T": torch.tensor(traj["T"], dtype=torch.float32),
            "DeltaT": torch.tensor(traj["DeltaT"], dtype=torch.float32),
            "M": torch.tensor(traj["M"], dtype=torch.float32),
            "subject_id": traj["subject_id"],
            "hadm_id": traj["hadm_id"],
            "stay_id": traj["stay_id"],
            "los_hours": traj["los_hours"],
            "bp_modality": traj["bp_modality"],
        }
