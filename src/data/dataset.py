"""
PyTorch Dataset for Irregular Longitudinal EHR Sequences.

Handles batching, padding, and tensor formatting for irregularly sampled continuous-time sequences.
"""

from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class EHRSequenceDataset(Dataset):
    """
    PyTorch Dataset wrapper for continuous-time longitudinal EHR data.

    Returns for each item:
        - observations: Tensor of shape (seq_len, feature_dim)
        - timestamps: Tensor of shape (seq_len,)
        - time_gaps: Tensor of shape (seq_len,)
        - mask: Tensor of shape (seq_len,) indicating valid vs padded steps
    """

    def __init__(self, sequences: List[Dict[str, torch.Tensor]]):
        self.sequences = sequences

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.sequences[idx]


def collate_ehr_sequences(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to pad variable-length longitudinal EHR sequences into fixed-size batches.

    Args:
        batch: List of patient sequence dictionaries.

    Returns:
        Batched Tensors for observations, timestamps, time_gaps, and masks.
    """
    raise NotImplementedError("Batch padding and collation function stub.")
