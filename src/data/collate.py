"""
Custom Collate Function for Variable-Length Continuous-Time ICU Trajectories.

Handles batching, zero-padding, sequence length tracking, and separation between:
1. Clinical Observation Mask M (whether feature j was measured at timestamp i)
2. Sequence Padding Mask (whether timestamp i is valid vs padded in batch)
"""

from typing import Any, Dict, List
import torch


def collate_icu_trajectories(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collates a list of variable-length trajectory samples into a padded batch dictionary.

    Args:
        batch: List of trajectory sample dictionaries returned by ICUTrajectoryDataset.

    Returns:
        Batched dictionary containing:
            - "X_padded": Tensor [B, max_len, 5]
            - "T_padded": Tensor [B, max_len]
            - "DeltaT_padded": Tensor [B, max_len]
            - "M_padded": Tensor [B, max_len, 5] (Clinical observation mask)
            - "sequence_lengths": Tensor [B] (int64 actual trajectory lengths N)
            - "padding_mask": Tensor [B, max_len] (1.0 for valid step, 0.0 for padded step)
            - "subject_ids": Tensor [B]
            - "hadm_ids": Tensor [B]
            - "stay_ids": Tensor [B]
            - "los_hours": Tensor [B]
            - "bp_modalities": List of modality arrays per sample
    """
    B = len(batch)
    lengths = [sample["X"].shape[0] for sample in batch]
    max_len = max(lengths)
    num_features = 5

    # Allocate padded batch tensors
    X_padded = torch.zeros((B, max_len, num_features), dtype=torch.float32)
    T_padded = torch.zeros((B, max_len), dtype=torch.float32)
    DeltaT_padded = torch.zeros((B, max_len), dtype=torch.float32)
    M_padded = torch.zeros((B, max_len, num_features), dtype=torch.float32)
    padding_mask = torch.zeros((B, max_len), dtype=torch.float32)

    subject_ids = torch.tensor([s["subject_id"] for s in batch], dtype=torch.int64)
    hadm_ids = torch.tensor([s["hadm_id"] for s in batch], dtype=torch.int64)
    stay_ids = torch.tensor([s["stay_id"] for s in batch], dtype=torch.int64)
    los_hours = torch.tensor([s["los_hours"] for s in batch], dtype=torch.float32)
    sequence_lengths = torch.tensor(lengths, dtype=torch.int64)
    bp_modalities = [s["bp_modality"] for s in batch]

    for i, sample in enumerate(batch):
        N = lengths[i]
        if N > 0:
            X_padded[i, :N, :] = sample["X"]
            T_padded[i, :N] = sample["T"]
            DeltaT_padded[i, :N] = sample["DeltaT"]
            M_padded[i, :N, :] = sample["M"]
            padding_mask[i, :N] = 1.0  # Valid sequence steps

    return {
        "X_padded": X_padded,
        "T_padded": T_padded,
        "DeltaT_padded": DeltaT_padded,
        "M_padded": M_padded,
        "sequence_lengths": sequence_lengths,
        "padding_mask": padding_mask,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
        "los_hours": los_hours,
        "bp_modalities": bp_modalities,
    }
