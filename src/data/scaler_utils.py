"""
Feature Scaler Utilities for Physiological Observations.

Provides functions to load parameters from data/processed/scaler.json, normalize raw observations,
and denormalize predicted physiological outputs back to original clinical units.
"""

import json
from pathlib import Path
from typing import Dict, Tuple, Union
import numpy as np
import torch


def load_scaler_params(scaler_path: Union[str, Path] = "data/processed/scaler.json") -> Dict[str, Any]:
    """
    Loads normalization parameters (mean and std) from scaler.json.

    Args:
        scaler_path: Path to scaler.json file

    Returns:
        Dictionary containing 'feature_names', 'mean', 'std', 'observed_counts', 'random_seed'
    """
    scaler_path = Path(scaler_path)
    if not scaler_path.exists():
        raise FileNotFoundError(f"Scaler parameter file not found at: {scaler_path}")

    with open(scaler_path, "r") as f:
        scaler_data = json.load(f)

    return scaler_data


def normalize(
    X: Union[torch.Tensor, np.ndarray],
    mean: Union[torch.Tensor, np.ndarray, list],
    std: Union[torch.Tensor, np.ndarray, list],
    eps: float = 1e-8,
) -> Union[torch.Tensor, np.ndarray]:
    """
    Normalizes raw physiological values: (X - mean) / std.

    Args:
        X: Physiological observation tensor/array of shape (..., D)
        mean: Channel-wise mean vector of shape (D,)
        std: Channel-wise standard deviation vector of shape (D,)
        eps: Small constant to avoid zero division

    Returns:
        Normalized observations of same type and shape as X
    """
    if isinstance(X, torch.Tensor):
        mean_t = torch.tensor(mean, device=X.device, dtype=X.dtype)
        std_t = torch.tensor(std, device=X.device, dtype=X.dtype)
        return (X - mean_t) / (std_t + eps)
    else:
        mean_arr = np.array(mean, dtype=X.dtype)
        std_arr = np.array(std, dtype=X.dtype)
        return (X - mean_arr) / (std_arr + eps)


def denormalize(
    X_hat: Union[torch.Tensor, np.ndarray],
    mean: Union[torch.Tensor, np.ndarray, list],
    std: Union[torch.Tensor, np.ndarray, list],
) -> Union[torch.Tensor, np.ndarray]:
    """
    Denormalizes predictions back to original clinical units: X_hat * std + mean.

    Args:
        X_hat: Normalized predictions tensor/array of shape (..., D)
        mean: Channel-wise mean vector of shape (D,)
        std: Channel-wise standard deviation vector of shape (D,)

    Returns:
        Denormalized predictions of same type and shape as X_hat
    """
    if isinstance(X_hat, torch.Tensor):
        mean_t = torch.tensor(mean, device=X_hat.device, dtype=X_hat.dtype)
        std_t = torch.tensor(std, device=X_hat.device, dtype=X_hat.dtype)
        return X_hat * std_t + mean_t
    else:
        mean_arr = np.array(mean, dtype=X_hat.dtype)
        std_arr = np.array(std, dtype=X_hat.dtype)
        return X_hat * std_arr + mean_arr
