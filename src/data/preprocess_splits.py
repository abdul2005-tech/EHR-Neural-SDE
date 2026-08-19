"""
Leakage-Safe Patient-Level Dataset Splitting & Normalization Module.

Splits ICU trajectories by unique subject_id into Train (70%), Validation (15%), and Test (15%)
with a fixed random seed to guarantee zero data leakage.

Normalizes continuous vital signs using statistics derived EXCLUSIVELY from training set observations:
    x_norm = (x - mean_train) / std_train   (where M == 1)
    x_norm = 0.0                             (where M == 0)

Saves split datasets to `data/processed/datasets/` and scaler params to `data/processed/scaler.json`.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np


INPUT_PKL_PATH = Path("data/processed/trajectories/icu_trajectories.pkl")
DATASETS_DIR = Path("data/processed/datasets")
SCALER_JSON_PATH = Path("data/processed/scaler.json")

FEATURE_NAMES = ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp"]
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_patients(trajectories: List[Dict[str, Any]], seed: int = SEED) -> Tuple[set, set, set]:
    """
    Performs patient-level partition of subject_ids into train, val, and test sets.
    """
    unique_subjects = sorted(list(set(t["subject_id"] for t in trajectories)))
    rng = np.random.RandomState(seed)
    shuffled_subjects = rng.permutation(unique_subjects).tolist()

    n_total = len(shuffled_subjects)
    n_train = int(round(n_total * TRAIN_RATIO))
    n_val = int(round(n_total * VAL_RATIO))

    train_subjects = set(shuffled_subjects[:n_train])
    val_subjects = set(shuffled_subjects[n_train : n_train + n_val])
    test_subjects = set(shuffled_subjects[n_train + n_val :])

    # Assert zero overlap
    assert len(train_subjects.intersection(val_subjects)) == 0
    assert len(train_subjects.intersection(test_subjects)) == 0
    assert len(val_subjects.intersection(test_subjects)) == 0

    return train_subjects, val_subjects, test_subjects


def compute_train_scaler(train_trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes mean, std, and observed cell count for each feature ONLY using training observations.
    """
    means = []
    stds = []
    counts = []

    for feat_idx in range(5):
        observed_vals = []
        for traj in train_trajectories:
            if len(traj["X"]) > 0:
                mask = traj["M"][:, feat_idx] == 1
                vals = traj["X"][mask, feat_idx]
                observed_vals.extend(vals)

        obs_arr = np.array(observed_vals, dtype=np.float64)
        mean_val = float(np.mean(obs_arr))
        std_val = float(np.std(obs_arr, ddof=1))  # Sample standard deviation
        count_val = int(len(obs_arr))

        means.append(round(mean_val, 6))
        stds.append(round(std_val, 6))
        counts.append(count_val)

    scaler_info = {
        "feature_names": FEATURE_NAMES,
        "mean": means,
        "std": stds,
        "observed_counts": counts,
        "random_seed": SEED,
    }
    return scaler_info


def normalize_trajectory(traj: Dict[str, Any], means: List[float], stds: List[float]) -> Dict[str, Any]:
    """
    Normalizes a trajectory's X matrix using training scaler parameters.
    Unobserved values (M == 0) are set to 0.0. Observed values (M == 1) are standardized.
    """
    norm_traj = dict(traj)  # Copy metadata fields
    X_orig = traj["X"]
    M = traj["M"]

    N, D = X_orig.shape
    X_norm = np.zeros((N, D), dtype=np.float32)

    for j in range(D):
        mean_j = means[j]
        std_j = stds[j]
        mask_j = M[:, j] == 1

        if np.any(mask_j):
            X_norm[mask_j, j] = (X_orig[mask_j, j] - mean_j) / std_j

    # Ensure no NaN values exist in X_norm
    X_norm = np.nan_to_num(X_norm, nan=0.0)

    norm_traj["X"] = X_norm
    return norm_traj


def main():
    print(f"Loading trajectory dataset from: {INPUT_PKL_PATH}")
    with open(INPUT_PKL_PATH, "rb") as f:
        trajectories = pickle.load(f)

    print(f"Splitting 100 patients into Train (70%), Val (15%), Test (15%) with seed={SEED}...")
    train_subs, val_subs, test_subs = split_patients(trajectories, seed=SEED)

    train_trajs = [t for t in trajectories if t["subject_id"] in train_subs]
    val_trajs = [t for t in trajectories if t["subject_id"] in val_subs]
    test_trajs = [t for t in trajectories if t["subject_id"] in test_subs]

    print(f"Split Summary:")
    print(f"  Train: {len(train_subs)} patients, {len(train_trajs)} trajectories")
    print(f"  Val  : {len(val_subs)} patients, {len(val_trajs)} trajectories")
    print(f"  Test : {len(test_subs)} patients, {len(test_trajs)} trajectories")

    print("Calculating normalization statistics from training trajectories ONLY...")
    scaler_info = compute_train_scaler(train_trajs)

    print("Scaler Statistics:")
    for name, m, s, cnt in zip(
        scaler_info["feature_names"],
        scaler_info["mean"],
        scaler_info["std"],
        scaler_info["observed_counts"],
    ):
        print(f"  {name:<18}: Mean = {m:10.4f}, Std = {s:10.4f}, Count = {cnt:6d}")

    # Save scaler JSON
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCALER_JSON_PATH, "w") as f:
        json.dump(scaler_info, f, indent=2)
    print(f"Saved scaler statistics to: {SCALER_JSON_PATH}")

    # Normalize all splits using training statistics
    norm_train = [normalize_trajectory(t, scaler_info["mean"], scaler_info["std"]) for t in train_trajs]
    norm_val = [normalize_trajectory(t, scaler_info["mean"], scaler_info["std"]) for t in val_trajs]
    norm_test = [normalize_trajectory(t, scaler_info["mean"], scaler_info["std"]) for t in test_trajs]

    # Save split pickle datasets
    train_path = DATASETS_DIR / "train.pkl"
    val_path = DATASETS_DIR / "val.pkl"
    test_path = DATASETS_DIR / "test.pkl"

    with open(train_path, "wb") as f:
        pickle.dump(norm_train, f)
    with open(val_path, "wb") as f:
        pickle.dump(norm_val, f)
    with open(test_path, "wb") as f:
        pickle.dump(norm_test, f)

    print(f"Saved datasets:")
    print(f"  {train_path} ({len(norm_train)} trajectories)")
    print(f"  {val_path} ({len(norm_val)} trajectories)")
    print(f"  {test_path} ({len(norm_test)} trajectories)")
    print("Phase 4 dataset preprocessing complete!")


if __name__ == "__main__":
    main()
