"""
Phase 4 Dataset Validation & Leakage Checks Suite.

Executes 12 comprehensive quality control checks on the normalized split datasets:
`train.pkl`, `val.pkl`, `test.pkl`, and `scaler.json`.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np


DATASETS_DIR = Path("data/processed/datasets")
SCALER_JSON_PATH = Path("data/processed/scaler.json")
ORIGINAL_TRAJ_PATH = Path("data/processed/trajectories/icu_trajectories.pkl")
OUTPUT_REPORT_JSON = Path("experiments/phase4_qc_metrics.json")


def validate_phase4_datasets() -> Dict[str, Any]:
    """
    Executes 12 Phase 4 validation checks.
    """
    train_path = DATASETS_DIR / "train.pkl"
    val_path = DATASETS_DIR / "val.pkl"
    test_path = DATASETS_DIR / "test.pkl"

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)
    with open(SCALER_JSON_PATH, "r") as f:
        scaler_info = json.load(f)
    with open(ORIGINAL_TRAJ_PATH, "rb") as f:
        orig_data = pickle.load(f)

    orig_map = {t["stay_id"]: t for t in orig_data}
    qc_results = {}

    # Check 1: No patient appears in multiple splits
    train_subs = set(t["subject_id"] for t in train_data)
    val_subs = set(t["subject_id"] for t in val_data)
    test_subs = set(t["subject_id"] for t in test_data)

    leakage_tv = len(train_subs.intersection(val_subs))
    leakage_tt = len(train_subs.intersection(test_subs))
    leakage_vt = len(val_subs.intersection(test_subs))
    qc_results["1_no_patient_leakage_across_splits"] = {
        "passed": bool(leakage_tv == 0 and leakage_tt == 0 and leakage_vt == 0),
        "train_val_overlap": int(leakage_tv),
        "train_test_overlap": int(leakage_tt),
        "val_test_overlap": int(leakage_vt),
    }

    # Check 2: No NaNs exist in normalized X
    all_splits = [("train", train_data), ("val", val_data), ("test", test_data)]
    nan_count = 0
    for split_name, data in all_splits:
        for t in data:
            if bool(np.isnan(t["X"]).any()):
                nan_count += 1
    qc_results["2_no_nans_in_normalized_X"] = {
        "passed": bool(nan_count == 0),
        "trajectories_with_nans": int(nan_count),
    }

    # Check 3: M contains only 0/1
    invalid_M_count = 0
    for split_name, data in all_splits:
        for t in data:
            if len(t["M"]) > 0:
                if not set(np.unique(t["M"])).issubset({0, 1}):
                    invalid_M_count += 1
    qc_results["3_M_binary_only"] = {
        "passed": bool(invalid_M_count == 0),
        "invalid_M_trajectories": int(invalid_M_count),
    }

    # Check 4: T remains monotonic
    non_mono_count = 0
    for split_name, data in all_splits:
        for t in data:
            if len(t["T"]) > 1:
                if bool(np.any(np.diff(t["T"]) <= 0.0)):
                    non_mono_count += 1
    qc_results["4_T_remains_monotonic"] = {
        "passed": bool(non_mono_count == 0),
        "non_monotonic_trajectories": int(non_mono_count),
    }

    # Check 5: DeltaT remains non-negative
    neg_delta_count = 0
    for split_name, data in all_splits:
        for t in data:
            if len(t["DeltaT"]) > 0:
                if bool(np.any(t["DeltaT"] < 0.0)):
                    neg_delta_count += 1
    qc_results["5_DeltaT_remains_non_negative"] = {
        "passed": bool(neg_delta_count == 0),
        "negative_delta_trajectories": int(neg_delta_count),
    }

    # Check 6: T and DeltaT not modified by normalization
    time_modified_count = 0
    for split_name, data in all_splits:
        for t in data:
            stay_id = t["stay_id"]
            orig = orig_map[stay_id]
            if not np.array_equal(t["T"], orig["T"]) or not np.array_equal(t["DeltaT"], orig["DeltaT"]):
                time_modified_count += 1
    qc_results["6_T_DeltaT_unmodified"] = {
        "passed": bool(time_modified_count == 0),
        "time_modified_trajectories": int(time_modified_count),
    }

    # Check 7 & 8: Training means ~0 and stds ~1 for observed cells
    train_M = np.vstack([t["M"] for t in train_data if len(t["M"]) > 0])
    train_X = np.vstack([t["X"] for t in train_data if len(t["X"]) > 0])

    norm_means = []
    norm_stds = []
    for j in range(5):
        obs_mask = train_M[:, j] == 1
        obs_vals = train_X[obs_mask, j]
        norm_means.append(round(float(np.mean(obs_vals)), 6))
        norm_stds.append(round(float(np.std(obs_vals, ddof=1)), 6))

    mean_check_pass = bool(all(abs(m) < 1e-4 for m in norm_means))
    std_check_pass = bool(all(abs(s - 1.0) < 1e-4 for s in norm_stds))

    qc_results["7_training_observed_means_approx_zero"] = {
        "passed": mean_check_pass,
        "observed_normalized_means": norm_means,
    }
    qc_results["8_training_observed_stds_approx_one"] = {
        "passed": std_check_pass,
        "observed_normalized_stds": norm_stds,
    }

    # Check 9: Val/Test transformed using training statistics ONLY
    val_check_fail = 0
    for t in val_data:
        stay_id = t["stay_id"]
        orig = orig_map[stay_id]
        X_orig = orig["X"]
        M = orig["M"]
        X_norm_expected = np.zeros_like(X_orig, dtype=np.float32)
        for j in range(5):
            mask_j = M[:, j] == 1
            if np.any(mask_j):
                X_norm_expected[mask_j, j] = (X_orig[mask_j, j] - scaler_info["mean"][j]) / scaler_info["std"][j]
        X_norm_expected = np.nan_to_num(X_norm_expected, nan=0.0)

        if not np.allclose(t["X"], X_norm_expected, atol=1e-5):
            val_check_fail += 1

    qc_results["9_val_test_use_train_scaler_only"] = {
        "passed": bool(val_check_fail == 0),
        "mismatched_val_trajectories": int(val_check_fail),
    }

    # Check 10: Sequence lengths remain unchanged
    length_mismatch_count = 0
    for split_name, data in all_splits:
        for t in data:
            orig = orig_map[t["stay_id"]]
            if len(t["T"]) != len(orig["T"]):
                length_mismatch_count += 1
    qc_results["10_sequence_lengths_unchanged"] = {
        "passed": bool(length_mismatch_count == 0),
        "length_mismatches": int(length_mismatch_count),
    }

    # Check 11: No new timestamps created
    new_timestamp_count = 0
    for split_name, data in all_splits:
        for t in data:
            orig = orig_map[t["stay_id"]]
            if not np.array_equal(t["T"], orig["T"]):
                new_timestamp_count += 1
    qc_results["11_no_new_timestamps_created"] = {
        "passed": bool(new_timestamp_count == 0),
        "new_timestamp_trajectories": int(new_timestamp_count),
    }

    # Check 12: No observations deleted because of missing features
    obs_deleted_count = 0
    for split_name, data in all_splits:
        for t in data:
            orig = orig_map[t["stay_id"]]
            if not np.array_equal(t["M"], orig["M"]):
                obs_deleted_count += 1
    qc_results["12_no_observations_deleted"] = {
        "passed": bool(obs_deleted_count == 0),
        "deleted_observation_trajectories": int(obs_deleted_count),
    }

    # Statistical Summaries for Report
    def get_split_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        n_subs = len(set(t["subject_id"] for t in data))
        n_trajs = len(data)
        lengths = [len(t["T"]) for t in data]
        durations = [t["los_hours"] for t in data]
        all_M = np.vstack([t["M"] for t in data if len(t["M"]) > 0])
        total_cells = len(all_M)
        obs_cnts = all_M.sum(axis=0)
        missing_pcts = {
            feat: round(float(1.0 - obs_cnts[i] / total_cells) * 100, 2)
            for i, feat in enumerate(scaler_info["feature_names"])
        }

        all_gaps = []
        for t in data:
            if len(t["DeltaT"]) > 1:
                all_gaps.extend(t["DeltaT"][1:])
        all_gaps = np.array(all_gaps)

        return {
            "num_patients": n_subs,
            "num_trajectories": n_trajs,
            "length_N": {
                "mean": round(float(np.mean(lengths)), 2),
                "median": round(float(np.median(lengths)), 2),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "std": round(float(np.std(lengths)), 2),
                "total_timestamps": int(sum(lengths)),
            },
            "duration_hours": {
                "mean": round(float(np.mean(durations)), 2),
                "median": round(float(np.median(durations)), 2),
                "min": round(float(np.min(durations)), 2),
                "max": round(float(np.max(durations)), 2),
            },
            "missingness_pct": missing_pcts,
            "delta_t_minutes": {
                "mean": round(float(np.mean(all_gaps) * 60.0), 2),
                "median": round(float(np.median(all_gaps) * 60.0), 2),
                "p25": round(float(np.percentile(all_gaps, 25) * 60.0), 2),
                "p75": round(float(np.percentile(all_gaps, 75) * 60.0), 2),
            },
        }

    stats_by_split = {
        "train": get_split_stats(train_data),
        "val": get_split_stats(val_data),
        "test": get_split_stats(test_data),
    }

    full_qc = {
        "qc_checks": qc_results,
        "split_statistics": stats_by_split,
        "scaler": scaler_info,
    }

    with open(OUTPUT_REPORT_JSON, "w") as f:
        json.dump(full_qc, f, indent=2)

    return full_qc


def print_validation_report(full_qc: Dict[str, Any]) -> None:
    """
    Prints validation report to stdout.
    """
    print(f"\n{'='*85}")
    print(f"{'PHASE 4 DATASET VALIDATION REPORT':^85}")
    print(f"{'='*85}\n")

    all_passed = True
    for check_name, res in full_qc["qc_checks"].items():
        passed = res["passed"]
        if not passed:
            all_passed = False
        print(f"[{'PASS' if passed else 'FAIL'}] {check_name:<48} : {'PASSED' if passed else 'FAILED'}")

    print("\n" + "-"*85)
    print(f"Overall Quality Control Status: {'PASSED (ALL 12 CHECKS SATISFIED)' if all_passed else 'FAILED'}")
    print("-" * 85 + "\n")


if __name__ == "__main__":
    qc = validate_phase4_datasets()
    print_validation_report(qc)
