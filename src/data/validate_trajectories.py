"""
MIMIC-IV Continuous-Time Trajectory Validation & Quality Control Script.

Performs 18 strict quality control checks on the generated ICU trajectories
in `data/processed/trajectories/icu_trajectories.pkl` and `trajectory_preview.csv`.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


PROCESSED_DIR = Path("data/processed/trajectories")
PICKLE_PATH = PROCESSED_DIR / "icu_trajectories.pkl"
PREVIEW_CSV_PATH = PROCESSED_DIR / "trajectory_preview.csv"
OUTPUT_REPORT_JSON = Path("experiments/phase3_qc_metrics.json")


def validate_trajectories() -> Dict[str, Any]:
    """
    Executes the 18 Quality Control checks on saved trajectories.
    """
    if not PICKLE_PATH.exists():
        raise FileNotFoundError(f"Pickle file not found at '{PICKLE_PATH}'. Run sequence_builder.py first.")

    print(f"Loading pickle trajectory dataset from: {PICKLE_PATH}")
    with open(PICKLE_PATH, "rb") as f:
        trajectories = pickle.load(f)

    total_trajectories = len(trajectories)
    unique_subjects = len(set(t["subject_id"] for t in trajectories))
    unique_stays = len(set(t["stay_id"] for t in trajectories))

    qc_results = {}

    # 1. No negative T values
    neg_T_count = sum(bool(np.any(t["T"] < 0.0)) for t in trajectories if len(t["T"]) > 0)
    qc_results["1_no_negative_T"] = {
        "passed": bool(neg_T_count == 0),
        "failed_trajectories": int(neg_T_count),
    }

    # 2. T is monotonically increasing
    non_mono_count = 0
    for t in trajectories:
        if len(t["T"]) > 1:
            diffs = np.diff(t["T"])
            if bool(np.any(diffs <= 0.0)):  # Strictly increasing
                non_mono_count += 1
    qc_results["2_T_monotonically_increasing"] = {
        "passed": bool(non_mono_count == 0),
        "failed_trajectories": int(non_mono_count),
    }

    # 3. DeltaT[0] == 0
    delta0_fail = sum(
        bool(len(t["DeltaT"]) > 0 and t["DeltaT"][0] != 0.0) for t in trajectories
    )
    qc_results["3_DeltaT_0_is_zero"] = {
        "passed": bool(delta0_fail == 0),
        "failed_trajectories": int(delta0_fail),
    }

    # 4. DeltaT[i] >= 0
    neg_delta_count = sum(
        bool(np.any(t["DeltaT"] < 0.0)) for t in trajectories if len(t["DeltaT"]) > 0
    )
    qc_results["4_no_negative_DeltaT"] = {
        "passed": bool(neg_delta_count == 0),
        "failed_trajectories": int(neg_delta_count),
    }

    # 5. X and M have identical dimensions [N, 5]
    shape_mismatch_count = 0
    for t in trajectories:
        if t["X"].shape != t["M"].shape or t["X"].ndim != 2 or t["X"].shape[1] != 5:
            shape_mismatch_count += 1
    qc_results["5_X_and_M_identical_shape"] = {
        "passed": bool(shape_mismatch_count == 0),
        "failed_trajectories": int(shape_mismatch_count),
    }

    # 6. T and DeltaT have length N identical to X
    length_mismatch_count = 0
    for t in trajectories:
        N = len(t["X"])
        if len(t["T"]) != N or len(t["DeltaT"]) != N or len(t["bp_modality"]) != N:
            length_mismatch_count += 1
    qc_results["6_T_DeltaT_length_matches_X"] = {
        "passed": bool(length_mismatch_count == 0),
        "failed_trajectories": int(length_mismatch_count),
    }

    # 7. M contains only 0 and 1
    invalid_M_count = 0
    for t in trajectories:
        if len(t["M"]) > 0:
            unique_vals = set(np.unique(t["M"]))
            if not unique_vals.issubset({0, 1}):
                invalid_M_count += 1
    qc_results["7_M_binary_only"] = {
        "passed": bool(invalid_M_count == 0),
        "failed_trajectories": int(invalid_M_count),
    }

    # 8. Every observed value (~isnan(X)) has M==1
    observed_M_mismatch = 0
    for t in trajectories:
        if len(t["X"]) > 0:
            obs_mask = ~np.isnan(t["X"])
            if not np.array_equal(obs_mask.astype(np.uint8), (t["M"] == 1).astype(np.uint8)):
                observed_M_mismatch += 1
    qc_results["8_observed_value_has_M1"] = {
        "passed": bool(observed_M_mismatch == 0),
        "failed_trajectories": int(observed_M_mismatch),
    }

    # 9. Every missing value (isnan(X)) has M==0
    missing_M_mismatch = 0
    for t in trajectories:
        if len(t["X"]) > 0:
            nan_mask = np.isnan(t["X"])
            if not np.array_equal(nan_mask.astype(np.uint8), (t["M"] == 0).astype(np.uint8)):
                missing_M_mismatch += 1
    qc_results["9_missing_value_has_M0"] = {
        "passed": bool(missing_M_mismatch == 0),
        "failed_trajectories": int(missing_M_mismatch),
    }

    # 10. No observations outside ICU stay bounds [intime, outtime]
    outside_bounds_count = 0
    for t in trajectories:
        if len(t["T"]) > 0:
            if bool(np.any(t["T"] < 0.0)) or bool(np.any(t["T"] > (t["los_hours"] + 1e-4))):
                outside_bounds_count += 1
    qc_results["10_no_obs_outside_icu_stay"] = {
        "passed": bool(outside_bounds_count == 0),
        "failed_trajectories": int(outside_bounds_count),
    }

    # 11. No duplicate stay_id + time combinations
    duplicate_time_count = 0
    for t in trajectories:
        if len(t["T"]) > 0:
            if len(t["T"]) != len(np.unique(t["T"])):
                duplicate_time_count += 1
    qc_results["11_no_duplicate_timestamps"] = {
        "passed": bool(duplicate_time_count == 0),
        "failed_trajectories": int(duplicate_time_count),
    }

    # 12. No impossible array shapes
    impossible_shapes = 0
    for t in trajectories:
        if not isinstance(t["X"], np.ndarray) or not isinstance(t["M"], np.ndarray):
            impossible_shapes += 1
    qc_results["12_valid_array_structures"] = {
        "passed": bool(impossible_shapes == 0),
        "failed_trajectories": int(impossible_shapes),
    }

    # 13. No NaNs in T or DeltaT
    nan_in_T_count = sum(
        bool(np.isnan(t["T"]).any() or np.isnan(t["DeltaT"]).any())
        for t in trajectories
        if len(t["T"]) > 0
    )
    qc_results["13_no_nans_in_T_or_DeltaT"] = {
        "passed": bool(nan_in_T_count == 0),
        "failed_trajectories": int(nan_in_T_count),
    }

    # Statistical Aggregations (14 - 18)
    lengths = [len(t["T"]) for t in trajectories]
    durations = [t["los_hours"] for t in trajectories]

    # Compute missingness per feature across all trajectories
    all_M = np.vstack([t["M"] for t in trajectories if len(t["M"]) > 0])
    total_cells = len(all_M)
    observed_per_channel = all_M.sum(axis=0)
    missing_pct_per_channel = {
        "heart_rate": round(float(1.0 - observed_per_channel[0] / total_cells) * 100, 2),
        "respiratory_rate": round(float(1.0 - observed_per_channel[1] / total_cells) * 100, 2),
        "spo2": round(float(1.0 - observed_per_channel[2] / total_cells) * 100, 2),
        "systolic_bp": round(float(1.0 - observed_per_channel[3] / total_cells) * 100, 2),
        "diastolic_bp": round(float(1.0 - observed_per_channel[4] / total_cells) * 100, 2),
    }

    # Collect DeltaT values (excluding DeltaT[0] = 0)
    all_delta_t_hrs = []
    for t in trajectories:
        if len(t["DeltaT"]) > 1:
            all_delta_t_hrs.extend(t["DeltaT"][1:])
    all_delta_t_hrs = np.array(all_delta_t_hrs)

    summary_stats = {
        "14_total_trajectories": total_trajectories,
        "14_total_unique_patients": unique_subjects,
        "14_total_icu_stays": unique_stays,
        "15_trajectory_length_N": {
            "mean": round(float(np.mean(lengths)), 2),
            "median": round(float(np.median(lengths)), 2),
            "min": int(np.min(lengths)),
            "max": int(np.max(lengths)),
            "std": round(float(np.std(lengths)), 2),
            "total_timestamps": int(sum(lengths)),
        },
        "16_icu_duration_hours": {
            "mean": round(float(np.mean(durations)), 2),
            "median": round(float(np.median(durations)), 2),
            "min": round(float(np.min(durations)), 2),
            "max": round(float(np.max(durations)), 2),
            "std": round(float(np.std(durations)), 2),
        },
        "17_missingness_pct_per_feature": missing_pct_per_channel,
        "18_delta_t_hours": {
            "mean": round(float(np.mean(all_delta_t_hrs)), 4),
            "median": round(float(np.median(all_delta_t_hrs)), 4),
            "p25": round(float(np.percentile(all_delta_t_hrs, 25)), 4),
            "p75": round(float(np.percentile(all_delta_t_hrs, 75)), 4),
            "p90": round(float(np.percentile(all_delta_t_hrs, 90)), 4),
            "min": round(float(np.min(all_delta_t_hrs)), 4),
            "max": round(float(np.max(all_delta_t_hrs)), 4),
            "std": round(float(np.std(all_delta_t_hrs)), 4),
        },
        "18_delta_t_minutes": {
            "mean": round(float(np.mean(all_delta_t_hrs) * 60.0), 2),
            "median": round(float(np.median(all_delta_t_hrs) * 60.0), 2),
            "p25": round(float(np.percentile(all_delta_t_hrs, 25) * 60.0), 2),
            "p75": round(float(np.percentile(all_delta_t_hrs, 75) * 60.0), 2),
            "p90": round(float(np.percentile(all_delta_t_hrs, 90) * 60.0), 2),
            "min": round(float(np.min(all_delta_t_hrs) * 60.0), 2),
            "max": round(float(np.max(all_delta_t_hrs) * 60.0), 2),
        },
    }

    full_qc = {
        "qc_checks": qc_results,
        "summary_statistics": summary_stats,
    }

    with open(OUTPUT_REPORT_JSON, "w") as f:
        json.dump(full_qc, f, indent=2)

    return full_qc


def print_qc_report(full_qc: Dict[str, Any]) -> None:
    """
    Prints a clean QC validation summary to stdout.
    """
    print(f"\n{'='*85}")
    print(f"{'PHASE 3 TRAJECTORY QUALITY CONTROL VALIDATION REPORT':^85}")
    print(f"{'='*85}\n")

    qc = full_qc["qc_checks"]
    all_passed = True
    for check_name, res in qc.items():
        status = "PASSED" if res["passed"] else f"FAILED ({res['failed_trajectories']} trajectories)"
        if not res["passed"]:
            all_passed = False
        print(f"[{'PASS' if res['passed'] else 'FAIL'}] {check_name:<45} : {status}")

    print("\n" + "-"*85)
    print(f"Overall Quality Control Status: {'PASSED (ALL 13 AUTOMATED CHECKS SATISFIED)' if all_passed else 'FAILED'}")
    print("-" * 85 + "\n")

    stats = full_qc["summary_statistics"]
    print(f"Total Trajectories   : {stats['14_total_trajectories']}")
    print(f"Unique Patients      : {stats['14_total_unique_patients']}")
    print(f"Unique ICU Stays     : {stats['14_total_icu_stays']}")
    print(f"Total Timestamps (N) : {stats['15_trajectory_length_N']['total_timestamps']}")
    print(f"Trajectory Length N  : Mean={stats['15_trajectory_length_N']['mean']}, Median={stats['15_trajectory_length_N']['median']}, Range=[{stats['15_trajectory_length_N']['min']}, {stats['15_trajectory_length_N']['max']}]")
    print(f"ICU Stay Duration    : Mean={stats['16_icu_duration_hours']['mean']} hrs, Range=[{stats['16_icu_duration_hours']['min']}, {stats['16_icu_duration_hours']['max']}] hrs")
    print(f"DeltaT (Minutes)     : Mean={stats['18_delta_t_minutes']['mean']} min, Median={stats['18_delta_t_minutes']['median']} min, P25={stats['18_delta_t_minutes']['p25']} min, P75={stats['18_delta_t_minutes']['p75']} min")
    print(f"Missingness per Var  : {stats['17_missingness_pct_per_feature']}")
    print("\n" + "="*85 + "\n")


if __name__ == "__main__":
    qc_report = validate_trajectories()
    print_qc_report(qc_report)
