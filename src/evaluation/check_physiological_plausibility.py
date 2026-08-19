"""
Physiological Plausibility Auditor for Synthetic EHR Trajectories.

Evaluates generated physiological values against broad physiological sanity ranges:
- Heart Rate        : [20, 250] bpm
- Respiratory Rate  : [2, 80] breaths/min
- SpO2              : [50, 100] %
- Systolic BP       : [40, 250] mmHg
- Diastolic BP      : [20, 150] mmHg

Computes percentages inside/outside range, min, max, median, 1st and 99th percentiles
without applying any artificial pre-clipping.
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from src.data.scaler_utils import load_scaler_params, denormalize


# Broad physiological sanity bounds
SANITY_RANGES = {
    "heart_rate": (20.0, 250.0),
    "respiratory_rate": (2.0, 80.0),
    "spo2": (50.0, 100.0),
    "systolic_bp": (40.0, 250.0),
    "diastolic_bp": (20.0, 150.0),
}


def audit_physiological_plausibility(
    generated_dir: str = "outputs/generated",
    test_pkl_path: str = "data/processed/datasets/test.pkl",
    scaler_path: str = "data/processed/scaler.json",
) -> Dict[str, Any]:
    print("=== Phase 10: Auditing Physiological Plausibility ===")

    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    gen_base = Path(generated_dir)
    sample_files = list(gen_base.glob("test_stay_*/sample_*.pkl"))

    if not sample_files:
        raise FileNotFoundError(f"No generated sample files found in: {generated_dir}")

    print(f"Found {len(sample_files)} generated sample PKL files.")

    # 1. Collect all synthetic observations
    synth_feature_values = {name: [] for name in feature_names}

    for s_file in sample_files:
        with open(s_file, "rb") as f:
            data = pickle.load(f)
        X_synth = data["X_synthetic"]  # [N, 5] physical units

        for f_idx, f_name in enumerate(feature_names):
            vals = X_synth[:, f_idx]
            synth_feature_values[f_name].extend(vals)

    # 2. Collect all real test observations (denormalized to physical units)
    real_feature_values = {name: [] for name in feature_names}
    with open(test_pkl_path, "rb") as f:
        real_trajectories = pickle.load(f)

    for traj in real_trajectories:
        X_real_norm = traj["X"]  # [N, 5]
        M_real = traj["M"]      # [N, 5]
        X_real_phys = denormalize(X_real_norm, mean, std)

        for f_idx, f_name in enumerate(feature_names):
            # Only consider observed values
            obs_mask = M_real[:, f_idx] == 1.0
            obs_vals = X_real_phys[obs_mask, f_idx]
            real_feature_values[f_name].extend(obs_vals)

    # 3. Compute Plausibility Statistics
    report = {}

    print("\n" + "=" * 90)
    print(f"{'Feature':<18} | {'Dataset':<9} | {'% In Range':<10} | {'% Out Range':<11} | {'Min':<7} | {'Max':<7} | {'P1':<7} | {'P50':<7} | {'P99':<7}")
    print("=" * 90)

    for f_name in feature_names:
        low_bound, high_bound = SANITY_RANGES[f_name]

        for dtype, vals_list in [("Synthetic", synth_feature_values[f_name]), ("Real Test", real_feature_values[f_name])]:
            vals_arr = np.array(vals_list, dtype=np.float64)
            total_cnt = len(vals_arr)

            in_mask = (vals_arr >= low_bound) & (vals_arr <= high_bound)
            pct_in = (in_mask.sum() / total_cnt) * 100.0
            pct_out = 100.0 - pct_in

            val_min = np.min(vals_arr)
            val_max = np.max(vals_arr)
            p1 = np.percentile(vals_arr, 1)
            p50 = np.median(vals_arr)
            p99 = np.percentile(vals_arr, 99)

            print(
                f"{f_name:<18} | {dtype:<9} | {pct_in:9.2f}% | {pct_out:10.2f}% | "
                f"{val_min:7.1f} | {val_max:7.1f} | {p1:7.1f} | {p50:7.1f} | {p99:7.1f}"
            )

            report[f"{f_name}_{dtype.lower().replace(' ', '_')}"] = {
                "total_count": total_cnt,
                "pct_in_range": pct_in,
                "pct_out_of_range": pct_out,
                "min": float(val_min),
                "max": float(val_max),
                "p1": float(p1),
                "p50": float(p50),
                "p99": float(p99),
            }

    print("=" * 90 + "\n")
    return report


if __name__ == "__main__":
    audit_physiological_plausibility()
