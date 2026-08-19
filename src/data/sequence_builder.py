"""
MIMIC-IV Continuous-Time ICU Trajectory Builder.

Builds longitudinal trajectory records per ICU stay from MIMIC-IV Demo dataset:
- Temporal origin: icustays.intime (t_rel = 0.0 hours)
- Retains irregular observation timing across 5 physiological channels:
  Channel 0: Heart Rate (220045)
  Channel 1: Respiratory Rate (220210)
  Channel 2: SpO2 (220277)
  Channel 3: Systolic Blood Pressure (NIBP 220179 & Arterial 220050)
  Channel 4: Diastolic Blood Pressure (NIBP 220180 & Arterial 220051)
- Combines NIBP and Arterial blood pressure with Arterial preference at exact timestamp collisions.
- Constructs union-of-timestamps matrix X [N, 5], time vector T [N] (hours),
  inter-observation time gap vector DeltaT [N] (hours), binary observation mask M [N, 5],
  and blood pressure modality tracking array.
"""

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def compute_time_gaps(timestamps: np.ndarray) -> np.ndarray:
    """
    Compute inter-observation time gaps (delta_t) for a sequence of timestamps.

    Args:
        timestamps: 1D array of ordered observation timestamps (t_0, t_1, ..., t_N).

    Returns:
        1D array of time gaps where delta_t[0] = 0 and delta_t[i] = t_i - t_{i-1}.
    """
    if len(timestamps) == 0:
        return np.array([], dtype=np.float32)

    delta_t = np.zeros_like(timestamps, dtype=np.float32)
    delta_t[1:] = timestamps[1:] - timestamps[:-1]
    return delta_t



RAW_ICU_DIR = Path("data/raw/mimic_demo/icu")
CHARTEVENTS_PATH = RAW_ICU_DIR / "chartevents.csv.gz"
ICUSTAYS_PATH = RAW_ICU_DIR / "icustays.csv.gz"

PROCESSED_DIR = Path("data/processed/trajectories")
PICKLE_PATH = PROCESSED_DIR / "icu_trajectories.pkl"
PREVIEW_CSV_PATH = PROCESSED_DIR / "trajectory_preview.csv"

# Target MIMIC-IV Item IDs mapping to channel index and modality
ITEM_MAP = {
    220045: {"channel": 0, "name": "heart_rate", "modality": None},
    220210: {"channel": 1, "name": "respiratory_rate", "modality": None},
    220277: {"channel": 2, "name": "spo2", "modality": None},
    220179: {"channel": 3, "name": "systolic_bp", "modality": "non_invasive"},
    220050: {"channel": 3, "name": "systolic_bp", "modality": "arterial"},
    220180: {"channel": 4, "name": "diastolic_bp", "modality": "non_invasive"},
    220051: {"channel": 4, "name": "diastolic_bp", "modality": "arterial"},
}

TARGET_ITEM_IDS = list(ITEM_MAP.keys())


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads icustays and filters chartevents to target vital sign item IDs.
    """
    print(f"Loading ICU stays from: {ICUSTAYS_PATH}")
    icustays = pd.read_csv(ICUSTAYS_PATH)
    icustays["intime"] = pd.to_datetime(icustays["intime"])
    icustays["outtime"] = pd.to_datetime(icustays["outtime"])
    icustays["los_hours"] = (icustays["outtime"] - icustays["intime"]).dt.total_seconds() / 3600.0

    print(f"Loading chart events from: {CHARTEVENTS_PATH}")
    chunks = []
    for chunk in pd.read_csv(
        CHARTEVENTS_PATH,
        chunksize=50000,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum"],
        low_memory=False,
    ):
        filtered = chunk[
            (chunk["itemid"].isin(TARGET_ITEM_IDS)) & (chunk["valuenum"].notna())
        ]
        chunks.append(filtered)

    chartevents = pd.concat(chunks, ignore_index=True)
    chartevents["charttime"] = pd.to_datetime(chartevents["charttime"])

    return icustays, chartevents


def build_stay_trajectory(
    stay_info: pd.Series,
    stay_events: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Builds a single continuous-time trajectory dictionary for an ICU stay.
    """
    stay_id = int(stay_info["stay_id"])
    subject_id = int(stay_info["subject_id"])
    hadm_id = int(stay_info["hadm_id"])
    intime = stay_info["intime"]
    outtime = stay_info["outtime"]
    los_hours = float(stay_info["los_hours"])

    # Filter events strictly within ICU stay interval [intime, outtime]
    events = stay_events[
        (stay_events["charttime"] >= intime) & (stay_events["charttime"] <= outtime)
    ].copy()

    if events.empty:
        return {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": stay_id,
            "intime": intime.strftime("%Y-%m-%d %H:%M:%S"),
            "outtime": outtime.strftime("%Y-%m-%d %H:%M:%S"),
            "los_hours": los_hours,
            "X": np.empty((0, 5), dtype=np.float32),
            "T": np.empty((0,), dtype=np.float64),
            "DeltaT": np.empty((0,), dtype=np.float64),
            "M": np.empty((0, 5), dtype=np.uint8),
            "bp_modality": np.empty((0,), dtype=object),
        }

    # Handle BP duplicate resolution at exact same timestamp
    # If both arterial and non-invasive exist at same charttime for same channel, prefer arterial
    events["channel"] = events["itemid"].map(lambda x: ITEM_MAP[x]["channel"])
    events["modality"] = events["itemid"].map(lambda x: ITEM_MAP[x]["modality"])

    # Sort by charttime, channel, and modality so 'arterial' comes before 'non_invasive'
    # 'arterial' < 'non_invasive' alphabetically, so first after sort is arterial
    events["modality_order"] = events["modality"].map(
        {"arterial": 0, "non_invasive": 1, None: 2}
    )
    events = events.sort_values(by=["charttime", "channel", "modality_order"])

    # Deduplicate exact stay_id + charttime + channel collisions
    events = events.drop_duplicates(subset=["charttime", "channel"], keep="first")

    # Union of unique observation timestamps for this stay
    unique_timestamps = np.sort(events["charttime"].unique())
    N = len(unique_timestamps)

    # Compute relative time in hours from intime
    T = np.array([(ts - intime).total_seconds() / 3600.0 for ts in unique_timestamps], dtype=np.float64)

    # Compute DeltaT in hours
    DeltaT = np.zeros(N, dtype=np.float64)
    if N > 1:
        DeltaT[1:] = T[1:] - T[:-1]

    # Initialize matrices
    X = np.full((N, 5), np.nan, dtype=np.float32)
    M = np.zeros((N, 5), dtype=np.uint8)
    bp_modality = np.full((N,), "none", dtype=object)

    # Map timestamp to index
    ts_to_idx = {ts: i for i, ts in enumerate(unique_timestamps)}

    for _, row in events.iterrows():
        idx = ts_to_idx[row["charttime"]]
        ch = int(row["channel"])
        val = float(row["valuenum"])
        mod = row["modality"]

        X[idx, ch] = val
        M[idx, ch] = 1

        # Track BP modality if blood pressure event
        if ch in [3, 4] and mod is not None:
            # If already arterial or setting new, arterial takes precedence
            if bp_modality[idx] != "arterial":
                bp_modality[idx] = mod

    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": stay_id,
        "intime": intime.strftime("%Y-%m-%d %H:%M:%S"),
        "outtime": outtime.strftime("%Y-%m-%d %H:%M:%S"),
        "los_hours": los_hours,
        "X": X,
        "T": T,
        "DeltaT": DeltaT,
        "M": M,
        "bp_modality": bp_modality,
    }


def create_csv_preview(trajectories: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Creates a human-readable CSV dataframe preview for inspection.
    """
    rows = []
    for traj in trajectories:
        N = len(traj["T"])
        for i in range(N):
            rows.append({
                "subject_id": traj["subject_id"],
                "hadm_id": traj["hadm_id"],
                "stay_id": traj["stay_id"],
                "time_hours": round(float(traj["T"][i]), 4),
                "delta_time_hours": round(float(traj["DeltaT"][i]), 4),
                "heart_rate": round(float(traj["X"][i, 0]), 2) if traj["M"][i, 0] == 1 else None,
                "respiratory_rate": round(float(traj["X"][i, 1]), 2) if traj["M"][i, 1] == 1 else None,
                "spo2": round(float(traj["X"][i, 2]), 2) if traj["M"][i, 2] == 1 else None,
                "systolic_bp": round(float(traj["X"][i, 3]), 2) if traj["M"][i, 3] == 1 else None,
                "diastolic_bp": round(float(traj["X"][i, 4]), 2) if traj["M"][i, 4] == 1 else None,
                "heart_rate_observed": int(traj["M"][i, 0]),
                "respiratory_rate_observed": int(traj["M"][i, 1]),
                "spo2_observed": int(traj["M"][i, 2]),
                "systolic_bp_observed": int(traj["M"][i, 3]),
                "diastolic_bp_observed": int(traj["M"][i, 4]),
                "bp_modality": str(traj["bp_modality"][i]),
            })

    df = pd.DataFrame(rows)
    return df


def main():
    print("Starting Phase 3 ICU Trajectory Building...")
    icustays, chartevents = load_raw_data()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    trajectories = []

    total_stays = len(icustays)
    print(f"Building trajectories for {total_stays} ICU stays...")

    for idx, stay_info in icustays.iterrows():
        stay_id = stay_info["stay_id"]
        stay_events = chartevents[chartevents["stay_id"] == stay_id]
        traj = build_stay_trajectory(stay_info, stay_events)
        trajectories.append(traj)

    print(f"Saving primary structured pickle dataset to: {PICKLE_PATH}")
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(trajectories, f)

    print(f"Generating human-readable CSV preview at: {PREVIEW_CSV_PATH}")
    preview_df = create_csv_preview(trajectories)
    preview_df.to_csv(PREVIEW_CSV_PATH, index=False)

    print(f"Phase 3 Trajectory Building Complete!")
    print(f"Total Trajectories Saved: {len(trajectories)}")
    print(f"Total Observation Timestamps Generated: {len(preview_df)}")


if __name__ == "__main__":
    main()
