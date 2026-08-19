"""
MIMIC-IV Demo Feature Exploration & Candidate Item Analysis Script.

Explores candidate MIMIC-IV item IDs for 5 key target vital signs:
1. Heart Rate
2. Respiratory Rate
3. Oxygen Saturation (SpO2)
4. Systolic Blood Pressure (NIBP & Arterial)
5. Diastolic Blood Pressure (NIBP & Arterial)

Analyzes d_items.csv.gz, chartevents.csv.gz, and icustays.csv.gz.
Preserves irregular observation timing without resampling, forward-filling, or interpolation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd


RAW_ICU_DIR = Path("data/raw/mimic_demo/icu")
D_ITEMS_PATH = RAW_ICU_DIR / "d_items.csv.gz"
CHARTEVENTS_PATH = RAW_ICU_DIR / "chartevents.csv.gz"
ICUSTAYS_PATH = RAW_ICU_DIR / "icustays.csv.gz"
OUTPUT_JSON = Path("experiments/phase2_feature_exploration.json")


# Target variable keywords for dictionary search
TARGET_KEYWORDS = {
    "Heart Rate": ["heart rate"],
    "Respiratory Rate": ["respiratory rate"],
    "SpO2": ["spo2", "o2 saturation", "oxygen saturation"],
    "Systolic BP": ["systolic"],
    "Diastolic BP": ["diastolic"],
}

# Standard recommended MIMIC-IV Metavision item IDs for initial ICU vital sign dataset
RECOMMENDED_ITEM_IDS = {
    220045: "Heart Rate",
    220210: "Respiratory Rate",
    220277: "SpO2",
    220179: "Non Invasive Blood Pressure systolic",
    220180: "Non Invasive Blood Pressure diastolic",
    220050: "Arterial Blood Pressure systolic",
    220051: "Arterial Blood Pressure diastolic",
}


def search_d_items() -> pd.DataFrame:
    """
    Searches d_items.csv.gz for all candidate items matching target vital sign keywords.
    """
    d_items = pd.read_csv(D_ITEMS_PATH)
    results = []

    for var_name, keywords in TARGET_KEYWORDS.items():
        pattern = "|".join(keywords)
        matches = d_items[d_items["label"].str.contains(pattern, case=False, na=False)].copy()
        matches["target_variable"] = var_name
        results.append(matches)

    candidates = pd.concat(results, ignore_index=True)
    return candidates


def inspect_chartevents_counts(candidate_itemids: List[int]) -> pd.DataFrame:
    """
    Scans chartevents.csv.gz to compute exact observation counts, null statistics,
    and value ranges for candidate item IDs.
    """
    stats = []

    # Read chartevents in chunks for memory safety
    chunks = []
    for chunk in pd.read_csv(
        CHARTEVENTS_PATH,
        chunksize=50000,
        usecols=["subject_id", "hadm_id", "stay_id", "charttime", "itemid", "valuenum", "valueuom", "warning"],
        low_memory=False,
    ):
        filtered = chunk[chunk["itemid"].isin(candidate_itemids)]
        chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    return df


def analyze_feature_statistics(events_df: pd.DataFrame, d_items: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Computes summary statistics for each candidate item ID.
    """
    grouped_stats = []

    for itemid, group in events_df.groupby("itemid"):
        item_info = d_items[d_items["itemid"] == itemid]
        label = item_info["label"].values[0] if not item_info.empty else "Unknown"
        category = item_info["category"].values[0] if not item_info.empty else "Unknown"
        unit_dict = item_info["unitname"].values[0] if not item_info.empty else "Unknown"

        total_obs = len(group)
        null_valuenum = int(group["valuenum"].isna().sum())
        null_pct = round((null_valuenum / total_obs) * 100, 2) if total_obs > 0 else 0.0

        valid_vals = group["valuenum"].dropna()
        units_used = group["valueuom"].dropna().unique().tolist()

        min_val = float(valid_vals.min()) if not valid_vals.empty else None
        max_val = float(valid_vals.max()) if not valid_vals.empty else None
        mean_val = round(float(valid_vals.mean()), 2) if not valid_vals.empty else None
        median_val = round(float(valid_vals.median()), 2) if not valid_vals.empty else None

        grouped_stats.append({
            "itemid": int(itemid),
            "label": label,
            "category": category,
            "unit_dict": str(unit_dict),
            "units_used": units_used,
            "total_observations": total_obs,
            "null_valuenum_count": null_valuenum,
            "null_valuenum_pct": null_pct,
            "min_val": min_val,
            "max_val": max_val,
            "mean_val": mean_val,
            "median_val": median_val,
        })

    return sorted(grouped_stats, key=lambda x: x["total_observations"], reverse=True)


def analyze_time_gaps_and_stays(
    events_df: pd.DataFrame,
    icustays_df: pd.DataFrame,
    recommended_ids: List[int],
) -> Dict[str, Any]:
    """
    Computes time-gap statistics (delta_t) and ICU stay coverage for recommended vital signs.
    """
    df = events_df[events_df["itemid"].isin(recommended_ids)].copy()
    df["charttime"] = pd.to_datetime(df["charttime"])

    # Join with icustays to verify ICU stay bounds
    df = df.merge(icustays_df[["stay_id", "intime", "outtime"]], on="stay_id", how="left")
    df["intime"] = pd.to_datetime(df["intime"])
    df["outtime"] = pd.to_datetime(df["outtime"])

    # Check observations within ICU window
    df["in_window"] = (df["charttime"] >= df["intime"]) & (df["charttime"] <= df["outtime"])
    outside_window_count = int((~df["in_window"]).sum())

    # Sort chronologically per stay to calculate inter-observation delta_t
    df = df.sort_values(by=["stay_id", "charttime"])

    # Duplicate check: same stay_id, itemid, and charttime
    duplicates_count = int(df.duplicated(subset=["stay_id", "itemid", "charttime"]).sum())

    # Overall delta_t across all recommended vital signs per stay
    df["prev_charttime"] = df.groupby("stay_id")["charttime"].shift(1)
    df["delta_t_min"] = (df["charttime"] - df["prev_charttime"]).dt.total_seconds() / 60.0

    delta_t_valid = df["delta_t_min"].dropna()
    delta_t_valid = delta_t_valid[delta_t_valid > 0]  # Positive gaps

    # Per-variable time gaps
    per_var_delta_t = {}
    for itemid in recommended_ids:
        var_name = RECOMMENDED_ITEM_IDS.get(itemid, str(itemid))
        sub = df[df["itemid"] == itemid].copy()
        sub["prev_ts"] = sub.groupby("stay_id")["charttime"].shift(1)
        sub["gap_min"] = (sub["charttime"] - sub["prev_ts"]).dt.total_seconds() / 60.0
        gaps = sub["gap_min"].dropna()
        gaps = gaps[gaps > 0]

        if not gaps.empty:
            per_var_delta_t[var_name] = {
                "count": int(len(gaps)),
                "mean_min": round(float(gaps.mean()), 2),
                "median_min": round(float(gaps.median()), 2),
                "p25_min": round(float(gaps.quantile(0.25)), 2),
                "p75_min": round(float(gaps.quantile(0.75)), 2),
                "p90_min": round(float(gaps.quantile(0.90)), 2),
                "min_min": round(float(gaps.min()), 2),
                "max_min": round(float(gaps.max()), 2),
            }

    # Per stay coverage statistics
    stay_summary = []
    for stay_id, stay_group in df.groupby("stay_id"):
        obs_count = len(stay_group)
        first_time = stay_group["charttime"].min().strftime("%Y-%m-%d %H:%M:%S")
        last_time = stay_group["charttime"].max().strftime("%Y-%m-%d %H:%M:%S")
        duration_hrs = round((stay_group["charttime"].max() - stay_group["charttime"].min()).total_seconds() / 3600.0, 2)

        var_counts = {
            RECOMMENDED_ITEM_IDS.get(itemid, str(itemid)): int((stay_group["itemid"] == itemid).sum())
            for itemid in recommended_ids
        }

        stay_summary.append({
            "stay_id": int(stay_id),
            "total_observations": obs_count,
            "first_observation": first_time,
            "last_observation": last_time,
            "duration_hours": duration_hrs,
            "var_counts": var_counts,
        })

    return {
        "total_recommended_observations": len(df),
        "outside_icu_window_count": outside_window_count,
        "duplicate_measurements_count": duplicates_count,
        "overall_delta_t_minutes": {
            "mean": round(float(delta_t_valid.mean()), 2),
            "median": round(float(delta_t_valid.median()), 2),
            "p25": round(float(delta_t_valid.quantile(0.25)), 2),
            "p75": round(float(delta_t_valid.quantile(0.75)), 2),
            "p90": round(float(delta_t_valid.quantile(0.90)), 2),
            "p95": round(float(delta_t_valid.quantile(0.95)), 2),
            "min": round(float(delta_t_valid.min()), 2),
            "max": round(float(delta_t_valid.max()), 2),
            "std": round(float(delta_t_valid.std()), 2),
        },
        "per_variable_delta_t_minutes": per_var_delta_t,
        "stay_sample": stay_summary[:5],  # Sample of 5 ICU stays
    }


def main():
    print(f"Phase 2 Feature Exploration starting...")
    print(f"Reading dictionary: {D_ITEMS_PATH}")
    d_items = pd.read_csv(D_ITEMS_PATH)
    candidates = search_d_items()

    candidate_ids = candidates["itemid"].tolist()
    print(f"Found {len(candidate_ids)} candidate item IDs in d_items.csv.gz.")

    print(f"Filtering chartevents.csv.gz for candidate items...")
    events_df = inspect_chartevents_counts(candidate_ids)
    print(f"Extracted {len(events_df)} candidate observation records.")

    stats = analyze_feature_statistics(events_df, d_items)

    print(f"Reading icustays: {ICUSTAYS_PATH}")
    icustays_df = pd.read_csv(ICUSTAYS_PATH)

    rec_ids = list(RECOMMENDED_ITEM_IDS.keys())
    time_gap_analysis = analyze_time_gaps_and_stays(events_df, icustays_df, rec_ids)

    output_data = {
        "candidate_items_summary": stats,
        "recommended_item_ids": RECOMMENDED_ITEM_IDS,
        "time_gap_and_stay_analysis": time_gap_analysis,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nFeature exploration finished! Results saved to '{OUTPUT_JSON}'.")


if __name__ == "__main__":
    main()
