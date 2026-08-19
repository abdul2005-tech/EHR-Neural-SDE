"""
MIMIC-IV Demo Dataset Comprehensive Inspection Script.

Recursively inspects MIMIC-IV Demo tables in `data/raw/mimic_demo/`,
extracting metadata, row counts, unique subject/hadm/stay IDs, timestamp min/max,
null value percentages, and key clinical variable details.

Outputs detailed findings to `experiments/phase1_inspection_data.json` and prints a summary report.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd


RAW_DIR = Path("data/raw/mimic_demo")
OUTPUT_JSON = Path("experiments/phase1_inspection_data.json")


def inspect_file(file_path: Path) -> Dict[str, Any]:
    """
    Inspects a single dataset table file without loading entire file into memory.

    Args:
        file_path: Path to .csv or .csv.gz file.

    Returns:
        Dictionary of table metadata, unique identifier counts, and timestamp stats.
    """
    table_name = file_path.name.replace(".csv.gz", "").replace(".csv", "")
    module_name = file_path.parent.name

    # Read first 10 rows for schema and data types
    sample_df = pd.read_csv(file_path, nrows=10)
    columns = list(sample_df.columns)
    dtypes = {col: str(sample_df[col].dtype) for col in columns}

    total_rows = 0
    unique_subjects = set()
    unique_hadms = set()
    unique_stays = set()

    timestamp_cols = [
        col for col in columns
        if any(ts_kw in col.lower() for ts_kw in ["time", "date", "dob", "dod"])
    ]

    ts_min_max = {col: [None, None] for col in timestamp_cols}
    null_counts = {col: 0 for col in columns}

    # Memory-safe chunked iteration
    for chunk in pd.read_csv(file_path, chunksize=10000, low_memory=False):
        total_rows += len(chunk)

        if "subject_id" in chunk.columns:
            unique_subjects.update(chunk["subject_id"].dropna().astype(int).unique())
        if "hadm_id" in chunk.columns:
            unique_hadms.update(chunk["hadm_id"].dropna().astype(int).unique())
        if "stay_id" in chunk.columns:
            unique_stays.update(chunk["stay_id"].dropna().astype(int).unique())

        for col in columns:
            null_counts[col] += int(chunk[col].isna().sum())

        for col in timestamp_cols:
            parsed_ts = pd.to_datetime(chunk[col], errors="coerce").dropna()
            if not parsed_ts.empty:
                c_min = parsed_ts.min().strftime("%Y-%m-%d %H:%M:%S")
                c_max = parsed_ts.max().strftime("%Y-%m-%d %H:%M:%S")
                if ts_min_max[col][0] is None or c_min < ts_min_max[col][0]:
                    ts_min_max[col][0] = c_min
                if ts_min_max[col][1] is None or c_max > ts_min_max[col][1]:
                    ts_min_max[col][1] = c_max

    null_pcts = {
        col: round((count / total_rows) * 100, 2) if total_rows > 0 else 0.0
        for col, count in null_counts.items()
    }

    sample_records = sample_df.head(3).astype(object).where(pd.notnull(sample_df.head(3)), None).to_dict(orient="records")

    return {
        "module": module_name,
        "table_name": table_name,
        "filename": file_path.name,
        "total_rows": total_rows,
        "columns": columns,
        "dtypes": dtypes,
        "unique_subjects": len(unique_subjects),
        "unique_hadms": len(unique_hadms),
        "unique_stays": len(unique_stays),
        "timestamp_cols": timestamp_cols,
        "ts_min_max": ts_min_max,
        "null_pcts": null_pcts,
        "sample": sample_records,
    }


def print_summary_report(results: List[Dict[str, Any]]) -> None:
    """
    Prints a formatted summary report to stdout.
    """
    print(f"\n{'='*95}")
    print(f"{'MIMIC-IV DEMO DATASET SUMMARY':^95}")
    print(f"{'='*95}\n")
    print(f"{'Module':<12} | {'Table Name':<22} | {'Rows':<8} | {'Subjects':<8} | {'Admissions':<10} | {'Stays':<6}")
    print("-" * 95)

    for r in results:
        print(
            f"{r['module']:<12} | {r['table_name']:<22} | {r['total_rows']:<8} | "
            f"{r['unique_subjects']:<8} | {r['unique_hadms']:<10} | {r['unique_stays']:<6}"
        )

    print("-" * 95 + "\n")


def main():
    if not RAW_DIR.exists():
        print(f"Error: Path '{RAW_DIR}' does not exist.")
        return

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    all_meta = []

    print(f"Inspecting MIMIC-IV Demo directory: {RAW_DIR}")
    for root, _, files in os.walk(RAW_DIR):
        for f in sorted(files):
            if f.endswith(".csv") or f.endswith(".csv.gz"):
                file_path = Path(root) / f
                print(f"Inspecting {file_path.name}...")
                meta = inspect_file(file_path)
                all_meta.append(meta)

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_meta, f, indent=2)

    print_summary_report(all_meta)
    print(f"Inspection complete! Detailed metadata saved to '{OUTPUT_JSON}'.")


if __name__ == "__main__":
    main()
