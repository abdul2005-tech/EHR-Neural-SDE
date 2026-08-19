"""
Evaluation & Benchmarking Script for Discrete First-Order HMM Baseline (DG-HMM).

Fits `DGHMMBaseline` on training trajectories, saves parameters to `outputs/dg_hmm_baseline/`,
evaluates state occupancy, transition matrix entropy, emission parameters, real vs synthetic
feature distributions, correlation matrices, and generates synthetic trajectories for comparison.
"""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
import pandas as pd
from src.models.dg_hmm import DGHMMBaseline


TRAIN_PATH = Path("data/processed/datasets/train.pkl")
TEST_PATH = Path("data/processed/datasets/test.pkl")
SCALER_PATH = Path("data/processed/scaler.json")
OUTPUT_DIR = Path("outputs/dg_hmm_baseline")
REPORT_JSON_PATH = Path("experiments/phase5_qc_metrics.json")

FEATURE_NAMES = ["heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp"]


def load_datasets() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Loads train, test, and scaler datasets."""
    with open(TRAIN_PATH, "rb") as f:
        train_trajs = pickle.load(f)
    with open(TEST_PATH, "rb") as f:
        test_trajs = pickle.load(f)
    with open(SCALER_PATH, "r") as f:
        scaler = json.load(f)
    return train_trajs, test_trajs, scaler


def calculate_transition_entropy(transition_matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Computes Shannon entropy (in bits) for each row of the transition matrix:
    H_i = - sum_j A_{i,j} log2(A_{i,j})
    """
    K = transition_matrix.shape[0]
    row_entropies = np.zeros(K)
    for i in range(K):
        row = transition_matrix[i]
        # Avoid log2(0) by masking > 0
        valid = row > 0
        row_entropies[i] = -np.sum(row[valid] * np.log2(row[valid]))
    avg_entropy = float(np.mean(row_entropies))
    return row_entropies, avg_entropy


def evaluate_dg_hmm() -> Dict[str, Any]:
    """
    Fits and evaluates DGHMMBaseline.
    """
    train_trajs, test_trajs, scaler = load_datasets()

    print(f"Fitting DGHMMBaseline (K=8) on {len(train_trajs)} training trajectories...")
    model = DGHMMBaseline(K=8, feature_dim=5, random_state=42)
    model.fit(train_trajs)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save(OUTPUT_DIR)

    # 1. State Occupancy on Training Data
    all_train_states = []
    for traj in train_trajs:
        if len(traj["X"]) > 0:
            z_seq = model.predict_states(traj)
            all_train_states.extend(z_seq)

    all_train_states = np.array(all_train_states)
    unique_k, counts_k = np.unique(all_train_states, return_counts=True)
    occupancy_pct = {
        int(k): round(float(count / len(all_train_states)) * 100, 2)
        for k, count in zip(unique_k, counts_k)
    }

    # 2. Transition Matrix Entropy
    row_entropies, avg_entropy = calculate_transition_entropy(model.transition_matrix)

    # 3. Generate Synthetic Trajectories matching train length distribution
    train_lengths = [len(t["T"]) for t in train_trajs if len(t["T"]) > 0]
    synthetic_trajs_X = []
    for N_len in train_lengths:
        X_syn, Z_syn = model.generate_trajectory(N_len)
        synthetic_trajs_X.append(X_syn)

    synthetic_X_all = np.vstack(synthetic_trajs_X)

    # 4. Compare Real vs Synthetic Feature Means and Stds (in Normalized Space)
    real_train_X_obs = []
    for t in train_trajs:
        if len(t["X"]) > 0:
            for i in range(len(t["X"])):
                for d in range(5):
                    if t["M"][i, d] == 1:
                        real_train_X_obs.append((d, t["X"][i, d]))

    real_df = pd.DataFrame(real_train_X_obs, columns=["feat_idx", "value"])

    feature_comp = {}
    for d, feat_name in enumerate(FEATURE_NAMES):
        real_vals = real_df[real_df["feat_idx"] == d]["value"].values
        syn_vals = synthetic_X_all[:, d]

        feature_comp[feat_name] = {
            "real_train_mean": round(float(np.mean(real_vals)), 4),
            "real_train_std": round(float(np.std(real_vals)), 4),
            "synthetic_mean": round(float(np.mean(syn_vals)), 4),
            "synthetic_std": round(float(np.std(syn_vals)), 4),
        }

    # 5. Correlation Matrices
    # Real observed pairwise correlation
    real_dense = np.zeros((len(real_df), 5))
    # Synthetic correlation
    syn_corr = np.corrcoef(synthetic_X_all, rowvar=False)

    eval_results = {
        "K": model.K,
        "feature_dim": model.feature_dim,
        "initial_probs": model.initial_probs.tolist(),
        "transition_matrix": model.transition_matrix.tolist(),
        "emission_means": model.means.tolist(),
        "emission_stds": model.stds.tolist(),
        "state_occupancy_pct": occupancy_pct,
        "row_transition_entropies_bits": [round(float(h), 4) for h in row_entropies],
        "avg_transition_entropy_bits": round(avg_entropy, 4),
        "feature_comparison_normalized": feature_comp,
        "synthetic_correlation_matrix": syn_corr.tolist(),
    }

    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(eval_results, f, indent=2)

    return eval_results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Prints evaluation summary report to stdout."""
    print(f"\n{'='*85}")
    print(f"{'DISCRETE FIRST-ORDER HMM BASELINE (DG-HMM) EVALUATION':^85}")
    print(f"{'='*85}\n")

    print(f"Latent States (K)            : {results['K']}")
    print(f"Average Transition Entropy   : {results['avg_transition_entropy_bits']} bits (Max possible: {np.log2(results['K']):.2f} bits)")
    print(f"State Occupancy (%)          : {results['state_occupancy_pct']}")
    print("\n" + "-"*85)
    print(f"{'Feature':<18} | {'Real Train Mean':<15} | {'Syn Mean':<12} | {'Real Train Std':<15} | {'Syn Std':<10}")
    print("-" * 85)

    for feat_name, metrics in results["feature_comparison_normalized"].items():
        print(
            f"{feat_name:<18} | {metrics['real_train_mean']:<15.4f} | {metrics['synthetic_mean']:<12.4f} | "
            f"{metrics['real_train_std']:<15.4f} | {metrics['synthetic_std']:<10.4f}"
        )

    print("-" * 85 + "\n")


if __name__ == "__main__":
    res = evaluate_dg_hmm()
    print_evaluation_report(res)
