"""
Phase 11 Comprehensive Ablation Evaluation Suite.

Evaluates trained candidate models (11A Baseline, 11B Temporal, 11C Rate, 11F Probabilistic, 11G Combined)
against Validation data (for model selection) and Test data (for final research benchmark).

Generates:
- Validation Scorecard
- Final Test Benchmark Summary Table
- CSV ablation report: experiments/phase11_ablation_results.csv
- Six publication-quality plots in experiments/
"""

import argparse
import csv
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import ks_2samp, wasserstein_distance
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE


SANITY_RANGES = {
    "heart_rate": (20.0, 250.0),
    "respiratory_rate": (2.0, 80.0),
    "spo2": (50.0, 100.0),
    "systolic_bp": (40.0, 250.0),
    "diastolic_bp": (20.0, 150.0),
}


def load_dataset_trajectories(
    data_path: str,
    scaler_path: str = "data/processed/scaler.json",
    max_stays: int = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[str]]:
    """Loads dataset trajectories and denormalized flat feature arrays."""
    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    with open(data_path, "rb") as f:
        trajectories = pickle.load(f)

    if max_stays is not None and max_stays > 0:
        trajectories = trajectories[:max_stays]

    flat_dict = {name: [] for name in feature_names}
    for traj in trajectories:
        X_phys = denormalize(traj["X"], mean, std)
        M = traj["M"]
        for i, name in enumerate(feature_names):
            obs_mask = M[:, i] == 1.0
            flat_dict[name].extend(X_phys[obs_mask, i])

    flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in flat_dict.items()}
    return trajectories, flat_arr, feature_names


def generate_candidate_futures(
    model: EHRNeuralSDE,
    trajectories: List[Dict[str, Any]],
    scaler_path: str = "data/processed/scaler.json",
    n_samples: int = 20,
    base_seed: int = 42,
) -> Dict[int, List[np.ndarray]]:
    """Generates n_samples stochastic futures for each trajectory using candidate model."""
    device = next(model.parameters()).device
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    synth_by_stay = {}

    for idx, traj in enumerate(trajectories):
        stay_id = traj["stay_id"]
        X_real = traj["X"]
        T_real = traj["T"]
        DeltaT_real = traj["DeltaT"]
        M_real = traj["M"]

        X_0 = torch.tensor(X_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
        M_0 = torch.tensor(M_real[0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
        T_0 = torch.tensor(T_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
        DeltaT_0 = torch.tensor(DeltaT_real[0:1], dtype=torch.float32).unsqueeze(0).to(device)
        T_target = torch.tensor(T_real, dtype=torch.float32).unsqueeze(0).to(device)

        synth_by_stay[stay_id] = []

        with torch.no_grad():
            h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
            z_0 = model.latent_projection(h_0)

            for s_idx in range(n_samples):
                gen_s = torch.Generator(device=device).manual_seed(base_seed + idx * 50 + s_idx * 1000)
                z_traj = model.sde.integrate(z0=z_0, times=T_target, generator=gen_s, enable_noise=True)

                if model.use_probabilistic_decoder:
                    mu, _ = model.decoder(z_traj)
                    pred_norm = mu.squeeze(0).cpu().numpy()
                else:
                    X_hat = model.decoder(z_traj)
                    pred_norm = X_hat.squeeze(0).cpu().numpy()

                X_phys = denormalize(pred_norm, mean, std)
                synth_by_stay[stay_id].append(X_phys)

    return synth_by_stay


def compute_comprehensive_metrics(
    real_flat: Dict[str, np.ndarray],
    real_trajectories: List[Dict[str, Any]],
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Computes all distribution, correlation, temporal, diversity, and plausibility metrics."""
    # 1. Flatten synthetic observations
    synth_flat = {name: [] for name in feature_names}
    for stay_id, s_list in synth_by_stay.items():
        for X_s in s_list:
            for i, name in enumerate(feature_names):
                synth_flat[name].extend(X_s[:, i])

    synth_flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in synth_flat.items()}

    # 2. Plausibility Check
    in_range_counts = 0
    total_synth_obs = 0
    for f_name in feature_names:
        low, high = SANITY_RANGES[f_name]
        arr = synth_flat_arr[f_name]
        in_range_counts += ((arr >= low) & (arr <= high)).sum()
        total_synth_obs += len(arr)
    plausibility_pct = (in_range_counts / max(total_synth_obs, 1)) * 100.0

    # 3. Distributional Metrics & Distances
    wasserstein_list = []
    ks_stat_list = []
    synth_stds = {}

    for f_name in feature_names:
        r_arr = real_flat[f_name]
        s_arr = synth_flat_arr[f_name]

        w_d = wasserstein_distance(r_arr, s_arr)
        ks_s = ks_2samp(r_arr, s_arr).statistic

        wasserstein_list.append(w_d)
        ks_stat_list.append(ks_s)
        synth_stds[f_name] = float(np.std(s_arr))

    mean_wasserstein = float(np.mean(wasserstein_list))
    mean_ks_stat = float(np.mean(ks_stat_list))

    # 4. Correlation Matrices & Pairwise Error
    min_r = min(len(real_flat[f]) for f in feature_names)
    min_s = min(len(synth_flat_arr[f]) for f in feature_names)

    real_mat = np.column_stack([real_flat[f][:min_r] for f in feature_names])
    synth_mat = np.column_stack([synth_flat_arr[f][:min_s] for f in feature_names])

    corr_real = np.corrcoef(real_mat, rowvar=False)
    corr_synth = np.corrcoef(synth_mat, rowvar=False)
    corr_diff_mean = float(np.abs(corr_real - corr_synth).mean())

    # 5. Temporal Metrics
    real_diffs = []
    synth_diffs = []
    synth_autocorr1 = []

    for traj in real_trajectories:
        X_p = traj["X"]
        for i in range(len(feature_names)):
            d = np.diff(X_p[:, i])
            real_diffs.extend(d)

    for stay_id, s_list in synth_by_stay.items():
        for X_s in s_list:
            for i in range(len(feature_names)):
                d = np.diff(X_s[:, i])
                synth_diffs.extend(d)
                if len(X_s) > 3 and np.std(X_s[:-1, i]) > 1e-5:
                    ac1 = np.corrcoef(X_s[:-1, i], X_s[1:, i])[0, 1]
                    if not np.isnan(ac1):
                        synth_autocorr1.append(ac1)

    real_diff_std = float(np.std(real_diffs))
    synth_diff_std = float(np.std(synth_diffs))
    temporal_volatility_error = float(abs(real_diff_std - synth_diff_std))
    mean_synth_autocorr1 = float(np.mean(synth_autocorr1)) if synth_autocorr1 else 0.0

    # 6. Stochastic Multi-Future Diversity
    pairwise_rmses = []
    pairwise_maes = []
    for stay_id, s_list in synth_by_stay.items():
        K = len(s_list)
        for i in range(K):
            for j in range(i + 1, K):
                pairwise_rmses.append(np.sqrt(np.mean((s_list[i] - s_list[j]) ** 2)))
                pairwise_maes.append(np.mean(np.abs(s_list[i] - s_list[j])))

    diversity_rmse = float(np.mean(pairwise_rmses)) if pairwise_rmses else 0.0
    diversity_mae = float(np.mean(pairwise_maes)) if pairwise_maes else 0.0

    return {
        "plausibility_pct": plausibility_pct,
        "mean_wasserstein": mean_wasserstein,
        "mean_ks_stat": mean_ks_stat,
        "correlation_error": corr_diff_mean,
        "synth_diff_std": synth_diff_std,
        "temporal_volatility_error": temporal_volatility_error,
        "synth_autocorr1": mean_synth_autocorr1,
        "diversity_rmse": diversity_rmse,
        "diversity_mae": diversity_mae,
        "synth_stds": synth_stds,
        "corr_real": corr_real,
        "corr_synth": corr_synth,
    }


def evaluate_ablations(
    split: str = "val",
    checkpoint_dir: str = "outputs/checkpoints",
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
    n_samples: int = 20,
    max_stays: int = None,
    csv_out: str = None,
    is_quick: bool = False,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_quick:
        header_title = f"=== QUICK VALIDATION SCREENING EXPERIMENT (Split: {split.upper()}, Stays: {max_stays}, Futures: {n_samples}) ==="
    else:
        header_title = f"=== Phase 11 Evaluation Suite (Split: {split.upper()}) ==="
    print(header_title)

    data_path = f"data/processed/datasets/{split}.pkl"
    real_trajectories, real_flat, feature_names = load_dataset_trajectories(data_path, scaler_path, max_stays=max_stays)
    print(f"Loaded {len(real_trajectories)} trajectories for evaluation.")

    candidates = [
        ("Phase 10 Baseline", "best_model.pt", False),
        ("11A Baseline Control", "phase11_baseline.pt", False),
        ("11B Temporal Difference", "phase11_temporal.pt", False),
        ("11C Rate Loss", "phase11_rate.pt", False),
        ("11F Probabilistic Decoder", "phase11_probabilistic.pt", True),
        ("11G Combined Model", "phase11_combined.pt", True),
    ]

    results_table = []

    print("\n" + "=" * 120)
    print(
        f"{'Model Candidate':<26} | {'Plausible %':<11} | {'Wasserstein':<11} | {'KS Stat':<8} | "
        f"{'Corr Error':<10} | {r'std($\Delta X$)':<12} | {'Lag-1 AC':<9} | {'Diversity RMSE':<14}"
    )
    print("=" * 120)

    for label, ckpt_file, is_prob in candidates:
        ckpt_path = Path(checkpoint_dir) / ckpt_file
        if not ckpt_path.exists():
            print(f"Skipping candidate '{label}' (checkpoint not found: {ckpt_path})")
            continue

        model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=is_prob).to(device)
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        synth_by_stay = generate_candidate_futures(model, real_trajectories, scaler_path, n_samples=n_samples)
        metrics = compute_comprehensive_metrics(real_flat, real_trajectories, synth_by_stay, feature_names)

        print(
            f"{label:<26} | {metrics['plausibility_pct']:10.2f}% | {metrics['mean_wasserstein']:11.4f} | "
            f"{metrics['mean_ks_stat']:8.4f} | {metrics['correlation_error']:10.4f} | "
            f"{metrics['synth_diff_std']:12.4f} | {metrics['synth_autocorr1']:9.4f} | {metrics['diversity_rmse']:14.4f}"
        )

        row = {
            "label": label,
            "checkpoint": ckpt_file,
            "plausibility_pct": metrics["plausibility_pct"],
            "mean_wasserstein": metrics["mean_wasserstein"],
            "mean_ks_stat": metrics["mean_ks_stat"],
            "correlation_error": metrics["correlation_error"],
            "synth_diff_std": metrics["synth_diff_std"],
            "temporal_volatility_error": metrics["temporal_volatility_error"],
            "synth_autocorr1": metrics["synth_autocorr1"],
            "diversity_rmse": metrics["diversity_rmse"],
            "diversity_mae": metrics["diversity_mae"],
            "stds": metrics["synth_stds"],
            "corr_real": metrics["corr_real"],
            "corr_synth": metrics["corr_synth"],
        }
        results_table.append(row)

    print("=" * 120 + "\n")

    # Save CSV Report
    if csv_out:
        csv_path = Path(csv_out)
    elif is_quick:
        csv_path = Path(experiments_dir) / "phase11_quick_validation.csv"
    else:
        csv_path = Path(experiments_dir) / "phase11_ablation_results.csv"

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model_label", "checkpoint", "plausibility_pct", "wasserstein_distance",
            "ks_statistic", "correlation_error", "synth_diff_std", "temporal_volatility_error",
            "lag1_autocorr", "diversity_rmse", "diversity_mae"
        ])
        for r in results_table:
            writer.writerow([
                r["label"], r["checkpoint"], f"{r['plausibility_pct']:.2f}",
                f"{r['mean_wasserstein']:.4f}", f"{r['mean_ks_stat']:.4f}",
                f"{r['correlation_error']:.4f}", f"{r['synth_diff_std']:.4f}",
                f"{r['temporal_volatility_error']:.4f}", f"{r['synth_autocorr1']:.4f}",
                f"{r['diversity_rmse']:.4f}", f"{r['diversity_mae']:.4f}"
            ])
    print(f"Saved CSV report to: {csv_path}")

    # Generate Comparison Plot
    if results_table:
        labels = [r["label"] for r in results_table]
        wassersteins = [r["mean_wasserstein"] for r in results_table]
        corr_errs = [r["correlation_error"] for r in results_table]
        diversities = [r["diversity_rmse"] for r in results_table]

        # 1. Model Comparison Summary Chart
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        axes[0].barh(labels, wassersteins, color="skyblue")
        axes[0].set_title("Distribution Distance (Wasserstein)")
        axes[0].set_xlabel("Distance (Lower is better)")
        axes[0].grid(True, linestyle="--", alpha=0.5)

        axes[1].barh(labels, corr_errs, color="salmon")
        axes[1].set_title("Correlation Structure Error")
        axes[1].set_xlabel("Abs Pairwise Error (Lower is better)")
        axes[1].grid(True, linestyle="--", alpha=0.5)

        axes[2].barh(labels, diversities, color="lightgreen")
        axes[2].set_title("Stochastic Multi-Future Diversity")
        axes[2].set_xlabel("Pairwise RMSE (Higher is better)")
        axes[2].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plot_name = "phase11_quick_model_comparison.png" if is_quick else "phase11_model_comparison.png"
        model_plot_path = Path(experiments_dir) / plot_name
        plt.savefig(model_plot_path, dpi=300)
        plt.close()
        print(f"Saved model comparison plot to: {model_plot_path}")

        # 2. Correlation Comparison Figure
        if len(results_table) >= 2:
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            axes = axes.flatten()

            # Plot Real Correlation
            sns.heatmap(results_table[0]["corr_real"], ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                        xticklabels=[f[:4] for f in feature_names], yticklabels=[f[:4] for f in feature_names])
            axes[0].set_title("Real Data Correlation")

            for idx, r in enumerate(results_table[:5]):
                ax = axes[idx + 1]
                sns.heatmap(r["corr_synth"], ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                            xticklabels=[f[:4] for f in feature_names], yticklabels=[f[:4] for f in feature_names])
                ax.set_title(f"{r['label']} Correlation")

            plt.tight_layout()
            corr_plot_name = "phase11_quick_correlation_comparison.png" if is_quick else "phase11_correlation_comparison.png"
            corr_plot_path = Path(experiments_dir) / corr_plot_name
            plt.savefig(corr_plot_path, dpi=300)
            plt.close()
            print(f"Saved correlation comparison plot to: {corr_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--n-samples", type=int, default=20, help="Number of stochastic futures per trajectory")
    parser.add_argument("--max-stays", type=int, default=None, help="Maximum number of ICU stays to evaluate")
    parser.add_argument("--quick", action="store_true", help="Run quick validation screening (5 stays, 5 futures)")
    parser.add_argument("--csv-out", type=str, default=None, help="Custom output CSV path")
    args = parser.parse_args()

    if args.quick:
        n_samples = 5
        max_stays = 5
        split = "val"
        csv_out = args.csv_out or "experiments/phase11_quick_validation.csv"
        is_quick = True
    else:
        n_samples = args.n_samples
        max_stays = args.max_stays
        split = args.split
        csv_out = args.csv_out
        is_quick = False

    evaluate_ablations(
        split=split,
        n_samples=n_samples,
        max_stays=max_stays,
        csv_out=csv_out,
        is_quick=is_quick,
    )

