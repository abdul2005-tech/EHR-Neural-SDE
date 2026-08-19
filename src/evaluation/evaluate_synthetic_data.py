"""
Comprehensive Synthetic Data Evaluation Suite for EHR-Neural-SDE (Phase 10).

Executes a 5-dimensional evaluation comparing real MIMIC-III TEST data vs conditional synthetic trajectories:
1. Distributional Similarity (Percentiles, Mean, Std, Wasserstein distance, KS statistic)
2. Feature Correlation Structure (Pearson correlation matrices & absolute pair differences)
3. Temporal Dynamics (First differences \Delta X_t, Lag-1 & Lag-5 autocorrelations, std(\Delta X))
4. Stochastic Multi-Future Diversity (Pairwise RMSE/MAE, per-timestep variance)
5. Time-Awareness Experiment (Latent displacement vs elapsed integration interval dt)

Generates 5 publication-quality plot artifacts in experiments/:
- phase10_sample_trajectories.png
- phase10_distribution_comparison.png
- phase10_real_vs_synthetic_correlation.png
- phase10_temporal_comparison.png
- phase10_generation_diversity.png
"""

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


def load_real_and_synthetic_data(
    test_pkl_path: str = "data/processed/datasets/test.pkl",
    generated_dir: str = "outputs/generated",
    scaler_path: str = "data/processed/scaler.json",
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[Dict[str, Any]], Dict[int, List[np.ndarray]], List[str]]:
    """
    Loads all real test data observations and all synthetic generated trajectories.

    Returns:
        Tuple of:
            - real_flat: Dict mapping feature_name -> flat numpy array of observed physical values
            - synth_flat: Dict mapping feature_name -> flat numpy array of synthetic physical values
            - real_trajectories: List of raw real test dataset dicts
            - synth_by_stay: Dict mapping stay_id -> list of 20 synthetic physical trajectory arrays [N, 5]
            - feature_names: List of feature names
    """
    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    # 1. Load real test data
    with open(test_pkl_path, "rb") as f:
        real_trajectories = pickle.load(f)

    real_flat = {name: [] for name in feature_names}
    for traj in real_trajectories:
        X_norm = traj["X"]
        M = traj["M"]
        X_phys = denormalize(X_norm, mean, std)
        for i, name in enumerate(feature_names):
            obs_mask = M[:, i] == 1.0
            real_flat[name].extend(X_phys[obs_mask, i])

    real_flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in real_flat.items()}

    # 2. Load synthetic generated data
    gen_base = Path(generated_dir)
    stay_dirs = list(gen_base.glob("test_stay_*"))

    synth_flat = {name: [] for name in feature_names}
    synth_by_stay = {}

    for s_dir in stay_dirs:
        stay_id = int(s_dir.name.replace("test_stay_", ""))
        sample_files = sorted(list(s_dir.glob("sample_*.pkl")))
        synth_by_stay[stay_id] = []

        for sf in sample_files:
            with open(sf, "rb") as f:
                sample_dict = pickle.load(f)
            X_synth = sample_dict["X_synthetic"]  # [N, 5] physical units
            synth_by_stay[stay_id].append(X_synth)

            for i, name in enumerate(feature_names):
                synth_flat[name].extend(X_synth[:, i])

    synth_flat_arr = {name: np.array(vals, dtype=np.float64) for name, vals in synth_flat.items()}

    return real_flat_arr, synth_flat_arr, real_trajectories, synth_by_stay, feature_names


def evaluate_distribution_similarity(
    real_flat: Dict[str, np.ndarray],
    synth_flat: Dict[str, np.ndarray],
    feature_names: List[str],
    experiments_dir: str = "experiments",
) -> Dict[str, Any]:
    print("\n--- 1. Distributional Similarity Evaluation ---")

    metrics = {}
    print(f"{'Feature':<18} | {'Dist':<9} | {'Mean':<7} | {'Std':<7} | {'P1':<6} | {'P25':<6} | {'P50':<6} | {'P75':<6} | {'P99':<6} | {'Wasserstein':<11} | {'KS Stat (p-val)':<15}")
    print("-" * 125)

    for f_name in feature_names:
        r_vals = real_flat[f_name]
        s_vals = synth_flat[f_name]

        w_dist = wasserstein_distance(r_vals, s_vals)
        ks_res = ks_2samp(r_vals, s_vals)

        r_mean, r_std = np.mean(r_vals), np.std(r_vals)
        s_mean, s_std = np.mean(s_vals), np.std(s_vals)

        r_p = np.percentile(r_vals, [1, 25, 50, 75, 99])
        s_p = np.percentile(s_vals, [1, 25, 50, 75, 99])

        print(
            f"{f_name:<18} | Real      | {r_mean:7.2f} | {r_std:7.2f} | {r_p[0]:6.1f} | {r_p[1]:6.1f} | {r_p[2]:6.1f} | {r_p[3]:6.1f} | {r_p[4]:6.1f} | {'-':<11} | {'-':<15}"
        )
        print(
            f"{'':<18} | Synth     | {s_mean:7.2f} | {s_std:7.2f} | {s_p[0]:6.1f} | {s_p[1]:6.1f} | {s_p[2]:6.1f} | {s_p[3]:6.1f} | {s_p[4]:6.1f} | {w_dist:11.4f} | {ks_res.statistic:.4f} ({ks_res.pvalue:.1e})"
        )

        metrics[f_name] = {
            "real_mean": float(r_mean),
            "synth_mean": float(s_mean),
            "real_std": float(r_std),
            "synth_std": float(s_std),
            "wasserstein_distance": float(w_dist),
            "ks_statistic": float(ks_res.statistic),
            "ks_pvalue": float(ks_res.pvalue),
        }

    # Plot Distribution Comparison Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        sns.kdeplot(real_flat[f_name], ax=ax, color="blue", label="Real Test Data", fill=True, alpha=0.3)
        sns.kdeplot(synth_flat[f_name], ax=ax, color="red", label="Synthetic Neural-SDE", fill=True, alpha=0.3)
        ax.set_title(f"Distribution: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Value (Clinical Units)")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    # Hide extra unused subplot
    axes[5].axis("off")

    plt.tight_layout()
    dist_plot_path = Path(experiments_dir) / "phase10_distribution_comparison.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"Saved distribution comparison figure to: {dist_plot_path}")

    return metrics


def evaluate_feature_correlations(
    real_flat: Dict[str, np.ndarray],
    synth_flat: Dict[str, np.ndarray],
    feature_names: List[str],
    experiments_dir: str = "experiments",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    print("\n--- 2. Feature Correlation Structure Evaluation ---")

    # Stack flat arrays into matrices (N_obs, 5)
    # Use minimum matching observation length for aligned array creation
    min_r_len = min(len(real_flat[f]) for f in feature_names)
    min_s_len = min(len(synth_flat[f]) for f in feature_names)

    real_mat = np.column_stack([real_flat[f][:min_r_len] for f in feature_names])
    synth_mat = np.column_stack([synth_flat[f][:min_s_len] for f in feature_names])

    corr_real = np.corrcoef(real_mat, rowvar=False)
    corr_synth = np.corrcoef(synth_mat, rowvar=False)
    corr_diff = np.abs(corr_real - corr_synth)

    clean_names = [f.replace("_", " ").title() for f in feature_names]

    pairs = [
        ("Heart Rate", "Respiratory Rate", 0, 1),
        ("Heart Rate", "SpO2", 0, 2),
        ("Heart Rate", "Systolic BP", 0, 3),
        ("Heart Rate", "Diastolic BP", 0, 4),
        ("Respiratory Rate", "SpO2", 1, 2),
        ("Respiratory Rate", "Systolic BP", 1, 3),
        ("Respiratory Rate", "Diastolic BP", 1, 4),
        ("SpO2", "Systolic BP", 2, 3),
        ("SpO2", "Diastolic BP", 2, 4),
        ("Systolic BP", "Diastolic BP", 3, 4),
    ]

    print(f"{'Feature Pair':<32} | {'Real Corr':<10} | {'Synth Corr':<10} | {'Abs Diff':<10}")
    print("-" * 70)
    for p1, p2, i, j in pairs:
        r_c = corr_real[i, j]
        s_c = corr_synth[i, j]
        diff = corr_diff[i, j]
        print(f"{p1 + ' - ' + p2:<32} | {r_c:10.4f} | {s_c:10.4f} | {diff:10.4f}")

    # Plot Correlation Matrices Heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(corr_real, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, xticklabels=clean_names, yticklabels=clean_names)
    axes[0].set_title("Real Test Data Correlation")

    sns.heatmap(corr_synth, ax=axes[1], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, xticklabels=clean_names, yticklabels=clean_names)
    axes[1].set_title("Synthetic Neural-SDE Correlation")

    sns.heatmap(corr_diff, ax=axes[2], annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=1, xticklabels=clean_names, yticklabels=clean_names)
    axes[2].set_title("Absolute Correlation Difference")

    plt.tight_layout()
    corr_plot_path = Path(experiments_dir) / "phase10_real_vs_synthetic_correlation.png"
    plt.savefig(corr_plot_path, dpi=300)
    plt.close()
    print(f"Saved correlation structure figure to: {corr_plot_path}")

    return corr_real, corr_synth, corr_diff


def evaluate_temporal_dynamics(
    real_trajectories: List[Dict[str, Any]],
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
) -> Dict[str, Any]:
    print("\n--- 3. Temporal Dynamics Evaluation ---")

    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    # 1. First differences \Delta X_t
    real_diffs = {f: [] for f in feature_names}
    synth_diffs = {f: [] for f in feature_names}

    # Autocorrelations
    real_autocorr_lag1 = {f: [] for f in feature_names}
    synth_autocorr_lag1 = {f: [] for f in feature_names}
    real_autocorr_lag5 = {f: [] for f in feature_names}
    synth_autocorr_lag5 = {f: [] for f in feature_names}

    def compute_autocorr(series: np.ndarray, lag: int) -> float:
        if len(series) <= lag + 2:
            return np.nan
        y1 = series[:-lag]
        y2 = series[lag:]
        if np.std(y1) < 1e-6 or np.std(y2) < 1e-6:
            return 0.0
        return float(np.corrcoef(y1, y2)[0, 1])

    # Compute Real Temporal Metrics
    for traj in real_trajectories:
        X_norm = traj["X"]
        M = traj["M"]
        X_phys = denormalize(X_norm, mean, std)

        for i, f_name in enumerate(feature_names):
            obs_indices = np.where(M[:, i] == 1.0)[0]
            if len(obs_indices) > 1:
                vals = X_phys[obs_indices, i]
                d = np.diff(vals)
                real_diffs[f_name].extend(d)

                ac1 = compute_autocorr(vals, 1)
                if not np.isnan(ac1):
                    real_autocorr_lag1[f_name].append(ac1)
                ac5 = compute_autocorr(vals, 5)
                if not np.isnan(ac5):
                    real_autocorr_lag5[f_name].append(ac5)

    # Compute Synthetic Temporal Metrics
    for stay_id, samples_list in synth_by_stay.items():
        for X_synth in samples_list:  # [N, 5]
            for i, f_name in enumerate(feature_names):
                vals = X_synth[:, i]
                d = np.diff(vals)
                synth_diffs[f_name].extend(d)

                ac1 = compute_autocorr(vals, 1)
                if not np.isnan(ac1):
                    synth_autocorr_lag1[f_name].append(ac1)
                ac5 = compute_autocorr(vals, 5)
                if not np.isnan(ac5):
                    synth_autocorr_lag5[f_name].append(ac5)

    print(f"{'Feature':<18} | {'Real std(\Delta X)':<18} | {'Synth std(\Delta X)':<19} | {'Real Lag-1 AC':<13} | {'Synth Lag-1 AC':<14} | {'Real Lag-5 AC':<13} | {'Synth Lag-5 AC':<14}")
    print("-" * 120)

    temporal_results = {}
    for f_name in feature_names:
        r_d_std = np.std(real_diffs[f_name])
        s_d_std = np.std(synth_diffs[f_name])

        r_ac1 = np.mean(real_autocorr_lag1[f_name]) if real_autocorr_lag1[f_name] else 0.0
        s_ac1 = np.mean(synth_autocorr_lag1[f_name]) if synth_autocorr_lag1[f_name] else 0.0

        r_ac5 = np.mean(real_autocorr_lag5[f_name]) if real_autocorr_lag5[f_name] else 0.0
        s_ac5 = np.mean(synth_autocorr_lag5[f_name]) if synth_autocorr_lag5[f_name] else 0.0

        print(
            f"{f_name:<18} | {r_d_std:18.4f} | {s_d_std:19.4f} | {r_ac1:13.4f} | {s_ac1:14.4f} | {r_ac5:13.4f} | {s_ac5:14.4f}"
        )

        temporal_results[f_name] = {
            "real_diff_std": float(r_d_std),
            "synth_diff_std": float(s_d_std),
            "real_autocorr_lag1": float(r_ac1),
            "synth_autocorr_lag1": float(s_ac1),
            "real_autocorr_lag5": float(r_ac5),
            "synth_autocorr_lag5": float(s_ac5),
        }

    # Plot Temporal Comparison Figure
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        sns.kdeplot(real_diffs[f_name], ax=ax, color="blue", label=r"Real First Diff ($\Delta X$)", fill=True, alpha=0.3)
        sns.kdeplot(synth_diffs[f_name], ax=ax, color="red", label=r"Synthetic First Diff ($\Delta X$)", fill=True, alpha=0.3)
        ax.set_title(f"Temporal Change $\Delta X$: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel(r"$\Delta X_t = X_t - X_{t-1}$")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[5].axis("off")
    plt.tight_layout()
    temp_plot_path = Path(experiments_dir) / "phase10_temporal_comparison.png"
    plt.savefig(temp_plot_path, dpi=300)
    plt.close()
    print(f"Saved temporal dynamics figure to: {temp_plot_path}")

    return temporal_results


def evaluate_stochastic_diversity(
    synth_by_stay: Dict[int, List[np.ndarray]],
    real_trajectories: List[Dict[str, Any]],
    feature_names: List[str],
    experiments_dir: str = "experiments",
) -> Dict[str, Any]:
    print("\n--- 4. Multiple Future Diversity Evaluation ---")

    all_pairwise_rmse = []
    all_pairwise_mae = []

    # Compute pairwise metrics across 20 futures for each stay
    for stay_id, samples_list in synth_by_stay.items():
        K = len(samples_list)
        for i in range(K):
            for j in range(i + 1, K):
                s1 = samples_list[i]
                s2 = samples_list[j]
                rmse = np.sqrt(np.mean((s1 - s2) ** 2))
                mae = np.mean(np.abs(s1 - s2))
                all_pairwise_rmse.append(rmse)
                all_pairwise_mae.append(mae)

    mean_rmse = np.mean(all_pairwise_rmse)
    mean_mae = np.mean(all_pairwise_mae)

    print(f"Mean Pairwise Trajectory RMSE across 20 Futures : {mean_rmse:.4f}")
    print(f"Mean Pairwise Trajectory MAE across 20 Futures  : {mean_mae:.4f}")

    # Plot Multi-Future Fan Chart and Per-Timestamp Variance for a sample stay
    sample_stay_id = list(synth_by_stay.keys())[0]
    sample_futures = np.stack(synth_by_stay[sample_stay_id], axis=0)  # [20, N, 5]

    # Find matching real trajectory
    real_traj_sample = next(t for t in real_trajectories if t["stay_id"] == sample_stay_id)
    T_sample = real_traj_sample["T"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        fut_f = sample_futures[:, :, idx]  # [20, N]

        # Plot individual 20 futures
        for k in range(20):
            ax.plot(T_sample, fut_f[k], color="red", alpha=0.15, linewidth=1.0)

        # Plot mean synthetic future and std band
        mean_fut = np.mean(fut_f, axis=0)
        std_fut = np.std(fut_f, axis=0)

        ax.plot(T_sample, mean_fut, color="darkred", linewidth=2.0, label="Synthetic Mean (20 Futures)")
        ax.fill_between(T_sample, mean_fut - std_fut, mean_fut + std_fut, color="red", alpha=0.2, label=r"$\pm 1$ Std Fan")

        ax.set_title(f"20 Futures Fan Chart: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Physical Unit")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend()

    axes[5].axis("off")
    plt.tight_layout()
    div_plot_path = Path(experiments_dir) / "phase10_generation_diversity.png"
    plt.savefig(div_plot_path, dpi=300)
    plt.close()
    print(f"Saved generation diversity fan chart figure to: {div_plot_path}")

    return {
        "mean_pairwise_rmse": float(mean_rmse),
        "mean_pairwise_mae": float(mean_mae),
    }


def run_time_awareness_experiment(
    checkpoint_path: str = "outputs/checkpoints/best_model.pt",
    scaler_path: str = "data/processed/scaler.json",
    test_pkl_path: str = "data/processed/datasets/test.pkl",
) -> Dict[str, Any]:
    print("\n--- 5. Time-Awareness Controlled Experiment ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EHRNeuralSDE(max_step_size=0.25).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    with open(test_pkl_path, "rb") as f:
        test_trajectories = pickle.load(f)

    traj = test_trajectories[0]
    X_0 = torch.tensor(traj["X"][0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
    M_0 = torch.tensor(traj["M"][0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
    T_0 = torch.tensor(traj["T"][0:1], dtype=torch.float32).unsqueeze(0).to(device)
    DeltaT_0 = torch.tensor(traj["DeltaT"][0:1], dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
        z_0 = model.latent_projection(h_0)  # [1, 32]
        x_hat_0_norm = model.decoder(z_0).squeeze(0).cpu().numpy()
        x_hat_0_phys = denormalize(x_hat_0_norm, mean, std)

    dt_values = [0.05, 1.0, 5.0, 10.0]
    exp_results = {}

    print(f"Fixed Initial Latent State z_0 Norm: {torch.norm(z_0).item():.4f}")
    print(f"{'Integration dt (Hours)':<22} | {'||z(T0+dt) - z(T0)||_2':<22} | {'||X_hat(T0+dt) - X_hat(T0)||_2':<28}")
    print("-" * 76)

    for dt in dt_values:
        t_eval = torch.tensor([[0.0, dt]], dtype=torch.float32, device=device)
        gen = torch.Generator(device=device).manual_seed(42)

        with torch.no_grad():
            z_traj = model.sde.integrate(z0=z_0, times=t_eval, generator=gen, enable_noise=True)
            z_end = z_traj[:, 1, :]  # [1, 32]
            z_disp = torch.norm(z_end - z_0).item()

            x_hat_end_norm = model.decoder(z_end).squeeze(0).cpu().numpy()
            x_hat_end_phys = denormalize(x_hat_end_norm, mean, std)
            x_disp = float(np.linalg.norm(x_hat_end_phys - x_hat_0_phys))

        print(f"dt = {dt:5.2f} hours             | {z_disp:22.4f} | {x_disp:28.4f}")
        exp_results[dt] = {"latent_displacement": z_disp, "physical_displacement": x_disp}

    return exp_results


def plot_sample_trajectories(
    real_trajectories: List[Dict[str, Any]],
    synth_by_stay: Dict[int, List[np.ndarray]],
    feature_names: List[str],
    scaler_path: str = "data/processed/scaler.json",
    experiments_dir: str = "experiments",
):
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    sample_stay_id = list(synth_by_stay.keys())[0]
    real_traj = next(t for t in real_trajectories if t["stay_id"] == sample_stay_id)

    X_real_phys = denormalize(real_traj["X"], mean, std)
    T_real = real_traj["T"]
    M_real = real_traj["M"]

    synth_samples = synth_by_stay[sample_stay_id][:5]  # Take 5 synthetic samples

    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)

    for i, f_name in enumerate(feature_names):
        ax = axes[i]

        # Plot observed real points
        obs_mask = M_real[:, i] == 1.0
        ax.plot(T_real[obs_mask], X_real_phys[obs_mask, i], "bo-", label="Real Observations", linewidth=2.0, markersize=5)

        # Plot 5 synthetic conditional trajectories
        for k, synth_X in enumerate(synth_samples):
            label = f"Synthetic Sample {k+1}" if k == 0 else None
            ax.plot(T_real, synth_X[:, i], color="red", linestyle="--", alpha=0.6, label=label)

        ax.set_ylabel(f_name.replace("_", " ").title())
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (Hours)")
    plt.suptitle(f"Real vs Synthetic Vital Sign Trajectories (Stay ID: {sample_stay_id})", fontsize=14, y=0.99)
    plt.tight_layout()

    sample_plot_path = Path(experiments_dir) / "phase10_sample_trajectories.png"
    plt.savefig(sample_plot_path, dpi=300)
    plt.close()
    print(f"Saved sample trajectories comparison figure to: {sample_plot_path}")


def run_full_evaluation_suite():
    print("=== Phase 10: Full Synthetic Data Evaluation Suite ===")

    # 1. Load Data
    real_flat, synth_flat, real_trajectories, synth_by_stay, feature_names = load_real_and_synthetic_data()

    # 2. Distributional Similarity
    dist_metrics = evaluate_distribution_similarity(real_flat, synth_flat, feature_names)

    # 3. Feature Correlations
    evaluate_feature_correlations(real_flat, synth_flat, feature_names)

    # 4. Temporal Dynamics
    temp_metrics = evaluate_temporal_dynamics(real_trajectories, synth_by_stay, feature_names)

    # 5. Stochastic Multi-Future Diversity
    div_metrics = evaluate_stochastic_diversity(synth_by_stay, real_trajectories, feature_names)

    # 6. Time-Awareness Controlled Experiment
    run_time_awareness_experiment()

    # 7. Sample Trajectories Visualization
    plot_sample_trajectories(real_trajectories, synth_by_stay, feature_names)

    print("\nSUCCESS: All Phase 10 evaluations and visualizations generated.")


if __name__ == "__main__":
    run_full_evaluation_suite()
