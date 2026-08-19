"""
Plot Generator Script for Phase 11 Variance-Aware Training.

Generates the 4 remaining required visual figure artifacts:
1. experiments/phase11_distribution_comparison.png
2. experiments/phase11_temporal_comparison.png
3. experiments/phase11_generation_diversity.png
4. experiments/phase11_real_vs_synthetic_trajectories.png
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.models.ehr_neural_sde import EHRNeuralSDE


def generate_all_phase11_figures(
    test_path: str = "data/processed/datasets/test.pkl",
    scaler_path: str = "data/processed/scaler.json",
    checkpoint_dir: str = "outputs/checkpoints",
    experiments_dir: str = "experiments",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Generating Phase 11 Detailed Visualization Artifacts ===")

    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    with open(test_path, "rb") as f:
        real_trajectories = pickle.load(f)

    # 1. Flatten real test observations
    real_flat = {f: [] for f in feature_names}
    real_diffs = {f: [] for f in feature_names}
    for traj in real_trajectories:
        X_p = denormalize(traj["X"], mean, std)
        M = traj["M"]
        for i, name in enumerate(feature_names):
            obs = X_p[M[:, i] == 1.0, i]
            real_flat[name].extend(obs)
            if len(X_p) > 1:
                real_diffs[name].extend(np.diff(X_p[:, i]))

    # Load Phase 10 Baseline, Phase 11 Probabilistic, and Phase 11 Combined models
    models_to_plot = [
        ("Phase 10 Baseline", "best_model.pt", False, "red"),
        ("Phase 11 Probabilistic", "phase11_probabilistic.pt", True, "green"),
        ("Phase 11 Combined", "phase11_combined.pt", True, "purple"),
    ]

    model_outputs = {}

    for label, ckpt_file, is_prob, color in models_to_plot:
        ckpt_p = Path(checkpoint_dir) / ckpt_file
        if not ckpt_p.exists():
            continue

        model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=is_prob).to(device)
        checkpoint = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        synth_flat = {f: [] for f in feature_names}
        synth_diffs = {f: [] for f in feature_names}
        synth_by_stay = {}

        for idx, traj in enumerate(real_trajectories):
            stay_id = traj["stay_id"]
            X_0 = torch.tensor(traj["X"][0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            M_0 = torch.tensor(traj["M"][0:1, :], dtype=torch.float32).unsqueeze(0).to(device)
            T_0 = torch.tensor(traj["T"][0:1], dtype=torch.float32).unsqueeze(0).to(device)
            DeltaT_0 = torch.tensor(traj["DeltaT"][0:1], dtype=torch.float32).unsqueeze(0).to(device)
            T_target = torch.tensor(traj["T"], dtype=torch.float32).unsqueeze(0).to(device)

            synth_by_stay[stay_id] = []

            with torch.no_grad():
                h_0 = model.encoder(X_0, M_0, T_0, DeltaT_0).squeeze(1)
                z_0 = model.latent_projection(h_0)

                for s_idx in range(20):
                    gen_s = torch.Generator(device=device).manual_seed(42 + idx * 50 + s_idx * 1000)
                    z_traj = model.sde.integrate(z0=z_0, times=T_target, generator=gen_s, enable_noise=True)

                    if is_prob:
                        mu, _ = model.decoder(z_traj)
                        pred_norm = mu.squeeze(0).cpu().numpy()
                    else:
                        X_hat = model.decoder(z_traj)
                        pred_norm = X_hat.squeeze(0).cpu().numpy()

                    X_phys = denormalize(pred_norm, mean, std)
                    synth_by_stay[stay_id].append(X_phys)

                    for i, f_name in enumerate(feature_names):
                        synth_flat[f_name].extend(X_phys[:, i])
                        if len(X_phys) > 1:
                            synth_diffs[f_name].extend(np.diff(X_phys[:, i]))

        model_outputs[label] = {
            "flat": synth_flat,
            "diffs": synth_diffs,
            "by_stay": synth_by_stay,
            "color": color,
        }

    # --- FIGURE 1: Distribution Comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        sns.kdeplot(real_flat[f_name], ax=ax, color="blue", label="Real Test Data", fill=True, alpha=0.25, linewidth=2.0)
        for label, m_data in model_outputs.items():
            sns.kdeplot(m_data["flat"][f_name], ax=ax, color=m_data["color"], label=label, linewidth=2.0)

        ax.set_title(f"Distribution: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    dist_plot_p = Path(experiments_dir) / "phase11_distribution_comparison.png"
    plt.savefig(dist_plot_p, dpi=300)
    plt.close()
    print(f"Saved distribution comparison figure to: {dist_plot_p}")

    # --- FIGURE 2: Temporal Comparison (First Differences \Delta X) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        sns.kdeplot(real_diffs[f_name], ax=ax, color="blue", label=r"Real First Diff ($\Delta X$)", fill=True, alpha=0.25, linewidth=2.0)
        for label, m_data in model_outputs.items():
            sns.kdeplot(m_data["diffs"][f_name], ax=ax, color=m_data["color"], label=f"{label} $\Delta X$", linewidth=2.0)

        ax.set_title(f"Temporal Volatility $\Delta X$: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel(r"$\Delta X_t = X_t - X_{t-1}$")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    temp_plot_p = Path(experiments_dir) / "phase11_temporal_comparison.png"
    plt.savefig(temp_plot_p, dpi=300)
    plt.close()
    print(f"Saved temporal comparison figure to: {temp_plot_p}")

    # --- FIGURE 3: Generation Diversity (Probabilistic Decoder Multi-Future Fan Chart) ---
    if "Phase 11 Probabilistic" in model_outputs:
        sample_stay_id = list(model_outputs["Phase 11 Probabilistic"]["by_stay"].keys())[0]
        sample_futures = np.stack(model_outputs["Phase 11 Probabilistic"]["by_stay"][sample_stay_id], axis=0) # [20, N, 5]
        real_sample = next(t for t in real_trajectories if t["stay_id"] == sample_stay_id)
        T_sample = real_sample["T"]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, f_name in enumerate(feature_names):
            ax = axes[idx]
            fut_f = sample_futures[:, :, idx]
            for k in range(20):
                ax.plot(T_sample, fut_f[k], color="green", alpha=0.15, linewidth=1.0)

            mean_fut = np.mean(fut_f, axis=0)
            std_fut = np.std(fut_f, axis=0)

            ax.plot(T_sample, mean_fut, color="darkgreen", linewidth=2.0, label="Probabilistic Mean (20 Futures)")
            ax.fill_between(T_sample, mean_fut - std_fut, mean_fut + std_fut, color="green", alpha=0.2, label=r"$\pm 1$ Std Fan")
            ax.set_title(f"Stochastic Diversity: {f_name.replace('_', ' ').title()}")
            ax.set_xlabel("Time (Hours)")
            ax.set_ylabel("Physical Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

        axes[5].axis("off")
        plt.tight_layout()
        div_plot_p = Path(experiments_dir) / "phase11_generation_diversity.png"
        plt.savefig(div_plot_p, dpi=300)
        plt.close()
        print(f"Saved generation diversity figure to: {div_plot_p}")

    # --- FIGURE 4: Real vs Synthetic Trajectories Comparison Overlay ---
    sample_stay_id = list(real_trajectories[0]["stay_id"] for _ in [0])[0]
    real_sample = real_trajectories[0]
    X_real_phys = denormalize(real_sample["X"], mean, std)
    M_real = real_sample["M"]
    T_sample = real_sample["T"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        obs_mask = M_real[:, idx] == 1.0
        ax.plot(T_sample[obs_mask], X_real_phys[obs_mask, idx], "o-", color="blue", label="Real Observations", linewidth=2.0, markersize=5)

        for label, m_data in model_outputs.items():
            if sample_stay_id in m_data["by_stay"]:
                fut_sample = m_data["by_stay"][sample_stay_id][0] # first generated future
                ax.plot(T_sample, fut_sample[:, idx], "--", color=m_data["color"], label=f"{label} Future", linewidth=1.8)

        ax.set_title(f"Trajectory Overlay: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Physical Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    traj_plot_p = Path(experiments_dir) / "phase11_real_vs_synthetic_trajectories.png"
    plt.savefig(traj_plot_p, dpi=300)
    plt.close()
    print(f"Saved real vs synthetic trajectories overlay figure to: {traj_plot_p}")


if __name__ == "__main__":
    generate_all_phase11_figures()
