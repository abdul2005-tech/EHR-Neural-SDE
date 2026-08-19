"""
Plot Generator Script for Phase 12 Observation Noise & Temporal Realism Investigation.

Generates 6 publication-quality visual figure artifacts:
1. experiments/phase12_distribution_comparison.png
2. experiments/phase12_temporal_comparison.png
3. experiments/phase12_correlation_comparison.png
4. experiments/phase12_generation_diversity.png
5. experiments/phase12_sigma_analysis.png
6. experiments/phase12_real_vs_synthetic_trajectories.png
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


def generate_all_phase12_figures(
    val_path: str = "data/processed/datasets/val.pkl",
    scaler_path: str = "data/processed/scaler.json",
    checkpoint_dir: str = "outputs/checkpoints",
    experiments_dir: str = "experiments",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== Generating Phase 12 Detailed Visualization Figure Artifacts ===")

    scaler_data = load_scaler_params(scaler_path)
    feature_names = scaler_data["feature_names"]
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    with open(val_path, "rb") as f:
        real_trajectories = pickle.load(f)

    # Real observations
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

    models_to_plot = [
        ("12A Baseline (Phase 11)", "phase11_probabilistic.pt", True, None, "green"),
        ("12B Independent Noise", "phase12_independent.pt", False, "independent", "orange"),
        ("12C Heteroscedastic Noise", "phase12_heteroscedastic.pt", False, "heteroscedastic", "red"),
        ("12D Correlated Noise", "phase12_correlated.pt", False, "correlated", "purple"),
    ]

    model_outputs = {}

    for label, ckpt_file, is_prob, obs_mode, color in models_to_plot:
        ckpt_p = Path(checkpoint_dir) / ckpt_file
        if not ckpt_p.exists():
            continue

        model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=is_prob, observation_mode=obs_mode).to(device)
        checkpoint = torch.load(ckpt_p, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        synth_flat = {f: [] for f in feature_names}
        synth_diffs = {f: [] for f in feature_names}
        sigmas_flat = {f: [] for f in feature_names}
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

                for s_idx in range(5):
                    gen_s = torch.Generator(device=device).manual_seed(42 + idx * 50 + s_idx * 1000)
                    z_traj = model.sde.integrate(z0=z_0, times=T_target, generator=gen_s, enable_noise=True)

                    if model.observation_model is not None:
                        if model.observation_mode in ("independent", "heteroscedastic"):
                            mu, sigma = model.observation_model(z_traj)
                            X_syn_norm = model.observation_model.sample(z_traj, generator=gen_s).squeeze(0).cpu().numpy()
                            sigma_np = sigma.squeeze(0).cpu().numpy() * std
                        else:
                            mu, d_diag, U = model.observation_model(z_traj)
                            X_syn_norm = model.observation_model.sample(z_traj, generator=gen_s).squeeze(0).cpu().numpy()
                            sigma_np = torch.sqrt(d_diag).squeeze(0).cpu().numpy() * std
                    elif is_prob:
                        mu, sigma = model.decoder(z_traj)
                        pred_norm = mu.squeeze(0).cpu().numpy()
                        sigma_norm = sigma.squeeze(0).cpu().numpy()
                        X_syn_norm = pred_norm + sigma_norm * np.random.randn(*pred_norm.shape)
                        sigma_np = sigma_norm * std

                    X_phys = denormalize(X_syn_norm, mean, std)
                    synth_by_stay[stay_id].append(X_phys)

                    for i, f_name in enumerate(feature_names):
                        synth_flat[f_name].extend(X_phys[:, i])
                        sigmas_flat[f_name].extend(sigma_np[:, i])
                        if len(X_phys) > 1:
                            synth_diffs[f_name].extend(np.diff(X_phys[:, i]))

        model_outputs[label] = {
            "flat": synth_flat,
            "diffs": synth_diffs,
            "sigmas": sigmas_flat,
            "by_stay": synth_by_stay,
            "color": color,
        }

    # --- FIGURE 1: Distribution Comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        sns.kdeplot(real_flat[f_name], ax=ax, color="blue", label="Real Validation Data", fill=True, alpha=0.25, linewidth=2.0)
        for label, m_data in model_outputs.items():
            sns.kdeplot(m_data["flat"][f_name], ax=ax, color=m_data["color"], label=label, linewidth=2.0)

        ax.set_title(f"Distribution: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(Path(experiments_dir) / "phase12_distribution_comparison.png", dpi=300)
    plt.close()
    print("Saved distribution comparison figure.")

    # --- FIGURE 2: Temporal Volatility Comparison ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        sns.kdeplot(real_diffs[f_name], ax=ax, color="blue", label=r"Real First Diff ($\Delta X$)", fill=True, alpha=0.25, linewidth=2.0)
        for label, m_data in model_outputs.items():
            sns.kdeplot(m_data["diffs"][f_name], ax=ax, color=m_data["color"], label=f"{label} $\Delta X$", linewidth=2.0)

        ax.set_title(f"Temporal Change $\Delta X$: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel(r"$\Delta X_t = X_t - X_{t-1}$")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(Path(experiments_dir) / "phase12_temporal_comparison.png", dpi=300)
    plt.close()
    print("Saved temporal comparison figure.")

    # --- FIGURE 3: Correlation Comparison ---
    if len(model_outputs) >= 2:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        min_r = min(len(real_flat[f]) for f in feature_names)
        real_mat = np.column_stack([real_flat[f][:min_r] for f in feature_names])
        sns.heatmap(np.corrcoef(real_mat, rowvar=False), ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    xticklabels=[f[:4] for f in feature_names], yticklabels=[f[:4] for f in feature_names])
        axes[0].set_title("Real Data Correlation")

        for idx, (label, m_data) in enumerate(list(model_outputs.items())[:5]):
            ax = axes[idx + 1]
            min_s = min(len(m_data["flat"][f]) for f in feature_names)
            synth_mat = np.column_stack([m_data["flat"][f][:min_s] for f in feature_names])
            sns.heatmap(np.corrcoef(synth_mat, rowvar=False), ax=ax, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                        xticklabels=[f[:4] for f in feature_names], yticklabels=[f[:4] for f in feature_names])
            ax.set_title(f"{label} Correlation")

        plt.tight_layout()
        plt.savefig(Path(experiments_dir) / "phase12_correlation_comparison.png", dpi=300)
        plt.close()
        print("Saved correlation comparison figure.")

    # --- FIGURE 4: Generation Diversity ---
    if "12C Heteroscedastic Noise" in model_outputs:
        sample_stay_id = list(model_outputs["12C Heteroscedastic Noise"]["by_stay"].keys())[0]
        sample_futures = np.stack(model_outputs["12C Heteroscedastic Noise"]["by_stay"][sample_stay_id], axis=0)
        real_sample = next(t for t in real_trajectories if t["stay_id"] == sample_stay_id)
        T_sample = real_sample["T"]

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()

        for idx, f_name in enumerate(feature_names):
            ax = axes[idx]
            fut_f = sample_futures[:, :, idx]
            for k in range(len(fut_f)):
                ax.plot(T_sample, fut_f[k], color="red", alpha=0.25, linewidth=1.0)

            mean_fut = np.mean(fut_f, axis=0)
            std_fut = np.std(fut_f, axis=0)

            ax.plot(T_sample, mean_fut, color="darkred", linewidth=2.0, label="12C Synthetic Mean")
            ax.fill_between(T_sample, mean_fut - std_fut, mean_fut + std_fut, color="red", alpha=0.2, label=r"$\pm 1$ Std Fan")
            ax.set_title(f"Stochastic Diversity: {f_name.replace('_', ' ').title()}")
            ax.set_xlabel("Time (Hours)")
            ax.set_ylabel("Physical Value")
            ax.grid(True, linestyle="--", alpha=0.5)
            ax.legend(fontsize=8)

        axes[5].axis("off")
        plt.tight_layout()
        plt.savefig(Path(experiments_dir) / "phase12_generation_diversity.png", dpi=300)
        plt.close()
        print("Saved generation diversity figure.")

    # --- FIGURE 5: Observation Sigma Analysis ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, f_name in enumerate(feature_names):
        ax = axes[idx]
        for label, m_data in model_outputs.items():
            if m_data["sigmas"][f_name]:
                sns.kdeplot(m_data["sigmas"][f_name], ax=ax, color=m_data["color"], label=f"{label} $\sigma$", linewidth=2.0)

        ax.set_title(f"Learned Observation Noise $\sigma$: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Physical Sigma Magnitude")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(Path(experiments_dir) / "phase12_sigma_analysis.png", dpi=300)
    plt.close()
    print("Saved observation sigma analysis figure.")

    # --- FIGURE 6: Real vs Synthetic Trajectories Overlay ---
    sample_stay_id = real_trajectories[0]["stay_id"]
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
                fut_sample = m_data["by_stay"][sample_stay_id][0]
                ax.plot(T_sample, fut_sample[:, idx], "--", color=m_data["color"], label=f"{label} Future", linewidth=1.8)

        ax.set_title(f"Trajectory Overlay: {f_name.replace('_', ' ').title()}")
        ax.set_xlabel("Time (Hours)")
        ax.set_ylabel("Physical Value")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    axes[5].axis("off")
    plt.tight_layout()
    plt.savefig(Path(experiments_dir) / "phase12_real_vs_synthetic_trajectories.png", dpi=300)
    plt.close()
    print("Saved real vs synthetic trajectories overlay figure.")


if __name__ == "__main__":
    generate_all_phase12_figures()
