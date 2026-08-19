"""
Phase 13 Plot Generation Script.

Generates 5 publication-quality figures for Phase 13 OU Residual Process evaluation:
1. experiments/phase13_ou_volatility_comparison.png
2. experiments/phase13_ou_autocorrelation_comparison.png
3. experiments/phase13_ou_distribution_comparison.png
4. experiments/phase13_ou_sample_trajectories.png
5. experiments/phase13_ou_residual_analysis.png
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
from src.models.temporal_residual import TemporalResidualModel
from src.evaluation.fit_phase13_residuals import estimate_ou_parameters_from_train
from src.evaluation.evaluate_phase13 import (
    load_dataset_trajectories,
    precompute_latent_trajectories,
    generate_phase13_futures_cached,
    compute_phase13_metrics,
)


plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#cccccc"
plt.rcParams["axes.linewidth"] = 1.0


def generate_all_phase13_plots(
    val_path: str = "data/processed/datasets/val.pkl",
    checkpoint_path: str = "outputs/checkpoints/phase11_probabilistic.pt",
    scaler_path: str = "data/processed/scaler.json",
    output_dir: str = "experiments",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating Phase 13 Publication-Quality Figures ===")

    # Load validation data & scaler
    real_trajectories, real_flat, feature_names = load_dataset_trajectories(val_path, scaler_path)
    scaler_data = load_scaler_params(scaler_path)
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)

    # Fit OU parameters on TRAIN data
    lambdas, sigmas, _ = estimate_ou_parameters_from_train(device=device)
    ou_model = TemporalResidualModel(
        output_dim=len(feature_names),
        lambda_val=lambdas,
        sigma_val=sigmas,
    ).to(device)

    # Load frozen Phase 11 model
    ckpt_p = Path(checkpoint_path)
    model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    checkpoint = torch.load(ckpt_p, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Generate synthetic futures for subset of stays (5 stays x 5 futures) for plotting
    subset_trajectories = real_trajectories[:5]
    cached_preds = precompute_latent_trajectories(model, subset_trajectories, n_samples=5)

    synth_13A0 = generate_phase13_futures_cached(subset_trajectories, cached_preds, "13A0_continuous_mean", ou_model, scale=1.0, n_samples=5)
    synth_13A1 = generate_phase13_futures_cached(subset_trajectories, cached_preds, "13A1_probabilistic", ou_model, scale=1.0, n_samples=5)
    synth_13B  = generate_phase13_futures_cached(subset_trajectories, cached_preds, "13B_independent", ou_model, scale=1.0, n_samples=5)
    synth_13C  = generate_phase13_futures_cached(subset_trajectories, cached_preds, "13C_ou_full", ou_model, scale=1.0, n_samples=5)
    synth_13D  = generate_phase13_futures_cached(subset_trajectories, cached_preds, "13D_ou_scaled", ou_model, scale=0.50, n_samples=5)


    # Compute metrics for plotting
    m_13A0 = compute_phase13_metrics(real_flat, subset_trajectories, synth_13A0, feature_names)
    m_13A1 = compute_phase13_metrics(real_flat, subset_trajectories, synth_13A1, feature_names)
    m_13B  = compute_phase13_metrics(real_flat, subset_trajectories, synth_13B, feature_names)
    m_13C  = compute_phase13_metrics(real_flat, subset_trajectories, synth_13C, feature_names)
    m_13D  = compute_phase13_metrics(real_flat, subset_trajectories, synth_13D, feature_names)

    labels = ["Real Data", "Phase 11 (Mean)", "Phase 11 (Prob)", "Phase 12 (Indep)", "Phase 13 (OU Full)", "Phase 13 (OU Scaled 0.5)"]
    colors = ["#2b5c8f", "#7f7f7f", "#d95f02", "#e7298a", "#7570b3", "#1b9e77"]

    # -------------------------------------------------------------
    # Figure 1: Volatility Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    vols = [
        m_13A0["real_diff_std"],
        m_13A0["synth_diff_std"],
        m_13A1["synth_diff_std"],
        m_13B["synth_diff_std"],
        m_13C["synth_diff_std"],
        m_13D["synth_diff_std"],
    ]
    bars = plt.bar(labels, vols, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    plt.ylabel(r"Step Volatility std($\Delta X$) (Physical Units)", fontsize=12, fontweight="bold")
    plt.title("Phase 13: Step Volatility Comparison Across Models", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=20, ha="right", fontsize=11)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 0.2, f"{height:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    
    plt.tight_layout()
    fig1_p = out_dir / "phase13_ou_volatility_comparison.png"
    plt.savefig(fig1_p, dpi=300)
    plt.close()
    print(f"Saved: {fig1_p}")

    # -------------------------------------------------------------
    # Figure 2: Autocorrelation Comparison
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    ac1s = [
        0.4581,  # Real reference
        m_13A0["mean_ac1"],
        m_13A1["mean_ac1"],
        m_13B["mean_ac1"],
        m_13C["mean_ac1"],
        m_13D["mean_ac1"],
    ]
    bars = plt.bar(labels, ac1s, color=colors, alpha=0.85, edgecolor="black", linewidth=1.2)
    plt.axhline(0.4581, color="#2b5c8f", linestyle="--", linewidth=1.5, label="Real Test Target (+0.4581)")
    plt.axhline(0.0, color="black", linestyle="-", linewidth=0.8)
    plt.ylabel("Lag-1 Autocorrelation", fontsize=12, fontweight="bold")
    plt.title("Phase 13: Preservation of Temporal Structure (Lag-1 Autocorrelation)", fontsize=14, fontweight="bold", pad=15)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=20, ha="right", fontsize=11)
    plt.ylim(-0.15, 1.05)
    plt.legend(loc="upper right", fontsize=11)

    for bar in bars:
        height = bar.get_height()
        va = "bottom" if height >= 0 else "top"
        y_pos = height + 0.02 if height >= 0 else height - 0.04
        plt.text(bar.get_x() + bar.get_width() / 2.0, y_pos, f"{height:+.3f}", ha="center", va=va, fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig2_p = out_dir / "phase13_ou_autocorrelation_comparison.png"
    plt.savefig(fig2_p, dpi=300)
    plt.close()
    print(f"Saved: {fig2_p}")

    # -------------------------------------------------------------
    # Figure 3: Marginal Distribution Comparison
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for i, f_name in enumerate(feature_names):
        ax = axes[i]
        sns.kdeplot(real_flat[f_name], ax=ax, color="#2b5c8f", label="Real Data", linewidth=2.0)
        
        # Flatten synthetic values across stays for feature i
        s_13A0_flat = np.concatenate([s[:, i] for stay in synth_13A0.values() for s in stay])
        s_13B_flat = np.concatenate([s[:, i] for stay in synth_13B.values() for s in stay])
        s_13C_flat = np.concatenate([s[:, i] for stay in synth_13C.values() for s in stay])
        s_13D_flat = np.concatenate([s[:, i] for stay in synth_13D.values() for s in stay])

        sns.kdeplot(s_13A0_flat, ax=ax, color="#7f7f7f", label="Phase 11 (Mean)", linestyle="--", linewidth=1.5)
        sns.kdeplot(s_13B_flat, ax=ax, color="#e7298a", label="Phase 12 (Indep)", linestyle=":", linewidth=1.5)
        sns.kdeplot(s_13C_flat, ax=ax, color="#7570b3", label="Phase 13 (OU Full)", linewidth=1.8)
        sns.kdeplot(s_13D_flat, ax=ax, color="#1b9e77", label="Phase 13 (OU Scale 0.5)", linewidth=1.8)

        ax.set_title(f_name.replace("_", " ").title(), fontsize=12, fontweight="bold")
        ax.set_xlabel("Value", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right")

    # Hide 6th unused subplot
    axes[5].axis("off")

    plt.suptitle("Phase 13: Marginal Distribution Density Comparison", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout()
    fig3_p = out_dir / "phase13_ou_distribution_comparison.png"
    plt.savefig(fig3_p, dpi=300)
    plt.close()
    print(f"Saved: {fig3_p}")

    # -------------------------------------------------------------
    # Figure 4: Sample Multi-Future Trajectories
    # -------------------------------------------------------------
    stay_sample = subset_trajectories[0]
    stay_id = stay_sample["stay_id"]
    T_rel = stay_sample["T"]
    X_real_phys = denormalize(stay_sample["X"], mean, std)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Subplot 1: Heart Rate
    ax1 = axes[0]
    ax1.plot(T_rel, X_real_phys[:, 0], "o-", color="black", label="Real Observations", linewidth=2.0, markersize=5)
    for s_idx, X_s in enumerate(synth_13A0[stay_id]):
        ax1.plot(T_rel, X_s[:, 0], color="#7f7f7f", alpha=0.5, linestyle="--", label="Phase 11 Mean" if s_idx == 0 else "")
    for s_idx, X_s in enumerate(synth_13B[stay_id]):
        ax1.plot(T_rel, X_s[:, 0], color="#e7298a", alpha=0.3, label="Phase 12 (Indep Noise)" if s_idx == 0 else "")
    for s_idx, X_s in enumerate(synth_13D[stay_id]):
        ax1.plot(T_rel, X_s[:, 0], color="#1b9e77", alpha=0.7, label="Phase 13 (OU Residual)" if s_idx == 0 else "")
    ax1.set_ylabel("Heart Rate (bpm)", fontsize=12, fontweight="bold")
    ax1.set_title(f"Sample Trajectories for ICU Stay #{stay_id} (Heart Rate)", fontsize=13, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Subplot 2: Systolic BP
    ax2 = axes[1]
    ax2.plot(T_rel, X_real_phys[:, 3], "o-", color="black", label="Real Observations", linewidth=2.0, markersize=5)
    for s_idx, X_s in enumerate(synth_13A0[stay_id]):
        ax2.plot(T_rel, X_s[:, 3], color="#7f7f7f", alpha=0.5, linestyle="--", label="Phase 11 Mean" if s_idx == 0 else "")
    for s_idx, X_s in enumerate(synth_13B[stay_id]):
        ax2.plot(T_rel, X_s[:, 3], color="#e7298a", alpha=0.3, label="Phase 12 (Indep Noise)" if s_idx == 0 else "")
    for s_idx, X_s in enumerate(synth_13D[stay_id]):
        ax2.plot(T_rel, X_s[:, 3], color="#1b9e77", alpha=0.7, label="Phase 13 (OU Residual)" if s_idx == 0 else "")
    ax2.set_xlabel("Time (Hours)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Systolic BP (mmHg)", fontsize=12, fontweight="bold")
    ax2.set_title(f"Sample Trajectories for ICU Stay #{stay_id} (Systolic BP)", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig4_p = out_dir / "phase13_ou_sample_trajectories.png"
    plt.savefig(fig4_p, dpi=300)
    plt.close()
    print(f"Saved: {fig4_p}")

    # -------------------------------------------------------------
    # Figure 5: Residual Autocorrelation Analysis Decay Curves
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    dt_grid = np.linspace(0.0, 24.0, 200)

    for d, f_name in enumerate(feature_names):
        l_val = lambdas[d]
        theo_ac = np.exp(-l_val * dt_grid)
        plt.plot(dt_grid, theo_ac, label=f"{f_name} (λ={l_val:.3f}/hr)", linewidth=2.0)

    plt.xlabel("Continuous Time Gap Δt (Hours)", fontsize=12, fontweight="bold")
    plt.ylabel("Theoretical OU Autocorrelation exp(-λ Δt)", fontsize=12, fontweight="bold")
    plt.title("Phase 13: Fitted Continuous-Time OU Residual Autocorrelation Decay", fontsize=14, fontweight="bold", pad=15)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(fontsize=11, loc="upper right")
    plt.ylim(-0.05, 1.05)

    plt.tight_layout()
    fig5_p = out_dir / "phase13_ou_residual_analysis.png"
    plt.savefig(fig5_p, dpi=300)
    plt.close()
    print(f"Saved: {fig5_p}")

    print("=== All 5 Phase 13 Figures Generated Successfully ===")


if __name__ == "__main__":
    generate_all_phase13_plots()
