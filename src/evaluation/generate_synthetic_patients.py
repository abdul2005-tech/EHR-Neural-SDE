"""
Data-Driven Synthetic Patient Generator Pipeline.

Generates N synthetic EHR patient trajectories with:
- Data-driven irregular time grids (unique duration & sequence length per patient)
- Initial state conditioning sampled strictly from TRAINING data
- Continuous Neural SDE evaluation using Phase 15 Champion (15C_Rational_tau1.0)
- Clean CSV dataset exports + detailed JSON metadata

Usage:
    python -m src.evaluation.generate_synthetic_patients
"""

import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
import torch

from src.data.scaler_utils import load_scaler_params, denormalize
from src.data.temporal_profile_generator import TemporalProfileGenerator
from src.models.ehr_neural_sde import EHRNeuralSDE
from src.models.state_dependent_residual import StateDependentMultivariateTemporalResidualModel


def generate_synthetic_patients(
    n_patients: int = 5,
    output_dir: str = "outputs/synthetic_datasets",
    base_seed: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> List[Tuple[str, str]]:
    """
    Generates n_patients synthetic patients with data-driven temporal profiles.

    Args:
        n_patients: Number of synthetic patient datasets to generate.
        output_dir: Target directory for CSVs and JSON metadata.
        base_seed: Base random seed. If None, derived dynamically from time.time().
        device: PyTorch compute device.

    Returns:
        List of tuples (csv_filepath, metadata_filepath) for all generated patients.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Execution Device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    if base_seed is None:
        base_seed = int(time.time() * 1000) % 1_000_000

    # 1. Load Scaler
    scaler_data = load_scaler_params("data/processed/scaler.json")
    mean = np.array(scaler_data["mean"], dtype=np.float32)
    std = np.array(scaler_data["std"], dtype=np.float32)
    feature_names = scaler_data["feature_names"]

    # 2. Load Models
    ckpt_path = "outputs/checkpoints/phase11_probabilistic.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing required checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    p11_model = EHRNeuralSDE(max_step_size=0.25, use_probabilistic_decoder=True).to(device)
    p11_model.load_state_dict(ckpt["model_state_dict"])
    p11_model.eval()

    with open("outputs/checkpoints/phase14_config.json", "r") as f:
        p14_cov = np.array(json.load(f)["selected_cov_matrix"], dtype=np.float32)

    with open("experiments/phase15_train_scales.json", "r") as f:
        p15_scales = json.load(f)["calibrated_scales_safe"]

    res_champion = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        cov_matrix=p14_cov,
        feature_scales=p15_scales,
        global_scale=1.0,
        attenuation_mode="rational",
        attenuation_tau=1.0,
    ).to(device).eval()

    # 3. Fit / Load Temporal Profile Generator (fitted on train.pkl ONLY)
    tp_gen = TemporalProfileGenerator().fit()
    tp_gen.save_profile("outputs/checkpoints/fitted_temporal_profile.json")

    generated_files = []

    print("\n" + "=" * 65)
    print(f"       GENERATING {n_patients} SYNTHETIC PATIENT TRAJECTORIES")
    print("=" * 65)

    for i in range(1, n_patients + 1):
        patient_seed = base_seed + i * 1000 + i * 17
        
        # Sample unique temporal profile & initial condition
        profile = tp_gen.generate_profile(seed=patient_seed)
        T_synth = profile["T"]
        init_cond = profile["initial_condition"]

        # Format initial condition tensors
        X_0 = torch.tensor(init_cond["X_0"], dtype=torch.float32, device=device).view(1, 1, 5)
        M_0 = torch.tensor(init_cond["M_0"], dtype=torch.float32, device=device).view(1, 1, 5)
        T_0 = torch.tensor([[init_cond["T_0"]]], dtype=torch.float32, device=device)
        DeltaT_0 = torch.tensor([[init_cond["DeltaT_0"]]], dtype=torch.float32, device=device)

        T_synth_tensor = torch.tensor(T_synth, dtype=torch.float32, device=device).unsqueeze(0)
        generator = torch.Generator(device=device).manual_seed(patient_seed)

        with torch.no_grad():
            z_0 = p11_model.get_initial_state(X_0, M_0, T_0, DeltaT_0)
            z_traj = p11_model.sde.integrate(z0=z_0, times=T_synth_tensor, generator=generator)
            mu_t, sigma_t = p11_model.decoder(z_traj)

            X_syn_norm, _ = res_champion.apply_residual(
                mu_norm=mu_t,
                T=T_synth_tensor,
                sigma_decoder=sigma_t,
                generator=generator,
            )

        # Denormalize to physical units
        X_syn_phys = denormalize(X_syn_norm.squeeze(0).cpu().numpy(), mean, std)
        
        # Build DataFrame
        df_patient = pd.DataFrame(X_syn_phys, columns=feature_names)
        df_patient.insert(0, "time_hours", T_synth)

        patient_str = f"{i:03d}"
        csv_filename = os.path.join(output_dir, f"synthetic_patient_{patient_str}.csv")
        meta_filename = os.path.join(output_dir, f"synthetic_patient_{patient_str}_metadata.json")

        df_patient.to_csv(csv_filename, index=False)

        metadata = {
            "patient_number": i,
            "patient_id": f"synthetic_patient_{patient_str}",
            "random_seed": patient_seed,
            "num_time_points": int(profile["num_time_points"]),
            "duration_hours": float(profile["duration"]),
            "t_min_hours": float(T_synth[0]),
            "t_max_hours": float(T_synth[-1]),
            "generation_timestamp": datetime.now().isoformat(),
            "model_champion": "15C_Rational_tau1.0",
            "temporal_profile_method": "Empirical_Train_Distribution_Sampler",
        }
        with open(meta_filename, "w") as f:
            json.dump(metadata, f, indent=2)

        generated_files.append((csv_filename, meta_filename))

        print(f"\n====================================================")
        print(f"Synthetic Patient #{i}")
        print(f"Seed: {patient_seed}")
        print(f"Time points: {profile['num_time_points']}")
        print(f"Duration: {profile['duration']:.2f} hours")
        print(f"\nDataFrame Preview (First 5 Rows):")
        
        try:
            from IPython.display import display
            display(df_patient.head(5))
        except ImportError:
            print(df_patient.head(5).to_string(index=False))

        print(f"\nSaved:")
        print(f"{os.path.basename(csv_filename)}")
        print(f"{os.path.basename(meta_filename)}")
        print(f"====================================================")

    print(f"\nSuccessfully generated {len(generated_files)} synthetic patient datasets in '{output_dir}/'.")
    return generated_files


def run_interactive_cli():
    """Interactive CLI prompting user for sample count."""
    print("=" * 65)
    print("       CONTINUOUS-TIME SYNTHETIC EHR PATIENT GENERATOR       ")
    print("=" * 65)
    
    user_input = input("\nHow many synthetic patients do you want to generate? ")
    try:
        n_patients = int(user_input)
        if n_patients <= 0:
            print("Please enter a positive integer greater than 0.")
            return
    except ValueError:
        print("Invalid input! Please enter a valid integer.")
        return

    generate_synthetic_patients(n_patients=n_patients)


if __name__ == "__main__":
    run_interactive_cli()
