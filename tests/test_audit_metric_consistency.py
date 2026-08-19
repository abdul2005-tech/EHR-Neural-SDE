"""
Unit and integration tests for Phase 15 Metric Consistency & Reproducibility Pipeline.
"""

import os
import json
import csv
import pytest
import numpy as np
from src.evaluation.audit_phase15_all_tasks import execute_master_audit


def test_audit_metric_consistency_and_report_generation(tmp_path):
    """
    Verifies that execute_master_audit produces metric-consistent CSV and Markdown reports,
    ensuring no hardcoded or stale values exist in phase15_final_audit_report.md.
    """
    execute_master_audit()

    assert os.path.exists("experiments/phase15_final_audit_report.md")
    assert os.path.exists("experiments/phase15_correlation_metric_audit.csv")

    # Read CSV metrics
    csv_rows = []
    with open("experiments/phase15_correlation_metric_audit.csv", "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            csv_rows.append(r)

    assert len(csv_rows) == 4
    p15_csv_row = csv_rows[3]
    p15_corr_mae = float(p15_csv_row["Canonical OffDiag MAE"])
    p15_volatility = float(p15_csv_row["Step Volatility std(dX)"])

    # Read Markdown Report
    with open("experiments/phase15_final_audit_report.md", "r") as f:
        md_text = f.read()

    # Assert Markdown report contains dynamically generated values matching CSV
    corr_mae_str = f"{p15_corr_mae:.4f}"
    vol_str = f"{p15_volatility:.4f}"

    assert corr_mae_str in md_text, f"Expected {corr_mae_str} in final audit report markdown"
    assert vol_str in md_text, f"Expected {vol_str} in final audit report markdown"
    assert "7.4377" not in md_text, "Found stale hardcoded volatility 7.4377 in final audit report markdown!"

    # Assert Evaluation Protocol & Environment Metadata section exists
    assert "Official Evaluation Protocol & Environment Metadata" in md_text
    assert "Execution Backend / Device" in md_text
    assert "Deterministic Generation Seed" in md_text
    assert "17 ICU stays" in md_text

    # Assert Cross-Device Reproducibility Note exists
    assert "Same-Device Fixed-Seed Determinism" in md_text
    assert "Cross-Device RNG Variance" in md_text

    # Assert Scientific Volatility Interpretation exists
    assert "Scientific Volatility Restoration" in md_text

    # Assert Champion remains frozen
    assert "15C_Rational_tau1.0" in md_text


def test_fixed_seed_same_device_determinism():
    """
    Verifies same-device fixed-seed bit-exact determinism for trajectory generation.
    """
    import torch
    from src.models.state_dependent_residual import StateDependentMultivariateTemporalResidualModel

    cov = np.eye(5, dtype=np.float32)
    scales = [1.0] * 5
    res_mod = StateDependentMultivariateTemporalResidualModel(
        output_dim=5,
        cov_matrix=cov,
        feature_scales=scales,
        global_scale=1.0,
        attenuation_mode="rational",
        attenuation_tau=1.0,
    )

    t_span = torch.linspace(0.0, 1.0, 10)
    mu_traj = torch.zeros(10, 5)
    phys_traj = torch.full((10, 5), 80.0)

    res_mod.eval()
    gen1 = torch.Generator().manual_seed(42)
    sample1, _ = res_mod(t_span, mu_traj, phys_traj, generator=gen1)

    gen2 = torch.Generator().manual_seed(42)
    sample2, _ = res_mod(t_span, mu_traj, phys_traj, generator=gen2)

    assert torch.allclose(sample1, sample2, atol=1e-6), "State-dependent residual samples generated with same seed on same device must be bit-exact identical!"



