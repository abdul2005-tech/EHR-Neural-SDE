"""
Unit test suite for Phase 15 Audit functions.
Verifies canonical correlation calculation, off-diagonal MAE, and safety attenuation properties.
"""

import numpy as np
import pytest
import torch
from src.evaluation.audit_phase15_all_tasks import (
    compute_canonical_correlation_metrics,
    SANITY_RANGES,
)
from src.models.state_dependent_residual import SmoothBoundaryAttenuation


def test_canonical_correlation_metrics_identity():
    # Identical matrices should produce zero error
    corr = np.eye(5)
    metrics = compute_canonical_correlation_metrics(corr, corr)
    assert metrics["canonical_corr_offdiag_mae"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["corr_offdiag_rmse"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["corr_max_abs_error"] == pytest.approx(0.0, abs=1e-6)


def test_canonical_correlation_metrics_known_diff():
    # Test known difference in off-diagonal entries
    corr_real = np.array([
        [1.0, 0.5],
        [0.5, 1.0]
    ])
    corr_synth = np.array([
        [1.0, 0.2],
        [0.2, 1.0]
    ])
    metrics = compute_canonical_correlation_metrics(corr_real, corr_synth)
    # Off-diagonal diff is |0.5 - 0.2| = 0.3
    assert metrics["canonical_corr_offdiag_mae"] == pytest.approx(0.3, abs=1e-5)
    assert metrics["corr_max_abs_error"] == pytest.approx(0.3, abs=1e-5)


def test_rational_attenuation_boundary_extinction():
    atten = SmoothBoundaryAttenuation(mode="rational", tau=1.0)
    mu_norm = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    val = atten(mu_norm)
    assert (val >= 0.0).all() and (val <= 1.0).all()


def test_sanity_ranges_defined():
    assert len(SANITY_RANGES) == 5
    assert "heart_rate" in SANITY_RANGES
    assert SANITY_RANGES["heart_rate"] == (20.0, 250.0)
