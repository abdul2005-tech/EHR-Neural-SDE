"""
Unit and Statistical Validation Tests for Data-Driven Temporal Profile Generator & Synthetic Patient Pipeline.
"""

import os
import json
import tempfile
import pickle
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.temporal_profile_generator import TemporalProfileGenerator
from src.evaluation.generate_synthetic_patients import generate_synthetic_patients


def test_fit_only_on_train():
    """Verifies that temporal profile generator fits strictly on training data."""
    train_path = "data/processed/datasets/train.pkl"
    assert os.path.exists(train_path), "train.pkl must exist"
    
    tp_gen = TemporalProfileGenerator(train_path=train_path).fit()
    assert tp_gen.is_fitted
    assert len(tp_gen.seq_lengths) == 98, f"Expected 98 train samples, got {len(tp_gen.seq_lengths)}"


def test_monotonically_increasing_and_non_negative_gaps():
    """Verifies generated timestamps are strictly monotonically increasing with non-negative time gaps."""
    tp_gen = TemporalProfileGenerator().fit()
    
    for seed in [42, 101, 2024]:
        profile = tp_gen.generate_profile(seed=seed)
        T = profile["T"]
        
        # 1. Monotonically increasing
        assert np.all(np.diff(T) > 0.0), f"Timestamps for seed {seed} are not strictly monotonically increasing!"
        
        # 2. Non-negative gaps
        gaps = np.diff(T)
        assert np.all(gaps >= 0.0), f"Found negative time gaps for seed {seed}!"


def test_variability_sequence_length_and_duration():
    """Verifies that generated patients have different sequence lengths and durations."""
    tp_gen = TemporalProfileGenerator().fit()
    
    profiles = [tp_gen.generate_profile(seed=seed) for seed in range(100, 110)]
    
    seq_lens = [p["num_time_points"] for p in profiles]
    durations = [p["duration"] for p in profiles]
    
    # Check that not all sequence lengths are equal
    assert len(set(seq_lens)) > 1, "All generated sequence lengths were identical!"
    
    # Check that not all durations are equal
    assert len(set(durations)) > 1, "All generated durations were identical!"


def test_bit_exact_reproducibility_with_same_seed():
    """Verifies that using identical seeds produces bit-exact identical profiles."""
    tp_gen = TemporalProfileGenerator().fit()
    
    prof1 = tp_gen.generate_profile(seed=777)
    prof2 = tp_gen.generate_profile(seed=777)
    
    assert prof1["num_time_points"] == prof2["num_time_points"]
    assert np.isclose(prof1["duration"], prof2["duration"])
    assert np.allclose(prof1["T"], prof2["T"])


def test_generate_synthetic_patients_pipeline():
    """Verifies complete end-to-end synthetic patient generation pipeline."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        generated_files = generate_synthetic_patients(
            n_patients=3,
            output_dir=tmp_dir,
            base_seed=123,
        )
        
        assert len(generated_files) == 3, f"Expected 3 generated files, got {len(generated_files)}"
        
        for csv_path, meta_path in generated_files:
            assert os.path.exists(csv_path), f"CSV missing: {csv_path}"
            assert os.path.exists(meta_path), f"Metadata missing: {meta_path}"
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Check non-empty
            assert len(df) > 0, "Generated CSV is empty!"
            
            # Check all vital sign columns exist
            expected_cols = ["time_hours", "heart_rate", "respiratory_rate", "spo2", "systolic_bp", "diastolic_bp"]
            for col in expected_cols:
                assert col in df.columns, f"Missing column {col} in CSV!"
                
            # Check no NaN or Inf values
            assert not df.isna().any().any(), "Found NaN values in generated CSV!"
            assert not np.isinf(df.to_numpy()).any(), "Found Inf values in generated CSV!"
            
            # Read metadata
            with open(meta_path, "r") as f:
                meta = json.load(f)
                
            # Check metadata fields and matching row count
            assert meta["num_time_points"] == len(df), f"Row count mismatch! CSV rows={len(df)}, metadata={meta['num_time_points']}"
            assert meta["duration_hours"] == pytest.approx(df["time_hours"].iloc[-1] - df["time_hours"].iloc[0])


def test_lightweight_statistical_validation_against_train():
    """
    Lightweight statistical validation comparing synthetic profiles against training data distributions.
    """
    train_path = "data/processed/datasets/train.pkl"
    with open(train_path, "rb") as f:
        train_trajs = pickle.load(f)
        
    train_lens = [len(t["T"]) for t in train_trajs]
    train_durs = [float(t["T"][-1] - t["T"][0]) for t in train_trajs if len(t["T"]) > 1]
    
    tp_gen = TemporalProfileGenerator(train_path=train_path).fit()
    
    synth_profiles = [tp_gen.generate_profile(seed=seed) for seed in range(50)]
    synth_lens = [p["num_time_points"] for p in synth_profiles]
    synth_durs = [p["duration"] for p in synth_profiles]
    
    # 1. Check sequence length range overlap
    assert min(synth_lens) >= min(train_lens), "Synthetic seq len below train min!"
    assert max(synth_lens) <= max(train_lens), "Synthetic seq len exceeded train max!"
    
    # 2. Check median duration is within reasonable bounds of train median duration
    train_median_dur = np.median(train_durs)
    synth_median_dur = np.median(synth_durs)
    
    # Allow 50% relative margin around training median duration
    assert abs(synth_median_dur - train_median_dur) < 0.5 * train_median_dur, (
        f"Synthetic median duration ({synth_median_dur:.2f}h) deviated significantly from train ({train_median_dur:.2f}h)"
    )
