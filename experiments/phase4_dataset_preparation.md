# Phase 4: Machine Learning Dataset Preparation & Leakage-Safe Splitting Report

**Project**: EHR-Neural-SDE  
**Date**: August 18, 2026  
**Input Dataset**: `data/processed/trajectories/icu_trajectories.pkl`  
**Output Datasets**:
- `data/processed/datasets/train.pkl` (98 trajectories)
- `data/processed/datasets/val.pkl` (25 trajectories)
- `data/processed/datasets/test.pkl` (17 trajectories)
- `data/processed/scaler.json` (Training-only normalization parameters)

---

## 1. Executive Summary

In Phase 4, we constructed a leakage-safe machine learning dataset from our Phase 3 continuous-time ICU trajectories. To prevent data leakage across ICU readmissions, the dataset was partitioned strictly at the **patient level** (`subject_id`).

Continuous vital sign variables were normalized using standardization ($x_{\text{norm}} = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$) computed **exclusively on observed training set values**. Missing observations are represented numerically as `0.0` in $X$ while preserving the binary observation mask $M \in \{0, 1\}^{N \times 5}$ unchanged.

---

## A. Patient Split Statistics

Patients were randomly assigned to splits using a fixed random seed (`seed = 42`):

| Split | Patient Count (`subject_id`) | Patient Percentage | Patient Leakage Check |
| :--- | :---: | :---: | :---: |
| **Train** | 70 | 70.0% | **0 Overlap** |
| **Validation** | 15 | 15.0% | **0 Overlap** |
| **Test** | 15 | 15.0% | **0 Overlap** |
| **Total** | **100** | **100.0%** | **PASSED** |

---

## B. Trajectory Split Statistics

Because patients may have multiple ICU stays, trajectories were partitioned according to their patient assignment:

| Split | Trajectory Count (`stay_id`) | Trajectory Percentage | Total Timestamps ($N$) |
| :--- | :---: | :---: | :---: |
| **Train** | 98 | 70.00% | 13,205 |
| **Validation** | 25 | 17.86% | 3,745 |
| **Test** | 17 | 12.14% | 2,607 |
| **Total** | **140** | **100.00%** | **19,557** |

---

## C. Training Normalization Statistics (`scaler.json`)

Normalization statistics were computed **exclusively using observed training values** where $M[i, j] = 1$:

| Feature Index | Feature Name | MIMIC Item IDs | Training Mean ($\mu_{\text{train}}$) | Training Std ($\sigma_{\text{train}}$) | Training Observed Count |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **`0`** | **Heart Rate** | `220045` | `91.0554 bpm` | `18.9470 bpm` | 9,355 |
| **`1`** | **Respiratory Rate** | `220210` | `19.7426 insp/min` | `5.5015 insp/min` | 9,368 |
| **`2`** | **SpO2** | `220277` | `96.8310 %` | `3.1075 %` | 9,061 |
| **`3`** | **Systolic BP** | `220179` (NIBP), `220050` (ART) | `113.0366 mmHg` | `21.5438 mmHg` | 9,089 |
| **`4`** | **Diastolic BP** | `220180` (NIBP), `220051` (ART) | `62.3311 mmHg` | `13.4937 mmHg` | 9,089 |

---

## D. Missingness Statistics by Split

The proportion of missing feature values ($M=0$) remains consistent across all three splits:

| Feature | Train Missing % ($M=0$) | Val Missing % ($M=0$) | Test Missing % ($M=0$) |
| :--- | :---: | :---: | :---: |
| **Heart Rate** | 29.16% | 27.64% | 31.02% |
| **Respiratory Rate** | 29.06% | 27.69% | 31.52% |
| **SpO2** | 31.38% | 29.35% | 31.38% |
| **Systolic BP** | 31.17% | 28.70% | 30.69% |
| **Diastolic BP** | 31.17% | 28.70% | 30.69% |

---

## E. Sequence Length Statistics by Split ($N$)

| Split | Mean $N$ | Median $N$ | Min $N$ | Max $N$ | Std $N$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | 134.74 | 74.0 | 1 | 817 | 152.02 |
| **Validation** | 149.80 | 66.0 | 4 | 569 | 163.69 |
| **Test** | 153.35 | 73.0 | 10 | 449 | 147.28 |

---

## F. Time-Gap ($\Delta T$) Statistics by Split (in Minutes)

Inter-observation time gaps $\Delta T$ retain their exact continuous distribution across all splits:

| Split | Mean $\Delta T$ | Median $\Delta T$ | $P_{25}$ $\Delta T$ | $P_{75}$ $\Delta T$ | Min $\Delta T$ | Max $\Delta T$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train** | 36.96 min | 58.00 min | 4.00 min | 60.00 min | 1.0 min | 2,626.0 min |
| **Validation** | 37.64 min | 58.00 min | 4.00 min | 60.00 min | 1.0 min | 868.0 min |
| **Test** | 37.94 min | 58.00 min | 4.00 min | 60.00 min | 1.0 min | 840.0 min |

---

## G. Data Leakage Checks

1. **Patient-Level Isolation**: $\text{Train}_{\text{subjects}} \cap \text{Val}_{\text{subjects}} = \emptyset$, $\text{Train}_{\text{subjects}} \cap \text{Test}_{\text{subjects}} = \emptyset$, and $\text{Val}_{\text{subjects}} \cap \text{Test}_{\text{subjects}} = \emptyset$. Zero patient leakage occurred.
2. **Scaler Isolation**: Validation and Test sets were transformed strictly using training set parameters ($\mu_{\text{train}}, \sigma_{\text{train}}$). Zero validation/test information was used during scaler fitting.

---

## H. Validation Results (`validate_dataset.py`)

All **12 automated quality control checks** passed:

| ID | Check Description | Status |
| :---: | :--- | :---: |
| **1** | No patient appears in multiple splits | **PASSED** |
| **2** | No NaNs exist in normalized matrix $X$ | **PASSED** |
| **3** | Mask $M$ contains exclusively binary values $\{0, 1\}$ | **PASSED** |
| **4** | Relative time $T$ remains strictly monotonic | **PASSED** |
| **5** | Inter-observation gap $\Delta T$ remains non-negative | **PASSED** |
| **6** | $T$ and $\Delta T$ remain unmodified by feature normalization | **PASSED** |
| **7** | Training set observed normalized means are $\approx 0.0$ | **PASSED** |
| **8** | Training set observed normalized stds are $\approx 1.0$ | **PASSED** |
| **9** | Validation and Test sets use training scaler ONLY | **PASSED** |
| **10** | Sequence lengths $N$ remain identical to Phase 3 | **PASSED** |
| **11** | Zero new timestamps created | **PASSED** |
| **12** | Zero observations deleted due to missing features | **PASSED** |

---

## I. Example Processed Trajectory Record

Below is a sample processed trajectory snippet from `train.pkl` (`stay_id: 32554129`):

```python
{
    "subject_id": 10020187,
    "hadm_id": 26842957,
    "stay_id": 32554129,
    "intime": "2110-01-01 00:00:00",
    "outtime": "2110-01-01 20:56:40",
    "los_hours": 20.94,
    "X": array([
        [-0.6889, -1.9527,  1.0198,  0.    ,  0.    ],  # Normalized values (0.0 where unobserved)
        [ 0.    ,  0.    ,  0.    ,  1.2516,  0.4942],
        [-1.5335, -0.6803, -0.5892,  0.9266,  0.1237]
    ], dtype=float32),
    "T": array([0.087222, 0.137222, 0.687222], dtype=float64),  # Relative hours
    "DeltaT": array([0.00, 0.05, 0.55], dtype=float64),        # Time gaps in hours
    "M": array([
        [1, 1, 1, 0, 0],  # Binary mask
        [0, 0, 0, 1, 1],
        [1, 1, 1, 1, 1]
    ], dtype=uint8),
    "bp_modality": array(['none', 'arterial', 'arterial'], dtype=object)
}
```

---

*Phase 4 Report for EHR-Neural-SDE.*
