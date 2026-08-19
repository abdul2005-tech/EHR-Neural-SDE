# Phase 3: Longitudinal ICU Trajectory Dataset Report

**Project**: EHR-Neural-SDE  
**Date**: August 18, 2026  
**Dataset Output**: `data/processed/trajectories/icu_trajectories.pkl`  
**CSV Preview Output**: `data/processed/trajectories/trajectory_preview.csv`  

---

## Executive Overview

In Phase 3, we constructed a model-ready continuous-time longitudinal ICU trajectory dataset from the MIMIC-IV Demo dataset. Each trajectory represents an independent ICU stay parameterized by continuous relative time $T$ (in hours from ICU arrival `intime`), inter-observation time gaps $\Delta T$ (in hours), 5 physiological vital sign channels ($X \in \mathbb{R}^{N \times 5}$), a binary observation mask ($M \in \{0, 1\}^{N \times 5}$), and a blood pressure modality tracking array.

**Core Rules Followed**:
- **Zero Fixed-Grid Resampling**: Irregular observation timing is strictly preserved.
- **Zero Forward-Filling / Imputation**: Missing channel values are represented as `NaN` in $X$ and `0` in mask $M$.
- **Zero Normalization / Model Fitting**: Values are retained in their natural physical units.
- **Arterial BP Preference**: Exact timestamp collisions between arterial line and non-invasive cuff blood pressure measurements resolve to arterial line readings.

---

## A. Number of Trajectories
- **Total Trajectories Created**: `140` (One independent trajectory per ICU stay).

---

## B. Number of Unique Patients
- **Total Unique Patients**: `100` (`subject_id`).

---

## C. Number of ICU Stays
- **Total Unique ICU Stays**: `140` (`stay_id`).

---

## D. Trajectory Length Distribution ($N$)

Across all 140 trajectories, a total of **19,557 unique observation timestamps** were generated:

| Metric | Trajectory Length $N$ (Timestamps per Stay) |
| :--- | :--- |
| **Mean $N$** | `139.69` |
| **Median $N$** | `73.50` |
| **25th Percentile ($P_{25}$)** | `35.00` |
| **75th Percentile ($P_{75}$)** | `175.00` |
| **Minimum $N$** | `1` |
| **Maximum $N$** | `817` |
| **Standard Deviation** | `153.28` |

---

## E. Duration Distribution (ICU Length of Stay)

Trajectory duration is measured in hours from `intime` ($t_{\text{rel}} = 0.0$) to `outtime`:

| Metric | ICU Stay Duration (Hours) | ICU Stay Duration (Days) |
| :--- | :--- | :--- |
| **Mean Duration** | `88.31 hours` | `3.68 days` |
| **Median Duration** | `60.10 hours` | `2.50 days` |
| **25th Percentile ($P_{25}$)** | `30.82 hours` | `1.28 days` |
| **75th Percentile ($P_{75}$)** | `114.77 hours` | `4.78 days` |
| **Minimum Duration** | `0.57 hours` | `0.02 days` (14 minutes) |
| **Maximum Duration** | `492.69 hours` | `20.53 days` |

---

## F. Missingness per Variable

In our **union-of-timestamps representation**, a row exists for every unique timestamp where *at least one* vital sign variable was logged. Missingness represents unmeasured channels at a given timestamp ($M[i, j] = 0$):

| Channel Index | Physiological Feature | MIMIC Item IDs | Total Observed Cells | Missingness Percentage ($M=0$) |
| :---: | :--- | :--- | :--- | :--- |
| **`X[:, 0]`** | **Heart Rate** | `220045` | 13,863 / 19,557 | **29.11%** |
| **`X[:, 1]`** | **Respiratory Rate** | `220210` | 13,861 / 19,557 | **29.12%** |
| **`X[:, 2]`** | **SpO2** | `220277` | 13,496 / 19,557 | **30.99%** |
| **`X[:, 3]`** | **Systolic Blood Pressure** | `220179` (NIBP), `220050` (ART) | 13,567 / 19,557 | **30.63%** |
| **`X[:, 4]`** | **Diastolic Blood Pressure** | `220180` (NIBP), `220051` (ART) | 13,567 / 19,557 | **30.63%** |

---

## G. Time-Gap ($\Delta T$) Statistics

Inter-observation time gaps $\Delta T[i] = T[i] - T[i-1]$ govern continuous-time latent state transitions:

| Metric | $\Delta T$ (Hours) | $\Delta T$ (Minutes) |
| :--- | :--- | :--- |
| **Mean $\Delta T$** | `0.6203 hours` | `37.22 minutes` |
| **Median $\Delta T$** | `0.9667 hours` | `58.00 minutes` |
| **25th Percentile ($P_{25}$)** | `0.0667 hours` | `4.00 minutes` |
| **75th Percentile ($P_{75}$)** | `1.0000 hour` | `60.00 minutes` |
| **90th Percentile ($P_{90}$)** | `1.0000 hour` | `60.00 minutes` |
| **Minimum $\Delta T$** | `0.0167 hours` | `1.00 minute` |
| **Maximum $\Delta T$** | `43.7667 hours` | `2,626.00 minutes` |

---

## H. Example Trajectories

### Example Trajectory 1: Short ICU Stay (`stay_id: 32554129`)
- **Patient**: `subject_id: 10020187`, **Admission**: `hadm_id: 26842957`
- **ICU Duration**: `20.94 hours`, **Total Timestamps ($N$)**: `26`

```
Time (hrs)  DeltaT (hrs)  HR (bpm)  RR (insp/min)  SpO2 (%)  SBP (mmHg)  DBP (mmHg)  BP Modality
------------------------------------------------------------------------------------------------
0.0872      0.0000        78        9              100       NaN         NaN         none
0.1372      0.0500        NaN       NaN            NaN       140         69          arterial
0.6872      0.5500        62        16             95        133         64          arterial
1.6872      1.0000        61        17             98        116         54          arterial
...
```

### Example Trajectory 2: Extended ICU Stay (`stay_id: 31269608`)
- **Patient**: `subject_id: 10000032`, **Admission**: `hadm_id: 22595853`
- **ICU Duration**: `184.86 hours` (~7.7 days), **Total Timestamps ($N$)**: `364`

---

## I. Quality-Control Results

All **13 automated quality control checks** in `src/data/validate_trajectories.py` passed cleanly:

| QC Check ID | Description | Validation Status |
| :---: | :--- | :---: |
| **Check 1** | No negative $T$ values ($T \ge 0$) | **PASSED** |
| **Check 2** | $T$ is monotonically increasing within every stay ($T[i] > T[i-1]$) | **PASSED** |
| **Check 3** | $\Delta T[0] == 0.0$ for all trajectories | **PASSED** |
| **Check 4** | $\Delta T[i] \ge 0.0$ for all steps | **PASSED** |
| **Check 5** | $X$ and $M$ have identical shape $[N, 5]$ | **PASSED** |
| **Check 6** | $T$ and $\Delta T$ have identical length $N$ matching $X$ | **PASSED** |
| **Check 7** | Mask $M$ contains exclusively binary values $\{0, 1\}$ | **PASSED** |
| **Check 8** | Every observed non-NaN cell in $X$ has $M=1$ | **PASSED** |
| **Check 9** | Every missing NaN cell in $X$ has $M=0$ | **PASSED** |
| **Check 10** | Zero observations fall outside ICU stay bounds $[t_{\text{intime}}, t_{\text{outtime}}]$ | **PASSED** |
| **Check 11** | Zero duplicate timestamps ($stay\_id + charttime$) remain | **PASSED** |
| **Check 12** | Array data structures match expected shapes | **PASSED** |
| **Check 13** | Zero `NaN` values present in $T$ or $\Delta T$ | **PASSED** |

---

## J. Warnings & Data Observations

1. **Short Trajectories**: 2 out of 140 ICU stays have $N < 5$ timestamps due to very short stay durations ($< 1$ hour). These are preserved without filtering to avoid data bias, but can be filtered by sequence length during model dataloading.
2. **BP Modality Shift**: In extended ICU stays, patients often transition from invasive arterial blood pressure lines to non-invasive cuffs as their health stabilizes. The `bp_modality` array captures these modality shifts dynamically per timestamp.

---

*Phase 3 Trajectory Dataset Report for EHR-Neural-SDE.*
