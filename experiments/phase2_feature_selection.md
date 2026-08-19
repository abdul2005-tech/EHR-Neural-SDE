# Phase 2: Feature Selection & Vital Sign Characterization Report

**Project**: EHR-Neural-SDE  
**Date**: August 18, 2026  
**Focus**: Identification and empirical characterization of continuous-time ICU vital sign features  
**Primary Tables Analyzed**:
- `data/raw/mimic_demo/icu/d_items.csv.gz`
- `data/raw/mimic_demo/icu/chartevents.csv.gz`
- `data/raw/mimic_demo/icu/icustays.csv.gz`

---

## 1. Executive Summary

In Phase 2, we conducted an empirical feature exploration of MIMIC-IV MetaVision ICU chart items to establish our candidate continuous-time observation variables. We searched `d_items.csv.gz` across 5 target clinical domains: **Heart Rate**, **Respiratory Rate**, **Oxygen Saturation (SpO2)**, **Systolic Blood Pressure**, and **Diastolic Blood Pressure**.

We extracted **80,364 candidate vital sign observations** across 140 ICU stays (100 unique patients) and analyzed their value ranges, units, missingness, measurement frequencies, and inter-observation time gaps ($\Delta t$).

---

## 2. Candidate Items in `d_items.csv.gz`

A dictionary search identified **41 matching item IDs** across the 5 target categories. The table below lists all candidate item IDs along with their total observation count in `chartevents.csv.gz`:

| Category | Item ID | Label | Category | Dict Unit | Total Observations | Range (`valuenum`) | Missingness (`valuenum`) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heart Rate** | **`220045`** | **Heart Rate** | **Routine Vital Signs** | **bpm** | **13,913** | **0 – 200 bpm** | **0.00%** |
| Heart Rate | `220046` | Heart Rate Alarm - High | Routine Vital Signs | bpm | 1,217 | 13 – 170 bpm | 0.00% |
| Heart Rate | `220047` | Heart Rate Alarm - Low | Routine Vital Signs | bpm | 1,208 | 34 – 5050 bpm | 0.00% |
| **Resp Rate** | **`220210`** | **Respiratory Rate** | **Respiratory** | **insp/min** | **13,913** | **0 – 58 insp/min**| **0.00%** |
| Resp Rate | `224690` | Respiratory Rate (Total)| Respiratory | insp/min | 1,331 | 0 – 44 insp/min | 0.00% |
| Resp Rate | `224689` | Respiratory Rate (spontaneous) | Respiratory | insp/min | 1,314 | 0 – 44 insp/min | 0.00% |
| Resp Rate | `224688` | Respiratory Rate (Set) | Respiratory | insp/min | 801 | 0 – 35 insp/min | 0.00% |
| **SpO2** | **`220277`** | **O2 saturation pulseoxymetry** | **Respiratory** | **%** | **13,540** | **29 – 100 %** | **0.00%** |
| SpO2 | `223770` | O2 Saturation Alarm - Low | Respiratory | % | 1,212 | 8 – 93 % | 0.00% |
| SpO2 | `223769` | O2 Saturation Alarm - High| Respiratory | % | 1,208 | 35 – 160 % | 0.00% |
| SpO2 | `226253` | SpO2 Desat Limit | Respiratory | % | 1,176 | 65 – 858 % | 0.00% |
| SpO2 | `220227` | Arterial O2 Saturation | Blood Gases | % | 104 | 43 – 99 % | 0.00% |
| **Systolic BP** | **`220179`** | **Non Invasive Blood Pressure systolic** | **Routine Vital Signs** | **mmHg** | **8,347** | **46 – 215 mmHg** | **0.00%** |
| **Systolic BP** | **`220050`** | **Arterial Blood Pressure systolic** | **Routine Vital Signs** | **mmHg** | **5,525** | **25 – 207 mmHg** | **0.00%** |
| Systolic BP | `225309` | ART BP Systolic | Routine Vital Signs | mmHg | 486 | 71 – 195 mmHg | 0.00% |
| Systolic BP | `227243` | Manual BP Systolic Right | Routine Vital Signs | mmHg | 5 | 99 – 152 mmHg | 0.00% |
| Systolic BP | `224167` | Manual BP Systolic Left | Routine Vital Signs | mmHg | 3 | 99 – 138 mmHg | 0.00% |
| **Diastolic BP**| **`220180`** | **Non Invasive Blood Pressure diastolic**| **Routine Vital Signs** | **mmHg** | **8,349** | **18 – 162 mmHg** | **0.00%** |
| **Diastolic BP**| **`220051`** | **Arterial Blood Pressure diastolic**| **Routine Vital Signs** | **mmHg** | **5,524** | **21 – 165 mmHg** | **0.00%** |
| Diastolic BP| `225310` | ART BP Diastolic | Routine Vital Signs | mmHg | 486 | 39 – 124 mmHg | 0.00% |
| Diastolic BP| `227242` | Manual BP Diastolic Right | Routine Vital Signs | mmHg | 5 | 50 – 78 mmHg | 0.00% |
| Diastolic BP| `224643` | Manual BP Diastolic Left | Routine Vital Signs | mmHg | 3 | 64 – 83 mmHg | 0.00% |

---

## 3. Recommended Item IDs & Rationale

We recommend selecting the following **7 primary Metavision Item IDs**:

### A. Recommended Core Features

1. **Heart Rate**: `itemid: 220045` (`Heart Rate`)
   - *Rationale*: Primary routine heart rate monitor signal. Contains 13,913 measurements across all 140 ICU stays with 0.0% missing values.
2. **Respiratory Rate**: `itemid: 220210` (`Respiratory Rate`)
   - *Rationale*: Primary respiratory rate signal. Contains 13,913 measurements with 0.0% missing values.
3. **Oxygen Saturation (SpO2)**: `itemid: 220277` (`O2 saturation pulseoxymetry`)
   - *Rationale*: Primary continuous pulse oximetry saturation reading. Contains 13,540 measurements with 0.0% missing values.
4. **Systolic Blood Pressure (NIBP)**: `itemid: 220179` (`Non Invasive Blood Pressure systolic`)
   - *Rationale*: Non-invasive automated blood pressure cuff reading for non-arterial line patients (8,347 measurements).
5. **Diastolic Blood Pressure (NIBP)**: `itemid: 220180` (`Non Invasive Blood Pressure diastolic`)
   - *Rationale*: Non-invasive automated blood pressure cuff reading (8,349 measurements).
6. **Systolic Blood Pressure (Arterial Line)**: `itemid: 220050` (`Arterial Blood Pressure systolic`)
   - *Rationale*: Direct invasive arterial line blood pressure monitoring for critically ill patients (5,525 measurements).
7. **Diastolic Blood Pressure (Arterial Line)**: `itemid: 220051` (`Arterial Blood Pressure diastolic`)
   - *Rationale*: Direct invasive arterial line blood pressure monitoring (5,524 measurements).

### B. Blood Pressure Representation & Modality Handling
In MIMIC-IV ICU, patients receive blood pressure monitoring either via **Non-Invasive Cuff (NIBP)** (`220179`/`220180`) or **Invasive Arterial Line (ABP)** (`220050`/`220051`). 
- Patients with arterial lines have more frequent blood pressure measurements (median gap ~34 minutes).
- Patients with NIBP cuffs have hourly measurements (median gap ~60 minutes).
- **Strategy**: Combine NIBP and Arterial Line measurements into unified `Systolic_BP` and `Diastolic_BP` continuous channels during sequence construction (using Arterial Line if both exist at the exact same timestamp), while maintaining a binary indicator for arterial line presence.

---

## 4. Empirical Data Quality & Missingness Analysis

### A. Missingness
For all 7 recommended item IDs, `valuenum` missingness is **0.00%**. Every chart event record for these items contains a valid floating-point numerical value.

### B. Units of Measurement
Units in `chartevents` are standardized across MetaVision items:
- Heart Rate: `bpm` (Beats Per Minute)
- Respiratory Rate: `insp/min` (Inspirations Per Minute)
- SpO2: `%` (Percentage)
- Blood Pressure: `mmHg` (Millimeters of Mercury)

### C. Physiological Range Check
The observed numerical ranges in the raw data match valid physiological bounds:
- Heart Rate: `[0.0, 200.0] bpm`
- Respiratory Rate: `[0.0, 58.0] insp/min`
- SpO2: `[29.0, 100.0] %`
- Systolic BP: `[25.0, 215.0] mmHg`
- Diastolic BP: `[18.0, 165.0] mmHg`

### D. Duplicate Check & ICU Stay Window Bounds
- **ICU Stay Bounds**: **0 observations** occurred outside `[intime, outtime]` bounds in `icustays.csv.gz`.
- **Duplicates**: Only **6 total duplicate measurements** (`stay_id`, `itemid`, `charttime`) exist across the dataset.

---

## 5. Continuous Time-Gap ($\Delta t$) Analysis

To preserve irregular timing for Neural SDE continuous-time modeling, we analyzed inter-observation time gaps:
$$\Delta t_i = t_i - t_{i-1} \quad \text{(in minutes)}$$

### A. Overall Inter-Observation Time-Gap Distribution
When considering all 7 vital sign channels combined per patient stay:

- **Mean $\Delta t$**: `37.26 minutes`
- **Median $\Delta t$**: `58.00 minutes`
- **25th Percentile ($P_{25}$)**: `4.00 minutes`
- **75th Percentile ($P_{75}$)**: `60.00 minutes`
- **90th Percentile ($P_{90}$)**: `60.00 minutes`
- **95th Percentile ($P_{95}$)**: `60.00 minutes`
- **Min $\Delta t$**: `1.00 minute`
- **Max $\Delta t$**: `2,626.00 minutes` (~43.7 hours)
- **Standard Deviation**: `34.13 minutes`

### B. Per-Variable Time-Gap Breakdown

| Target Variable | Item ID | Measurement Count | Median $\Delta t$ | $P_{25}$ $\Delta t$ | $P_{75}$ $\Delta t$ | Min $\Delta t$ | Max $\Delta t$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Heart Rate** | `220045` | 13,773 | 60.0 min | 60.0 min | 60.0 min | 1.0 min | 2,626.0 min |
| **Respiratory Rate** | `220210` | 13,773 | 60.0 min | 60.0 min | 60.0 min | 1.0 min | 2,626.0 min |
| **SpO2** | `220277` | 13,400 | 60.0 min | 60.0 min | 60.0 min | 1.0 min | 831.0 min |
| **Systolic NIBP** | `220179` | 8,208 | 60.0 min | 59.0 min | 60.0 min | 1.0 min | 7,921.0 min |
| **Diastolic NIBP** | `220180` | 8,210 | 60.0 min | 59.0 min | 60.0 min | 1.0 min | 7,921.0 min |
| **Systolic Arterial** | `220050` | 5,460 | 60.0 min | 34.0 min | 60.0 min | 1.0 min | 7,795.0 min |
| **Diastolic Arterial** | `220051` | 5,459 | 60.0 min | 34.0 min | 60.0 min | 1.0 min | 7,795.0 min |

### Key Temporal Insights
1. **Quasi-Regular Baseline Sampling**: The dominant measurement frequency is hourly logging ($\Delta t = 60$ minutes).
2. **Asynchronous Multi-Channel Shifts**: Because vital sign channels are recorded at slightly staggered times (e.g. HR at $t=0$, SpO2 at $t=4$ mins, NIBP at $t=5$ mins), the overall multi-channel $\Delta t$ median is **58 minutes** with a 25th percentile of **4 minutes**.
3. **High-Frequency Bursts**: During clinical deterioration, measurements occur in rapid succession ($\Delta t \in [1, 15]$ minutes).

---

## 6. Sample ICU Stay Trajectory Profile

Below is a breakdown of 5 representative ICU stays from the dataset:

| Stay ID | Duration (Hours) | Total Observations | Heart Rate Count | Resp Rate Count | SpO2 Count | Systolic BP Count | Diastolic BP Count |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `30000646` | 33.32 hrs | 201 | 35 | 34 | 34 | 33 | 33 |
| `30008064` | 308.17 hrs | 1,842 | 313 | 313 | 313 | 312 | 312 |
| `30008453` | 13.97 hrs | 87 | 15 | 15 | 15 | 14 | 14 |
| `30017180` | 27.60 hrs | 165 | 28 | 28 | 28 | 27 | 27 |
| `30022201` | 89.20 hrs | 525 | 90 | 90 | 90 | 89 | 89 |

---

## 7. Next Steps & Phase 3 Transition

### Recommended Next Preprocessing Step (Phase 3)
Build `src/data/sequence_builder.py` to construct patient-level sequence dictionary structures:
1. Filter `chartevents` for the 7 recommended vital sign item IDs.
2. Align timestamps relative to ICU admission: $t_{\text{rel}} = t_{\text{charttime}} - t_{\text{intime}}$ (in hours).
3. Compute exact inter-observation deltas $\Delta t_i = t_i - t_{i-1}$.
4. Construct observation tensor $X \in \mathbb{R}^{N \times D}$, timestamp tensor $T \in \mathbb{R}^N$, time-gap tensor $\Delta T \in \mathbb{R}^N$, and observation mask $M \in \{0, 1\}^{N \times D}$ without grid resampling or forward filling.

---

*Report generated automatically for EHR-Neural-SDE Phase 2.*
