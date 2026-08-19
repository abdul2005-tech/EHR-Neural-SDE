# Phase 1: MIMIC-IV Demo Dataset Inspection & Documentation Report

**Project**: EHR-Neural-SDE  
**Date**: August 18, 2026  
**Dataset**: MIMIC-IV Demo (v2.2)  
**Location**: `data/raw/mimic_demo/`  

---

## A. Dataset Overview

The **MIMIC-IV Demo** dataset is a de-identified, publicly accessible subset of the Medical Information Mart for Intensive Care IV (MIMIC-IV) database. It comprises detailed clinical records for **100 unique patients** admitted to Beth Israel Deaconess Medical Center (BIDMC).

### Key Dataset Statistics
- **Total Patients (`subject_id`)**: 100
- **Total Hospital Admissions (`hadm_id`)**: 275
- **Total ICU Stays (`stay_id`)**: 140
- **Total Ingested Records**: ~1,100,000 across 32 data tables
- **Date Range**: De-identified and shifted to future centuries (`2110-04-08` to `2202-08-26`) to preserve exact patient privacy while maintaining precise relative time intervals ($\Delta t$).

---

## B. Directory Structure

The dataset is stored in `data/raw/mimic_demo/` and partitioned into two primary functional subdirectories: `hosp/` (hospital-wide administrative, laboratory, and prescription records) and `icu/` (high-frequency bedside monitoring and ICU intervention events).

```
data/raw/mimic_demo/
├── LICENSE.txt
├── README.txt
├── SHA256SUMS.txt
├── demo_subject_id.csv
├── hosp/
│   ├── admissions.csv.gz
│   ├── d_hcpcs.csv.gz
│   ├── d_icd_diagnoses.csv.gz
│   ├── d_icd_procedures.csv.gz
│   ├── d_labitems.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── drgcodes.csv.gz
│   ├── emar.csv.gz
│   ├── emar_detail.csv.gz
│   ├── hcpcsevents.csv.gz
│   ├── labevents.csv.gz
│   ├── microbiologyevents.csv.gz
│   ├── omr.csv.gz
│   ├── patients.csv.gz
│   ├── pharmacy.csv.gz
│   ├── poe.csv.gz
│   ├── poe_detail.csv.gz
│   ├── prescriptions.csv.gz
│   ├── procedures_icd.csv.gz
│   ├── provider.csv.gz
│   ├── services.csv.gz
│   └── transfers.csv.gz
└── icu/
    ├── caregiver.csv.gz
    ├── chartevents.csv.gz
    ├── d_items.csv.gz
    ├── datetimeevents.csv.gz
    ├── icustays.csv.gz
    ├── ingredientevents.csv.gz
    ├── inputevents.csv.gz
    ├── outputevents.csv.gz
    └── procedureevents.csv.gz
```

---

## C. Table-by-Table Description

### Hospital Module (`hosp/`) — 22 Tables

| Table Name | Description / Purpose | Rows | Unique Subjects | Unique Admissions | Primary Identifiers | Key Timestamp Columns |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`patients`** | Demographics, gender, anchor age, date of death | 100 | 100 | 0 | `subject_id` | `dod` |
| **`admissions`** | Hospitalization records, admission/discharge times, admission type | 275 | 100 | 275 | `hadm_id` | `admittime`, `dischtime`, `deathtime` |
| **`transfers`** | Physical location tracking within hospital (wards, ED, ICU) | 1,190 | 100 | 275 | `transfer_id` | `intime`, `outtime` |
| **`labevents`** | Longitudinal blood chemistry, hematology, and lab test results | 107,727 | 100 | 252 | `labevent_id` | `charttime`, `storetime` |
| **`prescriptions`** | Medication prescriptions ordered by physicians | 18,087 | 100 | 250 | Foreign (`hadm_id`, `pharmacy_id`) | `starttime`, `stoptime` |
| **`pharmacy`** | Pharmacy system fulfillment and administration details | 15,306 | 100 | 250 | `pharmacy_id` | `starttime`, `stoptime`, `verifiedtime` |
| **`emar`** | Electronic Medication Administration Record headers | 35,835 | 65 | 181 | `emar_id` | `charttime`, `scheduletime`, `storetime` |
| **`emar_detail`** | Granular administration details (doses, route, site) | 72,018 | 65 | 0 | `emar_id`, `emar_seq` | None |
| **`diagnoses_icd`** | Billed ICD-9 / ICD-10 diagnosis codes for hospital stays | 4,506 | 100 | 275 | Foreign (`hadm_id`, `seq_num`) | None (linked via `admissions.dischtime`) |
| **`procedures_icd`** | Billed ICD-9 / ICD-10 procedure codes | 722 | 92 | 187 | Foreign (`hadm_id`, `seq_num`) | `chartdate` |
| **`microbiologyevents`**| Cultures (blood, urine, sputum) and antibiotic sensitivity tests | 2,899 | 97 | 171 | `microevent_id` | `chartdate`, `charttime`, `storetime` |
| **`omr`** | Outpatient medical records (vital signs, BMI, height, weight) | 2,964 | 79 | 0 | Foreign (`subject_id`) | `chartdate` |
| **`poe`** | Provider Order Entry headers (orders for labs, meds, consults) | 45,154 | 100 | 272 | `poe_id` | `ordertime` |
| **`poe_detail`** | Granular order entry action parameters | 3,795 | 100 | 0 | `poe_id` | None |
| **`services`** | Clinical service transfers (e.g. SURG, MED, NMED) | 319 | 100 | 275 | Foreign (`hadm_id`, `transfertime`)| `transfertime` |
| **`drgcodes`** | Diagnosis Related Group billing codes | 454 | 100 | 233 | Foreign (`hadm_id`) | None |
| **`hcpcsevents`** | HCPCS procedure billing events | 61 | 18 | 41 | Foreign (`hadm_id`) | `chartdate` |
| **`d_hcpcs`** | HCPCS billing code definitions lookup table | 89,200 | 0 | 0 | `code` | None |
| **`d_icd_diagnoses`** | ICD diagnosis code lookup table | 109,775 | 0 | 0 | `icd_code`, `icd_version` | None |
| **`d_icd_procedures`**| ICD procedure code lookup table | 85,257 | 0 | 0 | `icd_code`, `icd_version` | None |
| **`d_labitems`** | Laboratory item dictionary lookup table | 1,622 | 0 | 0 | `itemid` | None |
| **`provider`** | De-identified provider lookup table | 40,508 | 0 | 0 | `provider_id` | None |

---

### ICU Module (`icu/`) — 9 Tables

| Table Name | Description / Purpose | Rows | Unique Subjects | Unique Admissions | Unique Stays | Primary Identifiers | Key Timestamp Columns |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`icustays`** | ICU stay unit tracking, arrival and discharge timestamps | 140 | 100 | 128 | 140 | `stay_id` | `intime`, `outtime` |
| **`chartevents`** | High-frequency ICU bedside chart observations (vitals, GCS, ventilator) | 668,862 | 100 | 128 | 140 | Composite (`stay_id`, `itemid`, `charttime`) | `charttime`, `storetime` |
| **`inputevents`** | Continuous IV fluid infusions, drug titrations, and bolus administration | 20,404 | 99 | 127 | 138 | `orderid` | `starttime`, `endtime`, `storetime` |
| **`outputevents`** | Fluid outputs (urine, drains, tubes) | 9,362 | 100 | 126 | 137 | Composite (`stay_id`, `itemid`, `charttime`) | `charttime`, `storetime` |
| **`procedureevents`**| Bedside ICU procedures (intubation, mechanical ventilation, line placement) | 1,468 | 100 | 128 | 138 | `orderid` | `starttime`, `endtime`, `storetime` |
| **`datetimeevents`** | Date/time measurements recorded by ICU nurses | 15,280 | 100 | 127 | 138 | Composite (`stay_id`, `itemid`, `charttime`) | `charttime`, `storetime`, `value` |
| **`ingredientevents`**| Breakdown of active drug ingredients in IV infusions | 25,728 | 99 | 127 | 138 | Composite (`stay_id`, `itemid`, `starttime`) | `starttime`, `endtime`, `storetime` |
| **`d_items`** | ICU item lookup dictionary (`itemid` -> label, category, unit) | 4,014 | 0 | 0 | 0 | `itemid` | None |
| **`caregiver`** | De-identified caregiver lookup table | 15,468 | 0 | 0 | 0 | `caregiver_id` | None |

---

## D. Key Columns and Identifiers

The MIMIC-IV database enforces a strict 3-tier hierarchical identity structure:

```
[Patient Level]          subject_id (e.g. 10000032)
                                │
                                ▼
[Admission Level]        hadm_id (e.g. 22595853)
                                │
                                ▼
[ICU Stay Level]         stay_id (e.g. 39553978)
```

### Identifier Hierarchy & Usage
1. **`subject_id`**: Globally unique integer identifying an individual patient across all hospital admissions and outpatient encounters.
2. **`hadm_id`**: Unique integer identifying a specific hospital admission episode (from `admittime` to `dischtime`).
3. **`stay_id`**: Unique integer identifying a specific continuous stay within an ICU ward (from `intime` to `outtime`).
4. **`itemid`**: Foreign key linking observation events in `chartevents`, `labevents`, `inputevents`, `outputevents`, and `procedureevents` to their master metadata dictionaries (`d_items` or `d_labitems`).

---

## E. Timestamp Columns

Longitudinal modeling with Neural SDEs requires continuous-time parameterization ($t \in \mathbb{R}^+$) and inter-observation time-gap calculation:
$$\Delta t_i = t_i - t_{i-1}$$

### Overview of Primary Timestamps

| Table | Timestamp Field | Format / Resolution | Description |
| :--- | :--- | :--- | :--- |
| `admissions` | `admittime`, `dischtime` | `YYYY-MM-DD HH:MM:SS` | Hospital admission and discharge bounds |
| `icustays` | `intime`, `outtime` | `YYYY-MM-DD HH:MM:SS` | ICU arrival and transfer out bounds |
| `chartevents` | `charttime` | `YYYY-MM-DD HH:MM:SS` | Exact time vital sign or clinical chart event recorded |
| `labevents` | `charttime` | `YYYY-MM-DD HH:MM:SS` | Specimen collection time |
| `inputevents` | `starttime`, `endtime` | `YYYY-MM-DD HH:MM:SS` | Duration interval for IV medication/infusion |
| `prescriptions`| `starttime`, `stoptime` | `YYYY-MM-DD HH:MM:SS` | Prescribed medication active window |

### Time Shifting Notice
MIMIC-IV timestamps are shifted per patient by a random offset into future years (`2110` – `2202`). However:
- Relative time gaps $\Delta t$ between events for a given patient are **100% preserved**.
- Day-of-week and time-of-day (circadian rhythms) are preserved.

---

## F. Table Relationships

```
                        ┌───────────────────┐
                        │     patients      │
                        │   (subject_id)    │
                        └─────────┬─────────┘
                                  │
                                  ▼
                        ┌───────────────────┐
                        │    admissions     │
                        │ (subject_id,      │
                        │  hadm_id)         │
                        └─────────┬─────────┘
                                  │
      ┌───────────────────────────┼───────────────────────────┐
      │                           │                           │
      ▼                           ▼                           ▼
┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│  labevents  │            │prescriptions│            │  icustays   │
│ (hadm_id,   │            │ (hadm_id,   │            │ (hadm_id,   │
│  charttime) │            │ starttime)  │            │  stay_id)   │
└─────────────┘            └─────────────┘            └──────┬──────┘
                                                             │
                  ┌──────────────────────┬───────────────────┴───────────────────┐
                  ▼                      ▼                                       ▼
           ┌─────────────┐        ┌─────────────┐                         ┌─────────────┐
           │ chartevents │        │ inputevents │                         │outputevents │
           │ (stay_id,   │        │ (stay_id,   │                         │ (stay_id,   │
           │  charttime) │        │ starttime)  │                         │  charttime) │
           └─────────────┘        └─────────────┘                         └─────────────┘
```

---

## G. Candidate Longitudinal Data Sources

For learning continuous-time latent representations with Neural SDEs, candidate features were evaluated across 6 clinical domains:

### 1. Vital Signs (`icu/chartevents`)
High-frequency, irregularly sampled continuous features recorded by ICU bedside monitors:
- **Heart Rate** (`itemid: 220045`)
- **Respiratory Rate** (`itemid: 220210`)
- **O2 Saturation / Pulse Oximetry** (`itemid: 220277`)
- **Systolic Non-Invasive Blood Pressure** (`itemid: 220179`)
- **Diastolic Non-Invasive Blood Pressure** (`itemid: 220180`)
- **Mean Non-Invasive Blood Pressure** (`itemid: 220181`)
- **Body Temperature** (`itemid: 223761` / `223762`)

### 2. Laboratory Measurements (`hosp/labevents`)
Medium-frequency, irregularly sampled diagnostic biomarkers:
- **Potassium** (`itemid: 50971`)
- **Sodium** (`itemid: 50983`)
- **Creatinine** (`itemid: 50912`)
- **Blood Urea Nitrogen (BUN)** (`itemid: 51006`)
- **Hematocrit** (`itemid: 51221`)
- **Hemoglobin** (`itemid: 51222`)
- **Platelet Count** (`itemid: 51265`)

### 3. Medications (`icu/inputevents` & `hosp/prescriptions`)
Interval-based treatment interventions:
- Continuous IV Infusions (e.g. Norepinephrine, Propofol, Insulin, 0.9% NaCl).

### 4. Clinical Events & Interventions (`icu/procedureevents`)
Continuous procedure time windows (e.g., Mechanical Ventilation, Invasive Arterial Line).

### 5. Diagnoses (`hosp/diagnoses_icd`)
Static admission-level ICD-9/10 codes (used for conditional generation or evaluation targets).

### 6. Procedures (`hosp/procedures_icd`)
Static admission-level billed surgical and medical procedure codes.

---

## H. Missing-Data Observations

Empirical null value inspection revealed distinct patterns:

1. **`chartevents` Null Distribution**:
   - `valuenum` (numeric measurement): **57.89% null** overall across all 668,862 rows.
   - *Rationale*: `chartevents` stores both numeric vitals and text/categorical nurse notes (`value` is populated in **96.77%** of rows). When filtering specifically for continuous vital sign `itemid`s (e.g. Heart Rate), `valuenum` null percentage drops to **< 1.5%**.
2. **`labevents` Null Distribution**:
   - `valuenum`: **11.67% null** (representing text lab comments such as "CANCELLED" or "UNABLE TO REPORT").
   - `flag` (abnormal flag): **62.68% null** (only populated when a result falls outside reference limits).
3. **`admissions` & `patients` Structural Nulls**:
   - `admissions.deathtime`: **94.55% null** (only present for patients who died in hospital).
   - `patients.dod` (date of death): **69.00% null** (31 out of 100 demo patients are deceased).

---

## I. Potential Problems for Modeling

1. **Irregular & Asynchronous Sampling**: Vital signs are recorded every 1–4 hours, labs every 12–24 hours, and prescriptions over continuous start/stop intervals. Neural SDE models must handle multi-rate continuous-time dynamics.
2. **Informative Missingness (Sampling Bias)**: Clinical measurements are not missing at random (MAR). Clinicians order labs and vitals when a patient's state deteriorates.
3. **Mixing Continuous & Categorical Channels**: Vital signs are continuous $\mathbb{R}^d$, while medications and diagnoses are discrete state transitions.
4. **Time Scale Variance**: Time gaps $\Delta t$ range from minutes during acute ICU events to months between distinct hospital readmissions.

---

## J. Recommended Next Step

For **Phase 2**, we recommend focusing dataset construction on **ICU stay continuous-time vital sign trajectories** (`icu/chartevents` linked with `icu/icustays` and `hosp/patients`).

### Tradeoff Analysis
- **Option A (Full Multi-Admission EHR)**: Spans months/years across `admissions`, `labevents`, and `omr`. High sparsity, massive time gaps, difficult initial SDE convergence.
- **Option B (ICU Stay Continuous Trajectories - RECOMMENDED)**: Dense, high-frequency continuous vital signs (Heart Rate, Blood Pressure, SpO2, Resp Rate) over 24–72 hour ICU stays. Ideal for establishing Neural SDE drift/diffusion stability before scaling to sparse multi-admission trajectories.

---

*Report generated automatically from MIMIC-IV Demo Dataset Inspection.*
