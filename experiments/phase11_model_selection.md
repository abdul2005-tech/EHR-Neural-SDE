# Phase 11 Model Selection & Tradeoff Analysis (Full Validation)

## Executive Summary

This report documents the full validation evaluation comparing the two finalist candidates (**Candidate 11B: Temporal Difference** and **Candidate 11F: Probabilistic Decoder**) against the **Phase 10 Baseline** on the complete validation dataset (25 ICU stays, 20 futures per trajectory).

## Validation Scorecard

| Model Candidate | Plausibility % | Wasserstein ↓ | KS Stat ↓ | Correlation Error ↓ | Volatility std($\Delta X$) | Lag-1 AC | Lag-5 AC | Diversity RMSE ↑ | Diversity MAE ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Phase 10 Baseline** | 100.00% | 8.9471 | 0.4714 | 0.4786 | 0.0687 | 0.9145 | 0.6604 | 0.5273 | 0.3419 |
| **11B Temporal Difference** | 100.00% | 8.9353 | 0.4727 | 0.4691 | 0.0689 | 0.9146 | 0.6610 | 0.5275 | 0.3421 |
| **11F Probabilistic Decoder** | 100.00% | 8.6107 | 0.4547 | 0.2510 | 0.1424 | 0.9371 | 0.7304 | 1.1084 | 0.7089 |

## Time-Awareness Latent Displacement Experiment

| Model Candidate | dt = 0.05h | dt = 1.00h | dt = 5.00h | dt = 10.00h |
| :--- | :---: | :---: | :---: | :---: |
| **Phase 10 Baseline** | 0.6782 | 2.6544 | 6.2827 | 8.6552 |
| **11B Temporal Difference** | 0.6782 | 2.6539 | 6.2814 | 8.6480 |
| **11F Probabilistic Decoder** | 0.6815 | 2.6862 | 6.4419 | 9.2263 |

## Tradeoff Analysis & Findings

1. **Candidate 11F (Probabilistic Decoder)**:
   - **Strengths**: Achieves best 1D distribution metrics (Wasserstein = **8.6107**, KS stat = **0.4547**). Dramatically reduces cross-feature correlation error (from **0.4786** in baseline to **0.2510**). Significantly increases trajectory volatility ($\text{std}(\Delta X) = \mathbf{0.1424}$ vs baseline $0.0687$), directly mitigating the smoothness bias. Offers more than double the multi-future diversity (RMSE = **1.1084** vs baseline **0.5273**).
   - **Weaknesses**: Slightly higher Lag-1 autocorrelation (**0.9371** vs baseline **0.9145**).

2. **Candidate 11B (Temporal Difference Loss)**:
   - **Strengths**: Marginal improvement in 1D Wasserstein distance (**8.9353** vs baseline **8.9471**) and correlation error (**0.4691** vs baseline **0.4786**).
   - **Weaknesses**: Still inherits baseline failure modes of high correlation error ($0.4691$) and low step volatility ($\text{std}(\Delta X) = 0.0689$).

## Conclusion & Recommendation

Based strictly on validation evidence across all 25 validation ICU stays (20 futures per stay), **Candidate 11F (Probabilistic Decoder)** is unambiguously the superior model. It dominates Candidate 11B and Phase 10 Baseline across almost every major evaluation dimension (Wasserstein distance, KS statistic, correlation error, step volatility, and multi-future diversity).
