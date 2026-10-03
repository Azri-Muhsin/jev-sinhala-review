# Stage 3: Sentiment Primitive Equivalence Lab — Quantitative Report

**Target Model:** `jev-latest`  
**Sample Size:** $N = 150$ (Dataset A: Sinhala News-Comment Sentiment)  
**Timestamp:** `2026-10-03T11:38:57.700017+00:00`  

## 1. Classification Performance: Choice vs Noul

| Metric | Choice (4-way Categorical) | Noul (4-way Argmax) | Discrepancy |
| :--- | :---: | :---: | :---: |
| **Accuracy** | **60.7%** | **56.7%** | 4.0% |
| **Macro-F1** | 0.461 | 0.445 | 0.017 |
| **Weighted-F1** | 0.575 | 0.544 | 0.031 |
| **Brier Score** | 0.561 | 0.825 | 0.264 |
| **ECE (Calibration Error)** | 0.132 | 0.230 | 0.099 |

### Per-Class Performance Breakdown

| Class | Support | Choice Precision | Choice Recall | Choice F1 | Noul Precision | Noul Recall | Noul F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `POSITIVE` | 40 | 0.729 | 0.875 | 0.795 | 0.750 | 0.900 | 0.818 |
| `NEGATIVE` | 42 | 0.507 | 0.905 | 0.650 | 0.507 | 0.833 | 0.631 |
| `NEUTRAL` | 68 | 0.818 | 0.265 | 0.400 | 0.824 | 0.206 | 0.329 |
| `CONFLICT` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## 2. RQ2 Primitive Consistency Analysis

- **Argmax Agreement Rate:** **89.3%** (134/150 items agreed)
- **Disagreement Count:** 16 items
- **Single-Belief Coherence (Coherent):** 52.7% (79 items with exactly one Noul $\ge 0.50$)
- **Contradiction Rate (Multi-belief $\ge 0.50$):** **44.7%** (67 items)
- **Zero-Belief Rate (No Noul $\ge 0.50$):** 2.7% (4 items)

### Probability Correlation across Decision Spaces

| Class | Pearson $r$ | Pearson $p$-value | Spearman $\rho$ | Mean Absolute Diff |
| :--- | :---: | :---: | :---: | :---: |
| `POSITIVE` | 0.956 | 0.0000 | 0.956 | 0.117 |
| `NEGATIVE` | 0.958 | 0.0000 | 0.966 | 0.102 |
| `NEUTRAL` | 0.935 | 0.0000 | 0.925 | 0.090 |
| `CONFLICT` | 0.580 | 0.0000 | 0.750 | 0.354 |

### Score vs Choice Ordinal Alignment (3-class Ordered Subset)

- **Evaluated Examples:** 145 (excluding CONFLICT)
- **Mean Absolute Error (MAE):** 0.224
- **Exact Rounded Match Rate:** 84.1%
- **Spearman Rank Correlation ($\rho$):** **0.899** ($p = 0.0000$)
- **Pearson Correlation ($r$):** 0.939

## 3. Selective Classification & Risk-Coverage (Choice)

| Confidence Threshold ($\tau$) | Retained Count | Coverage (%) | Accuracy (%) | Risk (%) |
| :---: | :---: | :---: | :---: | :---: |
| $\ge 0.50$ | 119/150 | 79.3% | **68.9%** | 31.1% |
| $\ge 0.70$ | 90/150 | 60.0% | **75.6%** | 24.4% |
| $\ge 0.80$ | 70/150 | 46.7% | **77.1%** | 22.9% |
| $\ge 0.90$ | 57/150 | 38.0% | **86.0%** | 14.0% |
| $\ge 0.95$ | 38/150 | 25.3% | **89.5%** | 10.5% |

## 4. Telemetry Profile

- **Total Calls:** 150
- **Total Decision Records:** 450
- **Median Latency ($p_50$):** 304.4 ms
- **95th Percentile Latency ($p_95$):** 409.6 ms
- **99th Percentile Latency ($p_99$):** 475.5 ms
- **Mean Latency:** 332.2 ms
