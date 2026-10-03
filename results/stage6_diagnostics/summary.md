# Stage 6: Primitive Diagnostics & Robustness Report

**Timestamp:** 2026-10-03T12:59:39.675704+00:00  
**Model:** `jev-latest`  

## 1. Option-Order Permutation Stability Test (N=40)

- **Overall Stability Rate:** **90.0%** (all 3 permutations identical)

| Task | Sample Size | Stability Rate | Orig vs Shifted Flip | Orig vs Inverted Flip | Mean Entropy Δ (Shift) | Position Bias Stat (p-val) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NSINA Categories (4-way)** | 20 | **100.0%** | 0.0% | 0.0% | +0.0428 bits | No (p=0.4020) |
| **SinhalaMMLU (4-option QA)** | 20 | **80.0%** | 10.0% | 15.0% | +0.0107 bits | No (p=0.0620) |

## 2. Multi-Pass Repeatability / Stochasticity Test (N=100, 3 Runs)

| Primitive | N | Exact Repeatability | Mean Conf σ | Max Drift | Deterministic Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Choice (Sentiment 4-way)** | 100 | **94.0%** | 0.020589 | 0.190000 | Slight Variance |
| **Noul (Positive Binary)** | 100 | **100.0%** | 0.009342 | 0.110000 | Slight Variance |
| **Score (Ordinal Expectation)** | 100 | **15.0%** | 0.023930 | 0.320000 | Slight Variance |

## 3. Aggregate Selective Risk-Coverage Analysis

| Task | Baseline Acc | τ ≥ 0.50 (Cov / Acc) | τ ≥ 0.70 (Cov / Acc) | τ ≥ 0.80 (Cov / Acc) | τ ≥ 0.90 (Cov / Acc) | τ ≥ 0.95 (Cov / Acc) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment (Choice)** | 60.7% | 79% / **68.9%** | 60% / **75.6%** | 47% / **77.1%** | 38% / **86.0%** | 25% / **89.5%** |
| **Sentiment (Noul)** | 56.7% | 97% / **57.5%** | 79% / **60.2%** | 54% / **71.6%** | 31% / **80.4%** | 11% / **93.8%** |
| **SOLD (Choice)** | 64.7% | 65% / **76.3%** | 47% / **84.5%** | 38% / **86.0%** | 23% / **94.3%** | 14% / **100.0%** |
| **SOLD (Noul)** | 64.7% | 100% / **64.7%** | 58% / **78.2%** | 41% / **77.0%** | 14% / **100.0%** | 5% / **100.0%** |
| **NSINA Categories (Choice)** | 83.0% | 96% / **84.4%** | 87% / **89.7%** | 84% / **90.5%** | 74% / **91.9%** | 71% / **93.0%** |
| **SinhalaMMLU (Choice)** | 73.3% | 57% / **91.9%** | 50% / **93.3%** | 43% / **95.4%** | 35% / **94.3%** | 28% / **92.9%** |
| **SalAngaBhava (Choice)** | 78.0% | 62% / **89.2%** | 45% / **89.5%** | 35% / **90.6%** | 22% / **97.0%** | 17% / **96.0%** |

## 4. Key Takeaways for Robustness (Stage 6)
1. **Option-Order Invariance:** Evaluates whether decision boundaries are resilient to option shuffling or position permutation in multiclass Choice.
2. **Stochasticity Bound:** Validates whether zero-shot decision requests yield reproducible, deterministic predictions across multiple passes.
3. **Trust & Safety Thresholding:** Demonstrates that selective classification at τ ≥ 0.90 consistently provides high accuracy across tasks.