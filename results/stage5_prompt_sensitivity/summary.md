# Stage 5: Language & Prompt Sensitivity Experiment (RQ4)

**Timestamp:** 2026-10-03T12:09:36.089619+00:00  
**Model:** `jev-latest`  
**Dataset A Sample Size:** N=50  
**Dataset B Sample Size:** N=50  

## 1. Sensitivity Summary (English vs. Sinhala Instruction)

| Task / Primitive | Acc (EN) | Acc (SI) | Δ Acc | Macro-F1 (EN) | Macro-F1 (SI) | Δ F1 | Agreement | Flip Rate | Conf Drift (SI - EN) | Prob MAD | Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment (Choice 4-way)** | 0.600 | 0.600 | +0.000 | 0.468 | 0.468 | +0.000 | 90.0% | 10.0% | -0.032 | 0.0269 | 0.9912 |
| **Sentiment (Noul 4-way)** | 0.620 | 0.540 | -0.080 | 0.482 | 0.423 | -0.059 | 86.0% | 14.0% | -0.023 | 0.0847 | 0.9735 |
| **SOLD (Noul Binary)** | 0.680 | 0.640 | -0.040 | 0.680 | 0.630 | -0.050 | 84.0% | 16.0% | -0.000 | 0.0800 | 0.9815 |
| **SOLD (Choice Binary)** | 0.700 | 0.680 | -0.020 | 0.697 | 0.678 | -0.019 | 98.0% | 2.0% | +0.016 | 0.0372 | 0.9963 |

## 2. Decision Flip Transitions

### Dataset A: Sentiment
- **Choice Flips (5/50):** {'NEUTRAL -> NEGATIVE': 1, 'NEGATIVE -> CONFLICT': 2, 'NEGATIVE -> POSITIVE': 1, 'CONFLICT -> NEGATIVE': 1}
- **Noul Flips (7/50):** {'NEUTRAL -> POSITIVE': 2, 'POSITIVE -> CONFLICT': 2, 'NEGATIVE -> CONFLICT': 2, 'CONFLICT -> NEGATIVE': 1}

### Dataset B: SOLD
- **Noul Flips (8/50):** {'OFF -> NOT': 8}
- **Choice Flips (1/50):** {'NOT -> OFF': 1}

## 3. Confidence Drift & Correlation

- **Sentiment Choice:** Mean Conf EN = `0.746`, SI = `0.714` (Δ = `-0.032`, Pearson r = `0.952`, p = `0.0000e+00`)
- **Sentiment Noul:** Mean Conf EN = `0.814`, SI = `0.791` (Δ = `-0.023`, Pearson r = `0.839`, p = `0.0000e+00`)
- **SOLD Noul:** Mean Conf EN = `0.740`, SI = `0.740` (Δ = `-0.000`, Pearson r = `0.679`, p = `0.0000e+00`)
- **SOLD Choice:** Mean Conf EN = `0.581`, SI = `0.597` (Δ = `+0.016`, Pearson r = `0.953`, p = `0.0000e+00`)

## 4. Key Takeaways for RQ4
1. **Language Transfer Stability:** Investigates whether framing in English vs. Sinhala preserves classification accuracy and decision boundaries.
2. **Calibration Invariance:** Quantifies whether confidence shifts systematically between English instructions and native Sinhala prompts.
3. **Representation Cosine Alignment:** Assesses whether the model's internal posterior probability distribution remains structurally aligned across language instruction conditions.