# Stage 7: Code-Mixed Stress Track Report (Dataset F: CMCS, N=150)

**Timestamp:** 2026-10-03T13:01:20.166691+00:00  
**Model:** `jev-latest`  
**Total Samples Evaluated:** N=150  

## 1. Multi-Task Classification Performance

| Sub-Task | Primitive | Classes | N | Accuracy | Macro-F1 | ECE | Brier Score | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment** | `choice` | 4 | 150 | **0.620** | 0.440 | 0.204 | 0.579 | 0.718 |
| **Sentiment** | `noul` | 4 | 150 | **0.573** | 0.412 | 0.191 | 0.716 | 0.764 |
| **Humour Detection** | `choice` | 2 | 150 | **0.933** | 0.789 | 0.153 | 0.119 | 0.780 |
| **Humour Detection** | `noul` | 2 | 150 | **0.920** | 0.778 | 0.158 | 0.184 | 0.762 |
| **Hate Speech** | `choice` | 3 | 150 | **0.907** | 0.453 | 0.108 | 0.167 | 0.835 |
| **Hate Speech** | `noul` | 3 | 150 | **0.893** | 0.317 | 0.070 | 0.173 | 0.845 |
| **Single-Aspect QA** | `choice` | 5 | 47 | **0.851** | 0.816 | 0.130 | 0.223 | 0.759 |

## 2. Multi-Label Aspect Extraction (Parallel Noul)

| Aspect Domain | True Positives | Predicted Positives | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Billing or price** | 9 | 21 | 0.893 | 0.333 | 0.778 | **0.467** |
| **Customer service** | 10 | 24 | 0.880 | 0.333 | 0.800 | **0.471** |
| **Data** | 17 | 25 | 0.920 | 0.600 | 0.882 | **0.714** |
| **Network** | 12 | 26 | 0.867 | 0.346 | 0.750 | **0.474** |
| **Package** | 5 | 17 | 0.920 | 0.294 | 1.000 | **0.455** |

## 3. Script Type Degradation Analysis (Sentiment Choice)

| Script Category | N | Accuracy | Macro-F1 | ECE | Mean Confidence |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sinhala_in_Latin** | 111 | **0.604** | 0.430 | 0.209 | 0.719 |
| **Code_Mixed** | 17 | **0.706** | 0.586 | 0.205 | 0.668 |
| **Pure_Sinhala** | 22 | **0.636** | 0.436 | 0.304 | 0.751 |

## 4. Key Takeaways for Stage 7
1. **Code-Mixed Sentiment Resilience:** Assesses whether Jev maintains sentiment comprehension when Sinhala is interleaved with English and colloquialisms.
2. **Safety & Toxicity Sensitivity:** Compares 3-way hate speech detection against binary offensive language on informal social data.
3. **Script Sensitivity:** Measures the quantitative performance penalty between pure Sinhala script, Singlish (Latin), and code-switched text.