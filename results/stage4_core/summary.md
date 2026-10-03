# Stage 4: Core Pure-Sinhala Tasks — Quantitative Report

**Target Model:** `jev-latest`  
**Total Examples Evaluated:** $N = 650$ across 5 tasks  
**Total Decision Records:** 1500  
**Timestamp:** `2026-10-03T11:50:40.698221+00:00`  

## 1. Cross-Task Performance Overview

| Task | Domain | Decision Space | Primary Primitive | Accuracy | Macro-F1 | ECE | Chance Baseline |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **SOLD** | Social (Twitter) | Binary (`OFF`/`NOT`) | Choice | **64.7%** | 0.640 | 0.124 | 50.0% |
| *SOLD (Noul)* | Social (Twitter) | Binary (`OFF`/`NOT`) | Noul | **64.7%** | 0.642 | 0.120 | 50.0% |
| **NSINA Categories** | News Articles | 4 Categories | Choice | **83.0%** | 0.823 | 0.098 | 25.0% |
| **NSINA Media** | News Articles | 10 Sources | Choice | **20.0%** | 0.158 | 0.070 | 10.0% |
| **SinhalaMMLU** | Multi-discipline QA | 4 Options (`A`/`B`/`C`/`D`) | Choice | **73.3%** | 0.736 | 0.153 | 25.0% |
| **SalAngaBhava** | Product Reviews | Ordinal (1–5 Stars) | Choice | **78.0%** | 0.344 | 0.191 | 20.0% |
| *SalAngaBhava (Score)* | Product Reviews | Ordinal (1–5 Stars) | Score | **31.3%** (MAE=1.054) | — | — | 20.0% |

## 2. Detailed Task Syntheses

### 2.1 SOLD: Sinhala Offensive Language Detection ($N=150$)
- **Choice Accuracy:** 64.7% (Macro-F1: 0.640)
- **Noul Accuracy:** 64.7% (Macro-F1: 0.642)
- **Calibration (ECE):** Choice: 0.124 vs Noul: 0.120

### 2.2 NSINA Categories: News Classification ($N=100$)
- **Choice Accuracy:** 83.0% (Macro-F1: 0.823)
- **ECE:** 0.098

### 2.3 NSINA Media: Publisher Identification 10-way ($N=100$)
- **Choice Accuracy:** 20.0% (Macro-F1: 0.158)
- **ECE:** 0.070

### 2.4 SinhalaMMLU: Academic Multiple-Choice QA ($N=150$)
- **Choice Accuracy:** 73.3% (Macro-F1: 0.736)
- **ECE:** 0.153

### 2.5 SalAngaBhava: Product Review Ratings ($N=150$)
- **Choice Accuracy (Exact 1–5):** 78.0%
- **Score Rounded Accuracy:** 31.3% (Mean Absolute Error: 1.054)
- **Noul High-Satisfaction (Rating $\ge 4$):** 88.7%
- **Noul Low-Satisfaction (Rating $\le 2$):** 90.7%

## 3. Telemetry Profile

- **Median Latency ($p_50$):** 304.5 ms
- **95th Percentile Latency ($p_95$):** 437.6 ms
- **99th Percentile Latency ($p_99$):** 541.2 ms
- **Mean Latency:** 327.8 ms
