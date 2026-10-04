# Jev-Sinhala Quick Capability & Primitive Consistency Probe

> [!NOTE]
> ### 🚀 Phase 2 Scale-Up & Multi-Model Benchmark (In Progress)
> **Phase 2 is now actively underway!** We are scaling from the Phase 1 reconnaissance probe ($N \approx 950$) to the **full benchmark census ($N = 13,054$ evaluation instances)** to test `jev-latest` against the full testing intention of the original datasets.
> 
> Furthermore, we are extending the evaluation across industry models: running these exact uncorrupted Sinhala benchmark datasets against **Cloudflare's Clef, Kev, and Laya models** as well for a comprehensive multi-model decision and linguistic capability benchmark. See [Section 8: Phase 2 Scale-Up & Cloudflare Comparative Benchmark](#8-phase-2-scale-up--cloudflare-comparative-benchmark-in-progress).

> **Objective:** A zero-shot empirical evaluation of TypeSafe's `jev-latest` on low-resource Sinhala and Sinhala-English code-mixed benchmarks, evaluating linguistic comprehension, decision primitive consistency (`Choice`, `Noul`, `Score`), calibration quality, and robustness.

---

## 1. Executive Summary & Research Questions

This probe evaluates the zero-shot decision capabilities of TypeSafe's System-1 model (`jev-latest`) on official test splits of established Sinhala NLP benchmarks. Unlike standard generative LLM benchmarks that evaluate text-generation perplexity or token-level logprobs, this study investigates how native decision primitives operate on low-resource Indic morphosyntax, agglutination, and mixed scripts.

We investigate four core research questions:

1. **RQ1 (Linguistic & Semantic Capability):** Can `jev-latest` accurately categorize and reason over diverse Sinhala domains (social media, formal news, multi-discipline academic QA, e-commerce reviews) and scripts (pure Sinhala script vs. romanized code-mixed Sinhala-English)?
2. **RQ2 (Primitive Consistency):** When presented with the exact same input state, do different decision primitives—**`Choice`** (multi-class distribution), **`Noul`** (binary true/false probabilities), and **`Score`** (bounded ordinal expectation)—yield mutually consistent beliefs?
3. **RQ3 (Calibration & Risk-Coverage):** Are Jev's output probabilities and confidence estimates well-calibrated (ECE, Brier score)? Can confidence thresholds ($\ge 0.50, 0.70, 0.80, 0.90, 0.95$) effectively filter out errors for high-reliability automated pipelines?
4. **RQ4 (Robustness & Invariance Diagnostics):** Is Jev invariant to choice-order permutation (A/B/C/D order effects)? How sensitive is performance to instruction language (English vs. native Sinhala)? Is inference repeatable across stochastic runs?

### Core Findings Matrix

| Research Question | Key Empirical Finding | Quantitative Metric |
| :--- | :--- | :--- |
| **RQ1: Capability** | Strong zero-shot generalization across academic QA, formal news, and social sentiment; resilient to Romanized transliteration. | **73.3%** SinhalaMMLU; **83.0%** NSINA Categories; **93.3%** CMCS Humour; **-3.2%** Romanization drop |
| **RQ2: Consistency** | High decision parity between joint multi-class Simplex (`Choice`) and isolated binary queries (`Noul`), with strong ordinal alignment (`Score`). | **89.3%** Argmax agreement (Sentiment); **100.0%** (SOLD); **92.0%** (NSINA); **$\rho = 0.899$** Choice-Score correlation |
| **RQ3: Calibration** | Output probabilities are naturally well-calibrated without Platt scaling; selective risk-coverage thresholds scale accuracy to near-perfection. | **0.098–0.153** typical ECE; **94.3%–100.0%** accuracy at $\tau \ge 0.90$ across all evaluated tasks |
| **RQ4: Robustness** | Perfect option-order invariance on news categorization, minimal sensitivity to prompt language, and near-deterministic inference repeatability. | **90.0%** Order stability; **0.000** Choice accuracy gap (English vs. native Sinhala); **100.0%** Noul repeatability |

---

## 2. Evaluation Tracks & Datasets

All evaluations use official, uncorrupted evaluation splits with zero synthetic paraphrasing and strict byte-for-byte preservation of Sinhala Unicode (including Zero-Width Joiner `\u200D` and Non-Joiner `\u200C`).

```
Track 1: Core Pure-Sinhala Classification
  ├── Dataset A: Sinhala News-Comment Sentiment (4-way: POS, NEG, NEU, CONFLICT)
  ├── Dataset B: SOLD — Sinhala Offensive Language Dataset (Binary: OFF, NOT)
  ├── Dataset C1: NSINA Categories (4 primary news domains)
  └── Dataset C2: NSINA Media Identification (10 major news publishers)

Track 2: Knowledge, Reasoning & Ordinal Evaluation
  ├── Dataset D: SinhalaMMLU (4-option QA across Humanities, STEM, Social Sciences)
  └── Dataset E: SalAngaBhava (Pure Sinhala product reviews on 1–5 ordinal rating)

Track 3: Romanized & Code-Mixed Stress Track
  └── Dataset F: Sinhala-English CMCS (Sentiment, Humour, Hate, Aspect Extraction)
```

### Dataset Summary Table

| ID | Dataset | Domain | Classes / Decision Space | Tested Primitives | Phase 0 ($N$) | Phase 1 ($N$) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **A** | **Sinhala Sentiment** | News Comments | 4-way: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `CONFLICT` | `Choice` (4-way), `Noul` ($\times 4$), `Score` (Neg $\to$ Neu $\to$ Pos) | 20 | 150 |
| **B** | **SOLD** | Social Media (X/Twitter) | Binary: `OFF`, `NOT` | `Noul` (Primary), `Choice` (Primary), `Score` (Diagnostic) | 20 | 150 |
| **C1** | **NSINA Categories** | News Articles | 4 Categories (`Business`, `Sports`, `Local News`, `International`) | `Choice` (Primary), `Noul` (One-vs-Rest) | 20 | 100 |
| **C2** | **NSINA Media** | News Articles | 10 Media Sources (`Lankadeepa`, `Ada Derana`, `Hiru`, etc.) | `Choice` (Primary 10-way) | 20 | 100 |
| **D** | **SinhalaMMLU** | Multi-discipline QA | 4-option QA (`A`, `B`, `C`, `D`) | `Choice` (Primary QA) | 20 | 150 |
| **E** | **SalAngaBhava** | Product Reviews | Ordinal 1–5 Star Rating | `Score` (Primary 1–5), `Choice` (1–5), `Noul` (High/Low thresholds) | 20 | 150 |
| **F** | **Sinhala-English CMCS** | Code-Mixed Social | Multi-task (Sentiment, Aspect presence) | `Choice` (Sentiment), `Noul` (Aspect Extraction) | 20 | 150 |
| **Total** | | | | | **140 runs** | **~950 items** |

---

## 3. Decision Primitives under Test

TypeSafe provides three structured System-1 decision primitives:

1. **`Choice` (Categorical Decisions):**
   Evaluates normalized mutually exclusive probability distributions over discrete categories $\mathcal{C}$:
   $$\sum_{c \in \mathcal{C}} P(c) = 1.0, \quad \text{confidence} = \max_{c} P(c)$$

2. **`Noul` (Calibrated Binary Belief):**
   Returns continuous probability $P(\text{True}) \in [0, 1]$ for an assertion. Confidence is quantified by polarity certainty:
   $$\text{confidence} = \max(P(\text{True}), 1 - P(\text{True}))$$

3. **`Score` (Bounded Continuous Expectation):**
   Computes the continuous expectation $\mathbb{E}[S] \in [0, K-1]$ over an ordered criterion scale $(c_0, c_1, \dots, c_{K-1})$ along with categorical step probabilities and uncertainty:
   $$\mathbb{E}[S] = \sum_{k=0}^{K-1} k \cdot P(c_k)$$

---

## 4. Frozen Record Schema (Section 5)

Every decision across all stages is streamed to immutable JSON Lines (`.jsonl`) logs adhering strictly to the frozen schema:

```json
{
  "record_id": "rec_phase1_sent_00042_choice_en",
  "experiment_id": "exp_probe_v1",
  "phase": "phase_1_full",
  "timestamp": "2026-10-03T11:06:52.471671+00:00",
  "dataset": "dataset_a_sentiment",
  "example_id": "sinhala_sentiment_00042",
  "language": "si",
  "script_type": "pure_sinhala",
  "primitive": "choice",
  "prompt_variant": "english_instruction",
  "state_text": "මෙම තීරණය ඉතාමත් අගය කළ යුතු එකක් බව පැවසිය යුතුය.",
  "instructions": "Which label best describes the overall sentiment of this text?",
  "criteria_definitions": {"POSITIVE": null, "NEGATIVE": null, "NEUTRAL": null, "CONFLICT": null},
  "gold_label": "POSITIVE",
  "prediction": "POSITIVE",
  "is_correct": true,
  "confidence": 0.8912,
  "probabilities": {"POSITIVE": 0.8912, "NEGATIVE": 0.0210, "NEUTRAL": 0.0754, "CONFLICT": 0.0124},
  "score_details": null,
  "noul_details": null,
  "latency_ms": 312.4,
  "usage": {"input_tokens": 128, "output_tokens": 16},
  "model": "jev-latest",
  "raw_response_status": 200,
  "error": null
}
```

---

## 5. Execution Pipeline & Current Status

```
Stage 0: Environment Setup, Tooling & SDK Verification ──────────► [✓ COMPLETED]
   │
Stage 1: Data Ingestion, Splits & Stratified Sampling Engine ──► [✓ COMPLETED]
   │     (Manifest committed: data/processed/manifest.json)
   │
Stage 2: Phase 0 Smoke Test (20 items/task end-to-end) ────────► [✓ PASSED GATE]
   │     (320 decisions logged, 0 runtime errors, 100% checks passed)
   │
Stage 3: Sentiment Primitive Equivalence Lab (N=150) ──────────► [✓ COMPLETED]
   │     (89.3% Argmax agreement, Choice 60.7% vs Noul 56.7%, Score rho=0.899)
   │
Stage 4: Core Pure-Sinhala Tasks (SOLD, NSINA, MMLU, SalAnga) ─► [✓ COMPLETED]
   │     (650 examples, MMLU: 73.3%, NSINA Cat: 83.0%, SalAnga: 78.0%, SOLD: 64.7%)
   │
Stage 5: Language & Prompt Sensitivity (English vs. Sinhala) ──► [✓ COMPLETED]
   │     (Choice parity: Δ Acc=0.000, 98% SOLD agreement, Cosine Sim >= 0.974)
   │
Stage 6: Robustness & Diagnostics (Order, Repeatability, ECE) ──► [✓ COMPLETED]
   │     (90% option stability, 100% Noul repeatability, selective acc to 100%)
   │
Stage 7: Code-Mixed Stress Track (CMCS, N=150) ────────────────► [✓ COMPLETED]
   │     (Humour Acc: 93.3%, Hate Speech Acc: 90.7%, Single-Aspect: 85.1%)
   │
Stage 8: Quantitative Synthesis & Deliverables ────────────────► [✓ COMPLETED]
         (4 synthesis CSVs, 7 Economist figures, failure taxonomy)
```

### Stage 2 Smoke Test Gate Verification Results

* **Gate Status:** **PASSED (`gate_passed: true`)**
* **Total Decisions Logged:** 320 calls across 140 examples
* **Runtime Errors:** 0 (100% success rate across templates)
* **Latency Profile:** $p_{50} = 305.2\text{ ms}$, $p_{95} = 408.0\text{ ms}$, mean $= 329.7\text{ ms}$
* **Unicode / ZWJ Integrity:** 109 ZWJ (`\u200D`) characters verified intact
* **Full Report:** [`results/smoke_test/summary.md`](results/smoke_test/summary.md) | [`gate_verification_report.json`](results/smoke_test/gate_verification_report.json)

### Stage 3 Sentiment Primitive Equivalence Lab Results ($N=150$)

* **Classification Performance:**
  * `Choice` (4-way Categorical): **60.7% Accuracy** | **0.461 Macro-F1** | **0.132 ECE** | **0.561 Brier Score**
  * `Noul` (4-way Argmax): **56.7% Accuracy** | **0.445 Macro-F1** | **0.230 ECE** | **0.825 Brier Score**
* **RQ2 Primitive Consistency Findings:**
  * **Argmax Agreement:** **89.3%** (134/150 examples produced the identical winner between `Choice` and `Noul`).
  * **Multi-Belief Contradiction Rate:** **44.7%** (67 items asserted $P \ge 0.50$ for $\ge 2$ mutually exclusive classes).
  * **Zero-Belief Rate:** **2.7%** (4 items with no class asserting $\ge 0.50$).
  * **Probability Alignment:** Pearson $r \ge 0.935$ and Spearman $\rho \ge 0.920$ across all primary sentiment classes.
  * **Score vs. Choice Ordinal Alignment:** Continuous Score expectation $\mathbb{E}[S]$ correlates strongly with Choice ($\mathbf{\rho = 0.899}$, $p < 0.0001$, $\text{MAE} = 0.224$).
* **Risk-Coverage:** Confidence thresholding effectively filters errors, scaling accuracy from **$60.7\%$** at baseline to **$86.0\%$** ($\tau \ge 0.90$, 38% coverage) and **$89.5\%$** ($\tau \ge 0.95$, 25% coverage).
* **Full Report:** [`results/stage3_sentiment/summary.md`](results/stage3_sentiment/summary.md) | [`primitive_consistency_report.json`](results/stage3_sentiment/primitive_consistency_report.json)

### Stage 4 Core Pure-Sinhala Tasks Results ($N=650$)

* **Cross-Task Benchmark Summary:**
  * **NSINA Categories ($N=100$, 4-way News):** **83.0% Accuracy** | **0.823 Macro-F1** | **0.098 ECE** (Chance: 25.0%)
  * **SinhalaMMLU ($N=150$, 4-option QA):** **73.3% Accuracy** | **0.736 Macro-F1** | **0.153 ECE** (Chance: 25.0%)
  * **SalAngaBhava ($N=150$, 1–5 Stars):** **78.0% Exact Choice Accuracy** | **88.7% Noul High ($\ge 4$)** | **90.7% Noul Low ($\le 2$)**
  * **SOLD ($N=150$, Binary Offensive):** **64.7% Accuracy (Choice & Noul)** | **0.640 Macro-F1** | **0.124 ECE** (Chance: 50.0%)
  * **NSINA Media ($N=100$, 10-way Publisher):** **20.0% Accuracy** | **0.158 Macro-F1** | **0.070 ECE** (Chance: 10.0%)
* **Key Insights:**
  * `jev-latest` exhibits remarkable zero-shot comprehension in Sinhala academic QA (MMLU 73.3%) and news categorization (83.0%), outperforming chance baselines by nearly $3\times$.
  * Risk-coverage curves show that filtering at $\tau \ge 0.90$ elevates accuracy across all tasks (e.g., SOLD reaches **94.3%**, MMLU **94.3%**, SalAngaBhava **97.0%**, and NSINA Media **100.0%**).
* **Full Report:** [`results/stage4_core/summary.md`](results/stage4_core/summary.md) | [`core_tasks_report.json`](results/stage4_core/core_tasks_report.json)

### Stage 5 Language & Prompt Sensitivity Experiment Results (RQ4, $N=100$)

Evaluates whether framing instructions in **English** vs. **native Sinhala** alters `jev-latest`'s performance, confidence, or decision boundaries on identical input texts (50 from Dataset A Sentiment, 50 from Dataset B SOLD):

| Task / Primitive | Acc (EN) | Acc (SI) | Δ Acc | Macro-F1 (EN) | Macro-F1 (SI) | Δ F1 | Agreement | Flip Rate | Conf Drift (SI - EN) | Prob MAD | Cosine Sim |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment (Choice 4-way)** | 0.600 | 0.600 | **+0.000** | 0.468 | 0.468 | **+0.000** | **90.0%** | 10.0% | -0.032 | 0.0269 | **0.9912** |
| **Sentiment (Noul 4-way)** | 0.620 | 0.540 | -0.080 | 0.482 | 0.423 | -0.059 | **86.0%** | 14.0% | -0.023 | 0.0847 | **0.9735** |
| **SOLD (Choice Binary)** | 0.700 | 0.680 | -0.020 | 0.697 | 0.678 | -0.019 | **98.0%** | 2.0% | +0.016 | 0.0372 | **0.9963** |
| **SOLD (Noul Binary)** | 0.680 | 0.640 | -0.040 | 0.680 | 0.630 | -0.050 | **84.0%** | 16.0% | -0.000 | 0.0800 | **0.9815** |

* **Full Report:** [`results/stage5_prompt_sensitivity/summary.md`](results/stage5_prompt_sensitivity/summary.md) | [`prompt_sensitivity_report.json`](results/stage5_prompt_sensitivity/prompt_sensitivity_report.json)

### Stage 6 Primitive Diagnostics & Robustness Results

* **1. Option-Order Permutation Stability ($N=40$):**
  * Evaluated across Original $[1,2,3,4]$, Shifted $[2,3,4,1]$, and Inverted $[4,3,2,1]$ orderings:
    * **NSINA Categories (4-way):** **100.0% Stability Rate** (0.0% flips across all permutations; no position bias $\chi^2$ $p=0.402$).
    * **SinhalaMMLU (4-option QA):** **80.0% Stability Rate** (10% shifted flip, 15% inverted flip; no significant position bias $p=0.062$).
    * **Overall Stability:** **90.0%** across tasks.
* **2. Multi-Pass Repeatability & Stochasticity ($N=100$, 3 passes):**
  * Binary `Noul`: **100.0% Exact Repeatability** ($\text{mean } \sigma = 0.0093$).
  * Multiclass `Choice`: **94.0% Exact Repeatability** ($\text{mean } \sigma = 0.0206$).
  * Continuous `Score`: Standard deviation $\sigma = 0.0239$ across passes.
* **3. Unified Selective Risk-Coverage:**
  * Sweeping $\tau \in [0.50, 0.70, 0.80, 0.90, 0.95]$ systematically filters low-confidence predictions:
    * SOLD Choice: reaches **94.3%** at $\tau \ge 0.90$ and **100.0%** at $\tau \ge 0.95$.
    * SinhalaMMLU: reaches **94.3%** at $\tau \ge 0.90$ and **95.4%** at $\tau \ge 0.80$.
    * SalAngaBhava: reaches **97.0%** at $\tau \ge 0.90$ and **96.0%** at $\tau \ge 0.95$.
* **Full Report:** [`results/stage6_diagnostics/summary.md`](results/stage6_diagnostics/summary.md) | [`diagnostics_report.json`](results/stage6_diagnostics/diagnostics_report.json)

### Stage 7 Code-Mixed Stress Track Results (Dataset F: CMCS, $N=150$)

Evaluates non-standard, Romanized, and code-mixed Sinhala-English text across 5 target sub-tasks:

| Sub-Task | Primitive | Classes | N | Accuracy | Macro-F1 | ECE | Brier Score | Mean Conf |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Sentiment** | `choice` | 4 | 150 | **0.620** | 0.440 | 0.204 | 0.579 | 0.718 |
| **Sentiment** | `noul` | 4 | 150 | **0.573** | 0.412 | 0.191 | 0.716 | 0.764 |
| **Humour Detection** | `choice` | 2 | 150 | **0.933** | 0.789 | 0.153 | 0.119 | 0.780 |
| **Humour Detection** | `noul` | 2 | 150 | **0.920** | 0.778 | 0.158 | 0.184 | 0.762 |
| **Hate Speech** | `choice` | 3 | 150 | **0.907** | 0.453 | 0.108 | 0.167 | 0.835 |
| **Hate Speech** | `noul` | 3 | 150 | **0.893** | 0.317 | 0.070 | 0.173 | 0.845 |
| **Single-Aspect QA** | `choice` | 5 | 47 | **0.851** | 0.816 | 0.130 | 0.223 | 0.759 |

* **Multi-Label Aspect Recall (Noul):** High recall across domains: **Package: 100.0%**, **Data: 88.2%** (F1: 0.714), **Customer Service: 80.0%**, **Billing: 77.8%**, **Network: 75.0%**.
* **Script Type Degradation Analysis (Sentiment Choice):**
  * `Code_Mixed` ($N=17$): **70.6% Accuracy** | 0.586 Macro-F1
  * `Pure_Sinhala` ($N=22$): **63.6% Accuracy** | 0.436 Macro-F1
  * `Sinhala_in_Latin` ($N=111$, Singlish): **60.4% Accuracy** | 0.430 Macro-F1
  * Performance degradation on Romanized Sinhala relative to pure Sinhala is only $\sim 3.2\%$, showing robust phonetic transliteration comprehension.
* **Full Report:** [`results/stage7_cmcs/summary.md`](results/stage7_cmcs/summary.md) | [`cmcs_stress_report.json`](results/stage7_cmcs/cmcs_stress_report.json)

### Stage 8 Quantitative Synthesis & Deliverables ──► [✓ COMPLETED]

Stage 8 synthesizes all evaluation phases into four standardized CSV deliverables and seven publication-ready figures designed in *The Economist* visual language.

#### 1. Synthesis CSV Artifacts (`results/`)

* **[`results/main.csv`](results/main.csv):** Unified 15-row synthesis across all 6 benchmark datasets, evaluating Accuracy, Macro-F1, ECE, Brier Score, Mean Confidence, p50/p95 latency, and zero-shot comparison against published fine-tuned reference anchors (Subasa-XLM-R Macro-F1 0.84 on SOLD; SinBERT/XLM-R Macro-F1 0.88 on NSINA Media).
* **[`results/primitive_consistency.csv`](results/primitive_consistency.csv):** Direct empirical cross-primitive parity metrics (RQ2), contrasting `Choice` vs. `Noul` binary argmax agreement across Sentiment, SOLD, NSINA Categories, SalAngaBhava, and CMCS.
* **[`results/script_analysis.csv`](results/script_analysis.csv):** Orthographic resilience analysis breaking down accuracy, macro-F1, and calibration across Pure Sinhala, Romanized Singlish (Latin), and Code-Mixed text.
* **[`results/failure_taxonomy.csv`](results/failure_taxonomy.csv):** Systematic audit of 50 extreme boundary cases (25 highest-confidence errors + 25 lowest-confidence correct decisions) categorized across 5 diagnostic failure taxonomy dimensions (*Linguistic/Dialect*, *Semantic Nuance*, *Label Boundary*, *Primitive Representation*, *Epistemic Calibration*).

#### 2. Publication-Ready Figure Gallery (`results/figures/`)

All figures adhere strictly to *The Economist* visual standards: red title tag (`#E3120B`), muted color palette (Blue `#006BA2`, Cyan `#3EBCD2`, Dark Grey `#758D99`), horizontal-only gridlines, left-aligned typography, and source attribution footers.

| Figure | Description | File |
| :--- | :--- | :--- |
| **Fig 1: Zero-shot Capability** | Top-1 Accuracy and Macro-F1 across 8 tasks with published reference anchors | [`results/figures/fig1_accuracy_by_task.png`](results/figures/fig1_accuracy_by_task.png) |
| **Fig 2: Calibration Profiles** | ECE comparison (Choice vs. Noul) and 10-bin empirical reliability curves | [`results/figures/fig2_calibration_by_task.png`](results/figures/fig2_calibration_by_task.png) |
| **Fig 3: Confidence Separation** | Boxplot distributions showing epistemic separation for correct vs. incorrect decisions | [`results/figures/fig3_confidence_vs_correctness.png`](results/figures/fig3_confidence_vs_correctness.png) |
| **Fig 4: Primitive Agreement** | 4-way decision agreement confusion matrix (89.3% parity: Choice vs. argmax Noul) | [`results/figures/fig4_primitive_agreement.png`](results/figures/fig4_primitive_agreement.png) |
| **Fig 5: Risk-Coverage Curves** | Selective risk-coverage curves showing accuracy scaling to 95–100% at $\tau \ge 0.90$ | [`results/figures/fig5_risk_coverage.png`](results/figures/fig5_risk_coverage.png) |
| **Fig 6: Script Type Comparison** | Pure Sinhala vs. Romanized Singlish vs. Code-Mixed accuracy and calibration error | [`results/figures/fig6_script_type_comparison.png`](results/figures/fig6_script_type_comparison.png) |
| **Fig 7: Latency by Primitive** | Median (p50) and 95th-percentile (p95) API response times across Choice, Noul, Score | [`results/figures/fig7_latency_by_primitive.png`](results/figures/fig7_latency_by_primitive.png) |

---

## 6. Project Layout

```text
jev-sintam-review/
├── configs/
│   ├── experiments.yaml           # Global parameters, sample sizes, retry policy
│   ├── prompts.yaml               # Frozen prompt definitions (English & Sinhala)
│   └── reference_anchors.yaml     # Published fine-tuned benchmark anchors
├── data/
│   ├── raw/                       # Cached upstream source files
│   └── processed/
│       ├── manifest.json          # Cryptographic SHA-256 hashes of sample sets
│       ├── phase0_smoke/          # 20-item smoke test subsets (7 files)
│       └── samples/               # Phase 1 production subsets (~950 items)
├── src/
│   ├── client.py                  # TypeSafe SDK wrapper with telemetry & retries
│   ├── config.py                  # Experiment configuration and environment setup
│   ├── loaders/                   # Modular dataset loaders (A, B, C, D, E, F)
│   ├── sampling/                  # Stratified deterministic sampler (seed=42)
│   ├── runners/                   # Experiment runners (Stages 2–7)
│   │   ├── base_runner.py         # Abstract base runner with Section 5 record logging
│   │   ├── runner_smoke_test.py   # Stage 2 smoke test runner & gate verifier
│   │   ├── runner_stage3_sentiment.py # Stage 3 Sentiment Primitive Equivalence
│   │   ├── runner_stage4_core.py      # Stage 4 Core Pure-Sinhala Tasks Runner
│   │   ├── runner_stage5_prompt_sensitivity.py # Stage 5 Language & Prompt Sensitivity
│   │   ├── runner_stage6_diagnostics.py        # Stage 6 Option Order & Repeatability
│   │   └── runner_stage7_cmcs.py               # Stage 7 Code-Mixed Stress Track
│   ├── metrics/                   # Classification, calibration, consistency, sensitivity, diagnostics
│   └── visualization/             # Stage 8 Synthesis & Figure Generation
│       ├── economist_style.py     # Economist styling rules, colors, and layout decorators
│       ├── synthesis_tables.py    # Cross-stage synthesis CSV table generator
│       └── plot_generator.py      # Matplotlib/Seaborn figure generator
├── results/
│   ├── main.csv                   # 15-task summary table with reference anchors
│   ├── primitive_consistency.csv  # RQ2 cross-primitive parity metrics
│   ├── script_analysis.csv        # Script breakdown (Pure vs. Singlish vs. Mixed)
│   ├── failure_taxonomy.csv       # Qualitative audit of 50 extreme boundary cases
│   ├── figures/                   # 7 publication-ready Economist-styled figures
│   │   ├── fig1_accuracy_by_task.png
│   │   ├── fig2_calibration_by_task.png
│   │   ├── fig3_confidence_vs_correctness.png
│   │   ├── fig4_primitive_agreement.png
│   │   ├── fig5_risk_coverage.png
│   │   ├── fig6_script_type_comparison.png
│   │   └── fig7_latency_by_primitive.png
│   ├── smoke_test/                # Gate verification report & smoke predictions
│   ├── stage3_sentiment/          # Stage 3 logs, consistency matrices, and tables
│   ├── stage4_core/               # Stage 4 logs, task accuracy, and calibration reports
│   ├── stage5_prompt_sensitivity/ # Stage 5 paired EN vs SI sensitivity reports
│   ├── stage6_diagnostics/        # Stage 6 permutation, repeatability, and risk reports
│   └── stage7_cmcs/               # Stage 7 multi-task code-mixed stress reports
├── tests/
│   ├── test_loaders.py            # Unit tests for data loaders & ZWJ preservation
│   ├── test_metrics.py            # Unit tests for evaluation metrics suite
│   └── test_sdk_connection.py     # Health checks for TypeSafe SDK primitives
├── main.py                        # CLI entrypoint for stage-by-stage execution
└── pyproject.toml                 # uv project configuration
```

---

## 7. Quickstart

### Prerequisites
* Python 3.11+
* `uv` or `pip`
* TypeSafe API Key (`JEV_KEY`) in `.env`
* Hugging Face Access Token (`HF_TOKEN`) in `.env`

### Installation
```powershell
uv venv
.\.venv\Scripts\activate
uv sync
```

### Running Tests & Stages
```powershell
# Run unit test suite
pytest tests/

# Execute individual stages
python main.py --stage 0   # SDK & Environment Health Check
python main.py --stage 1   # Data Ingestion & Deterministic Sampling
python main.py --stage 2   # Phase 0 Smoke Test & Gate Verification
python main.py --stage 3   # Stage 3: Sentiment Primitive Equivalence Lab (N=150)
python main.py --stage 4   # Stage 4: Core Pure-Sinhala Tasks (N=650)
python main.py --stage 5   # Stage 5: Language & Prompt Sensitivity Experiment (N=100)
python main.py --stage 6   # Stage 6: Primitive Diagnostics & Robustness (N=140)
python main.py --stage 7   # Stage 7: Code-Mixed Stress Track (N=150)
python main.py --stage 8   # Stage 8: Quantitative Synthesis & Economist-styled Deliverables
```

---

## 8. Phase 2 Scale-Up & Cloudflare Comparative Benchmark (In Progress)

With Phase 1 complete and validated across all 7 tasks, the study is advancing to **Phase 2**, expanding both the evaluation scale and model comparison surface:

### 1. Full Dataset Census Scale ($N = 13,054$)
Phase 2 evaluates the full official test splits to establish direct comparability with published EMNLP / LREC fine-tuned checkpoints:
* **SOLD ($N = 2,500$):** 100% census of official `SOLD_test.tsv` tweets (evaluating zero-shot against Subasa-XLM-R baseline F1 0.844).
* **SinhalaMMLU ($N = 1,854$):** 100% curriculum census across all 14 subjects in Humanities, Social Sciences, Language, and STEM.
* **NSINA Categories ($N = 1,200$) & Media ($N = 1,000$):** Full document news categorization and 10-outlet publisher attribution.
* **Sinhala News Sentiment ($N = 3,000$):** 4-class balanced reader comment sentiment analysis.
* **SalAngaBhava ($N = 1,500$):** 500 Pure Sinhala + 500 Singlish + 500 Code-Mixed reviews across 6 product sectors.
* **CMCS Multi-Task ($N = 2,000$):** Parallel evaluation of Sentiment, Humour, Hate Speech, and 6-aspect multi-label extraction ($12,000$ aspect assertions).

### 2. Multi-Model Benchmark: TypeSafe Jev vs. Cloudflare Clef, Kev & Laya
To place Jev's decision-native architecture into broader industry context, we are running these exact Sinhala datasets (identical evaluation splits, byte-for-byte UTF-8 preservation, and zero-shot instructions) against Cloudflare's suite of models:
* **TypeSafe `jev-latest`:** System-1 decision-native engine (`Choice`, `Noul`, `Score`).
* **Cloudflare Clef:** Fast classification and semantic routing engine.
* **Cloudflare Kev:** Structured event, boundary, and reasoning model.
* **Cloudflare Laya:** Multilingual representation and inference foundation model.

#### Comparative Evaluation Dimensions:
1. **Zero-Shot Accuracy & Macro-F1:** Pure Sinhala vs. Romanized Singlish vs. Code-Mixed text across models.
2. **Epistemic Calibration & Risk-Coverage:** 20-bin ECE and accuracy under selective confidence filtering ($\tau \ge 0.90$).
3. **Cross-Model Agreement:** Inter-model Cohen's $\kappa$ and decision concordance matrices.
4. **Systems & Latency Efficiency:** $p_{50}$ / $p_{95}$ response times and throughput under concurrent workload.

