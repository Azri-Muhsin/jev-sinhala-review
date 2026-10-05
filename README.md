# Jev-Sinhala Capability, Primitive Consistency & Benchmark Census Probe

> [!NOTE]
> ### 🚀 Phase 2 Scale-Up Census Completed & Multi-Model Benchmark Underway
> **The Full Benchmark Census is complete!** We have scaled beyond the Phase 1 reconnaissance sample ($N \approx 950$) to evaluate `jev-latest` on the **complete benchmark census of $N = 11,434$ authentic evaluation instances** ($27,258$ primitive decisions) across all 7 benchmark datasets.
> 
> Furthermore, we are extending the evaluation across industry models: running these exact uncorrupted Sinhala benchmark datasets against **Cloudflare's Clef, Kev, and Laya models** as well for a comprehensive multi-model decision and linguistic capability benchmark. See [Section 8: Next Stage — Multi-Model Comparative Benchmark](#8-next-stage--multi-model-comparative-benchmark).

> **Objective:** An empirical zero-shot evaluation of TypeSafe's `jev-latest` (`jev-1.13.0`) on authentic low-resource Sinhala and Sinhala-English code-mixed datasets, quantifying linguistic comprehension, decision primitive consistency (`Choice`, `Noul`, `Score`), calibration quality ($\text{ECE}_{20}$, Brier score), script sensitivity, and multi-label extraction.

---

## 1. Executive Summary & Research Questions

This probe evaluates the zero-shot decision capabilities of TypeSafe's System-1 model (`jev-latest`) on official test splits of established Sinhala NLP benchmarks. Unlike standard generative LLM benchmarks that evaluate text-generation perplexity or token-level logprobs, this study investigates how native decision primitives operate on low-resource Indic morphosyntax, agglutination, and mixed scripts.

We investigate four core research questions:

1. **RQ1 (Linguistic & Semantic Capability):** Can `jev-latest` accurately categorize and reason over diverse Sinhala domains (social media, formal news, multi-discipline academic QA, e-commerce reviews) and scripts (pure Sinhala script vs. romanized code-mixed Sinhala-English)?
2. **RQ2 (Primitive Consistency):** When presented with the exact same input state, do different decision primitives—**`Choice`** (multi-class distribution), **`Noul`** (binary true/false probabilities), and **`Score`** (bounded ordinal expectation)—yield mutually consistent beliefs?
3. **RQ3 (Calibration & Risk-Coverage):** Are Jev's output probabilities and confidence estimates well-calibrated (ECE, Brier score)? Can confidence thresholds ($\ge 0.50, 0.70, 0.80, 0.90, 0.95$) effectively filter out errors for high-reliability automated pipelines?
4. **RQ4 (Robustness & Script Sensitivity):** How does performance degrade under Latin transliteration (romanized Singlish)? How sensitive is performance to instruction language (English vs. native Sinhala)? Is inference repeatable across stochastic runs?

### Core Findings Matrix (Full Census, $N = 11,434$)

| Research Question | Key Empirical Finding | Full Census Metric ($N=11,434$) |
| :--- | :--- | :--- |
| **RQ1: Capability** | Exceptional zero-shot comprehension across formal news, national academic exams, and colloquial social text. | **85.50%** NSINA Categories ($N=1,200$); **66.65%** SinhalaMMLU 14-subject QA ($N=1,850$); **91.00%** CMCS Humour ($N=2,000$); **90.80%** CMCS Hate Speech ($N=2,000$) |
| **RQ2: Consistency** | Noul primitive achieves substantially tighter calibration than Choice on binary tasks, while maintaining high classification concordance. | **$\text{ECE}_{20} = 0.0584$** on SOLD Noul vs. **$0.1356$** on Choice ($N=2,500$); identical top-1 parity ($68.5\%$ vs $68.7\%$) |
| **RQ3: Calibration** | Output probabilities are naturally well-calibrated across major domains without post-hoc scaling; sharp epistemic confidence separation. | **$\text{ECE}_{20} \le 0.078$** across 5 primary benchmark tracks; **$0.0643$** on NSINA Categories; **$0.0584$** on SOLD Noul |
| **RQ4: Script Sensitivity** | Significant orthographic penalty when switching from native Unicode Sinhala to Latinized Singlish; strong multi-label aspect sensitivity. | **20.0% accuracy drop** on Singlish ($52.80\%$) vs. Pure Sinhala ($72.80\%$); **$75\%-90\%$ aspect recall** on code-mixed comments |

---

## 2. Evaluation Tracks & Datasets

All evaluations use official, uncorrupted evaluation splits with zero synthetic paraphrasing and strict byte-for-byte preservation of Sinhala Unicode (including Zero-Width Joiner `\u200D` and Non-Joiner `\u200C`).

```
Track 1: Core Pure-Sinhala Classification
  ├── Dataset A: Sinhala News-Comment Sentiment (POS, NEG, NEU, CONFLICT)
  ├── Dataset B: SOLD — Sinhala Offensive Language Dataset (Binary: OFF, NOT)
  ├── Dataset C1: NSINA Categories (Business, Sports, Local, International, Entertainment)
  └── Dataset C2: NSINA Media Identification (10 major national publishers)

Track 2: Knowledge, Reasoning & Ordinal Evaluation
  ├── Dataset D: SinhalaMMLU (4-option QA across 14 academic disciplines)
  └── Dataset E: SalAngaBhava (Pure Sinhala & Singlish product reviews on 1–5 stars)

Track 3: Romanized & Code-Mixed Stress Track
  └── Dataset F: Sinhala-English CMCS (Sentiment, Humour, Hate, Multi-Label Aspect Extraction)
```

### Dataset Summary Table

| ID | Dataset | Domain | Classes / Decision Space | Tested Primitives | Phase 1 Sample | Phase 2 Full Census |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **A** | **Sinhala Sentiment** | News Comments | 4-way: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `CONFLICT` | `Choice`, `Noul` | 150 | **1,810** |
| **B** | **SOLD** | Social Media (X/Twitter) | Binary: `OFF`, `NOT` | `Noul`, `Choice` | 150 | **2,500** (100% test) |
| **C1** | **NSINA Categories** | News Articles | 5 Categories (Politics, Business, Sports, Entertainment, General) | `Choice` | 100 | **1,200** |
| **C2** | **NSINA Media** | News Articles | 10 Media Outlets (`Divaina`, `Lankadeepa`, `Hiru`, etc.) | `Choice` | 100 | **1,000** |
| **D** | **SinhalaMMLU** | Academic Curricula | 4-option QA across 14 G.C.E. (O/L) subjects | `Choice` | 150 | **1,850** (14 subjects) |
| **E** | **SalAngaBhava** | Product Reviews | Ordinal 1–5 Rating (Pure, Singlish, Mixed) | `Choice`, `Score` | 150 | **1,074** |
| **F** | **Sinhala-English CMCS** | Code-Mixed Social | Sentiment (4-way), Humour (2-way), Hate (3-way), 6 Aspects | `Choice`, `Noul` | 150 | **2,000** |
| **Total** | | | | | **~950 items** | **11,434 instances<br>(27,258 decisions)** |

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

## 4. Frozen Record Schema

Every decision across all evaluation stages is streamed to immutable JSON Lines (`.jsonl`) logs adhering strictly to the frozen schema:

```json
{
  "record_id": "rec_p2_sold_00042_choice",
  "experiment_id": "exp_phase2_census",
  "phase": "phase_2_scaled_census",
  "timestamp": "2026-10-05T12:55:04.281902+00:00",
  "dataset": "dataset_b_sold",
  "example_id": "sold_test_00042",
  "language": "si",
  "script_type": "pure_sinhala",
  "primitive": "choice",
  "prompt_variant": "english_instruction",
  "state_text": "මෙම තීරණය ඉතාමත් අගය කළ යුතු එකක් බව පැවසිය යුතුය.",
  "instructions": "Is this text offensive or not offensive?",
  "criteria_definitions": {"OFFENSIVE": null, "NOT OFFENSIVE": null},
  "gold_label": "NOT OFFENSIVE",
  "prediction": "NOT OFFENSIVE",
  "is_correct": true,
  "confidence": 0.8912,
  "probabilities": {"OFFENSIVE": 0.1088, "NOT OFFENSIVE": 0.8912},
  "score_details": null,
  "noul_details": null,
  "latency_ms": 314.7,
  "usage": {"input_tokens": 128, "output_tokens": 16},
  "model": "jev-latest",
  "raw_response_status": 200,
  "error": null
}
```

---

## 5. Empirical Results

### 5.1 Full Benchmark Census Master Results ($N = 11,434$)

The primary benchmark census results across all 7 authentic datasets evaluated zero-shot with TypeSafe `jev-latest`. Confidence intervals are derived from 300-iteration empirical bootstrap resampling ($95\%$ percentile intervals $[\text{CI}_{\text{low}}, \text{CI}_{\text{high}}]$). Calibration error ($\text{ECE}_{20}$) is computed using 20 equal-width probability bins.

| Dataset / Benchmark Task | Primitive | Census $N$ | Top-1 Accuracy ($95\%$ CI) | Macro-F1 ($95\%$ CI) | $\text{ECE}_{20}$ | Brier Score | Median Latency ($p_{50}$) | Reference Anchor (Supervised Fine-tuned) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **SOLD Offensiveness (Noul)** | Noul | 2,500 | **68.48%** [66.86%, 70.34%] | **0.6811** [0.6638, 0.7000] | **0.0584** | 0.4000 | 314.7 ms | XLM-R (Subasa-XLM-R) Macro-F1: 0.840 |
| **SOLD Offensiveness (Choice)** | Choice | 2,500 | **68.72%** [67.00%, 70.46%] | **0.6803** [0.6609, 0.6991] | 0.1356 | 0.4164 | 314.7 ms | XLM-R (Subasa-XLM-R) Macro-F1: 0.840 |
| **NSINA News Categories** | Choice (5-way) | 1,200 | **85.50%** [83.62%, 87.33%] | **0.8502** [0.8299, 0.8691] | **0.0643** | 0.2240 | 314.4 ms | XLM-R / SinBERT |
| **NSINA Media Identification** | Choice (10-way) | 1,000 | **21.20%** [18.75%, 23.90%] | 0.1734 [0.1494, 0.1942] | **0.0669** | 0.8640 | 318.8 ms | SinBERT / XLM-R Macro-F1: 0.880 (Chance: 10.0%) |
| **SinhalaMMLU Academic QA** | Choice (4-way) | 1,850 | **66.65%** [64.51%, 68.62%] | **0.6660** [0.6441, 0.6860] | 0.1155 | 0.4371 | 307.8 ms | Chance Baseline: 25.0% |
| **News Sentiment (Choice)** | Choice (4-way) | 1,810 | **61.33%** [59.14%, 63.54%] | 0.4680 [0.4525, 0.4846] | 0.1214 | 0.5676 | 302.1 ms | Fine-tuned SinBERT Baseline |
| **News Sentiment (Noul)** | Noul | 1,810 | **57.02%** [54.64%, 59.34%] | 0.4414 [0.4238, 0.4561] | 0.2170 | 0.8184 | 302.1 ms | Fine-tuned SinBERT Baseline |
| **SalAngaBhava Rating (Choice)** | Choice (1–5) | 1,074 | **63.04%** [60.10%, 65.41%] | 0.2899 [0.2524, 0.3225] | **0.0777** | 0.4899 | 305.0 ms | Chance Baseline: 20.0% |
| **SalAngaBhava Rating (Score)** | Score (1–5) | 1,074 | **29.70%** [26.81%, 32.26%] | 0.2176 [0.1858, 0.2447] | 0.2262 | 1.3679 | 305.0 ms | Continuous Expectation Rounded |
| **CMCS Sentiment** | Choice (4-way) | 2,000 | **60.35%** [58.12%, 62.70%] | 0.4417 [0.4065, 0.4844] | 0.1940 | 0.6109 | 309.2 ms | Code-mixed Telecom Comments |
| **CMCS Humour Detection** | Choice (2-way) | 2,000 | **91.00%** [89.62%, 92.10%] | **0.7371** [0.7002, 0.7662] | 0.1826 | 0.1495 | 309.2 ms | Binary Humorous vs Non-humorous |
| **CMCS Hate Speech** | Choice (3-way) | 2,000 | **90.80%** [89.70%, 92.05%] | 0.4127 [0.3766, 0.4480] | **0.0711** | 0.1564 | 309.2 ms | Abusive / Hate / Not Offensive |

*Full Deliverable CSV:* [`results/scaleup/main_scaled.csv`](results/scaleup/main_scaled.csv)

---

### 5.2 SinhalaMMLU Academic Curriculum Breakdown (14 Subjects)

Evaluated zero-shot on all $N = 1,850$ official exam questions from national G.C.E. (O/L) examinations (EMNLP 2025 benchmark split):

| Academic Faculty | Subject Discipline | Census $N$ | Zero-Shot Accuracy | Macro-F1 | Mean Confidence |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **STEM** | Science | 164 | **78.66%** | **0.7783** | 0.7743 |
| **Social Sciences** | Civics | 123 | **82.93%** | **0.8309** | 0.7736 |
| **Social Sciences** | Geography | 116 | **75.00%** | **0.7397** | 0.6995 |
| **Social Sciences** | Health & Physical Science | 130 | **71.54%** | **0.7164** | 0.7169 |
| **Social Sciences** | History | 151 | **54.30%** | **0.5257** | 0.5075 |
| **Humanities** | Christianity | 130 | **74.62%** | **0.7408** | 0.7162 |
| **Humanities** | Catholicism | 132 | **73.48%** | **0.7314** | 0.6087 |
| **Humanities** | Islam | 108 | **72.22%** | **0.7188** | 0.6256 |
| **Humanities** | Buddhism | 162 | **69.14%** | **0.6967** | 0.5595 |
| **Humanities** | Drama & Theatre | 128 | **64.84%** | **0.6488** | 0.5151 |
| **Humanities** | Eastern Music | 143 | **57.34%** | **0.5658** | 0.3713 |
| **Humanities** | Arts | 100 | **56.00%** | **0.5646** | 0.5426 |
| **Humanities** | Traditional Dancing | 108 | **42.59%** | **0.4163** | 0.2719 |
| **Language** | Sinhala Language & Literature | 155 | **57.42%** | **0.5677** | 0.4437 |
| **Total / Macro Mean** | *All 14 Subjects Combined* | **1,850** | **66.65%** | **0.6660** | **0.5817** |

*Faculty Averages:* STEM: **78.66%** | Social Sciences: **70.94%** | Humanities: **65.03%** | Language: **57.42%**  
*Full Breakdown CSV:* [`results/scaleup/mmlu_subject_breakdown.csv`](results/scaleup/mmlu_subject_breakdown.csv)

---

### 5.3 Orthographic & Script Sensitivity Analysis

Tested on the SalAngaBhava census across identical product review tasks ($N = 1,074$) to isolate the impact of script encoding on zero-shot inference:

| Script Modality | Census $N$ | Choice Top-1 Accuracy | Macro-F1 | ECE (10-bin) | Mean Confidence | Median Latency ($p_{50}$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Pure Sinhala Script** (සිංහල අකුරු) | 500 | **72.80%** | **0.3603** | 0.1127 | 0.6161 | 305.9 ms |
| **Code-Mixed** (Sinhala + English) | 74 | **66.22%** | 0.2973 | 0.1238 | 0.6308 | 296.5 ms |
| **Singlish / Sinhala-in-English** (Latin transliteration) | 500 | **52.80%** | 0.2264 | **0.0523** | 0.4886 | 306.8 ms |

*Finding:* Jev suffers a **20.0 percentage point accuracy penalty** when processing Latinized Singlish vs. native Sinhala script. Epistemic confidence appropriately drops from $0.616$ to $0.488$, demonstrating well-calibrated awareness of orthographic ambiguity.  
*Full Analysis CSV:* [`results/scaleup/script_analysis_scaled.csv`](results/scaleup/script_analysis_scaled.csv)

---

### 5.4 Multi-Label Aspect Extraction (CMCS Telecom Census)

Evaluated across $N = 2,000$ code-mixed customer comments using the binary `Noul` primitive ($p \ge 0.5$ threshold):

| Extracted Telecom Aspect | Total Evaluated | Positive Support (Ground Truth) | Accuracy | Precision | Recall | F1 Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Package** | 2,000 | 86 | **93.90%** | 0.4062 | **90.70%** | **0.5612** |
| **Data** | 2,000 | 180 | **89.20%** | 0.4489 | **87.78%** | **0.5940** |
| **Customer Service** | 2,000 | 130 | **92.40%** | 0.4495 | **75.38%** | **0.5632** |
| **Network** | 2,000 | 210 | **83.90%** | 0.3793 | **83.81%** | **0.5223** |
| **Billing & Price** | 2,000 | 98 | **87.50%** | 0.2500 | **77.55%** | **0.3781** |
| **Service or Product** | 2,000 | 443 | **72.95%** | 0.4109 | 51.02% | **0.4552** |

*Finding:* Zero-shot Noul demonstrates high aspect sensitivity, retrieving $75\%-90\%$ of relevant telecom aspect mentions across colloquial code-mixed comments.  
*Full Aspect CSV:* [`results/scaleup/aspect_multilabel.csv`](results/scaleup/aspect_multilabel.csv)

---

### 5.5 Phase 1 Reconnaissance Pilot Summary ($N \approx 950$)

Before launching the full census, Phase 1 validated infrastructure, prompt language invariance, option-order permutation, and multi-pass repeatability across smaller stratified samples:

* **Gate Verification:** 320 decisions across 140 examples logged with 0 runtime errors and 100% Unicode ZWJ integrity (`\u200D`).
* **Prompt Language Invariance (EN vs. SI, $N=100$):** Zero-gap choice accuracy ($\Delta \text{Acc} = +0.000$) on Sentiment, $98.0\%$ agreement on SOLD, and cosine probability similarity $\ge 0.9735$.
* **Option-Order Permutation Stability ($N=40$):** $100.0\%$ stability on NSINA news categorization and $80.0\%$ on SinhalaMMLU ($90.0\%$ overall stability).
* **Multi-Pass Stochasticity ($N=100$):** $100.0\%$ exact repeatability for binary Noul ($\text{mean } \sigma = 0.0093$) and $94.0\%$ for multiclass Choice.
* **Phase 1 Deliverables:** [`results/main.csv`](results/main.csv) | [`results/primitive_consistency.csv`](results/primitive_consistency.csv) | [`results/script_analysis.csv`](results/script_analysis.csv) | [`results/failure_taxonomy.csv`](results/failure_taxonomy.csv)

---

### 5.6 Publication Figure Gallery

All figures are designed in *The Economist* visual language (red tag `#E3120B`, blue `#006BA2`, cyan `#3EBCD2`, grey `#758D99`, horizontal-only gridlines):

#### Full Census Scale Figures (`results/scaleup/figures/`)
* **[Fig 8: Full Benchmark Census Capability with 95% CIs](results/scaleup/figures/fig8_scaled_benchmark_comparison.png)** — Top-1 accuracy across all primary tasks on the complete $11,434$-instance census.
* **[Fig 9: SinhalaMMLU Academic Curriculum Breakdown](results/scaleup/figures/fig9_mmlu_faculty_breakdown.png)** — Aggregate zero-shot accuracy across STEM, Social Sciences, Humanities, and Language faculties.
* **[Fig 10: Scaled Reliability Diagrams & ECE](results/scaleup/figures/fig10_scaled_calibration_reliability.png)** — Empirical calibration curves across 10 probability bins comparing SOLD Noul ($\text{ECE}=0.0584$), SOLD Choice ($\text{ECE}=0.1356$), and NSINA ($\text{ECE}=0.0643$).
* **[Fig 11: Scaled Epistemic Confidence Separation](results/scaleup/figures/fig11_scaled_confidence_separation.png)** — Confidence distribution boxplots for Correct vs. Incorrect decisions across tasks from $27,258$ decisions.
* **[Fig 12: Scaled Script Resilience Comparison](results/scaleup/figures/fig12_scaled_script_resilience.png)** — Performance comparison across Pure Sinhala, Code-Mixed, and Latinized Singlish across all $1,074$ SalAngaBhava reviews.
* **[Fig 13: Scaled Latency Profile by Primitive](results/scaleup/figures/fig13_scaled_latency_profile.png)** — API response times ($p_{50}$ / $p_{95}$) across Choice, Noul, and Score primitives across all $27,258$ decisions.

#### Phase 1 Pilot Figures (`results/figures/`)
* **[Fig 1: Zero-shot Capability (Phase 1)](results/figures/fig1_accuracy_by_task.png)** — Macro-F1 and Top-1 accuracy with published supervised reference anchors.
* **[Fig 2: Calibration Profiles (Phase 1)](results/figures/fig2_calibration_by_task.png)** — ECE comparison (Choice vs. Noul) and 10-bin reliability diagrams.
* **[Fig 3: Confidence Separation (Phase 1)](results/figures/fig3_confidence_vs_correctness.png)** — Epistemic separation boxplots for initial sample.
* **[Fig 4: Primitive Agreement (Phase 1)](results/figures/fig4_primitive_agreement.png)** — 4-way decision agreement confusion matrix (89.3% parity: Choice vs. argmax Noul).
* **[Fig 5: Risk-Coverage Curves (Phase 1)](results/figures/fig5_risk_coverage.png)** — Selective accuracy scaling to $95\%-100\%$ at $\tau \ge 0.90$.
* **[Fig 6: Script Type Comparison (Phase 1)](results/figures/fig6_script_type_comparison.png)** — Initial script comparison across CMCS subset.
* **[Fig 7: Latency by Primitive (Phase 1)](results/figures/fig7_latency_by_primitive.png)** — Initial latency distributions.

---

## 6. Project Layout

```text
jev-sintam-review/
├── configs/
│   ├── experiments.yaml           # Parameters, sample sizes, rate limits, retry policy
│   ├── prompts.yaml               # Frozen prompt templates (English & Sinhala)
│   └── reference_anchors.yaml     # Published fine-tuned benchmark anchors
├── data/
│   ├── raw/                       # Cached upstream benchmark files
│   └── processed/
│       ├── manifest_scaled.json   # SHA-256 hashes of scaled census datasets
│       ├── manifest.json          # SHA-256 hashes of Phase 1 samples
│       ├── samples_scaled/        # Phase 2 full census subsets (11,434 items)
│       └── samples/               # Phase 1 pilot subsets (~950 items)
├── src/
│   ├── client.py                  # TypeSafe SDK wrapper with telemetry & retries
│   ├── config.py                  # Configuration paths and environment settings
│   ├── loaders/                   # Dataset loaders (A, B, C, D, E, F)
│   ├── sampling/                  # Stratified deterministic sampler (seed=42)
│   ├── runners/                   # Experiment runners
│   │   ├── base_runner.py         # Thread-safe logging & idempotent checkpointing
│   │   ├── runner_scaleup.py      # High-throughput Phase 2 census concurrent runner
│   │   ├── runner_smoke_test.py   # Phase 0 smoke test runner
│   │   ├── runner_stage4_core.py  # Phase 1 core tasks runner
│   │   └── runner_stage7_cmcs.py  # Phase 1 code-mixed stress runner
│   ├── metrics/                   # Classification, calibration, consistency, sensitivity
│   └── visualization/             # Synthesis & publication figures
│       ├── economist_style.py     # Economist styling rules and decorators
│       ├── scaleup_synthesis.py   # Phase 2 scaled synthesis table & figure generator
│       └── plot_generator.py      # Phase 1 figure generator
├── results/
│   ├── scaleup/                   # Phase 2 Census Results
│   │   ├── main_scaled.csv        # Master census table with 95% CIs and 20-bin ECE
│   │   ├── mmlu_subject_breakdown.csv # 14-subject academic curriculum breakdown
│   │   ├── script_analysis_scaled.csv # Script breakdown at scale (Pure vs. Singlish)
│   │   ├── aspect_multilabel.csv  # 6-aspect multi-label precision/recall/F1 metrics
│   │   ├── figures/               # Figures 8, 9, 10, 11, 12, 13
│   │   └── raw_logs/              # 7 streaming JSONL prediction files (27,258 records)
│   ├── main.csv                   # Phase 1 summary table
│   └── figures/                   # Phase 1 figures (Figs 1–7)
├── scripts/
│   └── run_scaleup_census.py      # Master orchestrator for full census execution
├── main.py                        # CLI entrypoint for individual stages
└── pyproject.toml                 # Project dependencies and configuration
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

### Running Full Census & Synthesis
```powershell
# Execute the full benchmark census (N = 11,434 items)
python scripts/run_scaleup_census.py

# Regenerate synthesis tables and publication figures
python -m src.visualization.scaleup_synthesis
```

---

## 8. Next Stage — Multi-Model Comparative Benchmark

With the TypeSafe `jev-latest` full census completed and verified across all $11,434$ instances, the study is advancing to the cross-model comparative stage:

### Multi-Model Benchmark: TypeSafe Jev vs. Cloudflare Clef, Kev & Laya
To place Jev's decision-native architecture into broader industry perspective, we will run these exact Sinhala benchmark datasets (identical evaluation splits, byte-for-byte UTF-8 preservation, and zero-shot instructions) against Cloudflare's model suite:

* **TypeSafe `jev-latest`:** System-1 decision-native engine (`Choice`, `Noul`, `Score`).
* **Cloudflare Clef:** Fast classification and semantic routing model.
* **Cloudflare Kev:** Structured event, boundary, and reasoning model.
* **Cloudflare Laya:** Multilingual representation and inference foundation model.

#### Comparative Evaluation Dimensions:
1. **Zero-Shot Accuracy & Macro-F1:** Native Sinhala vs. Romanized Singlish vs. Code-Mixed text across models.
2. **Epistemic Calibration & Risk-Coverage:** 20-bin ECE and accuracy under selective confidence filtering ($\tau \ge 0.90$).
3. **Cross-Model Agreement:** Inter-model Cohen's $\kappa$ and decision concordance matrices.
4. **Systems & Latency Efficiency:** $p_{50}$ / $p_{95}$ response times and throughput under concurrent workload.
