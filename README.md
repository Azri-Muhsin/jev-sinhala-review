# Jev-Sinhala Quick Capability & Primitive Consistency Probe

> **Objective:** A zero-shot empirical evaluation of TypeSafe's `jev-latest` on low-resource Sinhala and Sinhala-English code-mixed benchmarks, evaluating linguistic comprehension, decision primitive consistency (`Choice`, `Noul`, `Score`), calibration quality, and robustness.

---

## 1. Executive Summary & Research Questions

This probe evaluates the zero-shot decision capabilities of TypeSafe's System-1 model (`jev-latest`) on official test splits of established Sinhala NLP benchmarks. Unlike standard generative LLM benchmarks that evaluate text-generation perplexity or token-level logprobs, this study investigates how native decision primitives operate on low-resource Indic morphosyntax, agglutination, and mixed scripts.

We investigate four core research questions:

1. **RQ1 (Linguistic & Semantic Capability):** Can `jev-latest` accurately categorize and reason over diverse Sinhala domains (social media, formal news, multi-discipline academic QA, e-commerce reviews) and scripts (pure Sinhala script vs. romanized code-mixed Sinhala-English)?
2. **RQ2 (Primitive Consistency):** When presented with the exact same input state, do different decision primitives—**`Choice`** (multi-class distribution), **`Noul`** (binary true/false probabilities), and **`Score`** (bounded ordinal expectation)—yield mutually consistent beliefs?
3. **RQ3 (Calibration & Risk-Coverage):** Are Jev's output probabilities and confidence estimates well-calibrated (ECE, Brier score)? Can confidence thresholds ($\ge 0.50, 0.70, 0.80, 0.90, 0.95$) effectively filter out errors for high-reliability automated pipelines?
4. **RQ4 (Robustness & Invariance Diagnostics):** Is Jev invariant to choice-order permutation (A/B/C/D order effects)? How sensitive is performance to instruction language (English vs. native Sinhala)? Is inference repeatable across stochastic runs?

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
Stage 5: Language & Prompt Sensitivity (English vs. Sinhala) ──► [PENDING]
   │
Stage 6: Robustness & Diagnostics (Order, Repeatability, ECE) ──► [PENDING]
   │
Stage 7: Code-Mixed Stress Track (CMCS, N=150) ────────────────► [PENDING]
   │
Stage 8: Quantitative Synthesis & Final Report ────────────────► [PENDING]
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

---

## 6. Project Layout

```text
jev-sintam-review/
├── configs/
│   ├── experiments.yaml           # Global parameters, sample sizes, retry policy
│   └── prompts.yaml               # Frozen prompt definitions (English & Sinhala)
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
│   ├── runners/                   # Experiment runners
│   │   ├── base_runner.py         # Abstract base runner with Section 5 record logging
│   │   ├── runner_smoke_test.py   # Stage 2 smoke test runner & gate verifier
│   │   ├── runner_stage3_sentiment.py # Stage 3 Sentiment Primitive Equivalence
│   │   └── runner_stage4_core.py      # Stage 4 Core Pure-Sinhala Tasks Runner
│   └── metrics/                   # Classification, calibration, consistency metrics
├── results/
│   ├── smoke_test/                # Gate verification report & smoke predictions
│   ├── stage3_sentiment/          # Stage 3 logs, consistency matrices, and tables
│   ├── stage4_core/               # Stage 4 logs, task accuracy, and calibration reports
│   └── figures/                   # Generated evaluation plots and diagrams
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
python main.py --stage 3   # Stage 3: Sentiment Primitive Equivalence Lab
python main.py --stage 4   # Stage 4: Core Pure-Sinhala Tasks (N=650)
```
