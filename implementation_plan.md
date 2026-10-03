# Jev-Sinhala: Quick Capability & Primitive Consistency Probe
## Master Implementation Plan (Fine-Grained & Staged)

---

### Executive Overview & Strategic Intent

This document specifies the complete engineering and experimental architecture for the **Jev-Sinhala Quick Capability and Primitive Consistency Probe**. 

The probe is a **rapid, zero-shot reconnaissance experiment** designed to evaluate whether Jev's decision-native interface (`Noul`, `Choice`, and `Score`) demonstrates coherent, calibrated, and robust behavior on real, low-resource Sinhala NLP tasks using the `'jev-latest'` model. It does not attempt to build a comprehensive multilingual benchmark; rather, it assesses whether TypeSafe's architectural promise of typed, parallel, calibrated decision primitives holds up when evaluated against authentic Sinhala text across diverse domains.

```
                                  SOURCE DATASETS (6)
             ┌─────────────────────────────┼─────────────────────────────┐
             ▼                             ▼                             ▼
    Pure Sinhala Core             Knowledge / Education          Code-Mixed Stress
  • Sentiment (News Comments)     • SinhalaMMLU (EMNLP '25)     • CMCS Multi-task
  • SOLD (Offensive Tweets)       • SalAngaBhava (Reviews)        (Aspect, Humour,
  • NSINA (Categories & Media)                                    Hate, LangID)
             │                             │                             │
             └─────────────────────────────┼─────────────────────────────┘
                                           ▼
                                 FROZEN TEST SAMPLING
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
          Phase 0 Smoke Test                           Phase 1 Full Probe
           (N = 20 / task)                         (Stratified Test, N ≈ 950)
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           ▼
                             TYPESAFE JEV RUNNER ENGINE
                     (Model: 'jev-latest' | Noul, Choice, Score)
                                           ▼
                      ┌────────────────────┼────────────────────┐
                      ▼                    ▼                    ▼
             Capability & Accuracy  Cross-Primitive     Calibration &
             (Macro-F1, Acc, NLL)     Consistency        Risk-Coverage
                                    (Heatmaps, Agree)    (ECE, Brier)
                                           │
                                           ▼
                       LIGHTWEIGHT SINHALA-NLP REFERENCE ANCHORS
                    (Logged alongside Jev from published checkpoints)
                                           │
                                           ▼
                               DELIVERABLES & REPORT
                      (main.csv, consistency.csv, 7 Figures)
```

---

## 1. Experimental Guardrails & Non-Negotiables

To ensure total scientific integrity, eliminate data leakage, and maintain defensibility, the implementation must adhere strictly to these hard rules:

1. **Model Specification:** The model target is fixed strictly to `'jev-latest'` via TypeSafe AI.
2. **Two-Phased Execution:** A **Phase 0 Smoke Test** ($N = 20$ examples per task) is executed end-to-end to validate prompt formatting, label mapping, raw SDK parsing, probabilities, Score behavior, and error handling before executing the full probe.
3. **Sinhala Only (Strict Boundary):** Absolutely no Tamil datasets or evaluation tracks. (Previous draft configurations involving Tamil are deprecated).
4. **Zero Synthetic / LLM-Generated Text:** No LLM-generated examples, synthetic variations, or synthetic expansions.
5. **No Paraphrasing or Translation:** Original Sinhala dataset strings must be preserved byte-for-byte (including unicode marks, ZWJ/ZWNJ).
6. **No Fine-Tuning:** Jev is evaluated strictly out-of-the-box (zero-shot).
7. **Preservation of Dataset Text & Labels:** Dataset splits, original gold labels, and text fields are strictly preserved without human alteration.
8. **Frozen & Documented Prompt Engineering:** English and Sinhala prompts are authored, documented, version-controlled, and frozen prior to running experiments.
9. **Appropriate Primitive Mapping:**
   * **`Choice`**: Mutually exclusive nominal categories.
   * **`Noul`**: Binary decisions, one-vs-rest questions, multi-label aspect presence, or threshold checks.
   * **`Score`**: Strictly reserved for inherently ordinal/continuous spaces (e.g., 1–5 customer ratings or 3-class sentiment subset: Negative $\to$ Neutral $\to$ Positive). **Never force Score onto nominal categorical spaces.**
10. **Lightweight Reference Anchors:** Reference performance metrics from published Hugging Face `sinhala-nlp` baseline checkpoints (e.g., SinBERT / XLM-R models for sentiment, SOLD, NSINA) are logged alongside Jev in summary deliverables to provide grounding without overcomplicating the pipeline.

---

## 2. Core Research Questions (RQs) & Empirical Mappings

| RQ | Dimension | Primary Metric / Evaluation Protocol | Primary Target Datasets |
| :--- | :--- | :--- | :--- |
| **RQ1** | **Sinhala Capability** | Macro-F1, Accuracy, Top-1 Exact Match across domains | All 6 Datasets (vs HF Sinhala-NLP anchors) |
| **RQ2** | **Primitive Consistency** | Argmax(Noul) vs Choice winner; Choice vs Score level; Noul contradiction rate ($\sum [\mathbb{P} \ge 0.5]$) | Dataset A (Sentiment), Dataset E (SalAngaBhava) |
| **RQ3** | **Calibration** | Expected Calibration Error (ECE), Brier Score, Multiclass NLL, Reliability Diagrams | Noul & Choice across all tasks |
| **RQ4** | **Prompt Sensitivity** | Performance & Confidence $\Delta$ between English instructions vs native Sinhala instructions | 50-example stratified probe on Datasets A & B |
| **RQ5** | **Failure Modes** | Taxonomy classification of 25 highest-confidence errors & 25 lowest-confidence correct predictions | Qualitative audit across tasks |

---

## 3. Dataset Portfolio & Ingestion Specifications

The probe utilizes six published Sinhala datasets, segmented into three distinct evaluation tracks:

```
Track 1: Pure-Sinhala Core (Datasets A, B, C)
Track 2: Knowledge & Ordinal Lab (Datasets D, E)
Track 3: Romanized / Code-Mixed Stress Track (Dataset F)
```

### Dataset Summary Table

| ID | Dataset Name | Domain | Primary Task & Classes | Primitives Tested | Split Rule | Phase 0 Smoke ($N$) | Phase 1 Probe ($N$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| **A** | Sinhala News-Comment Sentiment | News Comments | 4-way Sentiment: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `CONFLICT` | **Choice** (4-way), **Noul** ($\times 4$), **Score** (3-class subset: Neg $\to$ Neu $\to$ Pos) | Official evaluation/test split | 20 | 150 |
| **B** | SOLD (Sinhala Offensive Language) | Social (Twitter) | Binary Offensiveness: `OFF`, `NOT` | **Noul** (Primary), **Choice** (Primary), **Score** (Binary severity diagnostic: 0=NOT, 1=OFF) | Official test split | 20 | 150 |
| **C1** | NSINA Categories | News Articles | 10 News Categories (Politics, Sports, Business, etc.) | **Choice** (Primary), **Noul** (One-vs-rest diagnostic) | Official category test split | 20 | 100 |
| **C2** | NSINA Media Identification | News Articles | 10 Media Sources (Lankadeepa, Ada Derana, etc.) | **Choice** (Primary), **Noul** (One-vs-rest diagnostic) | Official 80/20 test split | 20 | 100 |
| **D** | SinhalaMMLU | Multi-discipline QA | 4-option Multiple Choice QA across 6 domains / 30 subjects | **Choice** (Primary A/B/C/D), **Noul** (Diagnostic on subset) | Test split (stratified across 6 domains) | 20 | 150 |
| **E** | SalAngaBhava | Product Reviews | Ordinal Rating: $1 \to 2 \to 3 \to 4 \to 5$ | **Score** (Primary 1–5), **Choice** (1–5), **Noul** (Thresholds: $\ge 4$, $\le 2$) | Filtered to `review_type == 'Pure_Sinhala'` | 20 | 150 |
| **F** | Sinhala-English CMCS | Code-mixed Social | 5 Tasks: Sentiment, Humour, Hate, Aspect (Multi-label), Language ID | **Choice**, **Noul** (Primary for multi-label Aspect), **Score** (Diagnostic) | Official evaluation split | 20 | 150 |
| **Total** | | | | | | **140 runs** | **~950 items** |

---

## 4. End-to-End System Architecture & Directory Layout

The project structure will be organized cleanly in the workspace:

```
jev-sintam-review/
├── configs/
│   ├── experiments.yaml             # Global experiment flags, sampling sizes, thresholds, model: 'jev-latest'
│   └── prompts.yaml                 # Frozen prompt definitions (English, Sinhala, Bilingual)
├── data/
│   ├── raw/                         # Ingested immutable source files
│   │   ├── dataset_a_sentiment/
│   │   ├── dataset_b_sold/
│   │   ├── dataset_c_nsina/
│   │   ├── dataset_d_sinhalammlu/
│   │   ├── dataset_e_salangabhava/
│   │   └── dataset_f_cmcs/
│   └── processed/
│       ├── manifest.json            # SHA-256 hashes of all frozen sample files
│       ├── phase0_smoke/            # 20-example subsets per task
│       └── samples/                 # Frozen Phase 1 production samples per dataset
├── src/
│   ├── __init__.py
│   ├── config.py                    # Environment & configuration parser (model: 'jev-latest')
│   ├── client.py                    # TypeSafe SDK wrapper with retry, latency, and telemetry
│   ├── loaders/                     # Modular dataset loaders
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── loader_a_sentiment.py
│   │   ├── loader_b_sold.py
│   │   ├── loader_c_nsina.py
│   │   ├── loader_d_sinhalammlu.py
│   │   ├── loader_e_salangabhava.py
│   │   └── loader_f_cmcs.py
│   ├── sampling/
│   │   ├── __init__.py
│   │   └── sampler.py               # Deterministic sampler with SHA-256 ID freezing (Phase 0 & 1)
│   ├── anchors/                     # Lightweight Hugging Face Sinhala-NLP reference anchor logging
│   │   ├── __init__.py
│   │   └── reference_anchors.py     # Pulls/records published metrics for SinBERT / XLM-R baselines
│   ├── runners/                     # Execution engines
│   │   ├── __init__.py
│   │   ├── base_runner.py
│   │   ├── runner_smoke_test.py     # Phase 0: 20 examples per task end-to-end smoke test
│   │   ├── runner_sentiment_lab.py  # Phase 1: Sentiment equivalence lab
│   │   ├── runner_core_tasks.py     # Phase 1: Core tasks runner (SOLD, NSINA, MMLU, SalAngaBhava)
│   │   ├── runner_prompts.py        # Phase 1: English vs Sinhala prompt sensitivity
│   │   ├── runner_diagnostics.py    # Phase 1: Option order, repeatability, confidence sweeps
│   │   └── runner_cmcs_stress.py    # Phase 1: Code-mixed multi-task stress track
│   ├── metrics/                     # Mathematical evaluation suite
│   │   ├── __init__.py
│   │   ├── accuracy.py              # Macro-F1, Accuracy, Top-1, MAE, RMSE
│   │   ├── calibration.py           # ECE, Brier score, Multiclass NLL, Reliability curves
│   │   ├── consistency.py           # Cross-primitive agreement & contradiction rates
│   │   └── latency.py               # p50, p95 latency profiler
│   └── visualization/
│       ├── __init__.py
│       └── plot_generator.py        # Publication figures (Matplotlib / Seaborn)
├── results/
│   ├── smoke_test/                  # Phase 0 validation outputs & gate verification log
│   ├── raw_logs/                    # Immutable JSONL prediction logs (one per query)
│   ├── main.csv                     # Core metrics table with HF Sinhala-NLP reference anchors
│   ├── primitive_consistency.csv    # RQ2 cross-primitive agreement
│   ├── script_analysis.csv          # Pure vs Romanized vs Code-mixed
│   ├── failure_taxonomy.csv         # 25 high-conf errors + 25 low-conf correct
│   └── figures/                     # 7 generated PNG charts
├── tests/
│   ├── test_loaders.py              # Unit tests for data loading & label mappings
│   ├── test_sdk_connection.py       # Live probe to TypeSafe API ('jev-latest')
│   └── test_metrics.py              # Verification of ECE, Brier, and agreement logic
├── pyproject.toml
└── README.md
```

---

## 5. Frozen Experiment Record Schema

Every single API interaction and decision is logged into an immutable raw JSONL stream conforming to this strict schema:

```json
{
  "record_id": "rec_sent_00142_choice_en",
  "experiment_id": "exp_probe_v1_core",
  "phase": "phase_1_full",
  "timestamp": "2026-10-02T04:15:22.184Z",
  "dataset": "dataset_a_sentiment",
  "example_id": "tallip_test_0481",
  "language": "si",
  "script_type": "pure_sinhala",
  "primitive": "choice",
  "prompt_variant": "english_instruction",
  "state_text": "මෙම තීරණය ඉතාමත් අගය කළ යුතු එකක් බව පැවසිය යුතුය.",
  "instructions": "Which label best describes the overall sentiment of this text?",
  "criteria_definitions": {
    "POSITIVE": null,
    "NEGATIVE": null,
    "NEUTRAL": null,
    "CONFLICT": null
  },
  "gold_label": "POSITIVE",
  "prediction": "POSITIVE",
  "is_correct": true,
  "confidence": 0.942,
  "probabilities": {
    "POSITIVE": 0.942,
    "NEUTRAL": 0.038,
    "NEGATIVE": 0.012,
    "CONFLICT": 0.008
  },
  "score_details": null,
  "noul_details": null,
  "latency_ms": 114,
  "usage": {
    "input_tokens": 84,
    "output_tokens": 1
  },
  "model": "jev-latest",
  "raw_response_status": 200,
  "error": null
}
```

---

## 6. Staged Implementation & Phased Execution

```
Stage 0: Tooling & SDK Verification (Model: 'jev-latest')
   │
Stage 1: Ingestion, Splits & Stratified Sampling Engine (Phase 0 & 1 with SHA-256)
   │
Stage 2: Phase 0 Smoke Test (20 items/task end-to-end) ──► [VERIFICATION GATE]
   │
Stage 3: Sentiment Primitive Equivalence Lab (Phase 1 Full Probe, N=150)
   │
Stage 4: Core Pure-Sinhala Tasks (SOLD, NSINA, MMLU, SalAngaBhava)
   │
Stage 5: Language & Prompt Sensitivity Experiment (English vs Sinhala, N=50)
   │
Stage 6: Robustness Diagnostics (Option Order, Repeatability, Calibration Sweeps)
   │
Stage 7: Code-Mixed Stress Track (CMCS, N=150)
   │
Stage 8: Quantitative Synthesis, Lightweight Reference Anchor Logging & Deliverables
```

---

### Stage 0: Environment Setup, Tooling & SDK Verification

* **Objective:** Ensure clean execution environment, install required scientific libraries, and verify TypeSafe API authentication.
* **Dependencies:**
  * `pandas`, `numpy`, `scipy`, `scikit-learn`
  * `datasets` (Hugging Face)
  * `matplotlib`, `seaborn`
  * `pyyaml`, `tqdm`, `pytest`
* **Health Check Execution:**
  * Query `TypeSafeClient.models.list()` to confirm access to Jev models, specifically validating `'jev-latest'`.
  * Run a test ping with minimal Noul and Choice questions to confirm credentials and latency tracking.

---

### Stage 1: Data Acquisition, Split Isolation & Sampling Engine

* **Objective:** Ingest all six datasets, isolate official evaluation splits, and generate deterministic, stratified subsets for both Phase 0 and Phase 1.
* **Dataset Ingestion Detail:**
  1. **Dataset A (Sinhala Sentiment):** Load `tallip` repository test split (15,059 items, labels: `POSITIVE`, `NEGATIVE`, `NEUTRAL`, `CONFLICT`).
  2. **Dataset B (SOLD):** Load Hugging Face `sinhala-nlp/SOLD` test split (labels: `OFF`, `NOT`). Sentence-level text only.
  3. **Dataset C (NSINA Categories & Media):** Load `sinhala-nlp/NSINA-Categories` test split (10 classes) and `Sinhala-News-Media-Identification` test split (10 sources).
  4. **Dataset D (SinhalaMMLU):** Load `naist-nlp/SinhalaMMLU` test split across 6 broad domains (Social Sciences, Humanities, STEM, etc.) and 30 subjects.
  5. **Dataset E (SalAngaBhava):** Load `lakshani005/SalAngaBhava`. Filter exclusively for `review_type == 'Pure_Sinhala'` for the core experiment. Extract numeric rating $1 \to 5$.
  6. **Dataset F (Sinhala-English CMCS):** Load `NLPC-UOM/Sinhala-English-Code-Mixed-Code-Switched-Dataset`. Filter evaluation set for the 5 target tasks.
* **Two-Tier Sampling Engine:**
  * **Phase 0 Smoke Subsets:** 20 stratified examples per task written to `data/processed/phase0_smoke/`.
  * **Phase 1 Full Probe Subsets:** Sentiment ($N=150$), SOLD ($N=150$), NSINA Cat ($N=100$), NSINA Media ($N=100$), SinhalaMMLU ($N=150$), SalAngaBhava ($N=150$), CMCS ($N=150$) written to `data/processed/samples/`.
  * Fixed seed: `42`.
  * Cryptographic hashing: Compute SHA-256 checksums for every sample file and commit to `data/processed/manifest.json`.

---

### Stage 2: Phase 0 Smoke Test (20 Examples per Task) & Verification Gate

* **Objective:** Run an end-to-end execution of all tasks on 20 examples each before launching the full probe.
* **Verification Checks:**
  1. Catch prompt syntax or wording mistakes in English and Sinhala templates.
  2. Verify label mapping bidirectionally (dataset gold label $\leftrightarrow$ SDK criteria key).
  3. Confirm raw response deserialization from `typesafe_sdk` (extracting `.choices[...].probabilities`, `.nouls[...].noul`, `.scores[...].score`, `.scores[...].confidence`).
  4. Validate Score behavior on 1–5 rubrics and 3-class ordered sentiment.
  5. Manually inspect 5 random Sinhala text inputs to ensure zero Unicode/ZWJ corruption occurred during ingestion.
* **Gate Decision:** Only when all 20-sample pipelines pass with zero runtime errors and valid probability distributions do we advance to the full Phase 1 probe.

---

### Stage 3: Sentiment Primitive Equivalence Lab (Phase 1 Full Probe, $N=150$)

* **Objective:** Test RQ2 (Primitive Consistency) on Dataset A with $N=150$ examples.
* **Design Matrix for the Exact Same State:**

```
                                  INPUT TEXT
             "මෙම ක්‍රියාව ඉතා පහත් එකක් බව සියලු දෙනාම දනිති."
                                       │
        ┌──────────────────────────────┼──────────────────────────────┐
        │                              │                              │
     Choice (1 Q)                 Noul (4 Qs)                    Score (1 Q)
Which sentiment?             Q1: Is it POSITIVE?              Negative -> Neutral
- POSITIVE                   Q2: Is it NEGATIVE?                 -> Positive
- NEGATIVE                   Q3: Is it NEUTRAL?                 (Subset without
- NEUTRAL                    Q4: Is it CONFLICT?                   CONFLICT)
- CONFLICT
```

* **SDK Payload Implementation (`model="jev-latest"`):**
  ```python
  questions = {
      "choice_sentiment": Choice(
          instructions="Which label best describes the overall sentiment of this text?",
          criteria={"POSITIVE": None, "NEGATIVE": None, "NEUTRAL": None, "CONFLICT": None},
      ),
      "noul_pos": Noul(instructions="Is the sentiment of this text positive?"),
      "noul_neg": Noul(instructions="Is the sentiment of this text negative?"),
      "noul_neu": Noul(instructions="Is the sentiment of this text neutral?"),
      "noul_cnf": Noul(instructions="Is the sentiment of this text conflict or mixed?"),
  }
  if gold_label in {"POSITIVE", "NEUTRAL", "NEGATIVE"}:
      questions["score_sentiment"] = Score(
          instructions="Rate the sentiment of this text on an ordered scale.",
          criteria=["Negative sentiment", "Neutral sentiment", "Positive sentiment"]
      )
  ```
* **Consistency Metrics:**
  * Argmax(Noul) vs Choice Winner agreement.
  * Contradiction rate: instances where 0 or $\ge 2$ Nouls return $\mathbb{P} \ge 0.50$.
  * Probability ordering check across classes.

---

### Stage 4: Core Pure-Sinhala Tasks Execution ($N \approx 650$)

* **Objective:** Scale evaluation to all core Sinhala tasks (Datasets B, C, D, E) using `model="jev-latest"`.
* **Task Formulations:**
  * **SOLD (Binary Offensive, $N=150$):**
    * *Noul:* `instructions="Is this post offensive?"` $\to \mathbb{P}(\text{True})$.
    * *Choice:* `criteria={"OFF": "Offensive", "NOT": "Not offensive"}`.
    * *Secondary Score:* `criteria=["Not offensive", "Offensive"]` (Binary severity scale).
  * **NSINA Categories (10-way Multiclass, $N=100$):**
    * *Choice:* 10 exact news category labels.
    * *Noul:* One-vs-rest on top 3 categories (`Politics`, `Sports`, `Business`).
    * *Rule:* **No Score.**
  * **NSINA Media (10-way Source Identification, $N=100$):**
    * *Choice:* 10 media house labels.
    * *Noul:* Diagnostic one-vs-rest.
    * *Rule:* **No Score.**
  * **SinhalaMMLU (Knowledge QA, $N=150$):**
    * *Choice:* `criteria={"A": None, "B": None, "C": None, "D": None}`. Preserve exact Sinhala options without modification.
    * *Noul Diagnostic (30 examples):* 4 independent questions per option (`Is option A the correct answer?`, etc.).
  * **SalAngaBhava (Ordinal Rating Lab, $N=150$):**
    * *Condition:* `review_type == Pure_Sinhala`.
    * *Score (Primary):*
      ```python
      criteria=["Rating level 1", "Rating level 2", "Rating level 3", "Rating level 4", "Rating level 5"]
      ```
    * *Choice:* `criteria={"1": None, "2": None, "3": None, "4": None, "5": None}`.
    * *Noul Thresholds:*
      * `noul_high`: "Is the rating of this review 4 or 5 (satisfied)?"
      * `noul_low`: "Is the rating of this review 1 or 2 (dissatisfied)?"
    * *Metrics:* Mean Absolute Error (MAE), RMSE, Exact Accuracy, Within-1 Accuracy, Choice-vs-Score agreement.

---

### Stage 5: Language & Prompt Sensitivity Experiment

* **Objective:** Address RQ4 by isolating whether Jev's performance changes when the task instruction is given in English vs Sinhala.
* **Sample:** 50 frozen Sinhala examples from Dataset A (Sentiment) and 50 from Dataset B (SOLD).
* **Controlled Conditions:**
  * **Condition A (English Instruction):** Task instruction in English; input text in original Sinhala.
  * **Condition B (Sinhala Instruction):** Task instruction in Sinhala; input text in original Sinhala.
    * Prompt: `"මෙම පාඨයේ සමස්ත හැඟීම වඩාත් හොඳින් විස්තර කරන්නේ කුමන ලේබලයද?"`
  * **Optional Condition C (Bilingual Scaffolding):** English framing accompanied by Sinhala translation.
* **Evaluation:** Accuracy delta ($\text{Acc}_{\text{si}} - \text{Acc}_{\text{en}}$), confidence drift, and choice flip rate.

---

### Stage 6: Primitive Diagnostics & Robustness

* **Objective:** Stress-test decision stability, option-order bias, and calibration boundaries.
* **1. Option-Order Permutation Test (multiclass Choice):**
  * $N = 40$ examples from SinhalaMMLU and NSINA Categories evaluated across 3 permutations: Original $[A, B, C, D]$, Shifted $[B, C, D, A]$, Inverted $[D, C, B, A]$.
  * Measure choice stability rate, probability entropy change, and position bias.
* **2. Repeatability / Stochasticity Test:**
  * $N = 100$ examples evaluated 3 independent times with identical parameters.
  * Calculate Standard Deviation of probabilities: $\sigma(p_{\text{Noul}})$, $\sigma(p_{\text{Choice}})$, $\sigma(\text{Score})$.
* **3. Confidence Threshold / Risk-Coverage Analysis:**
  * Sweep confidence thresholds $\tau \in [0.50, 0.70, 0.80, 0.90, 0.95]$.
  * Plot Coverage vs Accuracy / Error Rate curves.

---

### Stage 7: Code-Mixed Stress Track (Dataset F — CMCS, $N=150$)

* **Objective:** Evaluate Jev on non-standard, Romanized, and code-mixed Sinhala without contaminating the pure-Sinhala benchmark.
* **5 Target Sub-Tasks:**
  1. **Sentiment:** Choice (4 classes), Noul ($\times 4$), Score (3-class subset).
  2. **Humour:** Choice (`Humorous`, `Non-humorous`), Noul, optional binary Score.
  3. **Hate Speech:** Choice (`Hate-Inducing`, `Abusive`, `Not offensive`), Noul ($\times 3$).
  4. **Aspect Extraction (Multi-Label):**
     * Direct mapping via parallel **Noul** primitives (`noul_network`, `noul_billing`, `noul_package`, `noul_service`, `noul_data`).
     * Choice evaluated *only* on the subset of reviews annotated with exactly one positive aspect.
  5. **Language Identification:** Choice (Sinhala, English, Sin-Eng, Eng-Sin, Mixed, Symbol).

---

### Stage 8: Quantitative Synthesis, Lightweight Reference Anchor Logging & Deliverables

* **Lightweight Reference Anchor Logging:**
  * To provide external context without overcomplicating with local heavy model training, we extract and record the published benchmark test scores for existing Hugging Face `sinhala-nlp` checkpoints:
    * Sinhala Sentiment baseline (e.g., SinBERT / XLM-R test Macro-F1).
    * SOLD offensive baseline (e.g., SinBERT / RoBERTa test Macro-F1).
    * NSINA Category baseline (e.g., NSINA classifier benchmark Macro-F1).
  * These reference numbers are logged directly in `results/main.csv` under `hf_sinhala_nlp_anchor_f1` and `hf_sinhala_nlp_anchor_acc` for clear, side-by-side interpretation.
* **Mathematical Formulations:**
  * **Brier Score (Binary Noul):** $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2$
  * **Multiclass Brier Score (Choice):** $\text{BS}_{\text{multi}} = \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{ik} - y_{ik})^2$
  * **Expected Calibration Error (ECE):** $\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$ ($M=10$ bins).
  * **Score Metric:** $\text{MAE} = \frac{1}{N}\sum |s_i - y_i|, \quad \text{RMSE} = \sqrt{\frac{1}{N}\sum (s_i - y_i)^2}$
* **Qualitative Failure Taxonomy Audit:**
  * Audit the **25 highest-confidence errors** and **25 lowest-confidence correct predictions** across 5 categories: Language, Semantic, Classification, Primitive, and Confidence.
* **Script Comparison:**
  * Segment SalAngaBhava and CMCS into `Pure_Sinhala`, `Sinhala_in_English`, and `Code_mixed`.

---

## 7. Deliverables & Output Specifications

All outputs will be generated into the `results/` directory with strictly specified schemas:

### 1. `results/main.csv`
Includes both Jev performance and lightweight Hugging Face Sinhala-NLP reference anchors:
```csv
dataset,task,primitive,n,accuracy,macro_f1,nll,brier,ece,mean_confidence,p50_latency_ms,p95_latency_ms,hf_reference_model,hf_reference_acc,hf_reference_f1
sinhala_sentiment,sentiment,choice,150,0.760,0.732,0.612,0.314,0.068,0.892,112,240,sinhala-nlp/sinbert-sentiment,0.785,0.750
sinhala_sentiment,sentiment,noul,150,0.740,0.715,0.640,0.330,0.075,0.880,140,290,sinhala-nlp/sinbert-sentiment,0.785,0.750
sold,offensive,noul,150,0.820,0.818,0.480,0.240,0.052,0.910,105,210,sinhala-nlp/sold-sinbert,0.835,0.828
...
```

### 2. `results/primitive_consistency.csv`
```csv
task,n,choice_noul_agreement,noul_contradiction_rate,choice_score_agreement,prob_order_consistency
sentiment_news,150,0.842,0.080,0.891,0.912
salangabhava_rating,150,0.825,0.045,0.880,0.934
...
```

### 3. `results/script_analysis.csv`
```csv
dataset,script_type,n,accuracy,macro_f1,ece,mean_confidence,p50_latency_ms
salangabhava,Pure_Sinhala,150,0.813,0.790,0.054,0.884,115
salangabhava,Sinhala_in_English,150,0.680,0.645,0.112,0.810,122
salangabhava,Code_Mixed,150,0.710,0.680,0.095,0.840,118
...
```

### 4. 7 Publication-Ready Figures (`results/figures/`)
1. `fig1_accuracy_by_task.png`: Macro-F1 and Top-1 Accuracy across all 6 datasets (with HF reference anchors).
2. `fig2_calibration_by_task.png`: Reliability diagrams and ECE bars for Noul and Choice.
3. `fig3_confidence_vs_correctness.png`: Box/violin plots comparing confidence distributions for correct vs incorrect decisions.
4. `fig4_primitive_agreement.png`: Disagreement confusion matrix heatmap (Choice Winner vs Argmax Noul).
5. `fig5_risk_coverage.png`: Risk-coverage trade-off curves across confidence thresholds.
6. `fig6_script_type_comparison.png`: Performance and calibration degradation from Pure Sinhala to Romanized to Code-mixed.
7. `fig7_latency_by_primitive.png`: Latency distributions (p50, p95) across Noul, Choice, and Score.

---

## 8. Potential Pitfalls & Technical Mitigations

| Risk / Failure Point | Probability | Impact | Engineering Mitigation |
| :--- | :---: | :---: | :--- |
| **Sinhala Unicode Corruption** | High | Critical | Preserve raw UTF-8 encodings. Explicitly prevent string normalization that strips Sinhala Zero-Width Joiners (`\u200D` for bandi akuru/rakaransaya) or ZWNJ (`\u200C`). |
| **API Rate Limits / Timeouts** | Medium | Medium | Wrap `typesafe_sdk.TypeSafeClient` with tenacity retry policy (exponential backoff with jitter: 2s, 4s, 8s, max 5 attempts). Log every retry attempt. |
| **Data Leakage into Prompts** | Low | Critical | Automated check ensuring zero test set examples are embedded in prompt templates or few-shot exemplars (zero-shot only). |
| **Label Mapping Mismatches** | Medium | High | Unit tests verifying bidirectional mapping between original dataset labels and SDK criteria keys before initiating full runs. |
| **SDK Score Key Interpretation** | Low | High | Ensure Score answers properly access `.probabilities` (integer keys `0, 1, 2...`) and calculate continuous expectation vs discrete argmax accurately. |

---

## 9. Final Report Structure (Deliverable Report)

1. **Executive Summary & Motivation:** Jev (`jev-latest`) decision-native interface evaluated against low-resource Sinhala.
2. **Dataset Specifications:** Complete breakdown of the 6 authentic Sinhala benchmarks.
3. **Experimental Methodology:** Zero-shot setup, Phase 0 smoke test gate, Phase 1 full probe, prompt freezing, and metric definitions.
4. **Empirical Results (RQ1–RQ4):**
   * Macro performance across tasks compared with published Hugging Face Sinhala-NLP anchors.
   * Cross-primitive consistency matrices and contradiction rates.
   * Calibration curves, Brier scores, and ECE.
   * English vs Sinhala prompt sensitivity analysis.
5. **Qualitative Failure Mode Analysis (RQ5):** In-depth audit of 25 high-confidence errors and 25 low-confidence correct answers.
6. **Script & Orthography Findings:** Pure Sinhala vs Romanized vs Code-mixed performance drop.
7. **Systems & Latency Evaluation:** Practical throughput and response time across primitives.
8. **Threats to Validity & Limitations:** Sample size, lack of human re-annotation, domain specificity.
9. **Strategic Roadmap for Phase 2:** Decision gate recommendations on whether to proceed with deep multilingual pretraining, fine-tuning, or architectural interventions.
