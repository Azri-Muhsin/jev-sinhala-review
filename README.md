
# Jev-SI/TA: Quick and Dirty Sinhala & Tamil Capability Check

> **Purpose:** a fast, simple, zero-shot capability and limitation screening of TypeSafe for Jev on **existing Sinhala and Tamil datasets**.
>
> This is **not** a serious multilingual research benchmark. It is reconnaissance: determine whether Jev and "System-1 models" is worth deeper experimentation.

---

## 1. What this experiment is trying to answer

We are testing four practical questions:

1. **Can Jev understand Sinhala and Tamil well enough for real NLP classification tasks?**
2. **Are Jev's returned probabilities/confidence useful on these languages, or does it become over/under-confident?**
3. **Does the decision-native interface offer a practical benefit compared with a conventional autoregressive LLM and a multilingual encoder?**
4. **Where does Jev break: language, semantics, label space, calibration, or decision-interface effects?**

---

# 2. Scope

This first probe deliberately stays small.

### Languages

* Sinhala
* Tamil

### Tasks

#### Sinhala

* Sinhala news-comment sentiment
* Sinhala offensive-language detection (SOLD)
* Sinhala news-category classification (NSINA)

#### Tamil

* Tamil sentiment classification
* Tamil news classification
* Tamil abusive-comment classification

### Target sample


* **200 examples per dataset**
* **1,200 examples total**
* sample **stratified by gold label**

Sample size is ot exacty 200 if a class is too small or a dataset split makes that impractical.

---

# 3. Source datasets

The following sources are the basis for the probe.

## Sinhala

### 3.1 Sinhala Sentiment Dataset

A Sinhala news-comment sentiment dataset with four sentiment categories.

**Labels used in the source task should be preserved exactly.**

Source repository:

* https://github.com/LahiruSen/sinhala_sentiment_anlaysis_tallip

Research paper:

* https://arxiv.org/abs/2011.07280

---

### 3.2 SOLD — Sinhala Offensive Language Dataset

Sinhala Offensive Language Dataset (SOLD), containing manually annotated Sinhala social-media text for offensive-language detection.

For this probe, prefer the original manually annotated data rather than any later semi-supervised expansion.

Source repository:

* https://github.com/Sinhala-NLP/SOLD

Research paper:

* https://aclanthology.org/2022.dravidianlangtech-1.13/

---

### 3.3 NSINA / NSINA Categories

Sinhala news corpus and category classification resource.

Uses the released category task rather than constructing a new taxonomy.

Dataset:

* https://huggingface.co/datasets/sinhala-nlp/NSINA

Category dataset:

* https://huggingface.co/datasets/sinhala-nlp/NSINA-Categories

Research paper:

* https://aclanthology.org/2024.lrec-main.1076/

---

## Tamil

### 3.4 Tamil Sentiment — DravidianLangTech 2025

Tamil sentiment task data from the DravidianLangTech 2025 shared task.

Use a released labeled split suitable for local evaluation; do not rely on an undisclosed competition test set.

Research/task paper:

* https://aclanthology.org/2025.dravidianlangtech-1.124/

---

### 3.5 Tamil News Classification

Tamil news classification data available through MTEB and the related original resource.

MTEB dataset:

* https://huggingface.co/datasets/mteb/TamilNewsClassification

Original project:

* https://github.com/vanangamudi/tamil-news-classification

---

### 3.6 Tamil Abusive Comment Classification

Tamil abusive-comment classification from DravidianLangTech 2022.

Use the original released labels and split definitions.

Research/task paper:

* https://aclanthology.org/2022.dravidianlangtech-1.15/

DravidianLangTech 2022 collection:

* https://aclanthology.org/events/dravidianlangtech-2022/

---

# 4. Important methodological warning

This is **not** a controlled Sinhala-vs-Tamil benchmark.

The datasets cover different domains and label spaces:

| Dataset           | Language | Domain        | Decision space |
| ----------------- | -------- | ------------- | -------------- |
| Sinhala Sentiment | Sinhala  | News comments | 4-way          |
| SOLD              | Sinhala  | Social media  | Binary         |
| NSINA Categories  | Sinhala  | News          | Multi-class    |
| Tamil Sentiment   | Tamil    | Social media  | 4-way          |
| Tamil News        | Tamil    | News          | Multi-class    |
| Tamil Abuse       | Tamil    | Social media  | 8-way          |


---

# 5. Experiment matrix

The main comparison should stay small.

## Systems

### A. Jev

Native Jev decision interface:

* `Choice`
* `Noul`

Use `Score` only where the task genuinely supports an ordered/continuous interpretation.

### B. Autoregressive LLM baseline

Use one strong, available general LLM.

Recommended first baseline:

* Qwen family model available to you

Run:

1. concise classification prompt
2. constrained/structured output where practical

The baseline must use the same text and same gold labels.

### C. Multilingual encoder baseline

Use a released multilingual encoder/classifier checkpoint or an XLM-R-based classifier appropriate to the task.

The point is not to optimise the encoder. It is a **reference floor/anchor** showing what a model built for multilingual classification can do.

---

# 6. Main Jev evaluation

## 6.1 Choice tasks

For multi-class datasets, formulate:

```text
State:
<original dataset text>

Question:
Which label best describes this text?

Options:
<exact source labels>
```

exact label names from the dataset used.

---

## 6.2 Noul tasks

For binary tasks such as SOLD:

```text
State:
<original dataset text>

Question:
Does this text belong to the offensive-language class?

Noul:
yes / no
```

Map the returned decision back to the exact dataset label.

---

# 7. Prompt-language probe

The main evaluation should use **the same English task instruction across both languages**.

This isolates the language-understanding question.

Then run a small secondary probe:

* 50 Sinhala examples
* 50 Tamil examples

For each, compare:

### Run A

English instruction + Sinhala/Tamil input

### Run B

Native-language instruction + the same input

Do not change the underlying text.

### Question

Does performance or confidence change when the **instruction language** changes?

---

# 8. Small but important decision-interface tests

These are cheap and should be included because they test properties of the decision interface rather than just classification accuracy.

## 8.1 Choice-cardinality test

The datasets naturally give us different decision-space sizes:

* binary
* 4-way
* 6-way
* 8-way

Measure:

* accuracy
* ECE
* confidence
* latency

against number of choices.

Question:

> Does decision quality or confidence degrade as the number of choices increases?

---

## 8.2 Label-order sensitivity

Take a small 50-example subset from each multi-class task.

Run:

### Original order

```text
A
B
C
D
```

### Permuted order

```text
D
B
A
C
```

Keep everything else identical.

Measure:

* prediction agreement
* probability distribution change
* confidence change

This is a cheap probe for option-order sensitivity.

---

## 8.3 Repeatability / stochastic stability

Take 100 examples across the probe.

Run the same input/question/options three times.

Record:

* prediction agreement
* confidence mean
* confidence standard deviation
* probability distribution variation

The objective is not to assume deterministic behaviour; it is to measure it.

---

# 9. Metrics

Do **not** reduce the experiment to accuracy.

## Capability

* Accuracy
* Macro-F1
* Per-class precision/recall/F1
* Confusion matrix

## Probability quality

* NLL
* Brier score
* ECE
* Reliability diagram

## Automation-oriented view

Measure:

* accuracy at confidence >= 0.50
* accuracy at confidence >= 0.70
* accuracy at confidence >= 0.80
* accuracy at confidence >= 0.90
* coverage at each threshold

The main question is:

> If we only trust high-confidence Jev decisions, how often are they actually correct?

## Systems

* p50 latency
* p95 latency
* input tokens
* output tokens where applicable
* request cost where applicable

---

# 10. Raw prediction schema

Every model runner should save a row like:

```json
{
  "example_id": "sinhala_sentiment_00123",
  "dataset": "sinhala_sentiment",
  "language": "si",
  "text": "...",
  "gold_label": "negative",
  "predicted_label": "negative",
  "probabilities": {
    "positive": 0.03,
    "negative": 0.89,
    "neutral": 0.06,
    "conflict": 0.02
  },
  "confidence": 0.89,
  "latency_ms": 81,
  "input_tokens": 57,
  "output_tokens": null,
  "model": "jev",
  "prompt_variant": "english_instruction",
  "run_id": "2026-09-XX"
}
```

Store raw predictions even when the model fails.

Failures are data.

---

# 11. Repository structure

```text
jev-si-ta-probe/
│
├── README.md
│
├── data/
│   ├── README.md
│   ├── manifests/
│   │   ├── dataset_sources.yaml
│   │   └── sample_manifest.csv
│   │
│   └── frozen/
│       ├── sinhala_sentiment.jsonl
│       ├── sold.jsonl
│       ├── nsina_categories.jsonl
│       ├── tamil_sentiment.jsonl
│       ├── tamil_news.jsonl
│       └── tamil_abuse.jsonl
│
├── prompts/
│   ├── jev_choice.txt
│   ├── jev_noul.txt
│   ├── english_instruction.txt
│   └── native_instruction_probe.txt
│
├── runners/
│   ├── run_jev.py
│   ├── run_qwen.py
│   └── run_xlmr.py
│
├── evaluation/
│   ├── metrics.py
│   ├── calibration.py
│   ├── stability.py
│   ├── cardinality.py
│   └── order_sensitivity.py
│
├── results/
│   ├── raw/
│   ├── tables/
│   └── figures/
│
├── reports/
│   └── probe_report.md
│
└── pyproject.toml
```

---

# 12. TODO — execution checklist

## Phase 0 — Setup

* [ ] Create repository
* [ ] Create Python environment
* [ ] Record Python / package versions
* [ ] Choose Jev API access and authentication method
* [ ] Choose AR baseline
* [ ] Choose multilingual encoder baseline
* [ ] Record exact model versions/checkpoints
* [ ] Create experiment run-ID convention
* [ ] Create raw-results directory
* [ ] Create `dataset_sources.yaml`

### Done when

* [ ] One command can initialise the project
* [ ] Every model/version is documented
* [ ] No credentials are committed to Git

---

# Phase 1 — Acquire and inspect datasets

## Sinhala

* [ ] Download Sinhala Sentiment source data
* [ ] Verify source labels
* [ ] Download SOLD
* [ ] Verify source labels
* [ ] Download NSINA category data
* [ ] Verify source labels/splits

## Tamil

* [ ] Download Tamil Sentiment task data
* [ ] Verify source labels/splits
* [ ] Download Tamil News Classification
* [ ] Verify source labels
* [ ] Download Tamil abusive-comment data
* [ ] Verify source labels/splits

## Data audit

* [ ] Check encoding is valid UTF-8
* [ ] Check for missing text
* [ ] Check for missing labels
* [ ] Check duplicate IDs/text where detectable
* [ ] Check class distributions
* [ ] Check script/language assumptions
* [ ] Record original dataset split
* [ ] Record dataset license/usage notes
* [ ] Record source URL and citation metadata

### Done when

* [ ] Every dataset can be traced back to a public source
* [ ] Every row in the probe has a source dataset and source split
* [ ] No text has been rewritten

---

# Phase 2 — Build the frozen quick-test sample

For each dataset:

* [ ] Decide target sample size
* [ ] Create reproducible random seed
* [ ] Stratify by gold label
* [ ] Sample probe subset
* [ ] Save sample manifest
* [ ] Save frozen JSONL
* [ ] Compute class distribution
* [ ] Verify no accidental overlap between selected examples if relevant
* [ ] Record the exact sampling script and seed

### Done when

* [ ] The evaluation corpus is frozen
* [ ] Re-running the sampling script reproduces the same IDs
* [ ] No test examples are generated or altered

---

# Phase 3 — Implement Jev runner

* [ ] Implement Jev `Choice`
* [ ] Implement Jev `Noul`
* [ ] Map source labels to Jev options
* [ ] Preserve original gold labels
* [ ] Capture returned probabilities
* [ ] Capture Jev confidence
* [ ] Capture latency
* [ ] Capture API errors
* [ ] Capture raw request/response metadata needed for reproducibility
* [ ] Implement retries without silently duplicating result rows
* [ ] Add run IDs

### Done when

* [ ] One dataset can be evaluated end-to-end
* [ ] Raw prediction JSONL is produced
* [ ] Failed requests are logged separately

---

# Phase 4 — Run the first Jev smoke test

Start small.

* [ ] Run 10 Sinhala sentiment examples
* [ ] Run 10 Tamil sentiment examples
* [ ] Run 10 SOLD examples
* [ ] Run 10 Tamil abuse examples
* [ ] Inspect outputs manually
* [ ] Confirm label mappings
* [ ] Confirm probability format
* [ ] Confirm confidence extraction
* [ ] Confirm latency logging
* [ ] Look for obvious language misunderstanding

### Gate

* [ ] No schema/parsing problem in our runner
* [ ] No accidental label inversion
* [ ] No accidental English translation of inputs
* [ ] No prompt leakage from gold labels

If this gate passes:

* [ ] Run the full 1,200-example probe

---

# Phase 5 — Run baseline models

## AR baseline

* [ ] Build identical data loader
* [ ] Use same frozen examples
* [ ] Use same source labels
* [ ] Run concise classification prompt
* [ ] Run structured-output version if practical
* [ ] Save raw predictions
* [ ] Save latency
* [ ] Save token counts where available

## Multilingual encoder

* [ ] Select checkpoint
* [ ] Confirm task compatibility
* [ ] Run inference on same frozen examples
* [ ] Save predictions
* [ ] Record checkpoint/source

### Done when

* [ ] Jev, AR baseline, and encoder have predictions for the same examples
* [ ] Missing/failure cases are explicitly recorded

---

# Phase 6 — Compute metrics

* [ ] Accuracy
* [ ] Macro-F1
* [ ] Per-class metrics
* [ ] Confusion matrices
* [ ] NLL
* [ ] Brier
* [ ] ECE
* [ ] Reliability plots
* [ ] Confidence-threshold accuracy
* [ ] Coverage vs risk
* [ ] p50 latency
* [ ] p95 latency
* [ ] token counts
* [ ] cost estimate if available

### Produce

* [ ] `results/tables/main_results.csv`
* [ ] `results/tables/calibration.csv`
* [ ] `results/figures/accuracy_by_language.png`
* [ ] `results/figures/calibration_by_language.png`
* [ ] `results/figures/risk_coverage.png`
* [ ] `results/figures/latency.png`

---

# Phase 7 — Small diagnostic experiments

## Prompt-language probe

* [ ] Select 50 Sinhala examples
* [ ] Select 50 Tamil examples
* [ ] Run English instruction
* [ ] Run native-language instruction
* [ ] Compare prediction agreement
* [ ] Compare confidence
* [ ] Compare latency

## Choice-order probe

* [ ] Select 50 examples per multi-class task
* [ ] Run original option order
* [ ] Run shuffled option order
* [ ] Compare predictions
* [ ] Compare probability distributions
* [ ] Compare confidence

## Repeatability probe

* [ ] Select 100 examples
* [ ] Run each three times
* [ ] Calculate prediction agreement
* [ ] Calculate confidence variance
* [ ] Calculate probability variance

## Cardinality analysis

* [ ] Group tasks by number of options
* [ ] Compare accuracy
* [ ] Compare calibration
* [ ] Compare confidence
* [ ] Compare latency

---

# Phase 8 — Manual error inspection

Do **not** manually inspect all 1,200 examples.

Select approximately:

* [ ] 25 highest-confidence Jev errors
* [ ] 25 lowest-confidence correct predictions
* [ ] 10 largest Sinhala/Tamil confidence gaps
* [ ] 10 largest Jev-vs-baseline disagreements

For each selected example, classify the failure:

* [ ] Language comprehension
* [ ] Semantic ambiguity
* [ ] Class-boundary confusion
* [ ] Rare vocabulary
* [ ] Domain-specific wording
* [ ] Long/complex input
* [ ] Label interpretation
* [ ] Overconfidence
* [ ] Underconfidence
* [ ] Other

Keep the original text and gold label visible in the analysis record.

---

# Phase 9 — Build the quick report

Create `reports/probe_report.md`.

Include:

## 1. Objective

* [ ] What was tested
* [ ] What was explicitly not tested

## 2. Data

* [ ] Dataset sources
* [ ] Sample sizes
* [ ] Source splits
* [ ] Sampling procedure
* [ ] No-synthetic-data statement

## 3. Methods

* [ ] Jev task formulation
* [ ] AR baseline
* [ ] multilingual encoder
* [ ] prompt variants
* [ ] hardware/API context

## 4. Results

* [ ] Accuracy table
* [ ] Macro-F1 table
* [ ] Calibration table
* [ ] Latency table
* [ ] Confidence-threshold analysis

## 5. Diagnostics

* [ ] Prompt-language result
* [ ] Choice-order result
* [ ] Stability result
* [ ] Cardinality result
* [ ] Error taxonomy

## 6. Limitations

* [ ] Heterogeneous tasks
* [ ] Small probe sample
* [ ] Dataset-specific domain effects
* [ ] Any unavailable Jev features
* [ ] API/model version limitations

## 7. Conclusion

Answer only:

* [ ] What Jev can clearly do on Sinhala
* [ ] What Jev can clearly do on Tamil
* [ ] Where confidence is useful
* [ ] Where confidence appears unreliable
* [ ] Where Jev differs from the baselines
* [ ] Whether the evidence justifies the next-stage research

---

# 10. Success criteria for the quick probe

This experiment is successful even if Jev performs poorly.

The minimum success condition is:

### Data

* [ ] 6 real public datasets
* [ ] ~1,200 frozen examples
* [ ] zero synthetic examples

### Systems

* [ ] Jev
* [ ] one AR model
* [ ] one multilingual encoder

### Evidence

* [ ] accuracy/F1
* [ ] calibration
* [ ] confidence-threshold behaviour
* [ ] latency
* [ ] small diagnostic experiments
* [ ] manually inspected failure cases

### Reproducibility

* [ ] source dataset references
* [ ] frozen sampling manifest
* [ ] prompts
* [ ] model/version metadata
* [ ] raw predictions
* [ ] metric scripts

---

# 11. What we are deliberately NOT claiming

This quick probe cannot establish:

* [ ] that Jev is a better architecture than AR LLMs
* [ ] that Jev is better than multilingual encoders
* [ ] that Jev is calibrated in general
* [ ] that Jev is strong for all low-resource languages
* [ ] that Jev's internal architecture is known
* [ ] that Jev's RLCD implementation is reproduced
* [ ] that performance on these datasets predicts enterprise performance

The result is a **capability reconnaissance experiment**, not a final scientific benchmark.

---

# 12. Decision gate after the experiment

At the end, classify the findings into one of these research directions.

## Direction A — Capability is strong, calibration is interesting

Continue to:

> **Decision-native models for low-resource multilingual automation**

Focus on confidence, selective prediction, and enterprise workflow decisions.

## Direction B — Capability is strong, but confidence is weak

Continue to:

> **Are decision-native probabilities actually reliable across languages?**

This makes calibration the central research question.

## Direction C — Sinhala/Tamil capability drops substantially

Continue to:

> **How multilingual are decision-native models?**

Compare against mBERT/XLM-R/modern multilingual LLMs and investigate whether the interface or underlying model capability is responsible.

## Direction D — Jev is not clearly differentiated

Do not force the thesis.

Move to:

> **Comparative study of encoder, autoregressive, diffusion, and decision-native architectures for low-resource multilingual decisions.**

That is still a worthwhile research direction.

---

# 13. Later-stage research is explicitly out of scope

Only after the quick probe should we consider:

* controlled multilingual dataset construction
* independently authored human challenge sets
* larger Sinhala/Tamil experiments
* code-switched/transliterated evaluation
* DiffusionGemma/dLLM comparison
* decision-native model training
* calibration-aware fine-tuning
* policy DAGs
* structured-output systems benchmarking
* production latency experiments
* enterprise workflow simulation
* publication-grade statistical analysis

**Do not build the serious system before this probe tells us what is worth investigating.**

---

# 14. Suggested first command sequence

Once the repo is created:

```bash
# 1. Install
pip install -e .

# 2. Verify dataset sources
python scripts/verify_sources.py

# 3. Build the frozen sample
python scripts/build_probe_sample.py \
  --seed 20260923 \
  --per-dataset 200

# 4. Run a small smoke test
python runners/run_jev.py \
  --smoke-test

# 5. Inspect
python scripts/inspect_predictions.py \
  results/raw/jev_smoke.jsonl

# 6. Run full Jev probe
python runners/run_jev.py \
  --all

# 7. Run baselines
python runners/run_qwen.py --all
python runners/run_xlmr.py --all

# 8. Evaluate
python evaluation/run_all.py

# 9. Generate plots/tables
python evaluation/make_report.py
```

Adapt command names to the actual implementation; the important requirement is that every stage is scriptable and reproducible.

---

# 15. Final checklist

## Before running

* [ ] Sources verified
* [ ] Data licenses/usage checked
* [ ] Frozen sample created
* [ ] Prompts frozen
* [ ] Label mappings frozen
* [ ] Model versions recorded
* [ ] API credentials tested
* [ ] Raw logging enabled

## Before trusting results

* [ ] Gold labels never sent in prompt
* [ ] No synthetic/translated text introduced
* [ ] Same examples used across systems
* [ ] Failed calls separated from valid predictions
* [ ] No accidental duplicate runs counted as independent examples
* [ ] Calibration calculated from the correct probability outputs
* [ ] Latency measured consistently

## Before writing conclusions

* [ ] Check raw predictions
* [ ] Inspect high-confidence errors
* [ ] Check class imbalance
* [ ] Check dataset-specific effects
* [ ] Separate capability from calibration
* [ ] Separate model capability from interface behaviour
* [ ] Avoid generalising from one dataset
* [ ] Document every limitation

---

# 16. The one-sentence goal

> **In a few hours of clean experimentation, determine whether Jev demonstrates useful Sinhala and Tamil decision-making capability, whether its confidence is meaningful in those languages, and whether its decision-native interface exposes limitations or advantages that justify the much larger research project.**
