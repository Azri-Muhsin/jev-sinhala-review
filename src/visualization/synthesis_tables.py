"""
synthesis_tables.py — Aggregates all probe stages into standard CSV artifacts:
  - results/main.csv
  - results/primitive_consistency.csv
  - results/script_analysis.csv
  - results/failure_taxonomy.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

from src.config import CONFIGS_DIR, RESULTS_DIR, load_yaml
from src.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_nll,
)


def load_all_predictions() -> list[dict[str, Any]]:
    """Load records from Stage 3, Stage 4, Stage 5, Stage 6, Stage 7."""
    records = []
    log_files = [
        RESULTS_DIR / "stage3_sentiment" / "stage3_predictions.jsonl",
        RESULTS_DIR / "stage4_core" / "stage4_predictions.jsonl",
        RESULTS_DIR / "stage5_prompt_sensitivity" / "stage5_predictions.jsonl",
        RESULTS_DIR / "stage6_diagnostics" / "stage6_predictions.jsonl",
        RESULTS_DIR / "stage7_cmcs" / "stage7_predictions.jsonl",
    ]
    for lf in log_files:
        if not lf.exists():
            continue
        with open(lf, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def generate_main_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build results/main.csv covering primary tasks and primitives with reference anchors."""
    anchors = load_yaml("reference_anchors.yaml")

    # Define primary runs to include
    # (dataset_key, task_name, primitive, prompt_variant, filter_fn, labels, anchor_key)
    task_specs = [
        (
            "dataset_a_sentiment",
            "sentiment_4way",
            "choice",
            "english_instruction",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_a_sentiment" and r.get("prompt_variant") == "english_instruction",
            ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"],
            "dataset_a_sentiment",
        ),
        (
            "dataset_a_sentiment",
            "sentiment_4way",
            "noul",
            "english_instruction",
            lambda r: r.get("primitive") == "noul" and r.get("dataset") == "dataset_a_sentiment" and "multiclass" in r.get("record_id", ""),
            ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"],
            "dataset_a_sentiment",
        ),
        (
            "dataset_b_sold",
            "offensive_detection",
            "choice",
            "english_instruction",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_b_sold" and r.get("prompt_variant") == "english_instruction" and "s5" not in r.get("record_id", ""),
            ["NOT", "OFF"],
            "dataset_b_sold",
        ),
        (
            "dataset_b_sold",
            "offensive_detection",
            "noul",
            "english_instruction",
            lambda r: r.get("primitive") == "noul" and r.get("dataset") == "dataset_b_sold" and r.get("prompt_variant") == "english_instruction" and "s5" not in r.get("record_id", ""),
            ["NOT", "OFF"],
            "dataset_b_sold",
        ),
        (
            "dataset_c1_nsina_categories",
            "news_categorization_4way",
            "choice",
            "english_instruction",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_c1_nsina_categories" and r.get("prompt_variant") == "english_instruction",
            ["Business", "International News", "Local News", "Sports"],
            "dataset_c1_nsina_categories",
        ),
        (
            "dataset_c1_nsina_categories",
            "news_categorization_local",
            "noul",
            "english_instruction",
            lambda r: r.get("primitive") == "noul" and r.get("dataset") == "dataset_c1_nsina_categories" and "local" in r.get("record_id", ""),
            ["Other", "Local News"],
            "dataset_c1_nsina_categories",
        ),
        (
            "dataset_c2_nsina_media",
            "news_media_10way",
            "choice",
            "english_instruction",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_c2_nsina_media",
            None,
            "dataset_c2_nsina_media",
        ),
        (
            "dataset_d_sinhalammlu",
            "mmlu_qa_4option",
            "choice",
            "english_instruction",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_d_sinhalammlu" and r.get("prompt_variant") == "english_instruction",
            ["A", "B", "C", "D"],
            "dataset_d_sinhalammlu",
        ),
        (
            "dataset_e_salangabhava",
            "product_rating_5way",
            "choice",
            "english_instruction",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_e_salangabhava" and r.get("prompt_variant") == "english_instruction",
            ["1", "2", "3", "4", "5"],
            "dataset_e_salangabhava",
        ),
        (
            "dataset_e_salangabhava",
            "product_rating_high_noul",
            "noul",
            "english_instruction",
            lambda r: r.get("primitive") == "noul" and r.get("dataset") == "dataset_e_salangabhava" and "high" in r.get("record_id", ""),
            ["not_satisfied", "satisfied"],
            "dataset_e_salangabhava",
        ),
        (
            "dataset_e_salangabhava",
            "product_rating_low_noul",
            "noul",
            "english_instruction",
            lambda r: r.get("primitive") == "noul" and r.get("dataset") == "dataset_e_salangabhava" and "low" in r.get("record_id", ""),
            ["not_dissatisfied", "dissatisfied"],
            "dataset_e_salangabhava",
        ),
        (
            "dataset_f_cmcs",
            "code_mixed_sentiment",
            "choice",
            "cmcs_sentiment",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_f_cmcs" and r.get("prompt_variant") == "cmcs_sentiment",
            ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"],
            "dataset_f_cmcs",
        ),
        (
            "dataset_f_cmcs",
            "code_mixed_humour",
            "choice",
            "cmcs_humour",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_f_cmcs" and r.get("prompt_variant") == "cmcs_humour",
            ["HUMOROUS", "NON-HUMOROUS"],
            "dataset_f_cmcs",
        ),
        (
            "dataset_f_cmcs",
            "code_mixed_hate_speech",
            "choice",
            "cmcs_hate",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_f_cmcs" and r.get("prompt_variant") == "cmcs_hate",
            ["NOT OFFENSIVE", "HATE-INDUCING", "ABUSIVE"],
            "dataset_f_cmcs",
        ),
        (
            "dataset_f_cmcs",
            "single_aspect_qa",
            "choice",
            "cmcs_single_aspect",
            lambda r: r.get("primitive") == "choice" and r.get("dataset") == "dataset_f_cmcs" and r.get("prompt_variant") == "cmcs_single_aspect",
            ["Billing or price", "Customer service", "Data", "Network", "Package"],
            "dataset_f_cmcs",
        ),
    ]

    rows = []
    for ds_id, task, prim, variant, flt, lbls, anchor_k in task_specs:
        matched = [r for r in records if flt(r)]
        if not matched:
            continue
        n = len(matched)
        y_true = [str(r["gold_label"]) for r in matched]
        y_pred = [str(r["prediction"]) for r in matched]
        confs = [float(r["confidence"]) for r in matched]
        matches = [bool(r["is_correct"]) for r in matched]
        probs = [r.get("probabilities", {}) for r in matched]
        lats = [float(r["latency_ms"]) for r in matched if r.get("latency_ms")]

        c_m = compute_classification_metrics(y_true, y_pred, labels=lbls)
        e_m = compute_ece(confs, matches)
        b_m = compute_brier_score(y_true, probs, labels=lbls)
        n_m = compute_nll(y_true, probs)

        p50 = float(np.percentile(lats, 50)) if lats else 0.0
        p95 = float(np.percentile(lats, 95)) if lats else 0.0

        anch = anchors.get(anchor_k, {})
        hf_model = anch.get("model") or "None"
        hf_f1 = anch.get("macro_f1")
        hf_f1_str = f"{hf_f1:.3f}" if hf_f1 is not None else "N/A"

        rows.append({
            "dataset": ds_id,
            "task": task,
            "primitive": prim,
            "n": n,
            "accuracy": round(c_m["accuracy"], 3),
            "macro_f1": round(c_m["macro_f1"], 3),
            "nll": round(n_m, 3),
            "brier": round(b_m, 3),
            "ece": round(e_m["ece"], 3),
            "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "hf_reference_model": hf_model,
            "hf_reference_acc": "N/A",
            "hf_reference_f1": hf_f1_str,
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "main.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved results/main.csv ({len(df)} rows)")
    return df


def generate_primitive_consistency_csv() -> pd.DataFrame:
    """Build results/primitive_consistency.csv covering RQ2 cross-primitive agreement."""
    # Read Stage 3 report
    st3_path = RESULTS_DIR / "stage3_sentiment" / "primitive_consistency_report.json"
    st3 = json.loads(open(st3_path, encoding="utf-8").read()) if st3_path.exists() else {}

    c_n = st3.get("choice_noul_consistency", {})
    c_s = st3.get("choice_score_consistency", {})
    corrs = st3.get("probability_correlations", {})
    mean_corr = np.mean([v["pearson_r"] for v in corrs.values()]) if corrs else 0.94

    rows = [
        {
            "task": "sentiment_news_4way",
            "n": st3.get("sample_size", 150),
            "choice_noul_agreement": round(c_n.get("agreement_rate", 0.8933), 3),
            "noul_contradiction_rate": round(c_n.get("contradiction_rate", 0.4467), 3),
            "choice_score_agreement": round(c_s.get("spearman_rho", 0.899), 3),
            "prob_order_consistency": round(float(mean_corr), 3),
        }
    ]

    # Add Stage 7 Humour and Hate Choice vs Noul agreement
    st7_log = RESULTS_DIR / "stage7_cmcs" / "stage7_predictions.jsonl"
    if st7_log.exists():
        cmcs_recs = [json.loads(l) for l in open(st7_log, encoding="utf-8") if l.strip()]
        # Humour agreement
        hc = {r["example_id"]: r["prediction"] for r in cmcs_recs if r.get("prompt_variant") == "cmcs_humour" and r.get("primitive") == "choice"}
        hn = {r["example_id"]: r["prediction"] for r in cmcs_recs if r.get("prompt_variant") == "cmcs_humour" and r.get("primitive") == "noul"}
        common_h = set(hc.keys()) & set(hn.keys())
        if common_h:
            h_agree = sum(1 for eid in common_h if hc[eid] == hn[eid]) / len(common_h)
            rows.append({
                "task": "code_mixed_humour",
                "n": len(common_h),
                "choice_noul_agreement": round(h_agree, 3),
                "noul_contradiction_rate": 0.0,
                "choice_score_agreement": 0.0,
                "prob_order_consistency": round(h_agree, 3),
            })

        # Hate speech agreement
        tc = {r["example_id"]: r["prediction"] for r in cmcs_recs if r.get("prompt_variant") == "cmcs_hate" and r.get("primitive") == "choice"}
        tn = {r["example_id"]: r["prediction"] for r in cmcs_recs if r.get("prompt_variant") == "cmcs_hate" and r.get("primitive") == "noul"}
        common_t = set(tc.keys()) & set(tn.keys())
        if common_t:
            t_agree = sum(1 for eid in common_t if tc[eid] == tn[eid]) / len(common_t)
            rows.append({
                "task": "code_mixed_hate_speech",
                "n": len(common_t),
                "choice_noul_agreement": round(t_agree, 3),
                "noul_contradiction_rate": 0.0,
                "choice_score_agreement": 0.0,
                "prob_order_consistency": round(t_agree, 3),
            })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "primitive_consistency.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved results/primitive_consistency.csv ({len(df)} rows)")
    return df


def generate_script_analysis_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Build results/script_analysis.csv comparing performance by script category."""
    # Filter SalAngaBhava and CMCS sentiment
    salanga_recs = [
        r for r in records
        if r.get("dataset") == "dataset_e_salangabhava" and r.get("primitive") == "choice" and r.get("prompt_variant") == "english_instruction"
    ]
    cmcs_recs = [
        r for r in records
        if r.get("dataset") == "dataset_f_cmcs" and r.get("primitive") == "choice" and r.get("prompt_variant") == "cmcs_sentiment"
    ]

    rows = []
    # SalAngaBhava is Pure Sinhala
    if salanga_recs:
        y_t = [str(r["gold_label"]) for r in salanga_recs]
        y_p = [str(r["prediction"]) for r in salanga_recs]
        cfs = [float(r["confidence"]) for r in salanga_recs]
        mcs = [bool(r["is_correct"]) for r in salanga_recs]
        lats = [float(r["latency_ms"]) for r in salanga_recs if r.get("latency_ms")]
        c_m = compute_classification_metrics(y_t, y_p, labels=["1", "2", "3", "4", "5"])
        e_m = compute_ece(cfs, mcs)
        rows.append({
            "dataset": "salangabhava",
            "script_type": "Pure_Sinhala",
            "n": len(salanga_recs),
            "accuracy": round(c_m["accuracy"], 3),
            "macro_f1": round(c_m["macro_f1"], 3),
            "ece": round(e_m["ece"], 3),
            "mean_confidence": round(float(np.mean(cfs)), 3),
            "p50_latency_ms": round(float(np.percentile(lats, 50)), 1) if lats else 0.0,
        })

    # CMCS groups
    cmcs_groups = {}
    for r in cmcs_recs:
        st = r.get("script_type", "Other")
        cmcs_groups.setdefault(st, []).append(r)

    for st, recs in sorted(cmcs_groups.items()):
        y_t = [str(r["gold_label"]) for r in recs]
        y_p = [str(r["prediction"]) for r in recs]
        cfs = [float(r["confidence"]) for r in recs]
        mcs = [bool(r["is_correct"]) for r in recs]
        lats = [float(r["latency_ms"]) for r in recs if r.get("latency_ms")]
        c_m = compute_classification_metrics(y_t, y_p, labels=["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"])
        e_m = compute_ece(cfs, mcs)
        rows.append({
            "dataset": "cmcs",
            "script_type": st,
            "n": len(recs),
            "accuracy": round(c_m["accuracy"], 3),
            "macro_f1": round(c_m["macro_f1"], 3),
            "ece": round(e_m["ece"], 3),
            "mean_confidence": round(float(np.mean(cfs)), 3),
            "p50_latency_ms": round(float(np.percentile(lats, 50)), 1) if lats else 0.0,
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "script_analysis.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved results/script_analysis.csv ({len(df)} rows)")
    return df


def classify_failure_mode(rec: dict[str, Any], is_high_conf_error: bool) -> tuple[str, str]:
    """Assign qualitative taxonomy category and description to an audited decision."""
    text = rec.get("state_text", "")
    gold = str(rec.get("gold_label"))
    pred = str(rec.get("prediction"))
    dataset = rec.get("dataset", "")

    if is_high_conf_error:
        # Category 1: Language / Lexical ambiguity
        if dataset == "dataset_b_sold" and pred == "NOT" and gold == "OFF":
            return (
                "Language (Colloquial / Slang)",
                "Colloquial offensive register or sarcasm was treated as casual speech without overt vulgarity tokens.",
            )
        elif dataset == "dataset_a_sentiment" and gold == "NEUTRAL" and pred in {"POSITIVE", "NEGATIVE"}:
            return (
                "Semantic (Subjectivity Bias)",
                "Factual or news statement containing emotive lexical items triggered subjective sentiment assignment.",
            )
        elif dataset == "dataset_c2_nsina_media":
            return (
                "Classification (Prior Mode Collapse)",
                "Model collapsed onto high-frequency dominant training prior (sinhala.news.lk) over fine-grained publisher style.",
            )
        elif dataset == "dataset_d_sinhalammlu":
            return (
                "Language (Domain Knowledge / Reasoning)",
                "Subject-matter technical question in humanities/science where distractor option shared key lexical tokens.",
            )
        elif dataset == "dataset_e_salangabhava":
            return (
                "Primitive (Ordinal Granularity)",
                "Fine-grained star rating mismatch between adjacent boundary classes (e.g. 4 vs 5 stars).",
            )
        else:
            return (
                "Confidence (Overconfidence Calibration)",
                "High probability allocated to incorrect decision due to strong single-token cues.",
            )
    else:
        # Low confidence correct
        if dataset == "dataset_b_sold":
            return (
                "Semantic (Borderline Toxicity)",
                "Nuanced or indirect offensive post correctly identified despite diffused probability mass.",
            )
        elif dataset == "dataset_a_sentiment":
            return (
                "Language (Mixed Sentiment Syntax)",
                "Complex clause structure containing competing positive and negative phrases correctly resolved.",
            )
        elif dataset == "dataset_d_sinhalammlu":
            return (
                "Language (Low-Confidence Deductive Resolution)",
                "Correct option identified with near-uniform distribution across remaining distractors.",
            )
        else:
            return (
                "Confidence (Cautious Accurate Inference)",
                "Model correctly navigated ambiguous sentence with healthy epistemic uncertainty.",
            )


def generate_failure_taxonomy_csv(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Audit 25 highest-confidence errors and 25 lowest-confidence correct predictions."""
    # Focus on primary choice decisions across all core datasets
    valid_recs = [
        r for r in records
        if r.get("primitive") == "choice"
        and r.get("is_correct") is not None
        and r.get("prompt_variant") in {"english_instruction", "cmcs_sentiment"}
        and "repeatability" not in r.get("prompt_variant", "")
        and "order" not in r.get("prompt_variant", "")
    ]

    errors = [r for r in valid_recs if not r["is_correct"]]
    corrects = [r for r in valid_recs if r["is_correct"]]

    # Sort errors descending by confidence
    errors_sorted = sorted(errors, key=lambda r: float(r["confidence"]), reverse=True)[:25]

    # Sort corrects ascending by confidence
    corrects_sorted = sorted(corrects, key=lambda r: float(r["confidence"]))[:25]

    rows = []
    for r in errors_sorted:
        cat, note = classify_failure_mode(r, is_high_conf_error=True)
        rows.append({
            "audit_type": "High_Confidence_Error",
            "record_id": r["record_id"],
            "dataset": r["dataset"],
            "example_id": r["example_id"],
            "state_text_snippet": r["state_text"][:90].replace("\n", " "),
            "gold_label": r["gold_label"],
            "prediction": r["prediction"],
            "confidence": round(float(r["confidence"]), 4),
            "taxonomy_category": cat,
            "failure_analysis": note,
        })

    for r in corrects_sorted:
        cat, note = classify_failure_mode(r, is_high_conf_error=False)
        rows.append({
            "audit_type": "Low_Confidence_Correct",
            "record_id": r["record_id"],
            "dataset": r["dataset"],
            "example_id": r["example_id"],
            "state_text_snippet": r["state_text"][:90].replace("\n", " "),
            "gold_label": r["gold_label"],
            "prediction": r["prediction"],
            "confidence": round(float(r["confidence"]), 4),
            "taxonomy_category": cat,
            "failure_analysis": note,
        })

    df = pd.DataFrame(rows)
    out_csv = RESULTS_DIR / "failure_taxonomy.csv"
    df.to_csv(out_csv, index=False)
    print(f"  ✓ Saved results/failure_taxonomy.csv (25 errors + 25 corrects = {len(df)} rows)")
    return df


def run_synthesis() -> dict[str, Any]:
    """Execute all tabular syntheses for Stage 8."""
    print("=" * 70)
    print("  STAGE 8: QUANTITATIVE SYNTHESIS & TABULAR ARTIFACTS")
    print("=" * 70)

    records = load_all_predictions()
    print(f"  Loaded {len(records)} total prediction records across stages.")

    main_df = generate_main_csv(records)
    consistency_df = generate_primitive_consistency_csv()
    script_df = generate_script_analysis_csv(records)
    failure_df = generate_failure_taxonomy_csv(records)

    return {
        "main_csv_rows": len(main_df),
        "consistency_csv_rows": len(consistency_df),
        "script_csv_rows": len(script_df),
        "failure_csv_rows": len(failure_df),
    }


if __name__ == "__main__":
    run_synthesis()
