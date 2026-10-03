"""
runner_stage7_cmcs.py — Stage 7: Code-Mixed Stress Track (Dataset F: CMCS, N=150).

Evaluates Jev on non-standard, Romanized, and code-mixed Sinhala-English text across 5 target sub-tasks:
  1. Sentiment (Choice 4-way, Noul 4-way argmax)
  2. Humour (Choice binary, Noul binary)
  3. Hate Speech (Choice 3-way, Noul 3-way)
  4. Aspect Extraction (5 multi-label Noul primitives, Single-aspect Choice)
  5. Script Type Analysis (Pure Sinhala vs. Romanized/Singlish vs. Code-mixed performance)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from typesafe_sdk import Choice, Noul

from src.client import JevCallResult
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_nll,
)
from src.runners.base_runner import BaseRunner


def classify_script_type(text: str) -> str:
    """Classify text into Pure_Sinhala, Sinhala_in_Latin, or Code_Mixed."""
    has_sinhala = bool(re.search(r"[\u0D80-\u0DFF]", text))
    has_latin = bool(re.search(r"[a-zA-Z]", text))
    if has_sinhala and has_latin:
        return "Code_Mixed"
    elif has_sinhala:
        return "Pure_Sinhala"
    elif has_latin:
        return "Sinhala_in_Latin"
    return "Other"


class CMCSStressTrackRunner(BaseRunner):
    """Runner for Stage 7: Code-Mixed Stress Track (CMCS, N=150)."""

    def __init__(
        self,
        output_dir: Path | None = None,
        experiment_id: str = "exp_stage7_cmcs_stress",
    ) -> None:
        target_dir = output_dir or (RESULTS_DIR / "stage7_cmcs")
        super().__init__(
            experiment_id=experiment_id,
            phase="phase_1_full",
            output_dir=target_dir,
        )
        self.sample_path = DATA_PROCESSED_DIR / "samples" / "dataset_f_cmcs.jsonl"
        self.raw_log_path = self.output_dir / "stage7_predictions.jsonl"

        if self.raw_log_path.exists():
            self.raw_log_path.unlink()

    def load_samples(self) -> list[dict[str, Any]]:
        if not self.sample_path.exists():
            raise FileNotFoundError(f"CMCS sample file missing: {self.sample_path}")
        records = []
        with open(self.sample_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _ask_with_retry(
        self,
        state: str,
        questions: dict[str, Any],
        max_attempts: int = 3,
    ) -> JevCallResult | None:
        for attempt in range(max_attempts):
            try:
                return self.client.ask(state=state, questions=questions)
            except Exception as exc:
                if attempt == max_attempts - 1:
                    print(f"\n  [Error] Failed call after {max_attempts} attempts: {exc}")
                    return None
                time.sleep(1.5)
        return None

    def run(self) -> dict[str, Any]:
        """Execute Stage 7 CMCS stress evaluation."""
        print("=" * 70)
        print("  STAGE 7: CODE-MIXED STRESS TRACK (DATASET F: CMCS, N=150)")
        print(f"  Model: {self.config.model}")
        print(f"  Sample Source: {self.sample_path.name}")
        print(f"  Prediction Stream: {self.raw_log_path}")
        print("=" * 70)

        records = self.load_samples()
        print(f"  Loaded {len(records)} code-mixed records.")

        # Prompts
        p_sent_choice = self.prompts["cmcs_sentiment_choice"]["english"]
        p_pos = self.prompts["sentiment_noul"]["positive"]["english"]
        p_neg = self.prompts["sentiment_noul"]["negative"]["english"]
        p_neu = self.prompts["sentiment_noul"]["neutral"]["english"]
        p_cnf = self.prompts["sentiment_noul"]["conflict"]["english"]

        p_humour_choice = self.prompts["cmcs_humour_choice"]["english"]
        p_hate_choice = self.prompts["cmcs_hate_choice"]["english"]
        p_aspect = self.prompts["cmcs_aspect_noul"]["english"]

        criteria_sent = {"POSITIVE": None, "NEGATIVE": None, "NEUTRAL": None, "CONFLICT": None}
        criteria_humour = {"HUMOROUS": "Humorous", "NON-HUMOROUS": "Non-humorous"}
        criteria_hate = {"NOT OFFENSIVE": "Not offensive", "HATE-INDUCING": "Hate-inducing", "ABUSIVE": "Abusive"}

        aspect_targets = ["Billing or price", "Customer service", "Data", "Network", "Package"]

        # Result tracking
        records_sentiment_choice = []
        records_sentiment_noul = []
        records_humour_choice = []
        records_humour_noul = []
        records_hate_choice = []
        records_hate_noul = []
        aspect_eval_data: dict[str, dict[str, list]] = {asp: {"y_true": [], "y_pred": [], "conf": []} for asp in aspect_targets}
        single_aspect_choice_records = []

        for r in tqdm(records, desc="Executing CMCS Multi-Task Stress Track (N=150)"):
            state = r["text"]
            ex_id = r["example_id"]
            meta = r.get("metadata", {})
            gold_sent = meta.get("sentiment")
            gold_humour = meta.get("humor")
            gold_hate = meta.get("hate_speech")
            gold_aspects = meta.get("aspects", [])
            script_type = classify_script_type(state)

            questions: dict[str, Any] = {
                # 1. Sentiment
                "sent_choice": Choice(instructions=p_sent_choice, criteria=criteria_sent),
                "noul_pos": Noul(instructions=p_pos),
                "noul_neg": Noul(instructions=p_neg),
                "noul_neu": Noul(instructions=p_neu),
                "noul_cnf": Noul(instructions=p_cnf),
                # 2. Humour
                "humour_choice": Choice(instructions=p_humour_choice, criteria=criteria_humour),
                "noul_humour": Noul(instructions="Is this code-mixed text humorous?"),
                # 3. Hate Speech
                "hate_choice": Choice(instructions=p_hate_choice, criteria=criteria_hate),
                "noul_hate_ind": Noul(instructions="Is this code-mixed text hate-inducing?"),
                "noul_abusive": Noul(instructions="Is this code-mixed text abusive?"),
                "noul_not_off": Noul(instructions="Is this code-mixed text not offensive?"),
            }

            # 4. Aspects (5 Nouls)
            for asp in aspect_targets:
                safe_key = f"noul_asp_{asp.split()[0].lower()}"
                questions[safe_key] = Noul(instructions=p_aspect.format(aspect=asp))

            # Single-aspect Choice (only if exactly 1 active aspect among aspect_targets)
            active_target_aspects = [a for a in gold_aspects if a in aspect_targets]
            if len(active_target_aspects) == 1:
                questions["single_aspect_choice"] = Choice(
                    instructions="Which aspect is primarily discussed in this customer review?",
                    criteria={asp: None for asp in aspect_targets},
                )

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = (
                res.usage.model_dump()
                if hasattr(res.usage, "model_dump")
                else {"input_tokens": 0, "output_tokens": 0}
            )

            # ── 1. Sentiment Logging ─────────────────────────────────────
            if gold_sent:
                ans_sc = res.answers["sent_choice"]
                rec_sc = self.format_record(
                    record_id=f"rec_s7_{ex_id}_sent_choice",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="choice",
                    prompt_variant="cmcs_sentiment",
                    state_text=state,
                    instructions=p_sent_choice,
                    criteria_definitions=criteria_sent,
                    gold_label=gold_sent,
                    prediction=ans_sc.choice,
                    is_correct=(ans_sc.choice == gold_sent),
                    confidence=ans_sc.confidence,
                    probabilities=ans_sc.probabilities,
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_sc, self.raw_log_path)
                records_sentiment_choice.append(rec_sc)

                noul_sent_probs = {
                    "POSITIVE": res.answers["noul_pos"].noul,
                    "NEGATIVE": res.answers["noul_neg"].noul,
                    "NEUTRAL": res.answers["noul_neu"].noul,
                    "CONFLICT": res.answers["noul_cnf"].noul,
                }
                best_sent_noul = max(noul_sent_probs, key=noul_sent_probs.get)
                rec_sn = self.format_record(
                    record_id=f"rec_s7_{ex_id}_sent_noul",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="noul",
                    prompt_variant="cmcs_sentiment",
                    state_text=state,
                    instructions=p_pos,
                    criteria_definitions={"binary": ["False", "True"]},
                    gold_label=gold_sent,
                    prediction=best_sent_noul,
                    is_correct=(best_sent_noul == gold_sent),
                    confidence=noul_sent_probs[best_sent_noul],
                    probabilities=noul_sent_probs,
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_sn, self.raw_log_path)
                records_sentiment_noul.append(rec_sn)

            # ── 2. Humour Logging ────────────────────────────────────────
            if gold_humour:
                ans_hc = res.answers["humour_choice"]
                rec_hc = self.format_record(
                    record_id=f"rec_s7_{ex_id}_humour_choice",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="choice",
                    prompt_variant="cmcs_humour",
                    state_text=state,
                    instructions=p_humour_choice,
                    criteria_definitions=criteria_humour,
                    gold_label=gold_humour,
                    prediction=ans_hc.choice,
                    is_correct=(ans_hc.choice == gold_humour),
                    confidence=ans_hc.confidence,
                    probabilities=ans_hc.probabilities,
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_hc, self.raw_log_path)
                records_humour_choice.append(rec_hc)

                p_hum = res.answers["noul_humour"].noul
                n_hum_pred = "HUMOROUS" if p_hum >= 0.5 else "NON-HUMOROUS"
                conf_hum = p_hum if n_hum_pred == "HUMOROUS" else 1.0 - p_hum
                rec_hn = self.format_record(
                    record_id=f"rec_s7_{ex_id}_humour_noul",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="noul",
                    prompt_variant="cmcs_humour",
                    state_text=state,
                    instructions="Is this code-mixed text humorous?",
                    criteria_definitions={"binary": ["False", "True"]},
                    gold_label=gold_humour,
                    prediction=n_hum_pred,
                    is_correct=(n_hum_pred == gold_humour),
                    confidence=conf_hum,
                    probabilities={"NON-HUMOROUS": round(1.0 - p_hum, 4), "HUMOROUS": round(p_hum, 4)},
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_hn, self.raw_log_path)
                records_humour_noul.append(rec_hn)

            # ── 3. Hate Speech Logging ───────────────────────────────────
            if gold_hate:
                ans_tc = res.answers["hate_choice"]
                rec_tc = self.format_record(
                    record_id=f"rec_s7_{ex_id}_hate_choice",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="choice",
                    prompt_variant="cmcs_hate",
                    state_text=state,
                    instructions=p_hate_choice,
                    criteria_definitions=criteria_hate,
                    gold_label=gold_hate,
                    prediction=ans_tc.choice,
                    is_correct=(ans_tc.choice == gold_hate),
                    confidence=ans_tc.confidence,
                    probabilities=ans_tc.probabilities,
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_tc, self.raw_log_path)
                records_hate_choice.append(rec_tc)

                noul_hate_probs = {
                    "NOT OFFENSIVE": res.answers["noul_not_off"].noul,
                    "HATE-INDUCING": res.answers["noul_hate_ind"].noul,
                    "ABUSIVE": res.answers["noul_abusive"].noul,
                }
                best_hate_noul = max(noul_hate_probs, key=noul_hate_probs.get)
                rec_tn = self.format_record(
                    record_id=f"rec_s7_{ex_id}_hate_noul",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="noul",
                    prompt_variant="cmcs_hate",
                    state_text=state,
                    instructions="Is this code-mixed text hate-inducing / abusive / not offensive?",
                    criteria_definitions={"binary": ["False", "True"]},
                    gold_label=gold_hate,
                    prediction=best_hate_noul,
                    is_correct=(best_hate_noul == gold_hate),
                    confidence=noul_hate_probs[best_hate_noul],
                    probabilities=noul_hate_probs,
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_tn, self.raw_log_path)
                records_hate_noul.append(rec_tn)

            # ── 4. Aspect Logging ────────────────────────────────────────
            for asp in aspect_targets:
                safe_key = f"noul_asp_{asp.split()[0].lower()}"
                p_asp = res.answers[safe_key].noul
                has_asp_gold = asp in gold_aspects
                asp_pred = bool(p_asp >= 0.5)

                aspect_eval_data[asp]["y_true"].append(has_asp_gold)
                aspect_eval_data[asp]["y_pred"].append(asp_pred)
                aspect_eval_data[asp]["conf"].append(p_asp if asp_pred else 1.0 - p_asp)

            if len(active_target_aspects) == 1 and "single_aspect_choice" in res.answers:
                gold_asp_single = active_target_aspects[0]
                ans_ac = res.answers["single_aspect_choice"]
                rec_ac = self.format_record(
                    record_id=f"rec_s7_{ex_id}_aspect_single_choice",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type=script_type,
                    primitive="choice",
                    prompt_variant="cmcs_single_aspect",
                    state_text=state,
                    instructions="Which aspect is primarily discussed in this customer review?",
                    criteria_definitions={asp: None for asp in aspect_targets},
                    gold_label=gold_asp_single,
                    prediction=ans_ac.choice,
                    is_correct=(ans_ac.choice == gold_asp_single),
                    confidence=ans_ac.confidence,
                    probabilities=ans_ac.probabilities,
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_ac, self.raw_log_path)
                single_aspect_choice_records.append(rec_ac)

        # ── Compute Task Metrics ─────────────────────────────────────────
        def _calc_metrics(recs: list[dict[str, Any]], labels: list[str]) -> dict[str, Any]:
            if not recs:
                return {}
            yt = [r["gold_label"] for r in recs]
            yp = [r["prediction"] for r in recs]
            cf = [float(r["confidence"]) for r in recs]
            mc = [bool(r["is_correct"]) for r in recs]
            pb = [r.get("probabilities", {}) for r in recs]

            c_m = compute_classification_metrics(yt, yp, labels=labels)
            e_m = compute_ece(cf, mc)
            b_m = compute_brier_score(yt, pb, labels=labels)
            n_m = compute_nll(yt, pb)
            return {
                "sample_size": len(recs),
                "accuracy": c_m.get("accuracy", 0.0),
                "macro_f1": c_m.get("macro_f1", 0.0),
                "ece": e_m.get("ece", 0.0),
                "brier": b_m,
                "nll": n_m,
                "mean_confidence": round(float(np.mean(cf)), 4) if cf else 0.0,
            }

        sent_c_metrics = _calc_metrics(records_sentiment_choice, ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"])
        sent_n_metrics = _calc_metrics(records_sentiment_noul, ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"])
        humour_c_metrics = _calc_metrics(records_humour_choice, ["HUMOROUS", "NON-HUMOROUS"])
        humour_n_metrics = _calc_metrics(records_humour_noul, ["HUMOROUS", "NON-HUMOROUS"])
        hate_c_metrics = _calc_metrics(records_hate_choice, ["NOT OFFENSIVE", "HATE-INDUCING", "ABUSIVE"])
        hate_n_metrics = _calc_metrics(records_hate_noul, ["NOT OFFENSIVE", "HATE-INDUCING", "ABUSIVE"])

        # Aspect metrics
        aspect_results = {}
        for asp, data in aspect_eval_data.items():
            yt = data["y_true"]
            yp = data["y_pred"]
            cf = data["conf"]
            acc = float(accuracy_score(yt, yp)) if yt else 0.0
            p = float(precision_score(yt, yp, zero_division=0)) if yt else 0.0
            r = float(recall_score(yt, yp, zero_division=0)) if yt else 0.0
            f1 = float(f1_score(yt, yp, zero_division=0)) if yt else 0.0
            aspect_results[asp] = {
                "accuracy": round(acc, 4),
                "precision": round(p, 4),
                "recall": round(r, 4),
                "f1": round(f1, 4),
                "positive_count_true": int(sum(yt)),
                "positive_count_pred": int(sum(yp)),
            }

        single_aspect_choice_metrics = _calc_metrics(single_aspect_choice_records, aspect_targets)

        # ── Script Type Segmentation Analysis ────────────────────────────
        script_groups: dict[str, list[dict[str, Any]]] = {}
        for r in records_sentiment_choice:
            st = r.get("script_type", "Other")
            script_groups.setdefault(st, []).append(r)

        script_analysis = {}
        for st, recs in script_groups.items():
            yt = [r["gold_label"] for r in recs]
            yp = [r["prediction"] for r in recs]
            cf = [float(r["confidence"]) for r in recs]
            mc = [bool(r["is_correct"]) for r in recs]
            c_m = compute_classification_metrics(yt, yp, labels=["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"])
            e_m = compute_ece(cf, mc)
            script_analysis[st] = {
                "n": len(recs),
                "accuracy": c_m.get("accuracy", 0.0),
                "macro_f1": c_m.get("macro_f1", 0.0),
                "ece": e_m.get("ece", 0.0),
                "mean_confidence": round(float(np.mean(cf)), 4) if cf else 0.0,
            }

        report = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model,
            "sample_size": len(records),
            "sentiment_choice": sent_c_metrics,
            "sentiment_noul": sent_n_metrics,
            "humour_choice": humour_c_metrics,
            "humour_noul": humour_n_metrics,
            "hate_choice": hate_c_metrics,
            "hate_noul": hate_n_metrics,
            "aspect_nouls": aspect_results,
            "single_aspect_choice": single_aspect_choice_metrics,
            "script_analysis": script_analysis,
        }

        # Save JSON report
        report_path = self.output_dir / "cmcs_stress_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved JSON report: {report_path}")

        # Save Markdown summary
        summary_md = self.generate_markdown_summary(report)
        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"  ✓ Saved Markdown summary: {summary_path}")

        print("\n" + "=" * 70)
        print("  STAGE 7 CMCS STRESS TRACK HIGHLIGHTS")
        print("=" * 70)
        print(summary_md)

        return report

    def generate_markdown_summary(self, report: dict[str, Any]) -> str:
        """Render a clean summary table of Stage 7 CMCS in Markdown."""
        lines = [
            "# Stage 7: Code-Mixed Stress Track Report (Dataset F: CMCS, N=150)",
            "",
            f"**Timestamp:** {report['timestamp']}  ",
            f"**Model:** `{report['model']}`  ",
            f"**Total Samples Evaluated:** N={report['sample_size']}  ",
            "",
            "## 1. Multi-Task Classification Performance",
            "",
            "| Sub-Task | Primitive | Classes | N | Accuracy | Macro-F1 | ECE | Brier Score | Mean Conf |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        def _row(task: str, prim: str, classes: int, d: dict[str, Any]) -> str:
            if not d:
                return f"| **{task}** | {prim} | {classes} | 0 | — | — | — | — | — |"
            return (
                f"| **{task}** | `{prim}` | {classes} | {d['sample_size']} | "
                f"**{d['accuracy']:.3f}** | {d['macro_f1']:.3f} | {d['ece']:.3f} | "
                f"{d['brier']:.3f} | {d['mean_confidence']:.3f} |"
            )

        lines.append(_row("Sentiment", "choice", 4, report["sentiment_choice"]))
        lines.append(_row("Sentiment", "noul", 4, report["sentiment_noul"]))
        lines.append(_row("Humour Detection", "choice", 2, report["humour_choice"]))
        lines.append(_row("Humour Detection", "noul", 2, report["humour_noul"]))
        lines.append(_row("Hate Speech", "choice", 3, report["hate_choice"]))
        lines.append(_row("Hate Speech", "noul", 3, report["hate_noul"]))
        if report["single_aspect_choice"]:
            lines.append(_row("Single-Aspect QA", "choice", 5, report["single_aspect_choice"]))

        lines.extend([
            "",
            "## 2. Multi-Label Aspect Extraction (Parallel Noul)",
            "",
            "| Aspect Domain | True Positives | Predicted Positives | Accuracy | Precision | Recall | F1 Score |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
        ])

        for asp, ad in report["aspect_nouls"].items():
            lines.append(
                f"| **{asp}** | {ad['positive_count_true']} | {ad['positive_count_pred']} | "
                f"{ad['accuracy']:.3f} | {ad['precision']:.3f} | {ad['recall']:.3f} | **{ad['f1']:.3f}** |"
            )

        lines.extend([
            "",
            "## 3. Script Type Degradation Analysis (Sentiment Choice)",
            "",
            "| Script Category | N | Accuracy | Macro-F1 | ECE | Mean Confidence |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ])

        for st, sd in report.get("script_analysis", {}).items():
            lines.append(
                f"| **{st}** | {sd['n']} | **{sd['accuracy']:.3f}** | {sd['macro_f1']:.3f} | "
                f"{sd['ece']:.3f} | {sd['mean_confidence']:.3f} |"
            )

        lines.extend([
            "",
            "## 4. Key Takeaways for Stage 7",
            "1. **Code-Mixed Sentiment Resilience:** Assesses whether Jev maintains sentiment comprehension when Sinhala is interleaved with English and colloquialisms.",
            "2. **Safety & Toxicity Sensitivity:** Compares 3-way hate speech detection against binary offensive language on informal social data.",
            "3. **Script Sensitivity:** Measures the quantitative performance penalty between pure Sinhala script, Singlish (Latin), and code-switched text.",
        ])

        return "\n".join(lines)
