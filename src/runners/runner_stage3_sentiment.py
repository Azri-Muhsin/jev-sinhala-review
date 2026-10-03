"""
runner_stage3_sentiment.py — Stage 3: Sentiment Primitive Equivalence Lab (N=150).

Executes Phase 1 full probe on Dataset A (Sinhala News-Comment Sentiment) testing RQ2
(Primitive Consistency) across Choice, Noul, and Score on the exact same input states.
"""

from __future__ import annotations

import io
import json
import os
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
from tqdm import tqdm
from typesafe_sdk import Choice, Noul, Score

from src.client import JevCallResult
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_nll,
    compute_selective_risk_coverage,
    compute_sentiment_primitive_consistency,
)
from src.runners.base_runner import BaseRunner


class SentimentPrimitiveEquivalenceRunner(BaseRunner):
    """Runner for Stage 3: Sentiment Primitive Equivalence Lab."""

    def __init__(
        self,
        output_dir: Path | None = None,
        experiment_id: str = "exp_stage3_sentiment_primitive_lab",
    ) -> None:
        target_dir = output_dir or (RESULTS_DIR / "stage3_sentiment")
        super().__init__(
            experiment_id=experiment_id,
            phase="phase_1_full",
            output_dir=target_dir,
        )
        self.sample_file = DATA_PROCESSED_DIR / "samples" / "dataset_a_sentiment.jsonl"
        self.raw_log_path = self.output_dir / "stage3_predictions.jsonl"

        # Fresh predictions stream
        if self.raw_log_path.exists():
            self.raw_log_path.unlink()

    def load_sample_records(self) -> list[dict[str, Any]]:
        """Load frozen N=150 records from sample file."""
        if not self.sample_file.exists():
            raise FileNotFoundError(f"Sample file missing: {self.sample_file}")
        records = []
        with open(self.sample_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def run(self) -> dict[str, Any]:
        """Execute Stage 3 sentiment probe and compute RQ2 consistency analysis."""
        print("=" * 70)
        print("  STAGE 3: SENTIMENT PRIMITIVE EQUIVALENCE LAB (N=150)")
        print(f"  Target Model: {self.config.model}")
        print(f"  Sample Source: {self.sample_file}")
        print(f"  Prediction Stream: {self.raw_log_path}")
        print("=" * 70)

        records = self.load_sample_records()
        print(f"  Loaded {len(records)} frozen records from {self.sample_file.name}")

        p_choice = self.prompts["sentiment_choice"]["english"]
        p_pos = self.prompts["sentiment_noul"]["positive"]["english"]
        p_neg = self.prompts["sentiment_noul"]["negative"]["english"]
        p_neu = self.prompts["sentiment_noul"]["neutral"]["english"]
        p_cnf = self.prompts["sentiment_noul"]["conflict"]["english"]
        p_score = self.prompts["sentiment_score"]["english"]

        criteria_choice = {
            "POSITIVE": None,
            "NEGATIVE": None,
            "NEUTRAL": None,
            "CONFLICT": None,
        }
        score_criteria = ["Negative sentiment", "Neutral sentiment", "Positive sentiment"]

        choice_records: list[dict[str, Any]] = []
        noul_records: list[dict[str, Any]] = []
        score_records: list[dict[str, Any]] = []
        latencies: list[float] = []

        for r in tqdm(records, desc="Executing Stage 3 (Choice + Noul + Score)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions: dict[str, Any] = {
                "choice_sentiment": Choice(instructions=p_choice, criteria=criteria_choice),
                "noul_pos": Noul(instructions=p_pos),
                "noul_neg": Noul(instructions=p_neg),
                "noul_neu": Noul(instructions=p_neu),
                "noul_cnf": Noul(instructions=p_cnf),
            }

            if gold in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                questions["score_sentiment"] = Score(
                    instructions=p_score,
                    criteria=score_criteria,
                )

            res = None
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    res = self.client.ask(state=state, questions=questions)
                    break
                except Exception as exc:
                    if attempt == max_attempts - 1:
                        rec_err = self.format_record(
                            record_id=f"rec_p1_sent_{ex_id}_error",
                            dataset="dataset_a_sentiment",
                            example_id=ex_id,
                            primitive="choice",
                            prompt_variant="english_instruction",
                            state_text=state,
                            instructions=p_choice,
                            criteria_definitions=criteria_choice,
                            gold_label=gold,
                            prediction=None,
                            is_correct=False,
                            confidence=0.0,
                            probabilities={},
                            latency_ms=0.0,
                            usage={"input_tokens": 0, "output_tokens": 0},
                            model=self.config.model,
                            raw_response_status=500,
                            error=str(exc),
                        )
                        self.log_record(rec_err, self.raw_log_path)
                    else:
                        time.sleep(1.5)

            if res is None:
                continue

            latencies.append(res.latency_ms)
            usage = (
                res.usage.model_dump()
                if hasattr(res.usage, "model_dump")
                else {"input_tokens": 0, "output_tokens": 0}
            )

            # 1. Log Choice
            ans_c = res.answers["choice_sentiment"]
            rec_c = self.format_record(
                record_id=f"rec_p1_sent_{ex_id}_choice",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=ans_c.choice,
                is_correct=(ans_c.choice == gold),
                confidence=ans_c.confidence,
                probabilities=ans_c.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c, self.raw_log_path)
            choice_records.append(rec_c)

            # 2. Log Noul (Multi-class decision)
            noul_probs = {
                "POSITIVE": res.answers["noul_pos"].noul,
                "NEGATIVE": res.answers["noul_neg"].noul,
                "NEUTRAL": res.answers["noul_neu"].noul,
                "CONFLICT": res.answers["noul_cnf"].noul,
            }
            best_noul = max(noul_probs, key=noul_probs.get)
            rec_n = self.format_record(
                record_id=f"rec_p1_sent_{ex_id}_noul_multiclass",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_pos,
                criteria_definitions={"binary": ["False", "True"]},
                gold_label=gold,
                prediction=best_noul,
                is_correct=(best_noul == gold),
                confidence=noul_probs[best_noul],
                probabilities=noul_probs,
                noul_details=noul_probs,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, self.raw_log_path)
            noul_records.append(rec_n)

            # 3. Log Score if present
            if "score_sentiment" in res.answers:
                ans_s = res.answers["score_sentiment"]
                rec_s = self.format_record(
                    record_id=f"rec_p1_sent_{ex_id}_score",
                    dataset="dataset_a_sentiment",
                    example_id=ex_id,
                    primitive="score",
                    prompt_variant="english_instruction",
                    state_text=state,
                    instructions=p_score,
                    criteria_definitions=score_criteria,
                    gold_label=gold,
                    prediction=ans_s.score,
                    is_correct=None,
                    confidence=ans_s.confidence,
                    probabilities=ans_s.probabilities,
                    score_details={"legend": ans_s.legend, "score": ans_s.score},
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_s, self.raw_log_path)
                score_records.append(rec_s)

        # Quantitative Analysis & RQ2 Synthesis
        report = self.analyze_results(choice_records, noul_records, score_records, latencies)
        return report

    def analyze_results(
        self,
        choice_records: list[dict[str, Any]],
        noul_records: list[dict[str, Any]],
        score_records: list[dict[str, Any]],
        latencies: list[float],
    ) -> dict[str, Any]:
        """Compute comprehensive metrics across accuracy, calibration, and primitive consistency."""
        print("\n" + "─" * 70)
        print("  STAGE 3 SYNTHESIS & RQ2 CONSISTENCY EVALUATION")
        print("─" * 70)

        classes = ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"]
        y_true = [r["gold_label"] for r in choice_records]
        y_choice = [r["prediction"] for r in choice_records]
        y_noul = [r["prediction"] for r in noul_records]

        # 1. Classification Metrics
        metrics_choice = compute_classification_metrics(y_true, y_choice, labels=classes)
        metrics_noul = compute_classification_metrics(y_true, y_noul, labels=classes)

        print(f"  [Classification] Choice Accuracy:     {metrics_choice['accuracy']:.1%} | Macro-F1: {metrics_choice['macro_f1']:.3f}")
        print(f"  [Classification] Noul Argmax Acc:     {metrics_noul['accuracy']:.1%} | Macro-F1: {metrics_noul['macro_f1']:.3f}")

        # 2. Calibration Metrics
        choice_confs = [r["confidence"] for r in choice_records]
        choice_matches = [r["is_correct"] for r in choice_records]
        choice_probs = [r["probabilities"] for r in choice_records]
        calib_choice = compute_ece(choice_confs, choice_matches, num_bins=10)
        brier_choice = compute_brier_score(y_true, choice_probs, labels=classes)
        nll_choice = compute_nll(y_true, choice_probs)
        rc_choice = compute_selective_risk_coverage(choice_confs, choice_matches)

        noul_confs = [r["confidence"] for r in noul_records]
        noul_matches = [r["is_correct"] for r in noul_records]
        noul_probs = [r["probabilities"] for r in noul_records]
        calib_noul = compute_ece(noul_confs, noul_matches, num_bins=10)
        brier_noul = compute_brier_score(y_true, noul_probs, labels=classes)

        print(f"  [Calibration]    Choice ECE:          {calib_choice['ece']:.3f} | Brier: {brier_choice:.3f} | NLL: {nll_choice:.3f}")
        print(f"  [Calibration]    Noul Argmax ECE:     {calib_noul['ece']:.3f} | Brier: {brier_noul:.3f}")

        # 3. RQ2 Primitive Consistency
        consistency = compute_sentiment_primitive_consistency(
            choice_records, noul_records, score_records
        )
        agr = consistency["argmax_agreement"]["agreement_rate"]
        contra = consistency["contradictions"]["multi_belief_contradiction_rate"]
        zero_b = consistency["contradictions"]["zero_belief_rate"]
        print(f"  [RQ2 Consistency] Argmax Agreement:   {agr:.1%}")
        print(f"  [RQ2 Consistency] Contradiction Rate: {contra:.1%} (multi-belief >= 0.50)")
        print(f"  [RQ2 Consistency] Zero-belief Rate:   {zero_b:.1%} (no class >= 0.50)")

        if consistency.get("score_ordinal_alignment"):
            score_mae = consistency["score_ordinal_alignment"]["mae"]
            score_rho = consistency["score_ordinal_alignment"]["spearman_rho"]
            print(f"  [RQ2 Consistency] Score vs Choice:   MAE={score_mae:.3f} | Spearman Rho={score_rho:.3f}")

        # 4. Latency Telemetry
        lat_arr = np.array(latencies) if latencies else np.array([0.0])
        telemetry = {
            "total_requests": len(latencies),
            "total_records_logged": len(choice_records) + len(noul_records) + len(score_records),
            "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 1),
            "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 1),
            "latency_mean_ms": round(float(np.mean(lat_arr)), 1),
        }
        print(f"  [Telemetry]      Latency p50: {telemetry['latency_p50_ms']}ms | p95: {telemetry['latency_p95_ms']}ms")

        # Compile Full Report
        report = {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model,
            "sample_size": len(choice_records),
            "classification": {
                "choice": metrics_choice,
                "noul": metrics_noul,
            },
            "calibration": {
                "choice": {
                    "ece": calib_choice["ece"],
                    "mce": calib_choice["mce"],
                    "brier_score": brier_choice,
                    "nll": nll_choice,
                    "bins": calib_choice["bins"],
                    "risk_coverage": rc_choice,
                },
                "noul": {
                    "ece": calib_noul["ece"],
                    "mce": calib_noul["mce"],
                    "brier_score": brier_noul,
                    "bins": calib_noul["bins"],
                },
            },
            "primitive_consistency_rq2": consistency,
            "telemetry": telemetry,
        }

        # Save JSON report
        report_path = self.output_dir / "primitive_consistency_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save Markdown Summary
        summary_md_path = self.output_dir / "summary.md"
        self._write_summary_markdown(summary_md_path, report)

        print("─" * 70)
        print(f"  ✓ Stage 3 Finished Successfully")
        print(f"  Report:      {report_path}")
        print(f"  Summary:     {summary_md_path}")
        print(f"  Predictions: {self.raw_log_path}")
        print("─" * 70)
        return report

    def _write_summary_markdown(self, path: Path, report: dict[str, Any]) -> None:
        """Write a comprehensive Markdown summary of Stage 3 findings."""
        c_m = report["classification"]["choice"]
        n_m = report["classification"]["noul"]
        c_cal = report["calibration"]["choice"]
        n_cal = report["calibration"]["noul"]
        cons = report["primitive_consistency_rq2"]
        tel = report["telemetry"]

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Stage 3: Sentiment Primitive Equivalence Lab — Quantitative Report\n\n")
            f.write(f"**Target Model:** `{report['model']}`  \n")
            f.write(f"**Sample Size:** $N = {report['sample_size']}$ (Dataset A: Sinhala News-Comment Sentiment)  \n")
            f.write(f"**Timestamp:** `{report['timestamp']}`  \n\n")

            f.write("## 1. Classification Performance: Choice vs Noul\n\n")
            f.write("| Metric | Choice (4-way Categorical) | Noul (4-way Argmax) | Discrepancy |\n")
            f.write("| :--- | :---: | :---: | :---: |\n")
            f.write(f"| **Accuracy** | **{c_m['accuracy']:.1%}** | **{n_m['accuracy']:.1%}** | {abs(c_m['accuracy'] - n_m['accuracy']):.1%} |\n")
            f.write(f"| **Macro-F1** | {c_m['macro_f1']:.3f} | {n_m['macro_f1']:.3f} | {abs(c_m['macro_f1'] - n_m['macro_f1']):.3f} |\n")
            f.write(f"| **Weighted-F1** | {c_m['weighted_f1']:.3f} | {n_m['weighted_f1']:.3f} | {abs(c_m['weighted_f1'] - n_m['weighted_f1']):.3f} |\n")
            f.write(f"| **Brier Score** | {c_cal['brier_score']:.3f} | {n_cal['brier_score']:.3f} | {abs(c_cal['brier_score'] - n_cal['brier_score']):.3f} |\n")
            f.write(f"| **ECE (Calibration Error)** | {c_cal['ece']:.3f} | {n_cal['ece']:.3f} | {abs(c_cal['ece'] - n_cal['ece']):.3f} |\n\n")

            f.write("### Per-Class Performance Breakdown\n\n")
            f.write("| Class | Support | Choice Precision | Choice Recall | Choice F1 | Noul Precision | Noul Recall | Noul F1 |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for cls in ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"]:
                pc_c = c_m["per_class"].get(cls, {})
                pc_n = n_m["per_class"].get(cls, {})
                f.write(
                    f"| `{cls}` | {pc_c.get('support', 0)} | "
                    f"{pc_c.get('precision', 0):.3f} | {pc_c.get('recall', 0):.3f} | {pc_c.get('f1', 0):.3f} | "
                    f"{pc_n.get('precision', 0):.3f} | {pc_n.get('recall', 0):.3f} | {pc_n.get('f1', 0):.3f} |\n"
                )

            f.write("\n## 2. RQ2 Primitive Consistency Analysis\n\n")
            f.write(f"- **Argmax Agreement Rate:** **{cons['argmax_agreement']['agreement_rate']:.1%}** ({cons['argmax_agreement']['agreement_count']}/{report['sample_size']} items agreed)\n")
            f.write(f"- **Disagreement Count:** {cons['argmax_agreement']['disagreement_count']} items\n")
            f.write(f"- **Single-Belief Coherence (Coherent):** {cons['contradictions']['single_belief_coherent_rate']:.1%} ({cons['contradictions']['single_belief_coherent_count']} items with exactly one Noul $\\ge 0.50$)\n")
            f.write(f"- **Contradiction Rate (Multi-belief $\\ge 0.50$):** **{cons['contradictions']['multi_belief_contradiction_rate']:.1%}** ({cons['contradictions']['multi_belief_contradiction_count']} items)\n")
            f.write(f"- **Zero-Belief Rate (No Noul $\\ge 0.50$):** {cons['contradictions']['zero_belief_rate']:.1%} ({cons['contradictions']['zero_belief_count']} items)\n\n")

            f.write("### Probability Correlation across Decision Spaces\n\n")
            f.write("| Class | Pearson $r$ | Pearson $p$-value | Spearman $\\rho$ | Mean Absolute Diff |\n")
            f.write("| :--- | :---: | :---: | :---: | :---: |\n")
            for cls, corr in cons["probability_alignment"].items():
                f.write(
                    f"| `{cls}` | {corr['pearson_r']:.3f} | {corr['pearson_p']:.4f} | "
                    f"{corr['spearman_rho']:.3f} | {corr['mean_abs_diff']:.3f} |\n"
                )

            if cons.get("score_ordinal_alignment"):
                sc = cons["score_ordinal_alignment"]
                f.write("\n### Score vs Choice Ordinal Alignment (3-class Ordered Subset)\n\n")
                f.write(f"- **Evaluated Examples:** {sc['evaluated_count']} (excluding CONFLICT)\n")
                f.write(f"- **Mean Absolute Error (MAE):** {sc['mae']:.3f}\n")
                f.write(f"- **Exact Rounded Match Rate:** {sc['exact_rounded_match_rate']:.1%}\n")
                f.write(f"- **Spearman Rank Correlation ($\\rho$):** **{sc['spearman_rho']:.3f}** ($p = {sc['spearman_p']:.4f}$)\n")
                f.write(f"- **Pearson Correlation ($r$):** {sc['pearson_r']:.3f}\n")

            f.write("\n## 3. Selective Classification & Risk-Coverage (Choice)\n\n")
            f.write("| Confidence Threshold ($\\tau$) | Retained Count | Coverage (%) | Accuracy (%) | Risk (%) |\n")
            f.write("| :---: | :---: | :---: | :---: | :---: |\n")
            for row in c_cal["risk_coverage"]:
                f.write(
                    f"| $\\ge {row['threshold']:.2f}$ | {row['retained_count']}/{row['total_count']} | "
                    f"{row['coverage']:.1%} | **{row['accuracy']:.1%}** | {row['risk']:.1%} |\n"
                )

            f.write("\n## 4. Telemetry Profile\n\n")
            f.write(f"- **Total Calls:** {tel['total_requests']}\n")
            f.write(f"- **Total Decision Records:** {tel['total_records_logged']}\n")
            f.write(f"- **Median Latency ($p_{50}$):** {tel['latency_p50_ms']} ms\n")
            f.write(f"- **95th Percentile Latency ($p_{95}$):** {tel['latency_p95_ms']} ms\n")
            f.write(f"- **99th Percentile Latency ($p_{99}$):** {tel['latency_p99_ms']} ms\n")
            f.write(f"- **Mean Latency:** {tel['latency_mean_ms']} ms\n")
