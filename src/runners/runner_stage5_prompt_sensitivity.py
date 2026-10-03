"""
runner_stage5_prompt_sensitivity.py — Stage 5: Language & Prompt Sensitivity Experiment (RQ4).

Isolates whether Jev's performance, confidence, and internal representations change when the
task instruction is presented in English vs. native Sinhala on identical input states:
  - Dataset A: Sentiment (N=50, 4-way Choice & Noul)
  - Dataset B: SOLD (N=50, Binary Noul & Choice)
"""

from __future__ import annotations

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
from typesafe_sdk import Choice, Noul

from src.client import JevCallResult
from src.config import DATA_PROCESSED_DIR, RESULTS_DIR
from src.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_nll,
    compute_prompt_sensitivity,
)
from src.runners.base_runner import BaseRunner


class PromptSensitivityRunner(BaseRunner):
    """Runner for Stage 5: English vs Sinhala Language & Prompt Sensitivity (RQ4)."""

    def __init__(
        self,
        output_dir: Path | None = None,
        experiment_id: str = "exp_stage5_prompt_sensitivity",
        sample_n: int = 50,
    ) -> None:
        target_dir = output_dir or (RESULTS_DIR / "stage5_prompt_sensitivity")
        super().__init__(
            experiment_id=experiment_id,
            phase="phase_1_full",
            output_dir=target_dir,
        )
        self.sample_n = sample_n
        self.sample_a_path = DATA_PROCESSED_DIR / "samples" / "dataset_a_sentiment.jsonl"
        self.sample_b_path = DATA_PROCESSED_DIR / "samples" / "dataset_b_sold.jsonl"
        self.raw_log_path = self.output_dir / "stage5_predictions.jsonl"

        if self.raw_log_path.exists():
            self.raw_log_path.unlink()

    def _load_sample(self, file_path: Path, n: int) -> list[dict[str, Any]]:
        if not file_path.exists():
            raise FileNotFoundError(f"Sample file not found: {file_path}")
        records: list[dict[str, Any]] = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
                if len(records) >= n:
                    break
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

    def run_sentiment_sensitivity(
        self, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run paired English vs. Sinhala evaluation on Dataset A (Sentiment, N=50)."""
        p_choice_en = self.prompts["sentiment_choice"]["english"]
        p_choice_si = self.prompts["sentiment_choice"]["sinhala"]

        p_pos_en = self.prompts["sentiment_noul"]["positive"]["english"]
        p_neg_en = self.prompts["sentiment_noul"]["negative"]["english"]
        p_neu_en = self.prompts["sentiment_noul"]["neutral"]["english"]
        p_cnf_en = self.prompts["sentiment_noul"]["conflict"]["english"]

        p_pos_si = self.prompts["sentiment_noul"]["positive"]["sinhala"]
        p_neg_si = self.prompts["sentiment_noul"]["negative"]["sinhala"]
        p_neu_si = self.prompts["sentiment_noul"]["neutral"]["sinhala"]
        p_cnf_si = self.prompts["sentiment_noul"]["conflict"]["sinhala"]

        criteria_choice = {
            "POSITIVE": None,
            "NEGATIVE": None,
            "NEUTRAL": None,
            "CONFLICT": None,
        }

        records_choice_en = []
        records_choice_si = []
        records_noul_en = []
        records_noul_si = []

        print(f"\n  [1/2] Dataset A Sentiment Sensitivity (N={len(records)})")
        for r in tqdm(records, desc="Dataset A Sentiment (Paired EN vs SI)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                # English condition
                "choice_en": Choice(instructions=p_choice_en, criteria=criteria_choice),
                "noul_pos_en": Noul(instructions=p_pos_en),
                "noul_neg_en": Noul(instructions=p_neg_en),
                "noul_neu_en": Noul(instructions=p_neu_en),
                "noul_cnf_en": Noul(instructions=p_cnf_en),
                # Sinhala condition
                "choice_si": Choice(instructions=p_choice_si, criteria=criteria_choice),
                "noul_pos_si": Noul(instructions=p_pos_si),
                "noul_neg_si": Noul(instructions=p_neg_si),
                "noul_neu_si": Noul(instructions=p_neu_si),
                "noul_cnf_si": Noul(instructions=p_cnf_si),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = (
                res.usage.model_dump()
                if hasattr(res.usage, "model_dump")
                else {"input_tokens": 0, "output_tokens": 0}
            )

            # 1. Choice English
            ans_c_en = res.answers["choice_en"]
            rec_c_en = self.format_record(
                record_id=f"rec_s5_sent_{ex_id}_choice_en",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice_en,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=ans_c_en.choice,
                is_correct=(ans_c_en.choice == gold),
                confidence=ans_c_en.confidence,
                probabilities=ans_c_en.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c_en, self.raw_log_path)
            records_choice_en.append(rec_c_en)

            # 2. Choice Sinhala
            ans_c_si = res.answers["choice_si"]
            rec_c_si = self.format_record(
                record_id=f"rec_s5_sent_{ex_id}_choice_si",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="sinhala_instruction",
                state_text=state,
                instructions=p_choice_si,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=ans_c_si.choice,
                is_correct=(ans_c_si.choice == gold),
                confidence=ans_c_si.confidence,
                probabilities=ans_c_si.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c_si, self.raw_log_path)
            records_choice_si.append(rec_c_si)

            # 3. Noul English
            noul_probs_en = {
                "POSITIVE": res.answers["noul_pos_en"].noul,
                "NEGATIVE": res.answers["noul_neg_en"].noul,
                "NEUTRAL": res.answers["noul_neu_en"].noul,
                "CONFLICT": res.answers["noul_cnf_en"].noul,
            }
            best_noul_en = max(noul_probs_en, key=noul_probs_en.get)
            rec_n_en = self.format_record(
                record_id=f"rec_s5_sent_{ex_id}_noul_en",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_pos_en,
                criteria_definitions={"binary": ["False", "True"]},
                gold_label=gold,
                prediction=best_noul_en,
                is_correct=(best_noul_en == gold),
                confidence=noul_probs_en[best_noul_en],
                probabilities=noul_probs_en,
                noul_details=noul_probs_en,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n_en, self.raw_log_path)
            records_noul_en.append(rec_n_en)

            # 4. Noul Sinhala
            noul_probs_si = {
                "POSITIVE": res.answers["noul_pos_si"].noul,
                "NEGATIVE": res.answers["noul_neg_si"].noul,
                "NEUTRAL": res.answers["noul_neu_si"].noul,
                "CONFLICT": res.answers["noul_cnf_si"].noul,
            }
            best_noul_si = max(noul_probs_si, key=noul_probs_si.get)
            rec_n_si = self.format_record(
                record_id=f"rec_s5_sent_{ex_id}_noul_si",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="sinhala_instruction",
                state_text=state,
                instructions=p_pos_si,
                criteria_definitions={"binary": ["False", "True"]},
                gold_label=gold,
                prediction=best_noul_si,
                is_correct=(best_noul_si == gold),
                confidence=noul_probs_si[best_noul_si],
                probabilities=noul_probs_si,
                noul_details=noul_probs_si,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n_si, self.raw_log_path)
            records_noul_si.append(rec_n_si)

        labels = ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"]
        choice_sens = compute_prompt_sensitivity(records_choice_en, records_choice_si, labels=labels)
        noul_sens = compute_prompt_sensitivity(records_noul_en, records_noul_si, labels=labels)

        return {
            "sample_size": len(records_choice_en),
            "choice_sensitivity": choice_sens,
            "noul_sensitivity": noul_sens,
            "metrics": {
                "choice_en": self._eval_standard(records_choice_en, labels),
                "choice_si": self._eval_standard(records_choice_si, labels),
                "noul_en": self._eval_standard(records_noul_en, labels),
                "noul_si": self._eval_standard(records_noul_si, labels),
            },
        }

    def run_sold_sensitivity(
        self, records: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Run paired English vs. Sinhala evaluation on Dataset B (SOLD, N=50)."""
        p_noul_en = self.prompts["sold_noul"]["english"]
        p_noul_si = self.prompts["sold_noul"]["sinhala"]
        p_choice_en = self.prompts["sold_choice"]["english"]
        p_choice_si = self.prompts["sold_choice"]["sinhala"]

        criteria_choice = {"OFF": "Offensive", "NOT": "Not offensive"}
        labels = ["NOT", "OFF"]

        records_choice_en = []
        records_choice_si = []
        records_noul_en = []
        records_noul_si = []

        print(f"\n  [2/2] Dataset B SOLD Sensitivity (N={len(records)})")
        for r in tqdm(records, desc="Dataset B SOLD (Paired EN vs SI)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                # English condition
                "choice_en": Choice(instructions=p_choice_en, criteria=criteria_choice),
                "noul_en": Noul(instructions=p_noul_en),
                # Sinhala condition
                "choice_si": Choice(instructions=p_choice_si, criteria=criteria_choice),
                "noul_si": Noul(instructions=p_noul_si),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = (
                res.usage.model_dump()
                if hasattr(res.usage, "model_dump")
                else {"input_tokens": 0, "output_tokens": 0}
            )

            # 1. Choice English
            ans_c_en = res.answers["choice_en"]
            rec_c_en = self.format_record(
                record_id=f"rec_s5_sold_{ex_id}_choice_en",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice_en,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=ans_c_en.choice,
                is_correct=(ans_c_en.choice == gold),
                confidence=ans_c_en.confidence,
                probabilities=ans_c_en.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c_en, self.raw_log_path)
            records_choice_en.append(rec_c_en)

            # 2. Choice Sinhala
            ans_c_si = res.answers["choice_si"]
            rec_c_si = self.format_record(
                record_id=f"rec_s5_sold_{ex_id}_choice_si",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="sinhala_instruction",
                state_text=state,
                instructions=p_choice_si,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=ans_c_si.choice,
                is_correct=(ans_c_si.choice == gold),
                confidence=ans_c_si.confidence,
                probabilities=ans_c_si.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c_si, self.raw_log_path)
            records_choice_si.append(rec_c_si)

            # 3. Noul English
            p_true_en = res.answers["noul_en"].noul
            pred_n_en = "OFF" if p_true_en >= 0.5 else "NOT"
            conf_n_en = p_true_en if pred_n_en == "OFF" else (1.0 - p_true_en)
            rec_n_en = self.format_record(
                record_id=f"rec_s5_sold_{ex_id}_noul_en",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul_en,
                criteria_definitions={"False": "Not offensive", "True": "Offensive"},
                gold_label=gold,
                prediction=pred_n_en,
                is_correct=(pred_n_en == gold),
                confidence=conf_n_en,
                probabilities={"NOT": round(1.0 - p_true_en, 4), "OFF": round(p_true_en, 4)},
                noul_details={"p_true": p_true_en},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n_en, self.raw_log_path)
            records_noul_en.append(rec_n_en)

            # 4. Noul Sinhala
            p_true_si = res.answers["noul_si"].noul
            pred_n_si = "OFF" if p_true_si >= 0.5 else "NOT"
            conf_n_si = p_true_si if pred_n_si == "OFF" else (1.0 - p_true_si)
            rec_n_si = self.format_record(
                record_id=f"rec_s5_sold_{ex_id}_noul_si",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="sinhala_instruction",
                state_text=state,
                instructions=p_noul_si,
                criteria_definitions={"False": "Not offensive", "True": "Offensive"},
                gold_label=gold,
                prediction=pred_n_si,
                is_correct=(pred_n_si == gold),
                confidence=conf_n_si,
                probabilities={"NOT": round(1.0 - p_true_si, 4), "OFF": round(p_true_si, 4)},
                noul_details={"p_true": p_true_si},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n_si, self.raw_log_path)
            records_noul_si.append(rec_n_si)

        choice_sens = compute_prompt_sensitivity(records_choice_en, records_choice_si, labels=labels)
        noul_sens = compute_prompt_sensitivity(records_noul_en, records_noul_si, labels=labels)

        return {
            "sample_size": len(records_choice_en),
            "choice_sensitivity": choice_sens,
            "noul_sensitivity": noul_sens,
            "metrics": {
                "choice_en": self._eval_standard(records_choice_en, labels),
                "choice_si": self._eval_standard(records_choice_si, labels),
                "noul_en": self._eval_standard(records_noul_en, labels),
                "noul_si": self._eval_standard(records_noul_si, labels),
            },
        }

    def _eval_standard(
        self, records: list[dict[str, Any]], labels: list[str]
    ) -> dict[str, Any]:
        """Compute accuracy, macro-f1, ece, brier, and nll for a condition."""
        if not records:
            return {}
        y_true = [r["gold_label"] for r in records]
        y_pred = [r["prediction"] for r in records]
        confs = [float(r["confidence"]) for r in records]
        matches = [bool(r["is_correct"]) for r in records]
        probs = [r.get("probabilities", {}) for r in records]

        cls_m = compute_classification_metrics(y_true, y_pred, labels=labels)
        ece_m = compute_ece(confs, matches)
        brier = compute_brier_score(y_true, probs, labels=labels)
        nll = compute_nll(y_true, probs)

        return {
            "accuracy": cls_m.get("accuracy", 0.0),
            "macro_f1": cls_m.get("macro_f1", 0.0),
            "ece": ece_m.get("ece", 0.0),
            "brier": brier,
            "nll": nll,
            "mean_confidence": round(float(np.mean(confs)), 4) if confs else 0.0,
        }

    def generate_markdown_summary(self, report: dict[str, Any]) -> str:
        """Render a readable summary table and analysis in Markdown."""
        sent_c = report["sentiment"]["choice_sensitivity"]
        sent_n = report["sentiment"]["noul_sensitivity"]
        sold_n = report["sold"]["noul_sensitivity"]
        sold_c = report["sold"]["choice_sensitivity"]

        lines = [
            "# Stage 5: Language & Prompt Sensitivity Experiment (RQ4)",
            "",
            f"**Timestamp:** {report['timestamp']}  ",
            f"**Model:** `{report['model']}`  ",
            f"**Dataset A Sample Size:** N={report['sentiment']['sample_size']}  ",
            f"**Dataset B Sample Size:** N={report['sold']['sample_size']}  ",
            "",
            "## 1. Sensitivity Summary (English vs. Sinhala Instruction)",
            "",
            "| Task / Primitive | Acc (EN) | Acc (SI) | Δ Acc | Macro-F1 (EN) | Macro-F1 (SI) | Δ F1 | Agreement | Flip Rate | Conf Drift (SI - EN) | Prob MAD | Cosine Sim |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]

        def _row(name: str, sens: dict[str, Any]) -> str:
            acc = sens["accuracy"]
            f1 = sens["macro_f1"]
            agr = sens["agreement"]
            cnf = sens["confidence"]
            prb = sens["probability_alignment"]
            return (
                f"| **{name}** | {acc['english']:.3f} | {acc['sinhala']:.3f} | {acc['delta']:+.3f} | "
                f"{f1['english']:.3f} | {f1['sinhala']:.3f} | {f1['delta']:+.3f} | "
                f"{agr['agreement_rate']:.1%} | {agr['flip_rate']:.1%} | {cnf['delta']:+.3f} | "
                f"{prb['mean_absolute_difference']:.4f} | {prb['mean_cosine_similarity']:.4f} |"
            )

        lines.append(_row("Sentiment (Choice 4-way)", sent_c))
        lines.append(_row("Sentiment (Noul 4-way)", sent_n))
        lines.append(_row("SOLD (Noul Binary)", sold_n))
        lines.append(_row("SOLD (Choice Binary)", sold_c))

        lines.extend([
            "",
            "## 2. Decision Flip Transitions",
            "",
            "### Dataset A: Sentiment",
            f"- **Choice Flips ({sent_c['agreement']['flip_count']}/{sent_c['sample_size']}):** {sent_c['agreement']['flip_transitions']}",
            f"- **Noul Flips ({sent_n['agreement']['flip_count']}/{sent_n['sample_size']}):** {sent_n['agreement']['flip_transitions']}",
            "",
            "### Dataset B: SOLD",
            f"- **Noul Flips ({sold_n['agreement']['flip_count']}/{sold_n['sample_size']}):** {sold_n['agreement']['flip_transitions']}",
            f"- **Choice Flips ({sold_c['agreement']['flip_count']}/{sold_c['sample_size']}):** {sold_c['agreement']['flip_transitions']}",
            "",
            "## 3. Confidence Drift & Correlation",
            "",
            f"- **Sentiment Choice:** Mean Conf EN = `{sent_c['confidence']['mean_english']:.3f}`, SI = `{sent_c['confidence']['mean_sinhala']:.3f}` (Δ = `{sent_c['confidence']['delta']:+.3f}`, Pearson r = `{sent_c['confidence']['pearson_r']:.3f}`, p = `{sent_c['confidence']['pearson_p']:.4e}`)",
            f"- **Sentiment Noul:** Mean Conf EN = `{sent_n['confidence']['mean_english']:.3f}`, SI = `{sent_n['confidence']['mean_sinhala']:.3f}` (Δ = `{sent_n['confidence']['delta']:+.3f}`, Pearson r = `{sent_n['confidence']['pearson_r']:.3f}`, p = `{sent_n['confidence']['pearson_p']:.4e}`)",
            f"- **SOLD Noul:** Mean Conf EN = `{sold_n['confidence']['mean_english']:.3f}`, SI = `{sold_n['confidence']['mean_sinhala']:.3f}` (Δ = `{sold_n['confidence']['delta']:+.3f}`, Pearson r = `{sold_n['confidence']['pearson_r']:.3f}`, p = `{sold_n['confidence']['pearson_p']:.4e}`)",
            f"- **SOLD Choice:** Mean Conf EN = `{sold_c['confidence']['mean_english']:.3f}`, SI = `{sold_c['confidence']['mean_sinhala']:.3f}` (Δ = `{sold_c['confidence']['delta']:+.3f}`, Pearson r = `{sold_c['confidence']['pearson_r']:.3f}`, p = `{sold_c['confidence']['pearson_p']:.4e}`)",
            "",
            "## 4. Key Takeaways for RQ4",
            "1. **Language Transfer Stability:** Investigates whether framing in English vs. Sinhala preserves classification accuracy and decision boundaries.",
            "2. **Calibration Invariance:** Quantifies whether confidence shifts systematically between English instructions and native Sinhala prompts.",
            "3. **Representation Cosine Alignment:** Assesses whether the model's internal posterior probability distribution remains structurally aligned across language instruction conditions.",
        ])

        return "\n".join(lines)

    def run(self) -> dict[str, Any]:
        """Execute Stage 5 experiment end-to-end."""
        print("=" * 70)
        print("  STAGE 5: LANGUAGE & PROMPT SENSITIVITY EXPERIMENT (RQ4)")
        print(f"  Model: {self.config.model}")
        print(f"  N/Task: {self.sample_n}")
        print(f"  Prediction Stream: {self.raw_log_path}")
        print("=" * 70)

        records_a = self._load_sample(self.sample_a_path, self.sample_n)
        records_b = self._load_sample(self.sample_b_path, self.sample_n)

        sent_res = self.run_sentiment_sensitivity(records_a)
        sold_res = self.run_sold_sensitivity(records_b)

        report = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model,
            "sentiment": sent_res,
            "sold": sold_res,
        }

        # Save json report
        report_path = self.output_dir / "prompt_sensitivity_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved JSON report: {report_path}")

        # Save summary markdown
        summary_md = self.generate_markdown_summary(report)
        summary_path = self.output_dir / "summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_md)
        print(f"  ✓ Saved Markdown summary: {summary_path}")

        print("\n" + "=" * 70)
        print("  STAGE 5 SENSITIVITY HIGHLIGHTS")
        print("=" * 70)
        print(summary_md)

        return report
