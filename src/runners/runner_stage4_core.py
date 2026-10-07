"""
runner_stage4_core.py — Stage 4: Core Pure-Sinhala Tasks Execution (N=650).

Executes Phase 1 full probe across four core pure-Sinhala datasets (5 tasks, 650 examples):
  1. Dataset B: SOLD — Sinhala Offensive Language (N=150)
  2. Dataset C1: NSINA Categories (N=100)
  3. Dataset C2: NSINA Media Identification (N=100)
  4. Dataset D: SinhalaMMLU (N=150)
  5. Dataset E: SalAngaBhava (N=150)
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
)
from src.runners.base_runner import BaseRunner


class CorePureSinhalaRunner(BaseRunner):
    """Runner for Stage 4: Core Pure-Sinhala Tasks."""

    def __init__(
        self,
        output_dir: Path | None = None,
        experiment_id: str = "exp_stage4_core_sinhala",
    ) -> None:
        target_dir = output_dir or (RESULTS_DIR / "stage4_core")
        super().__init__(
            experiment_id=experiment_id,
            phase="phase_1_full",
            output_dir=target_dir,
        )
        self.sample_dir = DATA_PROCESSED_DIR / "samples"
        self.raw_log_path = self.output_dir / "stage4_predictions.jsonl"

        # Fresh predictions stream
        if self.raw_log_path.exists():
            self.raw_log_path.unlink()

    def _load_sample(self, filename: str) -> list[dict[str, Any]]:
        """Load frozen JSONL sample file."""
        filepath = self.sample_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Sample file missing: {filepath}")
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
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
        """Call client.ask with automated retry for network resilience."""
        for attempt in range(max_attempts):
            try:
                return self.client.ask(state=state, questions=questions)
            except Exception as exc:
                if attempt == max_attempts - 1:
                    return None
                time.sleep(1.5)
        return None

    # ── 1. Task B: SOLD (N=150) ──────────────────────────────────────────

    def run_sold_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task B: SOLD (Noul primary, Choice primary, Score diagnostic)."""
        p_noul = self.prompts["sold_noul"]["english"]
        p_choice = self.prompts["sold_choice"]["english"]
        p_score = self.prompts["sold_score"]["english"]

        criteria_choice = {"OFF": "Offensive", "NOT": "Not offensive"}
        criteria_score = ["Not offensive", "Offensive"]
        task_records = []

        for r in tqdm(records, desc="[1/5] SOLD (N=150)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "noul_off": Noul(instructions=p_noul),
                "choice_off": Choice(instructions=p_choice, criteria=criteria_choice),
                "score_off": Score(instructions=p_score, criteria=criteria_score),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # Noul decision
            ans_n = res.answers["noul_off"]
            p_true = ans_n.noul
            noul_pred = "OFF" if p_true >= 0.5 else "NOT"
            conf_n = p_true if noul_pred == "OFF" else (1.0 - p_true)

            rec_n = self.format_record(
                record_id=f"rec_p1_{ex_id}_noul",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul,
                criteria_definitions={"False": "Not offensive", "True": "Offensive"},
                gold_label=gold,
                prediction=noul_pred,
                is_correct=(noul_pred == gold),
                confidence=conf_n,
                probabilities={"NOT": round(1.0 - p_true, 4), "OFF": round(p_true, 4)},
                noul_details={"p_true": p_true},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, self.raw_log_path)
            task_records.append(rec_n)

            # Choice decision
            ans_c = res.answers["choice_off"]
            rec_c = self.format_record(
                record_id=f"rec_p1_{ex_id}_choice",
                dataset="dataset_b_sold",
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
            task_records.append(rec_c)

            # Score diagnostic
            ans_s = res.answers["score_off"]
            rec_s = self.format_record(
                record_id=f"rec_p1_{ex_id}_score",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="score",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_score,
                criteria_definitions=criteria_score,
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
            task_records.append(rec_s)

        return task_records

    # ── 2. Task C1: NSINA Categories (N=100) ──────────────────────────────

    def run_nsina_categories_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task C1: NSINA Categories (Choice primary, Noul one-vs-rest)."""
        p_choice = self.prompts["nsina_cat_choice"]["english"]
        p_noul = self.prompts["nsina_cat_noul"]["english"]

        categories = ["Business", "International News", "Local News", "Sports"]
        criteria_choice = {cat: None for cat in categories}
        task_records = []

        for r in tqdm(records, desc="[2/5] NSINA Categories (N=100)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "cat_choice": Choice(instructions=p_choice, criteria=criteria_choice),
                "noul_local": Noul(instructions=p_noul.format(category="Local News")),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["cat_choice"]
            rec_c = self.format_record(
                record_id=f"rec_p1_{ex_id}_choice",
                dataset="dataset_c1_nsina_categories",
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
            task_records.append(rec_c)

            # Noul Local News
            ans_n = res.answers["noul_local"]
            is_local = (gold == "Local News")
            rec_n = self.format_record(
                record_id=f"rec_p1_{ex_id}_noul_local",
                dataset="dataset_c1_nsina_categories",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul.format(category="Local News"),
                criteria_definitions={"True": "Local News", "False": "Other"},
                gold_label="Local News" if is_local else "Other",
                prediction="Local News" if ans_n.noul >= 0.5 else "Other",
                is_correct=((ans_n.noul >= 0.5) == is_local),
                confidence=ans_n.noul if ans_n.noul >= 0.5 else (1.0 - ans_n.noul),
                probabilities={"Other": round(1.0 - ans_n.noul, 4), "Local News": round(ans_n.noul, 4)},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, self.raw_log_path)
            task_records.append(rec_n)

        return task_records

    # ── 3. Task C2: NSINA Media (N=100) ──────────────────────────────────

    def run_nsina_media_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task C2: NSINA Media Identification (Choice 10-way)."""
        p_choice = self.prompts["nsina_media_choice"]["english"]

        media_sources = [
            "www.lankadeepa.lk", "ITN news", "www.vikalpa.org", "Lankatruth",
            "divaina", "Adaderana", "hirunews", "https://sinhala.news.lk",
            "dinamina", "Siyatha News"
        ]
        criteria_choice = {src: None for src in media_sources}
        task_records = []

        for r in tqdm(records, desc="[3/5] NSINA Media (N=100)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "media_choice": Choice(instructions=p_choice, criteria=criteria_choice),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["media_choice"]
            rec_c = self.format_record(
                record_id=f"rec_p1_{ex_id}_choice",
                dataset="dataset_c2_nsina_media",
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
            task_records.append(rec_c)

        return task_records

    # ── 4. Task D: SinhalaMMLU (N=150) ───────────────────────────────────

    def run_sinhalammlu_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task D: SinhalaMMLU 4-option QA (Choice A/B/C/D)."""
        p_choice = self.prompts["mmlu_choice"]["english"]
        criteria_choice = {"A": None, "B": None, "C": None, "D": None}
        task_records = []

        for r in tqdm(records, desc="[4/5] SinhalaMMLU (N=150)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "mmlu_choice": Choice(instructions=p_choice, criteria=criteria_choice),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["mmlu_choice"]
            rec_c = self.format_record(
                record_id=f"rec_p1_{ex_id}_choice",
                dataset="dataset_d_sinhalammlu",
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
            task_records.append(rec_c)

        return task_records

    # ── 5. Task E: SalAngaBhava (N=150) ──────────────────────────────────

    def run_salangabhava_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task E: SalAngaBhava (Score 1-5, Choice 1-5, Noul thresholds)."""
        p_score = self.prompts["salangabhava_score"]["english"]
        p_choice = self.prompts["salangabhava_choice"]["english"]
        p_noul_high = self.prompts["salangabhava_noul_high"]["english"]
        p_noul_low = self.prompts["salangabhava_noul_low"]["english"]

        score_criteria = [
            "Rating level 1",
            "Rating level 2",
            "Rating level 3",
            "Rating level 4",
            "Rating level 5",
        ]
        choice_criteria = {"1": None, "2": None, "3": None, "4": None, "5": None}
        task_records = []

        for r in tqdm(records, desc="[5/5] SalAngaBhava (N=150)"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]
            gold_num = int(gold)

            questions = {
                "score_rating": Score(instructions=p_score, criteria=score_criteria),
                "choice_rating": Choice(instructions=p_choice, criteria=choice_criteria),
                "noul_high": Noul(instructions=p_noul_high),
                "noul_low": Noul(instructions=p_noul_low),
            }

            res = self._ask_with_retry(state, questions)
            if res is None:
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # Score (1-based: score 0 -> 1, score 4 -> 5)
            ans_s = res.answers["score_rating"]
            score_1based = ans_s.score + 1.0
            rec_s = self.format_record(
                record_id=f"rec_p1_{ex_id}_score",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                primitive="score",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_score,
                criteria_definitions=score_criteria,
                gold_label=gold,
                prediction=score_1based,
                is_correct=(round(score_1based) == gold_num),
                confidence=ans_s.confidence,
                probabilities=ans_s.probabilities,
                score_details={"legend": ans_s.legend, "score_raw": ans_s.score, "score_1based": score_1based},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_s, self.raw_log_path)
            task_records.append(rec_s)

            # Choice
            ans_c = res.answers["choice_rating"]
            rec_c = self.format_record(
                record_id=f"rec_p1_{ex_id}_choice",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice,
                criteria_definitions=choice_criteria,
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
            task_records.append(rec_c)

            # Noul High (Rating >= 4)
            ans_nh = res.answers["noul_high"]
            is_high = (gold_num >= 4)
            rec_nh = self.format_record(
                record_id=f"rec_p1_{ex_id}_noul_high",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul_high,
                criteria_definitions={"True": "Rating >= 4", "False": "Rating < 4"},
                gold_label="satisfied" if is_high else "not_satisfied",
                prediction="satisfied" if ans_nh.noul >= 0.5 else "not_satisfied",
                is_correct=((ans_nh.noul >= 0.5) == is_high),
                confidence=ans_nh.noul if ans_nh.noul >= 0.5 else (1.0 - ans_nh.noul),
                probabilities={"not_satisfied": round(1.0 - ans_nh.noul, 4), "satisfied": round(ans_nh.noul, 4)},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_nh, self.raw_log_path)
            task_records.append(rec_nh)

            # Noul Low (Rating <= 2)
            ans_nl = res.answers["noul_low"]
            is_low = (gold_num <= 2)
            rec_nl = self.format_record(
                record_id=f"rec_p1_{ex_id}_noul_low",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul_low,
                criteria_definitions={"True": "Rating <= 2", "False": "Rating > 2"},
                gold_label="dissatisfied" if is_low else "not_dissatisfied",
                prediction="dissatisfied" if ans_nl.noul >= 0.5 else "not_dissatisfied",
                is_correct=((ans_nl.noul >= 0.5) == is_low),
                confidence=ans_nl.noul if ans_nl.noul >= 0.5 else (1.0 - ans_nl.noul),
                probabilities={"not_dissatisfied": round(1.0 - ans_nl.noul, 4), "dissatisfied": round(ans_nl.noul, 4)},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_nl, self.raw_log_path)
            task_records.append(rec_nl)

        return task_records

    # ── Main Run Execution ────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        """Execute Stage 4 across all 5 pure-Sinhala tasks (N=650)."""
        print("=" * 70)
        print("  STAGE 4: CORE PURE-SINHALA TASKS EXECUTION (N=650)")
        print(f"  Target Model: {self.config.model}")
        print(f"  Prediction Stream: {self.raw_log_path}")
        print("=" * 70)

        # 1. Dataset B: SOLD
        rec_b = self._load_sample("dataset_b_sold.jsonl")
        sold_records = self.run_sold_task(rec_b)

        # 2. Dataset C1: NSINA Categories
        rec_c1 = self._load_sample("dataset_c1_nsina_categories.jsonl")
        nsina_cat_records = self.run_nsina_categories_task(rec_c1)

        # 3. Dataset C2: NSINA Media (Deprecated: media outlet classification)
        # rec_c2 = self._load_sample("dataset_c2_nsina_media.jsonl")
        # nsina_media_records = self.run_nsina_media_task(rec_c2)
        nsina_media_records = []

        # 4. Dataset D: SinhalaMMLU
        rec_d = self._load_sample("dataset_d_sinhalammlu.jsonl")
        mmlu_records = self.run_sinhalammlu_task(rec_d)

        # 5. Dataset E: SalAngaBhava
        rec_e = self._load_sample("dataset_e_salangabhava.jsonl")
        salanga_records = self.run_salangabhava_task(rec_e)

        # Quantitative Synthesis
        report = self.analyze_all_tasks(
            sold_records,
            nsina_cat_records,
            nsina_media_records,
            mmlu_records,
            salanga_records,
        )
        return report

    def analyze_all_tasks(
        self,
        sold_records: list[dict[str, Any]],
        nsina_cat_records: list[dict[str, Any]],
        nsina_media_records: list[dict[str, Any]] | None = None,
        mmlu_records: list[dict[str, Any]] = None,
        salanga_records: list[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Synthesize metrics across all five core pure-Sinhala tasks."""
        print("\n" + "─" * 70)
        print("  STAGE 4 SYNTHESIS & METRICS REPORT")
        print("─" * 70)

        task_metrics: dict[str, Any] = {}

        # 1. SOLD Analysis
        sold_choice = [r for r in sold_records if r["primitive"] == "choice"]
        sold_noul = [r for r in sold_records if r["primitive"] == "noul"]
        m_sold_c = compute_classification_metrics(
            [r["gold_label"] for r in sold_choice],
            [r["prediction"] for r in sold_choice],
            labels=["NOT", "OFF"],
        )
        m_sold_n = compute_classification_metrics(
            [r["gold_label"] for r in sold_noul],
            [r["prediction"] for r in sold_noul],
            labels=["NOT", "OFF"],
        )
        cal_sold_c = compute_ece([r["confidence"] for r in sold_choice], [r["is_correct"] for r in sold_choice])
        cal_sold_n = compute_ece([r["confidence"] for r in sold_noul], [r["is_correct"] for r in sold_noul])
        rc_sold_c = compute_selective_risk_coverage([r["confidence"] for r in sold_choice], [r["is_correct"] for r in sold_choice])

        task_metrics["dataset_b_sold"] = {
            "choice": {**m_sold_c, "ece": cal_sold_c["ece"], "risk_coverage": rc_sold_c},
            "noul": {**m_sold_n, "ece": cal_sold_n["ece"]},
        }
        print(f"  [SOLD]            Choice Acc: {m_sold_c['accuracy']:.1%} (F1: {m_sold_c['macro_f1']:.3f}, ECE: {cal_sold_c['ece']:.3f}) | Noul Acc: {m_sold_n['accuracy']:.1%} (F1: {m_sold_n['macro_f1']:.3f}, ECE: {cal_sold_n['ece']:.3f})")

        # 2. NSINA Categories Analysis
        c1_choice = [r for r in nsina_cat_records if r["primitive"] == "choice"]
        m_c1 = compute_classification_metrics(
            [r["gold_label"] for r in c1_choice],
            [r["prediction"] for r in c1_choice],
        )
        cal_c1 = compute_ece([r["confidence"] for r in c1_choice], [r["is_correct"] for r in c1_choice])
        rc_c1 = compute_selective_risk_coverage([r["confidence"] for r in c1_choice], [r["is_correct"] for r in c1_choice])
        task_metrics["dataset_c1_nsina_categories"] = {
            "choice": {**m_c1, "ece": cal_c1["ece"], "risk_coverage": rc_c1}
        }
        print(f"  [NSINA Categories] Choice Acc: {m_c1['accuracy']:.1%} (F1: {m_c1['macro_f1']:.3f}, ECE: {cal_c1['ece']:.3f})")

        # 3. NSINA Media Analysis (Guarded)
        c2_choice = []
        if nsina_media_records:
            c2_choice = [r for r in nsina_media_records if r["primitive"] == "choice"]
            m_c2 = compute_classification_metrics(
                [r["gold_label"] for r in c2_choice],
                [r["prediction"] for r in c2_choice],
            )
            cal_c2 = compute_ece([r["confidence"] for r in c2_choice], [r["is_correct"] for r in c2_choice])
            rc_c2 = compute_selective_risk_coverage([r["confidence"] for r in c2_choice], [r["is_correct"] for r in c2_choice])
            task_metrics["dataset_c2_nsina_media"] = {
                "choice": {**m_c2, "ece": cal_c2["ece"], "risk_coverage": rc_c2}
            }
            print(f"  [NSINA Media]      Choice Acc: {m_c2['accuracy']:.1%} (F1: {m_c2['macro_f1']:.3f}, ECE: {cal_c2['ece']:.3f}) [Random: 10%]")

        # 4. SinhalaMMLU Analysis
        d_choice = [r for r in mmlu_records if r["primitive"] == "choice"]
        m_d = compute_classification_metrics(
            [r["gold_label"] for r in d_choice],
            [r["prediction"] for r in d_choice],
            labels=["A", "B", "C", "D"],
        )
        cal_d = compute_ece([r["confidence"] for r in d_choice], [r["is_correct"] for r in d_choice])
        rc_d = compute_selective_risk_coverage([r["confidence"] for r in d_choice], [r["is_correct"] for r in d_choice])
        task_metrics["dataset_d_sinhalammlu"] = {
            "choice": {**m_d, "ece": cal_d["ece"], "risk_coverage": rc_d}
        }
        print(f"  [SinhalaMMLU]     Choice Acc: {m_d['accuracy']:.1%} (F1: {m_d['macro_f1']:.3f}, ECE: {cal_d['ece']:.3f}) [Random: 25%]")

        # 5. SalAngaBhava Analysis
        e_choice = [r for r in salanga_records if r["primitive"] == "choice"]
        e_score = [r for r in salanga_records if r["primitive"] == "score"]
        e_noul_high = [r for r in salanga_records if "noul_high" in r["record_id"]]
        e_noul_low = [r for r in salanga_records if "noul_low" in r["record_id"]]

        m_e_c = compute_classification_metrics(
            [r["gold_label"] for r in e_choice],
            [r["prediction"] for r in e_choice],
            labels=["1", "2", "3", "4", "5"],
        )
        cal_e_c = compute_ece([r["confidence"] for r in e_choice], [r["is_correct"] for r in e_choice])
        rc_e_c = compute_selective_risk_coverage([r["confidence"] for r in e_choice], [r["is_correct"] for r in e_choice])

        # Score MAE and rounded match rate
        gold_ratings = np.array([int(r["gold_label"]) for r in e_score], dtype=float)
        pred_scores = np.array([float(r["prediction"]) for r in e_score], dtype=float)
        score_mae = float(np.mean(np.abs(gold_ratings - pred_scores)))
        score_rounded_acc = float(np.mean(np.round(pred_scores) == gold_ratings))

        # Noul High & Low accuracy
        acc_high = float(np.mean([r["is_correct"] for r in e_noul_high])) if e_noul_high else 0.0
        acc_low = float(np.mean([r["is_correct"] for r in e_noul_low])) if e_noul_low else 0.0

        task_metrics["dataset_e_salangabhava"] = {
            "choice": {**m_e_c, "ece": cal_e_c["ece"], "risk_coverage": rc_e_c},
            "score": {
                "mae": round(score_mae, 4),
                "rounded_accuracy": round(score_rounded_acc, 4),
                "total_evaluated": len(e_score),
            },
            "noul_high_accuracy": round(acc_high, 4),
            "noul_low_accuracy": round(acc_low, 4),
        }
        print(f"  [SalAngaBhava]    Choice Acc: {m_e_c['accuracy']:.1%} | Score Rounded Acc: {score_rounded_acc:.1%} (MAE: {score_mae:.3f})")
        print(f"                    Noul High (>=4) Acc: {acc_high:.1%} | Noul Low (<=2) Acc: {acc_low:.1%}")

        # All Latencies
        all_records = sold_records + nsina_cat_records + nsina_media_records + mmlu_records + salanga_records
        latencies = [r["latency_ms"] for r in all_records if r.get("latency_ms")]
        lat_arr = np.array(latencies) if latencies else np.array([0.0])

        report = {
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": self.config.model,
            "total_examples": len(sold_choice) + len(c1_choice) + len(c2_choice) + len(d_choice) + len(e_choice),
            "total_records_logged": len(all_records),
            "tasks": task_metrics,
            "telemetry": {
                "latency_p50_ms": round(float(np.percentile(lat_arr, 50)), 1),
                "latency_p95_ms": round(float(np.percentile(lat_arr, 95)), 1),
                "latency_p99_ms": round(float(np.percentile(lat_arr, 99)), 1),
                "latency_mean_ms": round(float(np.mean(lat_arr)), 1),
            },
        }

        # Save JSON
        report_path = self.output_dir / "core_tasks_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        # Save Markdown Summary
        summary_md_path = self.output_dir / "summary.md"
        self._write_summary_markdown(summary_md_path, report)

        print("─" * 70)
        print(f"  ✓ Stage 4 Complete")
        print(f"  Report:      {report_path}")
        print(f"  Summary:     {summary_md_path}")
        print(f"  Predictions: {self.raw_log_path}")
        print("─" * 70)
        return report

    def _write_summary_markdown(self, path: Path, report: dict[str, Any]) -> None:
        """Write a structured Markdown summary of Stage 4 findings."""
        tasks = report["tasks"]
        tel = report["telemetry"]

        with open(path, "w", encoding="utf-8") as f:
            f.write("# Stage 4: Core Pure-Sinhala Tasks — Quantitative Report\n\n")
            f.write(f"**Target Model:** `{report['model']}`  \n")
            f.write(f"**Total Examples Evaluated:** $N = {report['total_examples']}$ across 5 tasks  \n")
            f.write(f"**Total Decision Records:** {report['total_records_logged']}  \n")
            f.write(f"**Timestamp:** `{report['timestamp']}`  \n\n")

            f.write("## 1. Cross-Task Performance Overview\n\n")
            f.write("| Task | Domain | Decision Space | Primary Primitive | Accuracy | Macro-F1 | ECE | Chance Baseline |\n")
            f.write("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")

            t_sold = tasks["dataset_b_sold"]
            f.write(f"| **SOLD** | Social (Twitter) | Binary (`OFF`/`NOT`) | Choice | **{t_sold['choice']['accuracy']:.1%}** | {t_sold['choice']['macro_f1']:.3f} | {t_sold['choice']['ece']:.3f} | 50.0% |\n")
            f.write(f"| *SOLD (Noul)* | Social (Twitter) | Binary (`OFF`/`NOT`) | Noul | **{t_sold['noul']['accuracy']:.1%}** | {t_sold['noul']['macro_f1']:.3f} | {t_sold['noul']['ece']:.3f} | 50.0% |\n")

            t_c1 = tasks["dataset_c1_nsina_categories"]
            f.write(f"| **NSINA Categories** | News Articles | 4 Categories | Choice | **{t_c1['choice']['accuracy']:.1%}** | {t_c1['choice']['macro_f1']:.3f} | {t_c1['choice']['ece']:.3f} | 25.0% |\n")

            if "dataset_c2_nsina_media" in tasks:
                t_c2 = tasks["dataset_c2_nsina_media"]
                f.write(f"| **NSINA Media** | News Articles | 10 Sources | Choice | **{t_c2['choice']['accuracy']:.1%}** | {t_c2['choice']['macro_f1']:.3f} | {t_c2['choice']['ece']:.3f} | 10.0% |\n")

            t_d = tasks["dataset_d_sinhalammlu"]
            f.write(f"| **SinhalaMMLU** | Multi-discipline QA | 4 Options (`A`/`B`/`C`/`D`) | Choice | **{t_d['choice']['accuracy']:.1%}** | {t_d['choice']['macro_f1']:.3f} | {t_d['choice']['ece']:.3f} | 25.0% |\n")

            t_e = tasks["dataset_e_salangabhava"]
            f.write(f"| **SalAngaBhava** | Product Reviews | Ordinal (1–5 Stars) | Choice | **{t_e['choice']['accuracy']:.1%}** | {t_e['choice']['macro_f1']:.3f} | {t_e['choice']['ece']:.3f} | 20.0% |\n")
            f.write(f"| *SalAngaBhava (Score)* | Product Reviews | Ordinal (1–5 Stars) | Score | **{t_e['score']['rounded_accuracy']:.1%}** (MAE={t_e['score']['mae']:.3f}) | — | — | 20.0% |\n\n")

            f.write("## 2. Detailed Task Syntheses\n\n")

            # SOLD
            f.write("### 2.1 SOLD: Sinhala Offensive Language Detection ($N=150$)\n")
            f.write(f"- **Choice Accuracy:** {t_sold['choice']['accuracy']:.1%} (Macro-F1: {t_sold['choice']['macro_f1']:.3f})\n")
            f.write(f"- **Noul Accuracy:** {t_sold['noul']['accuracy']:.1%} (Macro-F1: {t_sold['noul']['macro_f1']:.3f})\n")
            f.write(f"- **Calibration (ECE):** Choice: {t_sold['choice']['ece']:.3f} vs Noul: {t_sold['noul']['ece']:.3f}\n\n")

            # NSINA Categories
            f.write("### 2.2 NSINA Categories: News Classification ($N=100$)\n")
            f.write(f"- **Choice Accuracy:** {t_c1['choice']['accuracy']:.1%} (Macro-F1: {t_c1['choice']['macro_f1']:.3f})\n")
            f.write(f"- **ECE:** {t_c1['choice']['ece']:.3f}\n\n")

            # NSINA Media
            f.write("### 2.3 NSINA Media: Publisher Identification 10-way ($N=100$)\n")
            f.write(f"- **Choice Accuracy:** {t_c2['choice']['accuracy']:.1%} (Macro-F1: {t_c2['choice']['macro_f1']:.3f})\n")
            f.write(f"- **ECE:** {t_c2['choice']['ece']:.3f}\n\n")

            # SinhalaMMLU
            f.write("### 2.4 SinhalaMMLU: Academic Multiple-Choice QA ($N=150$)\n")
            f.write(f"- **Choice Accuracy:** {t_d['choice']['accuracy']:.1%} (Macro-F1: {t_d['choice']['macro_f1']:.3f})\n")
            f.write(f"- **ECE:** {t_d['choice']['ece']:.3f}\n\n")

            # SalAngaBhava
            f.write("### 2.5 SalAngaBhava: Product Review Ratings ($N=150$)\n")
            f.write(f"- **Choice Accuracy (Exact 1–5):** {t_e['choice']['accuracy']:.1%}\n")
            f.write(f"- **Score Rounded Accuracy:** {t_e['score']['rounded_accuracy']:.1%} (Mean Absolute Error: {t_e['score']['mae']:.3f})\n")
            f.write(f"- **Noul High-Satisfaction (Rating $\\ge 4$):** {t_e['noul_high_accuracy']:.1%}\n")
            f.write(f"- **Noul Low-Satisfaction (Rating $\\le 2$):** {t_e['noul_low_accuracy']:.1%}\n\n")

            f.write("## 3. Telemetry Profile\n\n")
            f.write(f"- **Median Latency ($p_{50}$):** {tel['latency_p50_ms']} ms\n")
            f.write(f"- **95th Percentile Latency ($p_{95}$):** {tel['latency_p95_ms']} ms\n")
            f.write(f"- **99th Percentile Latency ($p_{99}$):** {tel['latency_p99_ms']} ms\n")
            f.write(f"- **Mean Latency:** {tel['latency_mean_ms']} ms\n")
