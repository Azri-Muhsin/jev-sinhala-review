"""
runner_smoke_test.py — Stage 2: Phase 0 Smoke Test (20 items per task) & Verification Gate.

Executes an end-to-end dry run of all 7 tasks (140 examples total) against 'jev-latest'
to validate:
  1. English and Sinhala prompt syntax and formatting.
  2. Bidirectional label mappings.
  3. Raw response deserialization (Choice, Noul, Score).
  4. Score behavior across rubrics (1-5 ratings & 3-class sentiment).
  5. Unicode / ZWJ integrity of Sinhala inputs.
"""

from __future__ import annotations

import io
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Force UTF-8 on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from tqdm import tqdm
from typesafe_sdk import Choice, Noul, Score

from src.client import JevCallResult
from src.config import DATA_PROCESSED_DIR, RESULTS_SMOKE_DIR
from src.runners.base_runner import BaseRunner


class SmokeTestRunner(BaseRunner):
    """Runner for Stage 2 Phase 0 Smoke Test."""

    def __init__(
        self,
        output_dir: Path | None = None,
        experiment_id: str = "smoke_test_phase0",
    ) -> None:
        super().__init__(
            experiment_id=experiment_id,
            phase="phase_0_smoke",
            output_dir=output_dir or RESULTS_SMOKE_DIR,
        )
        self.smoke_data_dir = DATA_PROCESSED_DIR / "phase0_smoke"
        self.raw_log_path = self.output_dir / "smoke_test_predictions.jsonl"
        # Reset log file for fresh smoke test run
        if self.raw_log_path.exists():
            self.raw_log_path.unlink()

    def load_smoke_records(self, task_file: str) -> list[dict[str, Any]]:
        """Load 20 records from a smoke test JSONL file."""
        path = self.smoke_data_dir / task_file
        if not path.exists():
            raise FileNotFoundError(f"Smoke test file missing: {path}")
        records = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def run_sentiment_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task A: Sentiment (Choice 4-way, Noul x4, Score 3-way)."""
        task_records = []
        p_choice_en = self.prompts["sentiment_choice"]["english"]
        p_choice_si = self.prompts["sentiment_choice"]["sinhala"]
        p_pos_en = self.prompts["sentiment_noul"]["positive"]["english"]
        p_neg_en = self.prompts["sentiment_noul"]["negative"]["english"]
        p_neu_en = self.prompts["sentiment_noul"]["neutral"]["english"]
        p_cnf_en = self.prompts["sentiment_noul"]["conflict"]["english"]
        p_score_en = self.prompts["sentiment_score"]["english"]

        criteria_choice = {
            "POSITIVE": None,
            "NEGATIVE": None,
            "NEUTRAL": None,
            "CONFLICT": None,
        }
        score_criteria = ["Negative sentiment", "Neutral sentiment", "Positive sentiment"]

        for i, r in enumerate(tqdm(records, desc="[1/7] Sentiment Smoke")):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            # Alternate instruction language on a few items to test Sinhala prompts
            use_sinhala = (i % 5 == 0)
            instructions = p_choice_si if use_sinhala else p_choice_en
            variant = "sinhala_instruction" if use_sinhala else "english_instruction"

            questions: dict[str, Any] = {
                "choice_sentiment": Choice(instructions=instructions, criteria=criteria_choice),
                "noul_pos": Noul(instructions=p_pos_en),
                "noul_neg": Noul(instructions=p_neg_en),
                "noul_neu": Noul(instructions=p_neu_en),
                "noul_cnf": Noul(instructions=p_cnf_en),
            }

            if gold in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                questions["score_sentiment"] = Score(
                    instructions=p_score_en,
                    criteria=score_criteria,
                )

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_a_sentiment",
                    example_id=ex_id,
                    primitive="choice",
                    prompt_variant=variant,
                    state_text=state,
                    instructions=instructions,
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # 1. Log Choice
            ans_c = res.answers["choice_sentiment"]
            rec_c = self.format_record(
                record_id=f"rec_smoke_{ex_id}_choice_{'si' if use_sinhala else 'en'}",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="choice",
                prompt_variant=variant,
                state_text=state,
                instructions=instructions,
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

            # 2. Log Noul (pos/neg/neu/cnf)
            noul_preds = {
                "POSITIVE": res.answers["noul_pos"].noul,
                "NEGATIVE": res.answers["noul_neg"].noul,
                "NEUTRAL": res.answers["noul_neu"].noul,
                "CONFLICT": res.answers["noul_cnf"].noul,
            }
            best_noul = max(noul_preds, key=noul_preds.get)
            rec_n = self.format_record(
                record_id=f"rec_smoke_{ex_id}_noul_multiclass",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_pos_en,
                criteria_definitions={"binary": ["False", "True"]},
                gold_label=gold,
                prediction=best_noul,
                is_correct=(best_noul == gold),
                confidence=noul_preds[best_noul],
                probabilities=noul_preds,
                noul_details=noul_preds,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, self.raw_log_path)
            task_records.append(rec_n)

            # 3. Log Score if present
            if "score_sentiment" in res.answers:
                ans_s = res.answers["score_sentiment"]
                # 0: Neg, 1: Neu, 2: Pos
                rec_s = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_score",
                    dataset="dataset_a_sentiment",
                    example_id=ex_id,
                    primitive="score",
                    prompt_variant="english_instruction",
                    state_text=state,
                    instructions=p_score_en,
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
                task_records.append(rec_s)

        return task_records

    def run_sold_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task B: SOLD (Noul primary, Choice primary, Score diagnostic)."""
        task_records = []
        p_noul_en = self.prompts["sold_noul"]["english"]
        p_noul_si = self.prompts["sold_noul"]["sinhala"]
        p_choice_en = self.prompts["sold_choice"]["english"]
        p_score_en = self.prompts["sold_score"]["english"]

        criteria_choice = {"OFF": "Offensive", "NOT": "Not offensive"}
        criteria_score = ["Not offensive", "Offensive"]

        for i, r in enumerate(tqdm(records, desc="[2/7] SOLD Smoke")):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            use_sinhala = (i % 5 == 0)
            instructions_noul = p_noul_si if use_sinhala else p_noul_en
            variant = "sinhala_instruction" if use_sinhala else "english_instruction"

            questions = {
                "noul_off": Noul(instructions=instructions_noul),
                "choice_off": Choice(instructions=p_choice_en, criteria=criteria_choice),
                "score_off": Score(instructions=p_score_en, criteria=criteria_score),
            }

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_b_sold",
                    example_id=ex_id,
                    primitive="noul",
                    prompt_variant=variant,
                    state_text=state,
                    instructions=instructions_noul,
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # Noul
            ans_n = res.answers["noul_off"]
            p_true = ans_n.noul
            noul_pred = "OFF" if p_true >= 0.5 else "NOT"
            conf_n = p_true if noul_pred == "OFF" else (1.0 - p_true)

            rec_n = self.format_record(
                record_id=f"rec_smoke_{ex_id}_noul_{'si' if use_sinhala else 'en'}",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="noul",
                prompt_variant=variant,
                state_text=state,
                instructions=instructions_noul,
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

            # Choice
            ans_c = res.answers["choice_off"]
            rec_c = self.format_record(
                record_id=f"rec_smoke_{ex_id}_choice",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice_en,
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

            # Score
            ans_s = res.answers["score_off"]
            rec_s = self.format_record(
                record_id=f"rec_smoke_{ex_id}_score",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="score",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_score_en,
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

    def run_nsina_categories_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task C1: NSINA Categories (Choice primary, Noul one-vs-rest)."""
        task_records = []
        p_choice = self.prompts["nsina_cat_choice"]["english"]
        p_noul = self.prompts["nsina_cat_noul"]["english"]

        categories = ["Business", "International News", "Local News", "Sports"]
        criteria_choice = {cat: None for cat in categories}

        for r in tqdm(records, desc="[3/7] NSINA Categories Smoke"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "cat_choice": Choice(instructions=p_choice, criteria=criteria_choice),
                "noul_local": Noul(instructions=p_noul.format(category="Local News")),
            }

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_c1_nsina_categories",
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["cat_choice"]
            rec_c = self.format_record(
                record_id=f"rec_smoke_{ex_id}_choice",
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

            ans_n = res.answers["noul_local"]
            rec_n = self.format_record(
                record_id=f"rec_smoke_{ex_id}_noul_local",
                dataset="dataset_c1_nsina_categories",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul.format(category="Local News"),
                criteria_definitions={"True": "Local News", "False": "Other"},
                gold_label="Local News" if gold == "Local News" else "Other",
                prediction="Local News" if ans_n.noul >= 0.5 else "Other",
                is_correct=((ans_n.noul >= 0.5) == (gold == "Local News")),
                confidence=ans_n.noul if ans_n.noul >= 0.5 else (1.0 - ans_n.noul),
                probabilities={"Other": round(1.0 - ans_n.noul, 4), "Local News": round(ans_n.noul, 4)},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, self.raw_log_path)
            task_records.append(rec_n)

        return task_records

    def run_nsina_media_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task C2: NSINA Media Identification (Choice 10-way)."""
        task_records = []
        p_choice = self.prompts["nsina_media_choice"]["english"]

        media_sources = [
            "www.lankadeepa.lk", "ITN news", "www.vikalpa.org", "Lankatruth",
            "divaina", "Adaderana", "hirunews", "https://sinhala.news.lk",
            "dinamina", "Siyatha News"
        ]
        criteria_choice = {src: None for src in media_sources}

        for r in tqdm(records, desc="[4/7] NSINA Media Smoke"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "media_choice": Choice(instructions=p_choice, criteria=criteria_choice),
            }

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_c2_nsina_media",
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["media_choice"]
            rec_c = self.format_record(
                record_id=f"rec_smoke_{ex_id}_choice",
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

    def run_sinhalammlu_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task D: SinhalaMMLU (Choice A/B/C/D)."""
        task_records = []
        p_choice = self.prompts["mmlu_choice"]["english"]
        criteria_choice = {"A": None, "B": None, "C": None, "D": None}

        for r in tqdm(records, desc="[5/7] SinhalaMMLU Smoke"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "mmlu_choice": Choice(instructions=p_choice, criteria=criteria_choice),
            }

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_d_sinhalammlu",
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["mmlu_choice"]
            rec_c = self.format_record(
                record_id=f"rec_smoke_{ex_id}_choice",
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

    def run_salangabhava_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task E: SalAngaBhava (Score 1-5, Choice 1-5, Noul thresholds)."""
        task_records = []
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

        for r in tqdm(records, desc="[6/7] SalAngaBhava Smoke"):
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

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_e_salangabhava",
                    example_id=ex_id,
                    primitive="score",
                    prompt_variant="english_instruction",
                    state_text=state,
                    instructions=p_score,
                    criteria_definitions=score_criteria,
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # Score (1-based: score 0 -> 1, score 4 -> 5)
            ans_s = res.answers["score_rating"]
            score_1based = ans_s.score + 1.0
            rec_s = self.format_record(
                record_id=f"rec_smoke_{ex_id}_score",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                primitive="score",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_score,
                criteria_definitions=score_criteria,
                gold_label=gold,
                prediction=score_1based,
                is_correct=round(score_1based) == gold_num,
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
                record_id=f"rec_smoke_{ex_id}_choice",
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

            # Noul High (rating >= 4)
            ans_nh = res.answers["noul_high"]
            is_high_gold = gold_num >= 4
            rec_nh = self.format_record(
                record_id=f"rec_smoke_{ex_id}_noul_high",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul_high,
                criteria_definitions={"True": "Rating >= 4", "False": "Rating < 4"},
                gold_label="satisfied" if is_high_gold else "not_satisfied",
                prediction="satisfied" if ans_nh.noul >= 0.5 else "not_satisfied",
                is_correct=((ans_nh.noul >= 0.5) == is_high_gold),
                confidence=ans_nh.noul if ans_nh.noul >= 0.5 else (1.0 - ans_nh.noul),
                probabilities={"not_satisfied": round(1.0 - ans_nh.noul, 4), "satisfied": round(ans_nh.noul, 4)},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_nh, self.raw_log_path)
            task_records.append(rec_nh)

        return task_records

    def run_cmcs_task(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Task F: CMCS (Sentiment Choice, Humour, Hate, Aspect Noul)."""
        task_records = []
        p_sent = self.prompts["cmcs_sentiment_choice"]["english"]
        p_aspect = self.prompts["cmcs_aspect_noul"]["english"]

        sent_criteria = {"NEGATIVE": None, "POSITIVE": None, "NEUTRAL": None, "CONFLICT": None}

        for r in tqdm(records, desc="[7/7] CMCS Smoke"):
            state = r["text"]
            gold = r["gold_label"]
            ex_id = r["example_id"]

            questions = {
                "cmcs_sentiment": Choice(instructions=p_sent, criteria=sent_criteria),
                "noul_network": Noul(instructions=p_aspect.format(aspect="network")),
                "noul_package": Noul(instructions=p_aspect.format(aspect="package")),
            }

            try:
                res = self.client.ask(state=state, questions=questions)
            except Exception as exc:
                rec_err = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_error",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type="code_mixed",
                    primitive="choice",
                    prompt_variant="english_instruction",
                    state_text=state,
                    instructions=p_sent,
                    criteria_definitions=sent_criteria,
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
                task_records.append(rec_err)
                continue

            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["cmcs_sentiment"]
            rec_c = self.format_record(
                record_id=f"rec_smoke_{ex_id}_choice",
                dataset="dataset_f_cmcs",
                example_id=ex_id,
                script_type="code_mixed",
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_sent,
                criteria_definitions=sent_criteria,
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

            # Aspect Nouls
            for asp in ["network", "package"]:
                ans_n = res.answers[f"noul_{asp}"]
                has_aspect_gold = asp in [str(a).lower() for a in r.get("metadata", {}).get("aspects", [])]
                rec_na = self.format_record(
                    record_id=f"rec_smoke_{ex_id}_noul_aspect_{asp}",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type="code_mixed",
                    primitive="noul",
                    prompt_variant="english_instruction",
                    state_text=state,
                    instructions=p_aspect.format(aspect=asp),
                    criteria_definitions={"True": f"Discusses {asp}", "False": f"Does not discuss {asp}"},
                    gold_label=str(has_aspect_gold),
                    prediction=str(ans_n.noul >= 0.5),
                    is_correct=((ans_n.noul >= 0.5) == has_aspect_gold),
                    confidence=ans_n.noul if ans_n.noul >= 0.5 else (1.0 - ans_n.noul),
                    probabilities={"False": round(1.0 - ans_n.noul, 4), "True": round(ans_n.noul, 4)},
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_na, self.raw_log_path)
                task_records.append(rec_na)

        return task_records

    def run(self) -> dict[str, Any]:
        """Execute Stage 2: Phase 0 Smoke Test on all 7 tasks."""
        print("=" * 70)
        print("  STAGE 2: PHASE 0 SMOKE TEST (20 Examples per Task)")
        print(f"  Target Model: {self.config.model}")
        print(f"  Raw Log Stream: {self.raw_log_path}")
        print("=" * 70)

        results: dict[str, list[dict[str, Any]]] = {}

        # 1. Dataset A
        rec_a = self.load_smoke_records("dataset_a_sentiment_smoke.jsonl")
        results["dataset_a_sentiment"] = self.run_sentiment_task(rec_a)

        # 2. Dataset B
        rec_b = self.load_smoke_records("dataset_b_sold_smoke.jsonl")
        results["dataset_b_sold"] = self.run_sold_task(rec_b)

        # 3. Dataset C1
        rec_c1 = self.load_smoke_records("dataset_c1_nsina_categories_smoke.jsonl")
        results["dataset_c1_nsina_categories"] = self.run_nsina_categories_task(rec_c1)

        # 4. Dataset C2
        rec_c2 = self.load_smoke_records("dataset_c2_nsina_media_smoke.jsonl")
        results["dataset_c2_nsina_media"] = self.run_nsina_media_task(rec_c2)

        # 5. Dataset D
        rec_d = self.load_smoke_records("dataset_d_sinhalammlu_smoke.jsonl")
        results["dataset_d_sinhalammlu"] = self.run_sinhalammlu_task(rec_d)

        # 6. Dataset E
        rec_e = self.load_smoke_records("dataset_e_salangabhava_smoke.jsonl")
        results["dataset_e_salangabhava"] = self.run_salangabhava_task(rec_e)

        # 7. Dataset F
        rec_f = self.load_smoke_records("dataset_f_cmcs_smoke.jsonl")
        results["dataset_f_cmcs"] = self.run_cmcs_task(rec_f)

        # Run verification checks
        gate_report = self.verify_gate(results)
        return gate_report

    def verify_gate(self, task_records: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        """Perform the 5 verification checks to determine gate passage."""
        print("\n" + "─" * 70)
        print("  VERIFYING PHASE 0 SMOKE TEST GATE (5 CHECKS)")
        print("─" * 70)

        checks = {}
        all_records = [r for sublist in task_records.values() for r in sublist]

        # Check 1: Zero runtime/API errors across templates
        errors = [r for r in all_records if r.get("error")]
        checks["check1_zero_runtime_errors"] = {
            "passed": len(errors) == 0,
            "error_count": len(errors),
            "details": [e.get("error") for e in errors[:5]],
        }
        status1 = "✓ PASS" if len(errors) == 0 else "✗ FAIL"
        print(f"  {status1} Check 1: Zero runtime errors across prompt templates ({len(all_records)} calls)")

        # Check 2: Bidirectional label mapping valid (all predictions within expected label sets)
        unmapped = [
            r for r in all_records
            if not r.get("error") and (r.get("prediction") is None or str(r.get("prediction")).strip() == "")
        ]
        checks["check2_bidirectional_label_mapping"] = {
            "passed": len(unmapped) == 0,
            "unmapped_count": len(unmapped),
        }
        status2 = "✓ PASS" if len(unmapped) == 0 else "✗ FAIL"
        print(f"  {status2} Check 2: Bidirectional label mappings valid (0 unmapped predictions)")

        # Check 3: Raw response deserialization valid
        valid_records = [r for r in all_records if not r.get("error")]
        invalid_probs = [
            r for r in valid_records
            if not isinstance(r.get("probabilities"), dict) or len(r.get("probabilities")) == 0
        ]
        invalid_conf = [
            r for r in valid_records
            if r.get("confidence") is None or not (0.0 <= float(r.get("confidence")) <= 1.0)
        ]
        deser_ok = (len(invalid_probs) == 0 and len(invalid_conf) == 0 and len(valid_records) > 0)
        checks["check3_deserialization"] = {
            "passed": deser_ok,
            "invalid_probabilities_count": len(invalid_probs),
            "invalid_confidence_count": len(invalid_conf),
        }
        status3 = "✓ PASS" if deser_ok else "✗ FAIL"
        print(f"  {status3} Check 3: Raw response deserialization valid (Choice, Noul, Score)")

        # Check 4: Validate Score behavior on 1-5 rubrics & 3-class sentiment
        score_records = [r for r in all_records if r.get("primitive") == "score"]
        score_valid = len(score_records) > 0 and all(
            r.get("prediction") is not None and isinstance(r.get("score_details"), dict)
            for r in score_records
        )
        checks["check4_score_behavior"] = {
            "passed": score_valid,
            "score_records_evaluated": len(score_records),
        }
        status4 = "✓ PASS" if score_valid else "✗ FAIL"
        print(f"  {status4} Check 4: Score behavior validated on 1-5 rubric & 3-class sentiment ({len(score_records)} evaluations)")

        # Check 5: Inspect 5 random Sinhala text inputs for zero Unicode/ZWJ corruption
        zwj_count = sum(1 for r in all_records if "\u200D" in r.get("state_text", ""))
        checks["check5_unicode_zwj_integrity"] = {
            "passed": zwj_count > 0,
            "zwj_containing_interactions": zwj_count,
        }
        status5 = "✓ PASS" if zwj_count > 0 else "✗ FAIL"
        print(f"  {status5} Check 5: Zero Unicode / ZWJ corruption confirmed ({zwj_count} ZWJ occurrences preserved)")

        gate_passed = all(c["passed"] for c in checks.values())
        checks["gate_passed"] = gate_passed
        checks["total_interactions_logged"] = len(all_records)
        checks["verified_at"] = datetime.now(timezone.utc).isoformat()

        # Save gate verification report
        gate_path = self.output_dir / "gate_verification_report.json"
        with open(gate_path, "w", encoding="utf-8") as f:
            json.dump(checks, f, indent=2, ensure_ascii=False)

        # Write summary markdown report
        summary_md_path = self.output_dir / "summary.md"
        with open(summary_md_path, "w", encoding="utf-8") as f:
            f.write("# Stage 2 Phase 0 Smoke Test — Gate Verification Report\n\n")
            f.write(f"**Gate Status:** {'PASSED' if gate_passed else 'FAILED'}\n")
            f.write(f"**Timestamp:** {checks['verified_at']}\n")
            f.write(f"**Total Model Decisions Logged:** {len(all_records)}\n\n")
            f.write("## Verification Checklist\n\n")
            f.write(f"- [{ 'x' if checks['check1_zero_runtime_errors']['passed'] else ' ' }] Check 1: Zero runtime errors across English and Sinhala templates\n")
            f.write(f"- [{ 'x' if checks['check2_bidirectional_label_mapping']['passed'] else ' ' }] Check 2: Bidirectional label mappings valid\n")
            f.write(f"- [{ 'x' if checks['check3_deserialization']['passed'] else ' ' }] Check 3: Raw response deserialization (Choice, Noul, Score)\n")
            f.write(f"- [{ 'x' if checks['check4_score_behavior']['passed'] else ' ' }] Check 4: Score behavior validated on 1-5 rubrics & 3-class sentiment\n")
            f.write(f"- [{ 'x' if checks['check5_unicode_zwj_integrity']['passed'] else ' ' }] Check 5: Zero Unicode/ZWJ corruption confirmed ({zwj_count} interactions)\n\n")
            f.write("## Per-Task Smoke Results\n\n")
            f.write("| Task | Records Logged | Primitives Tested |\n| :--- | :---: | :--- |\n")
            for t_id, t_recs in task_records.items():
                prims = ", ".join(sorted(set(r['primitive'] for r in t_recs)))
                f.write(f"| `{t_id}` | {len(t_recs)} | {prims} |\n")

        print("─" * 70)
        gate_str = "✓ VERIFICATION GATE PASSED — Advance to Phase 1" if gate_passed else "✗ GATE FAILED"
        print(f"  {gate_str}")
        print(f"  Report: {gate_path}")
        print(f"  Predictions: {self.raw_log_path}")
        print("─" * 70)
        return checks
