"""
runner_scaleup.py — High-Throughput Concurrent Runner for Phase 2 Full Benchmark Census.

Executes all scaled benchmark tasks with:
  - ThreadPoolExecutor concurrency (default max_workers=8)
  - Rate limiting (token bucket / pacing <= 25 req/sec)
  - Idempotent checkpointing (skips completed record_ids on resume)
  - Thread-safe append-only JSONL logging
  - Live progress display via tqdm
"""

from __future__ import annotations

import concurrent.futures
import io
import json
import os
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

# Force UTF-8 on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
from tqdm import tqdm
from typesafe_sdk import Choice, Noul, Score

from src.client import JevCallResult, JevClient
from src.config import (
    DATA_PROCESSED_SCALED_DIR,
    ExperimentConfig,
    RESULTS_SCALEUP_DIR,
    RESULTS_SCALEUP_LOGS_DIR,
)
from src.runners.base_runner import BaseRunner


class RateLimiter:
    """Thread-safe token bucket rate limiter."""

    def __init__(self, max_rate: float = 25.0) -> None:
        self.max_rate = max_rate
        self.min_interval = 1.0 / max_rate if max_rate > 0 else 0.0
        self._last_call = time.time()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.min_interval <= 0:
            return
        with self._lock:
            now = time.time()
            elapsed = now - self._last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self._last_call = time.time()


class ScaledCensusRunner(BaseRunner):
    """Production concurrent runner for Phase 2 full benchmark census."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        output_dir: Path | None = None,
        max_workers: int | None = None,
        rate_limit_rps: float | None = None,
    ) -> None:
        super().__init__(
            config=config,
            experiment_id="exp_phase2_census",
            phase="phase_2_scaled_census",
            output_dir=output_dir or RESULTS_SCALEUP_DIR,
        )
        self.logs_dir = RESULTS_SCALEUP_LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.scaled_dir = DATA_PROCESSED_SCALED_DIR

        self.max_workers = max_workers or self.config.max_workers
        self.rate_limiter = RateLimiter(max_rate=rate_limit_rps or self.config.rate_limit_rps)

    def _load_scaled_file(self, filename: str) -> list[dict[str, Any]]:
        """Load frozen scaled JSONL dataset."""
        filepath = self.scaled_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Scaled dataset file missing: {filepath}")
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def _execute_parallel(
        self,
        records: list[dict[str, Any]],
        worker_fn: Callable[[dict[str, Any]], None],
        desc: str,
    ) -> None:
        """Execute a worker function concurrently over records with tqdm progress."""
        with tqdm(total=len(records), desc=desc) as pbar:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(worker_fn, r): r for r in records}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as exc:
                        print(f"\n[Worker Error]: {exc}")
                    pbar.update(1)

    # ── Task 1: SOLD (N=2,500) ──────────────────────────────────────────────

    def run_sold(self, pilot_n: int | None = None) -> Path:
        """Task 1: SOLD Offensive Language Census (N=2,500)."""
        log_file = self.logs_dir / "sold_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_b_sold_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_noul = self.prompts["sold_noul"]["english"]
        p_choice = self.prompts["sold_choice"]["english"]
        p_score = self.prompts["sold_score"]["english"]
        criteria_choice = {"OFF": "Offensive", "NOT": "Not offensive"}
        criteria_score = ["Not offensive", "Offensive"]

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_noul = f"rec_p2_{ex_id}_noul"
            if rec_id_noul in completed:
                return

            state = r["text"]
            gold = r["gold_label"]
            questions = {
                "noul_off": Noul(instructions=p_noul),
                "choice_off": Choice(instructions=p_choice, criteria=criteria_choice),
                "score_off": Score(instructions=p_score, criteria=criteria_score),
            }

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # 1. Noul record
            p_true = res.answers["noul_off"].noul
            pred_n = "OFF" if p_true >= 0.5 else "NOT"
            conf_n = p_true if pred_n == "OFF" else (1.0 - p_true)
            rec_n = self.format_record(
                record_id=rec_id_noul,
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_noul,
                criteria_definitions={"False": "Not offensive", "True": "Offensive"},
                gold_label=gold,
                prediction=pred_n,
                is_correct=(pred_n == gold),
                confidence=conf_n,
                probabilities={"OFF": p_true, "NOT": 1.0 - p_true},
                noul_details={"p_true": p_true},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, log_file)

            # 2. Choice record
            ans_c = res.answers["choice_off"]
            pred_c = ans_c.choice
            conf_c = ans_c.confidence
            rec_c = self.format_record(
                record_id=f"rec_p2_{ex_id}_choice",
                dataset="dataset_b_sold",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=pred_c,
                is_correct=(pred_c == gold),
                confidence=conf_c,
                probabilities=ans_c.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c, log_file)

            # 3. Score record
            ans_s = res.answers["score_off"]
            rec_s = self.format_record(
                record_id=f"rec_p2_{ex_id}_score",
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
                score_details={"expectation": ans_s.score},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_s, log_file)

        self._execute_parallel(records, worker, desc=f"SOLD Census (N={len(records)})")
        return log_file

    # ── Task 2: NSINA Categories (N=1,200) ───────────────────────────────────

    def run_nsina_categories(self, pilot_n: int | None = None) -> Path:
        """Task 2: NSINA Categories Census (N=1,200)."""
        log_file = self.logs_dir / "nsina_categories_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_c1_nsina_categories_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_choice = self.prompts["nsina_cat_choice"]["english"]
        p_noul_tmpl = self.prompts["nsina_cat_noul"]["english"]
        categories = ["Business", "International News", "Local News", "Sports"]
        criteria_choice = {cat: None for cat in categories}

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_choice = f"rec_p2_{ex_id}_choice"
            if rec_id_choice in completed:
                return

            state = r["text"]
            gold = r["gold_label"]
            questions = {
                "category": Choice(instructions=p_choice, criteria=criteria_choice),
                "is_sports": Noul(instructions=p_noul_tmpl.format(category="Sports")),
                "is_local": Noul(instructions=p_noul_tmpl.format(category="Local News")),
            }

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["category"]
            pred_c = ans_c.choice
            conf_c = ans_c.confidence

            rec_c = self.format_record(
                record_id=rec_id_choice,
                dataset="dataset_c1_nsina_categories",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=pred_c,
                is_correct=(pred_c == gold),
                confidence=conf_c,
                probabilities=ans_c.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c, log_file)

        self._execute_parallel(records, worker, desc=f"NSINA Categories (N={len(records)})")
        return log_file

    # ── Task 3: NSINA Media (N=1,000) ────────────────────────────────────────

    def run_nsina_media(self, pilot_n: int | None = None) -> Path:
        """Task 3: NSINA Media Identification Census (N=1,000)."""
        log_file = self.logs_dir / "nsina_media_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_c2_nsina_media_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_choice = self.prompts["nsina_media_choice"]["english"]
        sources = [
            "www.lankadeepa.lk", "ITN news", "www.vikalpa.org", "Lankatruth",
            "divaina", "Adaderana", "hirunews", "https://sinhala.news.lk",
            "dinamina", "Siyatha News",
        ]
        criteria_choice = {src: None for src in sources}

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_choice = f"rec_p2_{ex_id}_choice"
            if rec_id_choice in completed:
                return

            state = r["text"]
            gold = r["gold_label"]
            questions = {
                "source": Choice(instructions=p_choice, criteria=criteria_choice)
            }

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["source"]
            pred_c = ans_c.choice
            conf_c = ans_c.confidence

            rec_c = self.format_record(
                record_id=rec_id_choice,
                dataset="dataset_c2_nsina_media",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=pred_c,
                is_correct=(pred_c == gold),
                confidence=conf_c,
                probabilities=ans_c.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c, log_file)

        self._execute_parallel(records, worker, desc=f"NSINA Media (N={len(records)})")
        return log_file

    # ── Task 4: SinhalaMMLU (N=1,850) ────────────────────────────────────────

    def run_mmlu(self, pilot_n: int | None = None) -> Path:
        """Task 4: SinhalaMMLU 14-Subject Curriculum Census (N=1,850)."""
        log_file = self.logs_dir / "mmlu_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_d_sinhalammlu_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_choice = self.prompts["mmlu_choice"]["english"]
        criteria_choice = {"A": None, "B": None, "C": None, "D": None}

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_choice = f"rec_p2_{ex_id}_choice"
            if rec_id_choice in completed:
                return

            state = r["text"]
            gold = r["gold_label"]
            questions = {
                "answer": Choice(instructions=p_choice, criteria=criteria_choice)
            }

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            ans_c = res.answers["answer"]
            pred_c = ans_c.choice
            conf_c = ans_c.confidence

            meta = r.get("metadata", {})
            rec_c = self.format_record(
                record_id=rec_id_choice,
                dataset="dataset_d_sinhalammlu",
                example_id=ex_id,
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_choice,
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=pred_c,
                is_correct=(pred_c == gold),
                confidence=conf_c,
                probabilities=ans_c.probabilities,
                score_details={
                    "subject": meta.get("subject", "general"),
                    "category": meta.get("category", "general"),
                },
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_c, log_file)

        self._execute_parallel(records, worker, desc=f"SinhalaMMLU Census (N={len(records)})")
        return log_file

    # ── Task 5: Sinhala Sentiment (N=1,810) ───────────────────────────────────

    def run_sentiment(self, pilot_n: int | None = None) -> Path:
        """Task 5: Sinhala News-Comment Sentiment Census (N=1,810)."""
        log_file = self.logs_dir / "sentiment_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_a_sentiment_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_choice = self.prompts["sentiment_choice"]["english"]
        p_pos = self.prompts["sentiment_noul"]["positive"]["english"]
        p_neg = self.prompts["sentiment_noul"]["negative"]["english"]
        p_neu = self.prompts["sentiment_noul"]["neutral"]["english"]
        p_cnf = self.prompts["sentiment_noul"]["conflict"]["english"]
        p_score = self.prompts["sentiment_score"]["english"]

        criteria_choice = {
            "POSITIVE": None, "NEGATIVE": None, "NEUTRAL": None, "CONFLICT": None
        }
        score_criteria = ["Negative sentiment", "Neutral sentiment", "Positive sentiment"]

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_choice = f"rec_p2_{ex_id}_choice"
            if rec_id_choice in completed:
                return

            state = r["text"]
            gold = r["gold_label"]
            questions: dict[str, Any] = {
                "choice_sentiment": Choice(instructions=p_choice, criteria=criteria_choice),
                "noul_pos": Noul(instructions=p_pos),
                "noul_neg": Noul(instructions=p_neg),
                "noul_neu": Noul(instructions=p_neu),
                "noul_cnf": Noul(instructions=p_cnf),
            }
            if gold in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
                questions["score_sentiment"] = Score(instructions=p_score, criteria=score_criteria)

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # 1. Choice record
            ans_c = res.answers["choice_sentiment"]
            rec_c = self.format_record(
                record_id=rec_id_choice,
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
            self.log_record(rec_c, log_file)

            # 2. Noul Argmax record
            noul_probs = {
                "POSITIVE": res.answers["noul_pos"].noul,
                "NEGATIVE": res.answers["noul_neg"].noul,
                "NEUTRAL": res.answers["noul_neu"].noul,
                "CONFLICT": res.answers["noul_cnf"].noul,
            }
            pred_n = max(noul_probs, key=noul_probs.get)
            conf_n = noul_probs[pred_n]
            rec_n = self.format_record(
                record_id=f"rec_p2_{ex_id}_noul",
                dataset="dataset_a_sentiment",
                example_id=ex_id,
                primitive="noul",
                prompt_variant="english_instruction",
                state_text=state,
                instructions="Argmax of 4 one-vs-rest Noul questions",
                criteria_definitions=criteria_choice,
                gold_label=gold,
                prediction=pred_n,
                is_correct=(pred_n == gold),
                confidence=conf_n,
                probabilities=noul_probs,
                noul_details=noul_probs,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_n, log_file)

            # 3. Score record if available
            if "score_sentiment" in res.answers:
                ans_s = res.answers["score_sentiment"]
                rec_s = self.format_record(
                    record_id=f"rec_p2_{ex_id}_score",
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
                    score_details={"expectation": ans_s.score},
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_s, log_file)

        self._execute_parallel(records, worker, desc=f"Sentiment Census (N={len(records)})")
        return log_file

    # ── Task 6: SalAngaBhava (N=1,074) ───────────────────────────────────────

    def run_salangabhava(self, pilot_n: int | None = None) -> Path:
        """Task 6: SalAngaBhava Ordinal & Script Census (N=1,074)."""
        log_file = self.logs_dir / "salangabhava_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_e_salangabhava_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_score = self.prompts["salangabhava_score"]["english"]
        p_choice = self.prompts["salangabhava_choice"]["english"]
        p_high = self.prompts["salangabhava_noul_high"]["english"]
        p_low = self.prompts["salangabhava_noul_low"]["english"]

        score_criteria = [
            "1 star: Terrible", "2 stars: Poor", "3 stars: Average",
            "4 stars: Good", "5 stars: Excellent",
        ]
        choice_criteria = {"1": None, "2": None, "3": None, "4": None, "5": None}

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_choice = f"rec_p2_{ex_id}_choice"
            if rec_id_choice in completed:
                return

            state = r["text"]
            gold = r["gold_label"]
            gold_int = int(gold)
            script_type = r.get("metadata", {}).get("review_type", "Pure_Sinhala")

            questions = {
                "score_rating": Score(instructions=p_score, criteria=score_criteria),
                "choice_rating": Choice(instructions=p_choice, criteria=choice_criteria),
                "noul_high": Noul(instructions=p_high),
                "noul_low": Noul(instructions=p_low),
            }

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # 1. Choice record
            ans_c = res.answers["choice_rating"]
            rec_c = self.format_record(
                record_id=rec_id_choice,
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                script_type=script_type,
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
            self.log_record(rec_c, log_file)

            # 2. Score record
            ans_s = res.answers["score_rating"]
            rec_s = self.format_record(
                record_id=f"rec_p2_{ex_id}_score",
                dataset="dataset_e_salangabhava",
                example_id=ex_id,
                script_type=script_type,
                primitive="score",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_score,
                criteria_definitions=score_criteria,
                gold_label=gold,
                prediction=ans_s.score + 1.0,  # Map 0..4 expectation to 1..5 scale
                is_correct=(round(ans_s.score + 1.0) == gold_int),
                confidence=ans_s.confidence,
                probabilities=ans_s.probabilities,
                score_details={"expectation_1to5": ans_s.score + 1.0},
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_s, log_file)

        self._execute_parallel(records, worker, desc=f"SalAngaBhava (N={len(records)})")
        return log_file

    # ── Task 7: CMCS Multi-Task (N=2,000) ────────────────────────────────────

    def run_cmcs(self, pilot_n: int | None = None) -> Path:
        """Task 7: Sinhala-English CMCS Multi-Task Stress Track (N=2,000)."""
        log_file = self.logs_dir / "cmcs_predictions.jsonl"
        completed = self.load_completed_record_ids(log_file)
        records = self._load_scaled_file("dataset_f_cmcs_scaled.jsonl")
        if pilot_n:
            records = records[:pilot_n]

        p_sent = self.prompts["cmcs_sentiment_choice"]["english"]
        p_hum = self.prompts["cmcs_humour_choice"]["english"]
        p_hate = self.prompts["cmcs_hate_choice"]["english"]
        p_asp = self.prompts["cmcs_aspect_noul"]["english"]

        sent_criteria = {"POSITIVE": None, "NEGATIVE": None, "NEUTRAL": None, "CONFLICT": None}
        hum_criteria = {"HUMOROUS": None, "NON-HUMOROUS": None}
        hate_criteria = {"NOT OFFENSIVE": None, "HATE-INDUCING": None, "ABUSIVE": None}

        aspect_keys = [
            ("Billing or price", "billing"),
            ("Customer service", "customer service"),
            ("Data", "data"),
            ("Network", "network"),
            ("Package", "package"),
            ("Service or product", "service"),
        ]

        def worker(r: dict[str, Any]) -> None:
            ex_id = r["example_id"]
            rec_id_sent = f"rec_p2_{ex_id}_sent"
            if rec_id_sent in completed:
                return

            state = r["text"]
            meta = r.get("metadata", {})
            gold_sent = meta.get("sentiment", "NEUTRAL")
            gold_hum = meta.get("humor", "NON-HUMOROUS")
            gold_hate = meta.get("hate_speech", "NOT OFFENSIVE")

            questions = {
                "sentiment": Choice(instructions=p_sent, criteria=sent_criteria),
                "humour": Choice(instructions=p_hum, criteria=hum_criteria),
                "hate": Choice(instructions=p_hate, criteria=hate_criteria),
            }
            # Add 6 aspect Nouls
            for _, asp_name in aspect_keys:
                questions[f"noul_{asp_name}"] = Noul(instructions=p_asp.format(aspect=asp_name))

            self.rate_limiter.acquire()
            res = self.client.ask(state=state, questions=questions)
            usage = res.usage.model_dump() if hasattr(res.usage, "model_dump") else {"input_tokens": 0, "output_tokens": 0}

            # Sentiment record
            ans_sent = res.answers["sentiment"]
            rec_sent = self.format_record(
                record_id=rec_id_sent,
                dataset="dataset_f_cmcs",
                example_id=ex_id,
                script_type="code_mixed",
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_sent,
                criteria_definitions=sent_criteria,
                gold_label=gold_sent,
                prediction=ans_sent.choice,
                is_correct=(ans_sent.choice == gold_sent),
                confidence=ans_sent.confidence,
                probabilities=ans_sent.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_sent, log_file)

            # Humour record
            ans_hum = res.answers["humour"]
            rec_hum = self.format_record(
                record_id=f"rec_p2_{ex_id}_humour",
                dataset="dataset_f_cmcs",
                example_id=ex_id,
                script_type="code_mixed",
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_hum,
                criteria_definitions=hum_criteria,
                gold_label=gold_hum,
                prediction=ans_hum.choice,
                is_correct=(ans_hum.choice == gold_hum),
                confidence=ans_hum.confidence,
                probabilities=ans_hum.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_hum, log_file)

            # Hate speech record
            ans_hate = res.answers["hate"]
            rec_hate = self.format_record(
                record_id=f"rec_p2_{ex_id}_hate",
                dataset="dataset_f_cmcs",
                example_id=ex_id,
                script_type="code_mixed",
                primitive="choice",
                prompt_variant="english_instruction",
                state_text=state,
                instructions=p_hate,
                criteria_definitions=hate_criteria,
                gold_label=gold_hate,
                prediction=ans_hate.choice,
                is_correct=(ans_hate.choice == gold_hate),
                confidence=ans_hate.confidence,
                probabilities=ans_hate.probabilities,
                latency_ms=res.latency_ms,
                usage=usage,
                model=res.model,
            )
            self.log_record(rec_hate, log_file)

            # Aspect Nouls
            active_aspects = meta.get("active_aspects", [])
            for raw_k, asp_name in aspect_keys:
                ans_asp = res.answers[f"noul_{asp_name}"]
                p_asp_true = ans_asp.noul
                pred_asp = "TRUE" if p_asp_true >= 0.5 else "FALSE"
                gold_asp = "TRUE" if raw_k in active_aspects else "FALSE"
                rec_asp = self.format_record(
                    record_id=f"rec_p2_{ex_id}_asp_{asp_name}",
                    dataset="dataset_f_cmcs",
                    example_id=ex_id,
                    script_type="code_mixed",
                    primitive="noul",
                    prompt_variant="english_instruction",
                    state_text=state,
                    instructions=f"Aspect: {asp_name}",
                    criteria_definitions={"True": asp_name, "False": "absent"},
                    gold_label=gold_asp,
                    prediction=pred_asp,
                    is_correct=(pred_asp == gold_asp),
                    confidence=p_asp_true if pred_asp == "TRUE" else (1.0 - p_asp_true),
                    probabilities={"TRUE": p_asp_true, "FALSE": 1.0 - p_asp_true},
                    noul_details={"aspect": asp_name, "p_true": p_asp_true},
                    latency_ms=res.latency_ms,
                    usage=usage,
                    model=res.model,
                )
                self.log_record(rec_asp, log_file)

        self._execute_parallel(records, worker, desc=f"CMCS Multi-Task (N={len(records)})")
        return log_file

    def run(self, pilot_n: int | None = None) -> dict[str, Any]:
        """Execute all scaled tasks sequentially or with a pilot limit."""
        print("=" * 70)
        print("  PHASE 2: FULL BENCHMARK CENSUS CONCURRENT EXECUTION")
        print(f"  Model: {self.config.model}  |  Workers: {self.max_workers}  |  Max RPS: {self.rate_limiter.max_rate}")
        print(f"  Log Directory: {self.logs_dir}")
        if pilot_n:
            print(f"  *** PILOT MODE ENABLED: N={pilot_n} items per task ***")
        print("=" * 70)

        tasks = [
            ("SOLD (Twitter)", self.run_sold),
            ("NSINA Categories", self.run_nsina_categories),
            # ("NSINA Media Identification", self.run_nsina_media),  # Deprecated: media outlet classification
            ("SinhalaMMLU Academic QA", self.run_mmlu),
            ("News-Comment Sentiment", self.run_sentiment),
            ("SalAngaBhava Reviews", self.run_salangabhava),
            ("CMCS Multi-Task", self.run_cmcs),
        ]

        summary = {}
        for name, task_fn in tasks:
            print(f"\n▶ Launching Task: {name}")
            t0 = time.time()
            log_path = task_fn(pilot_n=pilot_n)
            dt = time.time() - t0
            print(f"  ✓ Completed in {dt:.1f}s — Log: {log_path.name}")
            summary[name] = {"log": str(log_path), "duration_sec": round(dt, 1)}

        print("\n" + "=" * 70)
        print("  ALL SCALED CENSUS TASKS COMPLETED SUCCESSFULLY")
        print("=" * 70)
        return summary
