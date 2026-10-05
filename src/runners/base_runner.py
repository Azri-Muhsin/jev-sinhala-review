"""
base_runner.py — Abstract Base Runner for Jev Experiments.

Provides:
  - Configuration & prompt loading (prompts.yaml)
  - JevClient lifecycle management
  - Strict experiment record schema formatting (Section 5)
  - Immutable JSONL prediction stream logging
  - Telemetry & error handling
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.client import JevCallResult, JevClient
from src.config import (
    CONFIGS_DIR,
    DATA_PROCESSED_DIR,
    ExperimentConfig,
    RESULTS_DIR,
    RESULTS_RAW_LOGS_DIR,
    RESULTS_SMOKE_DIR,
    load_yaml,
)
from src.loaders.base import DatasetRecord


class BaseRunner(ABC):
    """Abstract base class for all experiment runners."""

    def __init__(
        self,
        config: ExperimentConfig | None = None,
        prompts_file: str = "prompts.yaml",
        experiment_id: str = "exp_probe_v1",
        phase: str = "phase_0_smoke",
        output_dir: Path | None = None,
    ) -> None:
        self.config = config or ExperimentConfig.from_yaml()
        self.prompts = load_yaml(prompts_file)
        self.experiment_id = experiment_id
        self.phase = phase
        self.output_dir = output_dir or RESULTS_RAW_LOGS_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.client = JevClient(config=self.config)
        self.records_logged: int = 0
        self._lock = threading.Lock()

    def close(self) -> None:
        """Close the underlying client connection."""
        self.client.close()

    def __enter__(self) -> BaseRunner:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def format_record(
        self,
        *,
        record_id: str,
        dataset: str,
        example_id: str,
        language: str = "si",
        script_type: str = "pure_sinhala",
        primitive: str,
        prompt_variant: str,
        state_text: str,
        instructions: str,
        criteria_definitions: dict[str, Any] | list[str] | None,
        gold_label: str,
        prediction: Any,
        is_correct: bool | None,
        confidence: float,
        probabilities: dict[str, float] | dict[int, float],
        score_details: dict[str, Any] | None = None,
        noul_details: dict[str, Any] | None = None,
        latency_ms: float,
        usage: dict[str, int],
        model: str,
        raw_response_status: int = 200,
        error: str | None = None,
    ) -> dict[str, Any]:
        """Format an interaction dictionary according to the frozen experiment record schema."""
        return {
            "record_id": record_id,
            "experiment_id": self.experiment_id,
            "phase": self.phase,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset": dataset,
            "example_id": example_id,
            "language": language,
            "script_type": script_type,
            "primitive": primitive,
            "prompt_variant": prompt_variant,
            "state_text": state_text,
            "instructions": instructions,
            "criteria_definitions": criteria_definitions,
            "gold_label": gold_label,
            "prediction": prediction,
            "is_correct": is_correct,
            "confidence": round(float(confidence), 4),
            "probabilities": {
                str(k): round(float(v), 4) for k, v in probabilities.items()
            },
            "score_details": score_details,
            "noul_details": noul_details,
            "latency_ms": round(float(latency_ms), 1),
            "usage": usage,
            "model": model,
            "raw_response_status": raw_response_status,
            "error": error,
        }

    def log_record(self, record: dict[str, Any], filepath: Path) -> None:
        """Append a single record to the target JSONL file (thread-safe)."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(line)
            self.records_logged += 1

    def load_completed_record_ids(self, filepath: Path) -> set[str]:
        """Load already completed record IDs from an existing JSONL file for idempotent resume."""
        completed: set[str] = set()
        if not filepath.exists():
            return completed
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "record_id" in obj:
                            completed.add(obj["record_id"])
                    except Exception:
                        continue
        return completed

    @abstractmethod
    def run(self) -> dict[str, Any]:
        """Execute the runner's experiment."""
        ...
