"""
Environment & Configuration Parser for the Jev-Sinhala Probe.

Loads experiment configuration from YAML files and environment variables.
The model target is fixed to 'jev-latest' via TypeSafe AI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_PROCESSED_SCALED_DIR = DATA_PROCESSED_DIR / "samples_scaled"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_RAW_LOGS_DIR = RESULTS_DIR / "raw_logs"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"
RESULTS_SMOKE_DIR = RESULTS_DIR / "smoke_test"
RESULTS_SCALEUP_DIR = RESULTS_DIR / "scaleup"
RESULTS_SCALEUP_LOGS_DIR = RESULTS_SCALEUP_DIR / "raw_logs"

# ---------------------------------------------------------------------------
# Constants (non-negotiable)
# ---------------------------------------------------------------------------

MODEL_NAME: str = "jev-latest"
RANDOM_SEED: int = 42
PHASE0_N_PER_TASK: int = 20
ECE_NUM_BINS: int = 10
PHASE2_ECE_NUM_BINS: int = 20

# Phase 1 sample sizes per dataset
PHASE1_SAMPLE_SIZES: dict[str, int] = {
    "dataset_a_sentiment": 150,
    "dataset_b_sold": 150,
    "dataset_c1_nsina_categories": 100,
    "dataset_c2_nsina_media": 100,
    "dataset_d_sinhalammlu": 150,
    "dataset_e_salangabhava": 150,
    "dataset_f_cmcs": 150,
}

# Phase 2 full benchmark census sizes per dataset (N = 13,054)
PHASE2_SAMPLE_SIZES: dict[str, int] = {
    "dataset_a_sentiment": 3000,
    "dataset_b_sold": 2500,
    "dataset_c1_nsina_categories": 1200,
    "dataset_c2_nsina_media": 1000,
    "dataset_d_sinhalammlu": 1854,
    "dataset_e_salangabhava": 1500,
    "dataset_f_cmcs": 2000,
}


# ---------------------------------------------------------------------------
# API Key
# ---------------------------------------------------------------------------

def get_api_key() -> str:
    """Retrieve the TypeSafe API key from environment.

    Loads .env at PROJECT_ROOT if it exists. The key is expected under
    the environment variable ``JEV_KEY``.

    Raises
    ------
    RuntimeError
        If the key is not found.
    """
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.environ.get("JEV_KEY")
    if not key:
        raise RuntimeError(
            "JEV_KEY not found in environment. "
            "Set it in .env or export JEV_KEY=<your-key>."
        )
    return key


# ---------------------------------------------------------------------------
# YAML Config Loader
# ---------------------------------------------------------------------------

def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML config from the configs/ directory."""
    path = CONFIGS_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# Experiment Config
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Parsed experiment configuration."""

    model: str = MODEL_NAME
    seed: int = RANDOM_SEED
    phase0_n: int = PHASE0_N_PER_TASK
    phase1_sizes: dict[str, int] = field(default_factory=lambda: dict(PHASE1_SAMPLE_SIZES))
    phase2_sizes: dict[str, int] = field(default_factory=lambda: dict(PHASE2_SAMPLE_SIZES))
    ece_bins: int = ECE_NUM_BINS
    phase2_ece_bins: int = PHASE2_ECE_NUM_BINS

    # Concurrency and rate limiting
    max_workers: int = 8
    rate_limit_rps: float = 25.0

    # Retry policy for TypeSafe SDK
    max_retries: int = 5
    backoff_initial: float = 2.0
    backoff_max: float = 30.0
    backoff_jitter: float = 0.5

    # Confidence thresholds for risk-coverage analysis
    confidence_thresholds: list[float] = field(
        default_factory=lambda: [0.50, 0.70, 0.80, 0.90, 0.95]
    )

    @classmethod
    def from_yaml(cls) -> ExperimentConfig:
        """Load from configs/experiments.yaml, falling back to defaults."""
        try:
            raw = load_yaml("experiments.yaml")
        except FileNotFoundError:
            return cls()
        concurrency = raw.get("concurrency", {})
        return cls(
            model=raw.get("model", MODEL_NAME),
            seed=raw.get("seed", RANDOM_SEED),
            phase0_n=raw.get("phase0_n", PHASE0_N_PER_TASK),
            phase1_sizes=raw.get("phase1_sizes", dict(PHASE1_SAMPLE_SIZES)),
            phase2_sizes=raw.get("phase2_sizes", dict(PHASE2_SAMPLE_SIZES)),
            ece_bins=raw.get("ece_bins", ECE_NUM_BINS),
            phase2_ece_bins=raw.get("phase2_ece_bins", PHASE2_ECE_NUM_BINS),
            max_workers=concurrency.get("max_workers", 8),
            rate_limit_rps=concurrency.get("rate_limit_rps", 25.0),
            max_retries=raw.get("max_retries", 5),
            backoff_initial=raw.get("backoff_initial", 2.0),
            backoff_max=raw.get("backoff_max", 30.0),
            backoff_jitter=raw.get("backoff_jitter", 0.5),
            confidence_thresholds=raw.get(
                "confidence_thresholds", [0.50, 0.70, 0.80, 0.90, 0.95]
            ),
        )
