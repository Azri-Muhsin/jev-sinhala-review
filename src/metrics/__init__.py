"""
src/metrics — Evaluation metrics suite for Jev experiments.
"""

from src.metrics.classification import compute_classification_metrics
from src.metrics.calibration import (
    compute_ece,
    compute_brier_score,
    compute_nll,
    compute_selective_risk_coverage,
)
from src.metrics.consistency import compute_sentiment_primitive_consistency
from src.metrics.sensitivity import compute_prompt_sensitivity

__all__ = [
    "compute_classification_metrics",
    "compute_ece",
    "compute_brier_score",
    "compute_nll",
    "compute_selective_risk_coverage",
    "compute_sentiment_primitive_consistency",
    "compute_prompt_sensitivity",
]
