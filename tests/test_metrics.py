"""
test_metrics.py — Unit tests for evaluation metrics suite (accuracy, calibration, consistency).
"""

from __future__ import annotations

import pytest
from src.metrics import (
    compute_brier_score,
    compute_classification_metrics,
    compute_ece,
    compute_nll,
    compute_selective_risk_coverage,
    compute_sentiment_primitive_consistency,
)


def test_classification_metrics_basic() -> None:
    y_true = ["POSITIVE", "NEGATIVE", "NEUTRAL", "POSITIVE"]
    y_pred = ["POSITIVE", "NEGATIVE", "POSITIVE", "POSITIVE"]

    res = compute_classification_metrics(y_true, y_pred, labels=["POSITIVE", "NEGATIVE", "NEUTRAL"])
    assert res["accuracy"] == 0.75
    assert 0.0 <= res["macro_f1"] <= 1.0
    assert len(res["confusion_matrix"]) == 3
    assert res["per_class"]["POSITIVE"]["precision"] == pytest.approx(2 / 3, abs=1e-3)


def test_calibration_ece_perfect() -> None:
    confidences = [0.8, 0.8, 0.8, 0.8, 0.8]
    matches = [True, True, True, True, False]  # 4/5 = 0.8 accuracy, matching confidence
    res = compute_ece(confidences, matches, num_bins=5)
    assert res["ece"] < 0.05
    assert res["total_evaluated"] == 5


def test_calibration_brier_and_nll() -> None:
    y_true = ["A", "B"]
    probs = [{"A": 0.9, "B": 0.1}, {"A": 0.2, "B": 0.8}]
    brier = compute_brier_score(y_true, probs, labels=["A", "B"])
    assert brier < 0.1  # Highly accurate predictions have low Brier score
    nll = compute_nll(y_true, probs)
    assert nll < 0.3


def test_selective_risk_coverage() -> None:
    confidences = [0.95, 0.85, 0.65, 0.40]
    matches = [True, True, False, False]
    rc = compute_selective_risk_coverage(confidences, matches, thresholds=[0.50, 0.80, 0.90])
    assert len(rc) == 3
    # At tau >= 0.80: 2 items selected ([0.95, 0.85]), both True -> accuracy = 1.0, coverage = 0.50
    tau_80 = next(r for r in rc if r["threshold"] == 0.80)
    assert tau_80["coverage"] == 0.50
    assert tau_80["accuracy"] == 1.0
    assert tau_80["risk"] == 0.0


def test_primitive_consistency() -> None:
    choice_recs = [
        {"example_id": "1", "prediction": "POSITIVE", "probabilities": {"POSITIVE": 0.9, "NEGATIVE": 0.1, "NEUTRAL": 0.0, "CONFLICT": 0.0}},
        {"example_id": "2", "prediction": "NEGATIVE", "probabilities": {"POSITIVE": 0.1, "NEGATIVE": 0.8, "NEUTRAL": 0.1, "CONFLICT": 0.0}},
        {"example_id": "3", "prediction": "POSITIVE", "probabilities": {"POSITIVE": 0.6, "NEGATIVE": 0.2, "NEUTRAL": 0.2, "CONFLICT": 0.0}},
    ]
    noul_recs = [
        {"example_id": "1", "prediction": "POSITIVE", "noul_details": {"POSITIVE": 0.85, "NEGATIVE": 0.1, "NEUTRAL": 0.05, "CONFLICT": 0.0}},
        {"example_id": "2", "prediction": "NEGATIVE", "noul_details": {"POSITIVE": 0.1, "NEGATIVE": 0.75, "NEUTRAL": 0.15, "CONFLICT": 0.0}},
        {"example_id": "3", "prediction": "NEUTRAL", "noul_details": {"POSITIVE": 0.6, "NEGATIVE": 0.1, "NEUTRAL": 0.7, "CONFLICT": 0.0}},  # contradiction: pos & neu >= 0.5
    ]
    score_recs = [
        {"example_id": "1", "prediction": 1.95},
        {"example_id": "2", "prediction": 0.05},
        {"example_id": "3", "prediction": 1.5},
    ]

    res = compute_sentiment_primitive_consistency(choice_recs, noul_recs, score_recs)
    assert res["total_paired_examples"] == 3
    assert res["argmax_agreement"]["agreement_count"] == 2
    assert res["contradictions"]["multi_belief_contradiction_count"] == 1  # item 3 has 2 beliefs >= 0.5
    assert res["score_ordinal_alignment"] is not None
    assert res["score_ordinal_alignment"]["mae"] < 0.5
