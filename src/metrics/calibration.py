"""
calibration.py — Probability calibration metrics (ECE, Brier Score, NLL, Risk-Coverage).
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np


def compute_ece(
    confidences: list[float],
    matches: list[bool],
    num_bins: int = 10,
) -> dict[str, Any]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values in [0, 1].
    matches : list[bool]
        Boolean indicator of correctness (True = correct, False = error).
    num_bins : int, default 10
        Number of equal-width probability bins.
    """
    n = len(confidences)
    if n == 0:
        return {"ece": 0.0, "mce": 0.0, "bins": []}

    confs = np.array(confidences, dtype=float)
    corrects = np.array(matches, dtype=bool)

    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bins_data = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        if bin_upper == 1.0:
            in_bin = (confs >= bin_lower) & (confs <= bin_upper)
        else:
            in_bin = (confs >= bin_lower) & (confs < bin_upper)

        bin_count = int(np.sum(in_bin))

        if bin_count > 0:
            bin_acc = float(np.mean(corrects[in_bin]))
            bin_conf = float(np.mean(confs[in_bin]))
            bin_error = abs(bin_acc - bin_conf)
            ece += (bin_count / n) * bin_error
            mce = max(mce, bin_error)
            bins_data.append({
                "lower": round(float(bin_lower), 2),
                "upper": round(float(bin_upper), 2),
                "count": bin_count,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "error": round(bin_error, 4),
            })
        else:
            bins_data.append({
                "lower": round(float(bin_lower), 2),
                "upper": round(float(bin_upper), 2),
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "error": 0.0,
            })

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "bins": bins_data,
        "num_bins": num_bins,
        "total_evaluated": n,
    }


def compute_brier_score(
    y_true: list[str],
    probabilities: list[dict[str, float]],
    labels: list[str] | None = None,
) -> float:
    """Compute multi-class Brier score.

    BS = 1/N * sum_{i} sum_{c} (P_i(c) - y_{ic})^2
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    all_labels = labels or sorted(list(set(y_true) | {k for p in probabilities for k in p.keys()}))
    total_sq_err = 0.0

    for gold, prob_dict in zip(y_true, probabilities):
        for lbl in all_labels:
            p_val = float(prob_dict.get(lbl, 0.0))
            y_val = 1.0 if lbl == gold else 0.0
            total_sq_err += (p_val - y_val) ** 2

    return round(float(total_sq_err / n), 4)


def compute_nll(
    y_true: list[str],
    probabilities: list[dict[str, float]],
    eps: float = 1e-15,
) -> float:
    """Compute Negative Log-Likelihood (NLL / Log Loss)."""
    n = len(y_true)
    if n == 0:
        return 0.0

    total_nll = 0.0
    for gold, prob_dict in zip(y_true, probabilities):
        p_val = max(float(prob_dict.get(gold, 0.0)), eps)
        total_nll -= math.log(p_val)

    return round(float(total_nll / n), 4)


def compute_selective_risk_coverage(
    confidences: list[float],
    matches: list[bool],
    thresholds: list[float] | None = None,
) -> list[dict[str, Any]]:
    """Compute accuracy and coverage at various confidence thresholds.

    Parameters
    ----------
    confidences : list[float]
        Predicted confidence values.
    matches : list[bool]
        Correctness indicators.
    thresholds : list[float], optional
        Confidence thresholds (defaults to [0.50, 0.70, 0.80, 0.90, 0.95]).
    """
    threshold_list = thresholds or [0.50, 0.70, 0.80, 0.90, 0.95]
    n = len(confidences)
    if n == 0:
        return []

    confs = np.array(confidences, dtype=float)
    corrects = np.array(matches, dtype=bool)

    results = []
    for tau in threshold_list:
        selected = confs >= tau
        sel_count = int(np.sum(selected))
        coverage = sel_count / n
        if sel_count > 0:
            accuracy = float(np.mean(corrects[selected]))
            risk = 1.0 - accuracy
        else:
            accuracy = 0.0
            risk = 0.0

        results.append({
            "threshold": tau,
            "coverage": round(float(coverage), 4),
            "retained_count": sel_count,
            "total_count": n,
            "accuracy": round(float(accuracy), 4),
            "risk": round(float(risk), 4),
        })

    return results
