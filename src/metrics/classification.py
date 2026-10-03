"""
classification.py — Classification metrics calculation (Accuracy, Macro-F1, Confusion Matrix).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: list[str],
    y_pred: list[str],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Parameters
    ----------
    y_true : list[str]
        Ground truth labels.
    y_pred : list[str]
        Model predictions.
    labels : list[str], optional
        List of unique label strings in deterministic order.
    """
    if len(y_true) == 0:
        return {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "per_class": {},
            "confusion_matrix": [],
            "labels": [],
        }

    unique_labels = labels or sorted(list(set(y_true) | set(y_pred)))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=unique_labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=unique_labels, average="weighted", zero_division=0))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=unique_labels, zero_division=0
    )

    per_class = {}
    for idx, lbl in enumerate(unique_labels):
        per_class[lbl] = {
            "precision": round(float(precision[idx]), 4),
            "recall": round(float(recall[idx]), 4),
            "f1": round(float(f1[idx]), 4),
            "support": int(support[idx]),
        }

    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "total_evaluated": len(y_true),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "labels": unique_labels,
    }
