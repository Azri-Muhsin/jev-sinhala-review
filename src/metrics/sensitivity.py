"""
sensitivity.py — Prompt Language & Instruction Sensitivity Metrics (RQ4).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score


def compute_prompt_sensitivity(
    records_en: list[dict[str, Any]],
    records_si: list[dict[str, Any]],
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Compute sensitivity metrics between English and Sinhala instruction prompts.

    Parameters
    ----------
    records_en : list[dict]
        Section 5 records under English instruction condition.
    records_si : list[dict]
        Section 5 records under Sinhala instruction condition.
    labels : list[str], optional
        List of candidate label strings.
    """
    map_en = {r["example_id"]: r for r in records_en}
    map_si = {r["example_id"]: r for r in records_si}

    common_ids = sorted(
        [
            eid
            for eid in (set(map_en.keys()) & set(map_si.keys()))
            if map_en[eid].get("prediction") is not None
            and map_si[eid].get("prediction") is not None
        ]
    )
    n = len(common_ids)
    if n == 0:
        return {"error": "No common valid examples between English and Sinhala conditions"}

    all_labels = labels or sorted(
        list(
            {r["gold_label"] for r in records_en}
            | {r["prediction"] for r in records_en}
            | {r["prediction"] for r in records_si}
        )
    )

    y_true = [map_en[eid]["gold_label"] for eid in common_ids]
    y_pred_en = [map_en[eid]["prediction"] for eid in common_ids]
    y_pred_si = [map_si[eid]["prediction"] for eid in common_ids]

    # 1. Classification & Deltas
    acc_en = float(accuracy_score(y_true, y_pred_en))
    acc_si = float(accuracy_score(y_true, y_pred_si))
    acc_delta = acc_si - acc_en

    f1_en = float(f1_score(y_true, y_pred_en, labels=all_labels, average="macro", zero_division=0))
    f1_si = float(f1_score(y_true, y_pred_si, labels=all_labels, average="macro", zero_division=0))
    f1_delta = f1_si - f1_en

    # 2. Agreement & Flips
    agreements = sum(1 for p_en, p_si in zip(y_pred_en, y_pred_si) if p_en == p_si)
    flips = n - agreements
    agreement_rate = agreements / n
    flip_rate = flips / n

    # Detailed transition matrix of flips
    flip_transitions: dict[str, int] = {}
    for p_en, p_si in zip(y_pred_en, y_pred_si):
        if p_en != p_si:
            key = f"{p_en} -> {p_si}"
            flip_transitions[key] = flip_transitions.get(key, 0) + 1

    # 3. Confidence Drift
    confs_en = np.array([float(map_en[eid]["confidence"]) for eid in common_ids])
    confs_si = np.array([float(map_si[eid]["confidence"]) for eid in common_ids])

    mean_conf_en = float(np.mean(confs_en))
    mean_conf_si = float(np.mean(confs_si))
    conf_delta = mean_conf_si - mean_conf_en

    # Confidence correlation
    if np.std(confs_en) > 1e-8 and np.std(confs_si) > 1e-8:
        conf_corr_r, conf_corr_p = pearsonr(confs_en, confs_si)
    else:
        conf_corr_r, conf_corr_p = 0.0, 1.0

    # 4. Probability Vector Alignment (Cosine & MAD across classes)
    mad_list = []
    cos_sim_list = []

    for eid in common_ids:
        p_dict_en = map_en[eid].get("probabilities", {})
        p_dict_si = map_si[eid].get("probabilities", {})

        vec_en = np.array([float(p_dict_en.get(lbl, 0.0)) for lbl in all_labels])
        vec_si = np.array([float(p_dict_si.get(lbl, 0.0)) for lbl in all_labels])

        mad_list.append(float(np.mean(np.abs(vec_en - vec_si))))

        norm_en = np.linalg.norm(vec_en)
        norm_si = np.linalg.norm(vec_si)
        if norm_en > 1e-8 and norm_si > 1e-8:
            cos_sim_list.append(float(np.dot(vec_en, vec_si) / (norm_en * norm_si)))
        else:
            cos_sim_list.append(1.0 if np.allclose(vec_en, vec_si) else 0.0)

    mean_mad = float(np.mean(mad_list)) if mad_list else 0.0
    mean_cosine_sim = float(np.mean(cos_sim_list)) if cos_sim_list else 1.0

    return {
        "sample_size": n,
        "labels": all_labels,
        "accuracy": {
            "english": round(acc_en, 4),
            "sinhala": round(acc_si, 4),
            "delta": round(acc_delta, 4),
        },
        "macro_f1": {
            "english": round(f1_en, 4),
            "sinhala": round(f1_si, 4),
            "delta": round(f1_delta, 4),
        },
        "agreement": {
            "agreement_count": agreements,
            "agreement_rate": round(agreement_rate, 4),
            "flip_count": flips,
            "flip_rate": round(flip_rate, 4),
            "flip_transitions": flip_transitions,
        },
        "confidence": {
            "mean_english": round(mean_conf_en, 4),
            "mean_sinhala": round(mean_conf_si, 4),
            "delta": round(conf_delta, 4),
            "pearson_r": round(float(conf_corr_r), 4),
            "pearson_p": round(float(conf_corr_p), 6),
        },
        "probability_alignment": {
            "mean_absolute_difference": round(mean_mad, 4),
            "mean_cosine_similarity": round(mean_cosine_sim, 4),
        },
    }
