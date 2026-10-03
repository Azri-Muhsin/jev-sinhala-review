"""
consistency.py — Primitive Consistency Metrics (RQ2: Choice vs Noul vs Score).
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_sentiment_primitive_consistency(
    choice_records: list[dict[str, Any]],
    noul_records: list[dict[str, Any]],
    score_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compute cross-primitive consistency metrics across Choice, Noul, and Score.

    Parameters
    ----------
    choice_records : list[dict]
        Section 5 records for Choice sentiment predictions.
    noul_records : list[dict]
        Section 5 records for Noul multi-class sentiment predictions.
    score_records : list[dict], optional
        Section 5 records for Score ordinal sentiment predictions.
    """
    # Key records by example_id
    choice_by_id = {r["example_id"]: r for r in choice_records}
    noul_by_id = {r["example_id"]: r for r in noul_records}
    score_by_id = {r["example_id"]: r for r in (score_records or [])}

    common_ids = sorted(list(set(choice_by_id.keys()) & set(noul_by_id.keys())))
    n_total = len(common_ids)

    if n_total == 0:
        return {"error": "No common examples between Choice and Noul"}

    # 1. Argmax Agreement: Choice vs Noul
    agreements = 0
    winner_pairs: list[tuple[str, str]] = []

    # 2. Contradiction tracking (Noul >= 0.50)
    zero_belief_count = 0
    single_belief_count = 0
    multi_belief_count = 0
    belief_distribution = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # 3. Probability pairs for correlation analysis
    classes = ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"]
    prob_pairs: dict[str, list[tuple[float, float]]] = {c: [] for c in classes}

    for ex_id in common_ids:
        r_c = choice_by_id[ex_id]
        r_n = noul_by_id[ex_id]

        c_pred = r_c["prediction"]
        n_pred = r_n["prediction"]
        winner_pairs.append((c_pred, n_pred))

        if c_pred == n_pred:
            agreements += 1

        # Check Noul beliefs
        noul_probs = r_n.get("noul_details") or r_n.get("probabilities", {})
        num_positive_beliefs = sum(1 for p in noul_probs.values() if float(p) >= 0.50)
        belief_distribution[min(num_positive_beliefs, 4)] += 1

        if num_positive_beliefs == 0:
            zero_belief_count += 1
        elif num_positive_beliefs == 1:
            single_belief_count += 1
        else:
            multi_belief_count += 1

        # Collect paired probabilities
        c_probs = r_c.get("probabilities", {})
        for c in classes:
            p_c = float(c_probs.get(c, 0.0))
            p_n = float(noul_probs.get(c, 0.0))
            prob_pairs[c].append((p_c, p_n))

    argmax_agreement_rate = round(agreements / n_total, 4)

    # Compute correlation metrics per class
    correlation_by_class = {}
    for c in classes:
        pairs = prob_pairs[c]
        vec_c = np.array([p[0] for p in pairs])
        vec_n = np.array([p[1] for p in pairs])

        mad = float(np.mean(np.abs(vec_c - vec_n)))

        # Standard deviations check for constant arrays
        if np.std(vec_c) > 1e-8 and np.std(vec_n) > 1e-8:
            r_val, r_pval = pearsonr(vec_c, vec_n)
            rho_val, rho_pval = spearmanr(vec_c, vec_n)
        else:
            r_val, r_pval = 0.0, 1.0
            rho_val, rho_pval = 0.0, 1.0

        correlation_by_class[c] = {
            "pearson_r": round(float(r_val), 4),
            "pearson_p": round(float(r_pval), 6),
            "spearman_rho": round(float(rho_val), 4),
            "spearman_p": round(float(rho_pval), 6),
            "mean_abs_diff": round(mad, 4),
        }

    # 4. Choice vs Score Ordinal Consistency
    score_comparison = None
    ordinal_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}

    score_common_ids = [eid for eid in common_ids if eid in score_by_id]
    if score_common_ids:
        choice_ordinals = []
        score_expectations = []
        exact_rounded_matches = 0

        for eid in score_common_ids:
            c_label = choice_by_id[eid]["prediction"]
            if c_label in ordinal_map:
                c_val = ordinal_map[c_label]
                s_val = float(score_by_id[eid]["prediction"])

                choice_ordinals.append(c_val)
                score_expectations.append(s_val)

                if round(s_val) == c_val:
                    exact_rounded_matches += 1

        if len(choice_ordinals) > 0:
            arr_c = np.array(choice_ordinals, dtype=float)
            arr_s = np.array(score_expectations, dtype=float)
            mae = float(np.mean(np.abs(arr_c - arr_s)))

            if np.std(arr_c) > 1e-8 and np.std(arr_s) > 1e-8:
                sp_rho, sp_pval = spearmanr(arr_c, arr_s)
                p_r, p_pval = pearsonr(arr_c, arr_s)
            else:
                sp_rho, sp_pval = 0.0, 1.0
                p_r, p_pval = 0.0, 1.0

            score_comparison = {
                "evaluated_count": len(choice_ordinals),
                "mae": round(mae, 4),
                "exact_rounded_match_rate": round(exact_rounded_matches / len(choice_ordinals), 4),
                "spearman_rho": round(float(sp_rho), 4),
                "spearman_p": round(float(sp_pval), 6),
                "pearson_r": round(float(p_r), 4),
                "pearson_p": round(float(p_pval), 6),
            }

    return {
        "total_paired_examples": n_total,
        "argmax_agreement": {
            "agreement_count": agreements,
            "agreement_rate": argmax_agreement_rate,
            "disagreement_count": n_total - agreements,
        },
        "contradictions": {
            "single_belief_coherent_count": single_belief_count,
            "single_belief_coherent_rate": round(single_belief_count / n_total, 4),
            "zero_belief_count": zero_belief_count,
            "zero_belief_rate": round(zero_belief_count / n_total, 4),
            "multi_belief_contradiction_count": multi_belief_count,
            "multi_belief_contradiction_rate": round(multi_belief_count / n_total, 4),
            "belief_count_histogram": belief_distribution,
        },
        "probability_alignment": correlation_by_class,
        "score_ordinal_alignment": score_comparison,
    }
