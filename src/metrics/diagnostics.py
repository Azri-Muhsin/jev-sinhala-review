"""
diagnostics.py — Primitive Diagnostics & Robustness Metrics (Stage 6).

Implements:
  1. Option-Order Permutation Test analysis (stability, position bias, entropy shift).
  2. Repeatability / Stochasticity test analysis (multi-run standard deviation).
  3. Aggregate Selective Risk-Coverage analysis across confidence thresholds.
"""

from __future__ import annotations

import math
from typing import Any
import numpy as np
from scipy.stats import chisquare


def compute_shannon_entropy(probs: list[float] | dict[str, float], eps: float = 1e-12) -> float:
    """Compute Shannon entropy in bits: H(P) = -sum p_i * log2(p_i)."""
    values = list(probs.values()) if isinstance(probs, dict) else list(probs)
    h = 0.0
    for p in values:
        p_val = max(float(p), eps)
        h -= p_val * math.log2(p_val)
    return round(float(h), 4)


def compute_option_order_diagnostics(
    records_orig: list[dict[str, Any]],
    records_shifted: list[dict[str, Any]],
    records_inverted: list[dict[str, Any]],
    order_orig: list[str],
    order_shifted: list[str],
    order_inverted: list[str],
) -> dict[str, Any]:
    """Analyze stability, position bias, and entropy under option order permutations."""
    map_orig = {r["example_id"]: r for r in records_orig if r.get("prediction") is not None}
    map_shifted = {r["example_id"]: r for r in records_shifted if r.get("prediction") is not None}
    map_inverted = {r["example_id"]: r for r in records_inverted if r.get("prediction") is not None}

    common_ids = sorted(list(set(map_orig.keys()) & set(map_shifted.keys()) & set(map_inverted.keys())))
    n = len(common_ids)
    if n == 0:
        return {"error": "No common valid records across all 3 permutations"}

    stable_count = 0
    flips_orig_shifted = 0
    flips_orig_inverted = 0

    position_counts_orig = [0] * len(order_orig)
    position_counts_shifted = [0] * len(order_shifted)
    position_counts_inverted = [0] * len(order_inverted)

    entropies_orig = []
    entropies_shifted = []
    entropies_inverted = []

    for eid in common_ids:
        p_o = map_orig[eid]["prediction"]
        p_s = map_shifted[eid]["prediction"]
        p_i = map_inverted[eid]["prediction"]

        # Stability
        if p_o == p_s == p_i:
            stable_count += 1
        if p_o != p_s:
            flips_orig_shifted += 1
        if p_o != p_i:
            flips_orig_inverted += 1

        # Position tracking
        if p_o in order_orig:
            position_counts_orig[order_orig.index(p_o)] += 1
        if p_s in order_shifted:
            position_counts_shifted[order_shifted.index(p_s)] += 1
        if p_i in order_inverted:
            position_counts_inverted[order_inverted.index(p_i)] += 1

        # Entropies
        entropies_orig.append(compute_shannon_entropy(map_orig[eid].get("probabilities", {})))
        entropies_shifted.append(compute_shannon_entropy(map_shifted[eid].get("probabilities", {})))
        entropies_inverted.append(compute_shannon_entropy(map_inverted[eid].get("probabilities", {})))

    # Aggregated position distribution across all 3 conditions
    total_positions = [
        o + s + i
        for o, s, i in zip(position_counts_orig, position_counts_shifted, position_counts_inverted)
    ]
    k = len(total_positions)
    expected_count = (n * 3) / k
    chi2_stat, chi2_p = chisquare(total_positions, f_exp=[expected_count] * k)

    mean_h_orig = float(np.mean(entropies_orig))
    mean_h_shifted = float(np.mean(entropies_shifted))
    mean_h_inverted = float(np.mean(entropies_inverted))

    delta_h_shifted = mean_h_shifted - mean_h_orig
    delta_h_inverted = mean_h_inverted - mean_h_orig

    return {
        "sample_size": n,
        "stability_rate": round(stable_count / n, 4),
        "stable_count": stable_count,
        "pairwise_flips": {
            "orig_vs_shifted_rate": round(flips_orig_shifted / n, 4),
            "orig_vs_shifted_count": flips_orig_shifted,
            "orig_vs_inverted_rate": round(flips_orig_inverted / n, 4),
            "orig_vs_inverted_count": flips_orig_inverted,
        },
        "position_bias": {
            "positions": [f"pos_{i}" for i in range(k)],
            "counts_orig": position_counts_orig,
            "counts_shifted": position_counts_shifted,
            "counts_inverted": position_counts_inverted,
            "total_counts": total_positions,
            "percentage_distribution": [round(c / (n * 3) * 100, 2) for c in total_positions],
            "chi2_statistic": round(float(chi2_stat), 4),
            "chi2_p_value": round(float(chi2_p), 6),
            "significant_position_bias": bool(chi2_p < 0.05),
        },
        "entropy": {
            "mean_entropy_original": round(mean_h_orig, 4),
            "mean_entropy_shifted": round(mean_h_shifted, 4),
            "mean_entropy_inverted": round(mean_h_inverted, 4),
            "mean_delta_shifted": round(delta_h_shifted, 4),
            "mean_delta_inverted": round(delta_h_inverted, 4),
        },
    }


def compute_repeatability_diagnostics(
    runs: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Analyze stochasticity and repeatability across multi-pass runs with identical parameters."""
    num_runs = len(runs)
    if num_runs < 2:
        return {"error": "At least 2 runs are required for repeatability analysis"}

    run_maps = [{r["example_id"]: r for r in run if r.get("prediction") is not None} for run in runs]
    common_ids = sorted(list(set.intersection(*[set(m.keys()) for m in run_maps])))
    n = len(common_ids)
    if n == 0:
        return {"error": "No common valid records across runs"}

    exact_repeat_count = 0
    std_confs = []
    std_prob_max = []
    max_drift_list = []

    for eid in common_ids:
        preds = [run_maps[idx][eid]["prediction"] for idx in range(num_runs)]
        confs = [float(run_maps[idx][eid]["confidence"]) for idx in range(num_runs)]

        if len(set(preds)) == 1:
            exact_repeat_count += 1

        std_confs.append(float(np.std(confs, ddof=1)) if num_runs > 1 else 0.0)
        max_drift_list.append(float(np.max(confs) - np.min(confs)))

    exact_repeatability_rate = exact_repeat_count / n
    mean_std_conf = float(np.mean(std_confs))
    max_observed_drift = float(np.max(max_drift_list)) if max_drift_list else 0.0

    return {
        "num_runs": num_runs,
        "sample_size": n,
        "exact_repeatability_rate": round(exact_repeatability_rate, 4),
        "exact_repeat_count": exact_repeat_count,
        "divergent_count": n - exact_repeat_count,
        "confidence_stochasticity": {
            "mean_std_confidence": round(mean_std_conf, 6),
            "max_confidence_drift": round(max_observed_drift, 6),
            "is_strictly_deterministic": bool(exact_repeatability_rate == 1.0 and mean_std_conf < 1e-6),
        },
    }
