"""
main.py — Jev-Sinhala Quick Capability & Primitive Consistency Probe

Entry point for running the probe stages. Use this for quick interactive checks.
For formal execution, use the pytest suite or individual runner scripts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import io
import os

# Force UTF-8 on Windows console
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.config import (
    ExperimentConfig,
    MODEL_NAME,
    PROJECT_ROOT,
    CONFIGS_DIR,
    DATA_DIR,
    RESULTS_DIR,
)
from src.client import JevClient

from typesafe_sdk import Choice, Noul, Score


def print_banner() -> None:
    """Print the experiment banner."""
    print("=" * 70)
    print("  Jev-Sinhala Quick Capability & Primitive Consistency Probe")
    print("  Model: jev-latest  |  Zero-shot  |  TypeSafe Decision Interface")
    print("=" * 70)


def stage0_health_check() -> bool:
    """Stage 0: Environment Setup, Tooling & SDK Verification.

    Returns True if all checks pass.
    """
    print("\n── Stage 0: SDK Health Check ──────────────────────────────────")

    config = ExperimentConfig.from_yaml()
    print(f"  Model target:     {config.model}")
    print(f"  Random seed:      {config.seed}")
    print(f"  Phase 0 N/task:   {config.phase0_n}")
    print(f"  Max retries:      {config.max_retries}")
    print(f"  Configs dir:      {CONFIGS_DIR}")

    # 1. Check API authentication & model availability
    print("\n  [1/4] Authenticating and listing models...")
    try:
        client = JevClient(config=config)
        models = client.list_models()
        model_names = [m["name"] for m in models]
        print(f"        Available models: {model_names}")

        if MODEL_NAME not in model_names:
            print(f"        FAIL: '{MODEL_NAME}' not found in available models!")
            return False
        print(f"        OK: '{MODEL_NAME}' confirmed available")
    except Exception as e:
        print(f"        ✗ FAIL: {e}")
        return False

    # 2. Test Noul primitive
    print("\n  [2/4] Testing Noul primitive (binary decision)...")
    try:
        result = client.ask(
            state="අද කාලගුණය ඉතා හොඳයි.",
            questions={
                "is_positive": Noul(
                    instructions="Is this text expressing a positive sentiment?"
                ),
            },
        )
        noul_val = result.answers["is_positive"].noul
        print(f"        Noul P(True) = {noul_val:.4f}  [{result.latency_ms:.0f}ms]")
        print(f"        ✓ Noul working correctly")
    except Exception as e:
        print(f"        ✗ FAIL: {e}")
        return False

    # 3. Test Choice primitive
    print("\n  [3/4] Testing Choice primitive (4-way sentiment)...")
    try:
        result = client.ask(
            state="මෙම තීරණය ඉතාමත් අගය කළ යුතු එකක් බව පැවසිය යුතුය.",
            questions={
                "sentiment": Choice(
                    instructions="Which label best describes the overall sentiment of this text?",
                    criteria={
                        "POSITIVE": None,
                        "NEGATIVE": None,
                        "NEUTRAL": None,
                        "CONFLICT": None,
                    },
                ),
            },
        )
        ans = result.answers["sentiment"]
        print(f"        Choice: {ans.choice} (confidence={ans.confidence:.4f})")
        print(f"        Probs: {ans.probabilities}")
        print(f"        Latency: {result.latency_ms:.0f}ms")
        print(f"        ✓ Choice working correctly")
    except Exception as e:
        print(f"        ✗ FAIL: {e}")
        return False

    # 4. Test Score primitive
    print("\n  [4/4] Testing Score primitive (3-level ordered sentiment)...")
    try:
        result = client.ask(
            state="The food was acceptable but not remarkable.",
            questions={
                "sentiment_score": Score(
                    instructions="Rate the sentiment of this text on an ordered scale.",
                    criteria=[
                        "Negative sentiment",
                        "Neutral sentiment",
                        "Positive sentiment",
                    ],
                ),
            },
        )
        ans = result.answers["sentiment_score"]
        print(f"        Score: {ans.score:.4f} (confidence={ans.confidence:.4f})")
        print(f"        Probs: {ans.probabilities}")
        print(f"        Legend: {ans.legend}")
        print(f"        Latency: {result.latency_ms:.0f}ms")
        print(f"        ✓ Score working correctly")
    except Exception as e:
        print(f"        ✗ FAIL: {e}")
        return False

    client.close()

    print("\n" + "─" * 70)
    print("  ✓ STAGE 0 PASSED — All SDK primitives verified")
    print(f"    Model: {MODEL_NAME}")
    print(f"    Primitives: Noul ✓  Choice ✓  Score ✓")
    print("─" * 70)
    return True


def stage1_data_acquisition() -> bool:
    """Stage 1: Data Acquisition, Split Isolation & Deterministic Sampling Engine.

    Returns True if all datasets are ingested and sampled with valid SHA-256 manifests.
    """
    print("\n── Stage 1: Data Ingestion & Deterministic Sampling ──────────")

    from src.loaders import (
        SentimentLoader,
        SOLDLoader,
        NSINACategoriesLoader,
        NSINAMediaLoader,
        SinhalaMMLULoader,
        SalAngaBhavaLoader,
        CMCSLoader,
    )
    from src.sampling import StratifiedSampler

    loaders = [
        SentimentLoader(),
        SOLDLoader(),
        NSINACategoriesLoader(),
        NSINAMediaLoader(),
        SinhalaMMLULoader(),
        SalAngaBhavaLoader(),
        CMCSLoader(),
    ]

    print(f"  Validating and ingesting {len(loaders)} datasets...")
    for loader in loaders:
        try:
            val = loader.validate()
            if not val["is_valid"]:
                print(f"  ✗ {loader.dataset_id} validation failed: {val['issues']}")
                return False
            print(f"  ✓ {loader.dataset_id:30s} | {val['total_records']:6d} items | ZWJ texts: {val['zwj_containing_texts']}")
        except Exception as e:
            print(f"  ✗ Failed to load {loader.dataset_id}: {e}")
            return False

    print("\n  Generating stratified Phase 0 (N=20/task) and Phase 1 subsets...")
    sampler = StratifiedSampler()
    manifest = sampler.generate_all_samples(loaders)

    print("\n" + "─" * 70)
    print("  ✓ STAGE 1 PASSED — Ingestion, Splits & Sampling Complete")
    print("    Manifest: data/processed/manifest.json")
    print(f"    Phase 0 Smoke: {len(manifest['phase0_smoke'])} tasks (N=20 each, 140 total)")
    print(f"    Phase 1 Probe: {len(manifest['phase1_samples'])} tasks (~950 total)")
    print("─" * 70)
    return True


def stage2_smoke_test() -> bool:
    """Stage 2: Phase 0 Smoke Test (20 items/task) & Verification Gate.

    Returns True if all 7 tasks pass with zero runtime errors and valid distributions.
    """
    from src.runners.runner_smoke_test import SmokeTestRunner

    runner = SmokeTestRunner()
    try:
        report = runner.run()
        return bool(report.get("gate_passed", False))
    finally:
        runner.close()


def stage3_sentiment_equivalence() -> bool:
    """Stage 3: Sentiment Primitive Equivalence Lab (Phase 1 Full Probe, N=150).

    Tests RQ2 (Primitive Consistency) across Choice, Noul, and Score.
    """
    from src.runners.runner_stage3_sentiment import SentimentPrimitiveEquivalenceRunner

    runner = SentimentPrimitiveEquivalenceRunner()
    try:
        report = runner.run()
        return report.get("sample_size", 0) > 0
    finally:
        runner.close()


def stage4_core_tasks() -> bool:
    """Stage 4: Core Pure-Sinhala Tasks Execution (N=650).

    Executes Phase 1 full probe across SOLD, NSINA Categories, NSINA Media,
    SinhalaMMLU, and SalAngaBhava.
    """
    from src.runners.runner_stage4_core import CorePureSinhalaRunner

    runner = CorePureSinhalaRunner()
    try:
        report = runner.run()
        return report.get("total_examples", 0) > 0
    finally:
        runner.close()


def stage5_prompt_sensitivity() -> bool:
    """Stage 5: Language & Prompt Sensitivity Experiment (RQ4, N=100 total).

    Evaluates English vs. native Sinhala instructions on identical input states:
      - 50 items from Dataset A (Sentiment)
      - 50 items from Dataset B (SOLD)
    """
    from src.runners.runner_stage5_prompt_sensitivity import PromptSensitivityRunner

    runner = PromptSensitivityRunner()
    try:
        report = runner.run()
        return bool(report.get("sentiment") and report.get("sold"))
    finally:
        runner.close()


def stage6_diagnostics() -> bool:
    """Stage 6: Primitive Diagnostics & Robustness.

      - Option-order permutation test (N=40 across MMLU & NSINA Categories)
      - Repeatability & stochasticity test (N=100 over 3 independent passes)
      - Aggregate selective risk-coverage curves (tau in [0.50..0.95])
    """
    from src.runners.runner_stage6_diagnostics import DiagnosticsRunner

    runner = DiagnosticsRunner()
    try:
        report = runner.run()
        return bool(report.get("option_order") and report.get("repeatability"))
    finally:
        runner.close()


def stage7_cmcs_stress() -> bool:
    """Stage 7: Code-Mixed Stress Track (Dataset F: CMCS, N=150).

    Evaluates non-standard, Romanized, and code-mixed Sinhala-English text across
    5 target sub-tasks: Sentiment, Humour, Hate Speech, Aspect Extraction, Script Analysis.
    """
    from src.runners.runner_stage7_cmcs import CMCSStressTrackRunner

    runner = CMCSStressTrackRunner()
    try:
        report = runner.run()
        return bool(report.get("sentiment_choice") and report.get("sample_size", 0) > 0)
    finally:
        runner.close()


def stage8_deliverables() -> bool:
    """Stage 8: Quantitative Synthesis, Reference Anchors & Economist-styled Figures.

    Generates:
      - results/main.csv (15 task/primitive configurations with reference anchors)
      - results/primitive_consistency.csv (RQ2 cross-primitive parity metrics)
      - results/script_analysis.csv (Pure Sinhala vs Singlish vs Code-mixed)
      - results/failure_taxonomy.csv (Audited failure taxonomy of 50 samples)
      - results/figures/fig1_accuracy_by_task.png through fig7_latency_by_primitive.png
    """
    from src.visualization.synthesis_tables import generate_all_synthesis_tables
    from src.visualization.plot_generator import generate_all_figures

    print("\n── Stage 8: Quantitative Synthesis & Economist-styled Deliverables ──")
    try:
        tables = generate_all_synthesis_tables()
        figures = generate_all_figures()
        print("\n" + "─" * 70)
        print("  ✓ STAGE 8 PASSED — All synthesis tables and figures generated")
        print(f"    Tables: {len(tables)} CSV artifacts in results/")
        print(f"    Figures: {len(figures)} Economist-styled PNGs in results/figures/")
        print("─" * 70)
        return True
    except Exception as e:
        print(f"  ✗ Stage 8 execution failed: {e}")
        return False


def main() -> None:
    """Run the requested probe stage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Jev-Sinhala Quick Capability & Primitive Consistency Probe"
    )
    parser.add_argument(
        "--stage",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6, 7, 8],
        default=None,
        help="Stage to execute: 0 (Health), 1 (Data), 2 (Smoke), 3 (Sentiment), 4 (Core Tasks), 5 (Sensitivity), 6 (Diagnostics), 7 (CMCS Stress), 8 (Synthesis & Figures)",
    )
    args = parser.parse_args()

    print_banner()

    if args.stage == 0:
        success = stage0_health_check()
    elif args.stage == 1:
        success = stage1_data_acquisition()
    elif args.stage == 2:
        success = stage2_smoke_test()
    elif args.stage == 3:
        success = stage3_sentiment_equivalence()
    elif args.stage == 4:
        success = stage4_core_tasks()
    elif args.stage == 5:
        success = stage5_prompt_sensitivity()
    elif args.stage == 6:
        success = stage6_diagnostics()
    elif args.stage == 7:
        success = stage7_cmcs_stress()
    elif args.stage == 8:
        success = stage8_deliverables()
    else:
        # Default run
        s0 = stage0_health_check()
        if not s0:
            sys.exit(1)
        s1 = stage1_data_acquisition()
        if not s1:
            sys.exit(1)
        s2 = stage2_smoke_test()
        if not s2:
            sys.exit(1)
        s3 = stage3_sentiment_equivalence()
        if not s3:
            sys.exit(1)
        s4 = stage4_core_tasks()
        if not s4:
            sys.exit(1)
        s5 = stage5_prompt_sensitivity()
        if not s5:
            sys.exit(1)
        s6 = stage6_diagnostics()
        if not s6:
            sys.exit(1)
        success = stage7_cmcs_stress()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

