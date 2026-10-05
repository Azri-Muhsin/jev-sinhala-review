"""
scripts/run_scaleup_census.py — Master orchestrator for Phase 2 Full Benchmark Census.

Executes all 11,434 census items across the 7 tasks, followed by
synthesis table generation and figure creation.
"""

import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.runner_scaleup import ScaledCensusRunner
from src.visualization.scaleup_synthesis import run_scaleup_synthesis


def main():
    print("=" * 70)
    print("  LAUNCHING PHASE 2: FULL BENCHMARK CENSUS EXECUTION (N = 11,434)")
    print("=" * 70)

    t_start = time.time()
    runner = ScaledCensusRunner(max_workers=8, rate_limit_rps=25.0)

    try:
        # Run all scaled tasks
        runner.run()
    finally:
        runner.close()

    duration = time.time() - t_start
    print(f"\n✓ Full Census Execution finished in {duration/60:.2f} minutes.")

    print("\nRunning Scaled Synthesis & Deliverables Generation...")
    synth_res = run_scaleup_synthesis()
    print("\nSynthesis Summary:", synth_res)


if __name__ == "__main__":
    main()
