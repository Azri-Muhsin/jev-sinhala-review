"""
scripts/test_scaleup_pilot.py — Pre-flight pilot verification gate.
Runs 5 items per task through ScaledCensusRunner.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.runner_scaleup import ScaledCensusRunner


def main():
    print("=" * 60)
    print("  LAUNCHING PRE-FLIGHT VERIFICATION PILOT (N=5 per task)")
    print("=" * 60)

    runner = ScaledCensusRunner(max_workers=4, rate_limit_rps=10.0)
    try:
        summary = runner.run(pilot_n=5)
        print("\nPilot Summary:")
        for name, info in summary.items():
            print(f"  {name:30s} -> {info['log']} ({info['duration_sec']}s)")
        print("\n✓ PRE-FLIGHT PILOT COMPLETED SUCCESSFULLY")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
