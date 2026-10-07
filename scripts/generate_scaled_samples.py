"""
scripts/generate_scaled_samples.py — Generate Phase 2 full benchmark census datasets.
"""

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


def main():
    print("Initializing loaders for Phase 2 census...")
    loaders = [
        SentimentLoader(),
        SOLDLoader(),
        NSINACategoriesLoader(),
        # NSINAMediaLoader(),  # Deprecated: media outlet classification
        SinhalaMMLULoader(),
        SalAngaBhavaLoader(filter_pure_sinhala=False),
        CMCSLoader(),
    ]

    sampler = StratifiedSampler()
    manifest = sampler.generate_scaled_samples(loaders)

    print("\n=== SCALED SAMPLING COMPLETE ===")
    total_items = 0
    for ds_id, meta in manifest["phase2_samples"].items():
        n = meta["sample_size"]
        total_items += n
        h = meta["sha256"][:12]
        print(f"  {ds_id:30s} | {n:5d} items | SHA256: {h}...")
    print(f"\nTotal Scaled Census Items: {total_items}")


if __name__ == "__main__":
    main()
