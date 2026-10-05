"""
sampler.py — Deterministic, Stratified Sampling Engine for Phase 0 & Phase 1.

Features:
  - Exact stratification across gold label classes.
  - Fixed seed (default 42).
  - Byte-for-byte UTF-8 preservation (ZWJ, ZWNJ, Sinhala diacritics).
  - Two-tier output:
      data/processed/phase0_smoke/ (20 examples per task)
      data/processed/samples/      (Production probe subsets per task)
  - Cryptographic verification: SHA-256 checksum manifest committed to
      data/processed/manifest.json.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import (
    DATA_PROCESSED_DIR,
    DATA_PROCESSED_SCALED_DIR,
    PHASE0_N_PER_TASK,
    PHASE1_SAMPLE_SIZES,
    PHASE2_SAMPLE_SIZES,
    RANDOM_SEED,
)
from src.loaders.base import BaseLoader, DatasetRecord


class StratifiedSampler:
    """Deterministic stratified sampler across dataset classes."""

    def __init__(
        self,
        seed: int = RANDOM_SEED,
        processed_dir: Path | None = None,
    ) -> None:
        self.seed = seed
        self.processed_dir = processed_dir or DATA_PROCESSED_DIR
        self.smoke_dir = self.processed_dir / "phase0_smoke"
        self.samples_dir = self.processed_dir / "samples"
        self.scaled_samples_dir = self.processed_dir / "samples_scaled"

        self.smoke_dir.mkdir(parents=True, exist_ok=True)
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.scaled_samples_dir.mkdir(parents=True, exist_ok=True)

    def sample_records(
        self,
        records: list[DatasetRecord],
        target_n: int,
    ) -> list[DatasetRecord]:
        """Draw a deterministically stratified sample of size `target_n`."""
        if not records:
            return []

        if len(records) <= target_n:
            # Not enough records to sample, return all sorted by ID for determinism
            return sorted(records, key=lambda r: r.example_id)

        # Group records by gold_label
        by_label: dict[str, list[DatasetRecord]] = {}
        for r in records:
            by_label.setdefault(r.gold_label, []).append(r)

        # Deterministically sort within each label bucket by example_id
        for lbl in by_label:
            by_label[lbl].sort(key=lambda r: r.example_id)

        # Calculate quota per label
        total_records = len(records)
        raw_quotas: dict[str, float] = {
            lbl: (len(items) / total_records) * target_n
            for lbl, items in by_label.items()
        }

        # Ensure every class with records gets at least 1 if target_n >= num_classes
        quotas: dict[str, int] = {}
        for lbl, raw_q in raw_quotas.items():
            if target_n >= len(by_label):
                quotas[lbl] = max(1, math.floor(raw_q))
            else:
                quotas[lbl] = math.floor(raw_q)

        # Distribute remaining quota to highest remainder classes
        remainder = target_n - sum(quotas.values())
        if remainder > 0:
            remainders = sorted(
                by_label.keys(),
                key=lambda lbl: (raw_quotas[lbl] - quotas[lbl], lbl),
                reverse=True,
            )
            for i in range(remainder):
                lbl = remainders[i % len(remainders)]
                quotas[lbl] += 1
        elif remainder < 0:
            # Over-allocated (due to max(1, ...)), decrement from largest classes
            deficit = -remainder
            largest = sorted(
                by_label.keys(),
                key=lambda lbl: (quotas[lbl], lbl),
                reverse=True,
            )
            for i in range(deficit):
                lbl = largest[i % len(largest)]
                if quotas[lbl] > 1:
                    quotas[lbl] -= 1

        # Deterministically select using seeded RandomState
        rng = np.random.RandomState(self.seed)
        sampled: list[DatasetRecord] = []

        # Sort labels alphabetically for absolute determinism
        for lbl in sorted(by_label.keys()):
            items = by_label[lbl]
            count = min(quotas[lbl], len(items))
            if count <= 0:
                continue

            indices = rng.choice(len(items), size=count, replace=False)
            indices.sort()
            for idx in indices:
                sampled.append(items[idx])

        # Return sorted by example_id for reproducible ordering
        return sorted(sampled, key=lambda r: r.example_id)

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        """Compute SHA-256 hash of a file."""
        hasher = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    def save_jsonl(
        self,
        records: list[DatasetRecord],
        filepath: Path,
    ) -> str:
        """Save records to JSONL preserving byte-for-byte UTF-8. Returns SHA-256."""
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                obj = {
                    "example_id": r.example_id,
                    "dataset": r.dataset,
                    "text": r.text,
                    "gold_label": r.gold_label,
                    "split": r.split,
                    "metadata": r.metadata,
                }
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

        return self._compute_sha256(filepath)

    def generate_all_samples(
        self,
        loaders: list[BaseLoader],
    ) -> dict[str, Any]:
        """Generate Phase 0 smoke test subsets and Phase 1 full probe subsets."""
        manifest: dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": self.seed,
                "phase0_target_per_task": PHASE0_N_PER_TASK,
                "phase1_targets": PHASE1_SAMPLE_SIZES,
            },
            "phase0_smoke": {},
            "phase1_samples": {},
        }

        for loader in loaders:
            ds_id = loader.dataset_id
            print(f"Sampling dataset: {ds_id}...")
            records = loader.load()

            # 1. Phase 1 Full Sample
            p1_n = PHASE1_SAMPLE_SIZES.get(ds_id, 150)
            p1_records = self.sample_records(records, target_n=p1_n)
            p1_path = self.samples_dir / f"{ds_id}.jsonl"
            p1_hash = self.save_jsonl(p1_records, p1_path)

            p1_dist = dict(Counter(r.gold_label for r in p1_records))
            manifest["phase1_samples"][ds_id] = {
                "file": str(p1_path.relative_to(self.processed_dir)).replace("\\", "/"),
                "sample_size": len(p1_records),
                "sha256": p1_hash,
                "label_distribution": p1_dist,
            }

            # 2. Phase 0 Smoke Sample (Stratified 20 from p1_records for end-to-end coherence)
            p0_records = self.sample_records(p1_records, target_n=PHASE0_N_PER_TASK)
            p0_path = self.smoke_dir / f"{ds_id}_smoke.jsonl"
            p0_hash = self.save_jsonl(p0_records, p0_path)

            p0_dist = dict(Counter(r.gold_label for r in p0_records))
            manifest["phase0_smoke"][ds_id] = {
                "file": str(p0_path.relative_to(self.processed_dir)).replace("\\", "/"),
                "sample_size": len(p0_records),
                "sha256": p0_hash,
                "label_distribution": p0_dist,
            }

            print(
                f"  -> Phase 1: {len(p1_records)} items (SHA256: {p1_hash[:10]}...)"
            )
            print(
                f"  -> Phase 0: {len(p0_records)} items (SHA256: {p0_hash[:10]}...)"
            )

        # Write manifest.json
        manifest_path = self.processed_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\nManifest committed to: {manifest_path}")
        return manifest

    def generate_scaled_samples(
        self,
        loaders: list[BaseLoader],
        target_sizes: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Generate Phase 2 full benchmark census sample files and manifest_scaled.json."""
        targets = target_sizes or PHASE2_SAMPLE_SIZES
        manifest: dict[str, Any] = {
            "metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "random_seed": self.seed,
                "phase2_targets": targets,
            },
            "phase2_samples": {},
        }

        for loader in loaders:
            ds_id = loader.dataset_id
            print(f"Sampling scaled dataset: {ds_id}...")

            # For SalAngaBhava, ensure we load all review types (filter_pure_sinhala=False)
            if ds_id == "dataset_e_salangabhava" and hasattr(loader, "filter_pure_sinhala"):
                loader.filter_pure_sinhala = False

            records = loader.load()
            p2_n = targets.get(ds_id, len(records))

            # Special stratification for SalAngaBhava: 500 Pure + 500 Singlish + 500 Code-Mixed
            if ds_id == "dataset_e_salangabhava":
                pure_recs = [r for r in records if r.metadata.get("review_type") == "Pure_Sinhala"]
                sing_recs = [
                    r for r in records
                    if r.metadata.get("review_type") in {"Sinhala_in_English", "Sinhala_in_Latin"}
                ]
                mixed_recs = [r for r in records if r.metadata.get("review_type") == "Code_Mixed"]

                s_pure = self.sample_records(pure_recs, target_n=500)
                s_sing = self.sample_records(sing_recs, target_n=500)
                s_mixed = self.sample_records(mixed_recs, target_n=500)
                p2_records = sorted(s_pure + s_sing + s_mixed, key=lambda r: r.example_id)
            else:
                p2_records = self.sample_records(records, target_n=p2_n)

            p2_path = self.scaled_samples_dir / f"{ds_id}_scaled.jsonl"
            p2_hash = self.save_jsonl(p2_records, p2_path)
            p2_dist = dict(Counter(r.gold_label for r in p2_records))

            manifest["phase2_samples"][ds_id] = {
                "file": str(p2_path.relative_to(self.processed_dir)).replace("\\", "/"),
                "sample_size": len(p2_records),
                "sha256": p2_hash,
                "label_distribution": p2_dist,
            }
            print(f"  -> Phase 2 Scaled: {len(p2_records)} items (SHA256: {p2_hash[:10]}...)")

        # Write manifest_scaled.json
        manifest_path = self.processed_dir / "manifest_scaled.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

        print(f"\nScaled Manifest committed to: {manifest_path}")
        return manifest
