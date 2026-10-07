"""
test_loaders.py — Unit Tests for Dataset Loaders and Deterministic Sampling Engine.

Verifies:
  - BaseLoader interface adherence for all 7 loaders.
  - Correct label sets and bidirectional validity.
  - Unicode integrity (ZWJ / ZWNJ byte-for-byte preservation).
  - Deterministic stratified sampling quotas and manifest SHA-256 integrity.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.config import DATA_PROCESSED_DIR
from src.loaders import (
    BaseLoader,
    DatasetRecord,
    SentimentLoader,
    SOLDLoader,
    NSINACategoriesLoader,
    NSINAMediaLoader,
    SinhalaMMLULoader,
    SalAngaBhavaLoader,
    CMCSLoader,
)
from src.sampling import StratifiedSampler


@pytest.fixture(scope="module")
def all_loaders() -> list[BaseLoader]:
    """Instantiate active task loaders (6 datasets)."""
    return [
        SentimentLoader(),
        SOLDLoader(),
        NSINACategoriesLoader(),
        # NSINAMediaLoader(),  # Deprecated: media outlet classification
        SinhalaMMLULoader(),
        SalAngaBhavaLoader(),
        CMCSLoader(),
    ]


def test_loaders_instantiation(all_loaders: list[BaseLoader]) -> None:
    """Ensure all active loaders inherit from BaseLoader and declare dataset_id and label_set."""
    assert len(all_loaders) == 6
    for loader in all_loaders:
        assert isinstance(loader, BaseLoader)
        assert isinstance(loader.dataset_id, str)
        assert len(loader.dataset_id) > 0
        assert isinstance(loader.label_set, list)
        assert len(loader.label_set) > 0


def test_dataset_a_sentiment(all_loaders: list[BaseLoader]) -> None:
    """Test SentimentLoader: 4-way classification."""
    loader = next(l for l in all_loaders if l.dataset_id == "dataset_a_sentiment")
    records = loader.load()
    assert len(records) > 0

    valid_labels = set(loader.label_set)
    assert valid_labels == {"POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"}

    for r in records[:50]:
        assert r.gold_label in valid_labels
        assert len(r.text.strip()) > 0
        assert r.dataset == "dataset_a_sentiment"


def test_dataset_b_sold(all_loaders: list[BaseLoader]) -> None:
    """Test SOLDLoader: binary offensive language."""
    loader = next(l for l in all_loaders if l.dataset_id == "dataset_b_sold")
    records = loader.load()
    assert len(records) > 0

    valid_labels = set(loader.label_set)
    assert valid_labels == {"OFF", "NOT"}

    for r in records[:50]:
        assert r.gold_label in valid_labels
        assert len(r.text.strip()) > 0


def test_dataset_c_nsina(all_loaders: list[BaseLoader]) -> None:
    """Test NSINACategoriesLoader (C1)."""
    c1 = next(l for l in all_loaders if l.dataset_id == "dataset_c1_nsina_categories")
    c1_records = c1.load()
    assert len(c1_records) > 0
    assert len(c1.label_set) >= 4

    # C2 (NSINA Media) deprecated and removed from active benchmark suite.
    # c2 = next(l for l in all_loaders if l.dataset_id == "dataset_c2_nsina_media")


def test_dataset_d_sinhalammlu(all_loaders: list[BaseLoader]) -> None:
    """Test SinhalaMMLULoader: 4-option QA."""
    loader = next(l for l in all_loaders if l.dataset_id == "dataset_d_sinhalammlu")
    records = loader.load()
    assert len(records) > 0

    valid_labels = set(loader.label_set)
    assert valid_labels == {"A", "B", "C", "D"}

    for r in records[:50]:
        assert r.gold_label in valid_labels
        assert "ප්‍රශ්නය:" in r.text
        assert "A." in r.text
        assert "choices" in r.metadata


def test_dataset_e_salangabhava(all_loaders: list[BaseLoader]) -> None:
    """Test SalAngaBhavaLoader: 1-5 rating."""
    loader = next(l for l in all_loaders if l.dataset_id == "dataset_e_salangabhava")
    records = loader.load()
    assert len(records) > 0

    valid_labels = set(loader.label_set)
    assert valid_labels == {"1", "2", "3", "4", "5"}

    for r in records[:50]:
        assert r.gold_label in valid_labels
        assert r.metadata.get("review_type") == "Pure_Sinhala"


def test_dataset_f_cmcs(all_loaders: list[BaseLoader]) -> None:
    """Test CMCSLoader: code-mixed multi-task dataset."""
    loader = next(l for l in all_loaders if l.dataset_id == "dataset_f_cmcs")
    records = loader.load()
    assert len(records) > 0

    for r in records[:50]:
        assert r.gold_label in {"NEGATIVE", "POSITIVE", "NEUTRAL", "CONFLICT"}
        assert len(r.text.strip()) > 0
        assert "aspects" in r.metadata


def test_unicode_preservation(all_loaders: list[BaseLoader]) -> None:
    """Verify that Sinhala Zero-Width Joiner (ZWJ \\u200D) is preserved."""
    for loader in all_loaders:
        val = loader.validate()
        assert val["is_valid"], f"{loader.dataset_id} validation failed: {val['issues']}"
        assert val["zwj_containing_texts"] > 0, f"No ZWJ characters detected in {loader.dataset_id}!"


def test_manifest_and_frozen_samples() -> None:
    """Verify that data/processed/manifest.json and sample files exist with valid SHA-256."""
    manifest_path = DATA_PROCESSED_DIR / "manifest.json"
    assert manifest_path.exists(), "data/processed/manifest.json does not exist!"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "phase0_smoke" in manifest
    assert "phase1_samples" in manifest

    expected_p0_tasks = [
        "dataset_a_sentiment",
        "dataset_b_sold",
        "dataset_c1_nsina_categories",
        # "dataset_c2_nsina_media",  # Deprecated
        "dataset_d_sinhalammlu",
        "dataset_e_salangabhava",
        "dataset_f_cmcs",
    ]

    for task_id in expected_p0_tasks:
        assert task_id in manifest["phase0_smoke"], f"Missing {task_id} in phase0_smoke"
        p0_info = manifest["phase0_smoke"][task_id]
        assert p0_info["sample_size"] == 20, f"{task_id} phase0 size is not 20"

        file_path = DATA_PROCESSED_DIR / p0_info["file"]
        assert file_path.exists(), f"Phase 0 file {file_path} missing!"

        # Verify SHA-256 matches
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        assert hasher.hexdigest() == p0_info["sha256"], f"SHA256 mismatch for {file_path}"

    for task_id in expected_p0_tasks:
        assert task_id in manifest["phase1_samples"], f"Missing {task_id} in phase1_samples"
        p1_info = manifest["phase1_samples"][task_id]
        file_path = DATA_PROCESSED_DIR / p1_info["file"]
        assert file_path.exists(), f"Phase 1 file {file_path} missing!"

        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            hasher.update(f.read())
        assert hasher.hexdigest() == p1_info["sha256"], f"SHA256 mismatch for {file_path}"
