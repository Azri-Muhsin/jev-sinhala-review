"""
base.py — Abstract base class for dataset loaders.

All dataset loaders inherit from ``BaseLoader`` and implement the
``load()`` and ``get_label_map()`` methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class DatasetRecord:
    """A single example from any dataset in the probe.

    Attributes
    ----------
    example_id : str
        Unique identifier (e.g., ``"tallip_test_0481"``).
    dataset : str
        Dataset key (e.g., ``"dataset_a_sentiment"``).
    text : str
        Original text, preserved byte-for-byte (no normalization).
    gold_label : str
        Ground-truth label from the original dataset.
    split : str
        Dataset split name (e.g., ``"test"``).
    metadata : dict
        Any extra fields (e.g., ``subject``, ``domain``, ``review_type``).
    """

    example_id: str
    dataset: str
    text: str
    gold_label: str
    split: str = "test"
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLoader(ABC):
    """Abstract base class for dataset loaders."""

    @property
    @abstractmethod
    def dataset_id(self) -> str:
        """Return the dataset key (e.g., ``'dataset_a_sentiment'``)."""
        ...

    @property
    @abstractmethod
    def label_set(self) -> list[str]:
        """Return the ordered list of valid gold labels."""
        ...

    @abstractmethod
    def load(self) -> list[DatasetRecord]:
        """Load and return all records from the evaluation split.

        Returns
        -------
        list[DatasetRecord]
            Every example in the official test/eval split.
        """
        ...

    def to_dataframe(self) -> pd.DataFrame:
        """Load data and convert to a DataFrame."""
        records = self.load()
        return pd.DataFrame([
            {
                "example_id": r.example_id,
                "dataset": r.dataset,
                "text": r.text,
                "gold_label": r.gold_label,
                "split": r.split,
                **r.metadata,
            }
            for r in records
        ])

    def validate(self) -> dict[str, Any]:
        """Run basic validation checks on the loaded data.

        Returns
        -------
        dict
            Validation results including counts, label distribution,
            encoding checks, and any issues found.
        """
        records = self.load()
        issues: list[str] = []

        # Check for empty text
        empty_text = [r for r in records if not r.text or not r.text.strip()]
        if empty_text:
            issues.append(f"{len(empty_text)} records have empty text")

        # Check for missing labels
        missing_labels = [r for r in records if not r.gold_label]
        if missing_labels:
            issues.append(f"{len(missing_labels)} records have missing labels")

        # Check for unexpected labels
        valid_labels = set(self.label_set)
        unexpected = [r for r in records if r.gold_label not in valid_labels]
        if unexpected:
            bad_labels = set(r.gold_label for r in unexpected)
            issues.append(
                f"{len(unexpected)} records have unexpected labels: {bad_labels}"
            )

        # Label distribution
        from collections import Counter
        label_dist = Counter(r.gold_label for r in records)

        # Unicode ZWJ/ZWNJ preservation check (sample first 5 texts)
        zwj_count = sum(1 for r in records if "\u200D" in r.text)
        zwnj_count = sum(1 for r in records if "\u200C" in r.text)

        return {
            "dataset_id": self.dataset_id,
            "total_records": len(records),
            "label_distribution": dict(label_dist),
            "valid_labels": self.label_set,
            "zwj_containing_texts": zwj_count,
            "zwnj_containing_texts": zwnj_count,
            "issues": issues,
            "is_valid": len(issues) == 0,
        }
