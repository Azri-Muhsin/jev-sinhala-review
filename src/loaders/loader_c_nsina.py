"""
loader_c_nsina.py — Dataset C: NSINA Categories (C1) and NSINA Media Identification (C2).

Source: Hugging Face `sinhala-nlp/NSINA-Categories` and `sinhala-nlp/NSINA-Media`.
Official test splits:
  - C1 Categories: news category prediction across categories.
  - C2 Media: media source identification across 10 news sources.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import DATA_RAW_DIR
from src.loaders.base import BaseLoader, DatasetRecord


class NSINACategoriesLoader(BaseLoader):
    """Loader for Dataset C1: NSINA Categories."""

    DATASET_ID = "dataset_c1_nsina_categories"

    def __init__(self, raw_dir: Path | None = None) -> None:
        self.raw_dir = raw_dir or (DATA_RAW_DIR / "dataset_c_nsina" / "categories")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._label_set: list[str] = []

    @property
    def dataset_id(self) -> str:
        return self.DATASET_ID

    @property
    def label_set(self) -> list[str]:
        if not self._label_set:
            # Preload or default
            self._label_set = ["International News", "Local News", "Sports", "Business"]
        return list(self._label_set)

    def load(self) -> list[DatasetRecord]:
        """Load test split for news category prediction."""
        cache_file = self.raw_dir / "test.tsv"

        if not cache_file.exists():
            token = os.environ.get("HF_TOKEN")
            p = hf_hub_download(
                "sinhala-nlp/NSINA-Categories",
                "test.tsv",
                repo_type="dataset",
                token=token,
            )
            shutil.copy(p, cache_file)

        df = pd.read_csv(cache_file, sep="\t")

        # Determine text and label columns
        text_col = "News Content" if "News Content" in df.columns else df.columns[0]
        label_col = "Category" if "Category" in df.columns else df.columns[-1]

        # Dynamically record unique labels
        unique_labels = sorted(df[label_col].dropna().unique().tolist())
        self._label_set = [str(lbl).strip() for lbl in unique_labels]

        records: list[DatasetRecord] = []
        for idx, row in df.iterrows():
            raw_text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
            raw_label = str(row[label_col]).strip() if pd.notna(row[label_col]) else ""
            if not raw_text or not raw_label:
                continue

            record = DatasetRecord(
                example_id=f"nsina_cat_{idx:05d}",
                dataset=self.dataset_id,
                text=raw_text,
                gold_label=raw_label,
                split="test",
                metadata={"category": raw_label},
            )
            records.append(record)

        return records


class NSINAMediaLoader(BaseLoader):
    """Loader for Dataset C2: NSINA Media Identification."""

    DATASET_ID = "dataset_c2_nsina_media"

    def __init__(self, raw_dir: Path | None = None) -> None:
        self.raw_dir = raw_dir or (DATA_RAW_DIR / "dataset_c_nsina" / "media")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self._label_set: list[str] = []

    @property
    def dataset_id(self) -> str:
        return self.DATASET_ID

    @property
    def label_set(self) -> list[str]:
        if not self._label_set:
            self._label_set = [
                "www.lankadeepa.lk",
                "ITN news",
                "www.vikalpa.org",
                "Lankatruth",
                "divaina",
                "Adaderana",
                "hirunews",
                "https://sinhala.news.lk",
                "dinamina",
                "Siyatha News",
            ]
        return list(self._label_set)

    def load(self) -> list[DatasetRecord]:
        """Load test split for news media source identification."""
        cache_file = self.raw_dir / "test.tsv"

        if not cache_file.exists():
            token = os.environ.get("HF_TOKEN")
            p = hf_hub_download(
                "sinhala-nlp/NSINA-Media",
                "test.tsv",
                repo_type="dataset",
                token=token,
            )
            shutil.copy(p, cache_file)

        df = pd.read_csv(cache_file, sep="\t")

        text_col = "News Content" if "News Content" in df.columns else df.columns[0]
        label_col = "Source" if "Source" in df.columns else df.columns[-1]

        unique_labels = sorted(df[label_col].dropna().unique().tolist())
        self._label_set = [str(lbl).strip() for lbl in unique_labels]

        records: list[DatasetRecord] = []
        for idx, row in df.iterrows():
            raw_text = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
            raw_label = str(row[label_col]).strip() if pd.notna(row[label_col]) else ""
            if not raw_text or not raw_label:
                continue

            record = DatasetRecord(
                example_id=f"nsina_media_{idx:05d}",
                dataset=self.dataset_id,
                text=raw_text,
                gold_label=raw_label,
                split="test",
                metadata={"source": raw_label},
            )
            records.append(record)

        return records

