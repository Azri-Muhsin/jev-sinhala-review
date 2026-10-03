"""
loader_b_sold.py — Dataset B: Sinhala Offensive Language Dataset (SOLD).

Source: Hugging Face `sinhala-nlp/SOLD` (file: `SOLD_test.tsv`).
Official test split: 2,500 examples with binary labels:
  - OFF (Offensive)
  - NOT (Not offensive)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

from src.config import DATA_RAW_DIR
from src.loaders.base import BaseLoader, DatasetRecord


class SOLDLoader(BaseLoader):
    """Loader for Dataset B: SOLD (Sinhala Offensive Language)."""

    DATASET_ID = "dataset_b_sold"
    LABELS = ["NOT", "OFF"]

    def __init__(self, raw_dir: Path | None = None) -> None:
        self.raw_dir = raw_dir or (DATA_RAW_DIR / self.DATASET_ID)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @property
    def dataset_id(self) -> str:
        return self.DATASET_ID

    @property
    def label_set(self) -> list[str]:
        return list(self.LABELS)

    def load(self) -> list[DatasetRecord]:
        """Load official evaluation/test split from SOLD."""
        cache_file = self.raw_dir / "SOLD_test.tsv"

        if not cache_file.exists():
            token = os.environ.get("HF_TOKEN")
            p = hf_hub_download(
                "sinhala-nlp/SOLD",
                "SOLD_test.tsv",
                repo_type="dataset",
                token=token,
            )
            # Copy to raw_dir for caching
            import shutil
            shutil.copy(p, cache_file)

        df = pd.read_csv(cache_file, sep="\t")

        # Column names: post_id, text, tokens, rationales, label
        text_col = "text" if "text" in df.columns else "post"
        id_col = "post_id" if "post_id" in df.columns else "ID"
        label_col = "label"

        records: list[DatasetRecord] = []
        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            raw_label = str(row[label_col]).strip().upper() if pd.notna(row[label_col]) else ""
            ex_id = str(row[id_col]) if id_col in row and pd.notna(row[id_col]) else f"sold_{idx:05d}"

            record = DatasetRecord(
                example_id=f"sold_{ex_id}",
                dataset=self.dataset_id,
                text=raw_text,
                gold_label=raw_label,
                split="test",
                metadata={
                    "post_id": ex_id,
                    "tokens": row.get("tokens") if "tokens" in row else None,
                },
            )
            records.append(record)

        return records
