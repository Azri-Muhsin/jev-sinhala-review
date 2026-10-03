"""
loader_f_cmcs.py — Dataset F: Sinhala-English Code-Mixed & Code-Switched Dataset (CMCS).

Source: Hugging Face `NLPC-UOM/Sinhala-English-Code-Mixed-Code-Switched-Dataset`.
Total sentences: 13,518 rows.
Annotated tasks:
  1. Sentiment (Negative, Positive, Neutral, Conflict)
  2. Humour / Humor (Humorous, Non-humorous)
  3. Hate Speech (Not offensive, Hate-Inducing, Abusive)
  4. Aspect Extraction (multi-label / presence of: billing, customer service, data, network, package, service)
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


class CMCSLoader(BaseLoader):
    """Loader for Dataset F: CMCS (Code-Mixed Multi-Task)."""

    DATASET_ID = "dataset_f_cmcs"
    LABELS = ["NEGATIVE", "POSITIVE", "NEUTRAL", "CONFLICT"]

    def __init__(
        self,
        raw_dir: Path | None = None,
        primary_task: str = "sentiment",
    ) -> None:
        self.raw_dir = raw_dir or (DATA_RAW_DIR / self.DATASET_ID)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.primary_task = primary_task.lower()

    @property
    def dataset_id(self) -> str:
        return self.DATASET_ID

    @property
    def label_set(self) -> list[str]:
        if self.primary_task == "sentiment":
            return ["NEGATIVE", "POSITIVE", "NEUTRAL", "CONFLICT"]
        elif self.primary_task in {"humour", "humor"}:
            return ["HUMOROUS", "NON-HUMOROUS"]
        elif self.primary_task in {"hate", "hate_speech"}:
            return ["NOT OFFENSIVE", "HATE-INDUCING", "ABUSIVE"]
        return list(self.LABELS)

    def load(self) -> list[DatasetRecord]:
        """Load code-mixed sentences and subtask annotations."""
        cache_file = self.raw_dir / "sentence-level-annotation.csv"

        if not cache_file.exists():
            token = os.environ.get("HF_TOKEN")
            p = hf_hub_download(
                "NLPC-UOM/Sinhala-English-Code-Mixed-Code-Switched-Dataset",
                "sentence-level-annotation.csv",
                repo_type="dataset",
                token=token,
            )
            shutil.copy(p, cache_file)

        df = pd.read_csv(cache_file)
        # Strip any extraneous quotes from column headers
        df.columns = [c.strip().strip("'\"") for c in df.columns]

        text_col = "Sentence" if "Sentence" in df.columns else "Text"

        records: list[DatasetRecord] = []
        for idx, row in df.iterrows():
            text_val = row.get(text_col)
            if pd.isna(text_val):
                continue
            text = str(text_val).strip()
            if not text:
                continue

            sentiment = str(row.get("Sentiment", "")).strip().upper() if pd.notna(row.get("Sentiment")) else None
            humor = str(row.get("Humor", "")).strip().upper() if pd.notna(row.get("Humor")) else None
            hate = str(row.get("Hate_speech", "")).strip().upper() if pd.notna(row.get("Hate_speech")) else None

            # Collect active aspects
            aspect_keys = [
                "Billing or price",
                "Customer service",
                "Data",
                "Network",
                "Package",
                "Service or product",
            ]
            active_aspects = [
                k for k in aspect_keys if k in row and row[k] == 1
            ]

            gold_label: str
            if self.primary_task == "sentiment":
                if not sentiment or sentiment not in {"NEGATIVE", "POSITIVE", "NEUTRAL", "CONFLICT"}:
                    continue
                gold_label = sentiment
            elif self.primary_task in {"humour", "humor"}:
                if not humor:
                    continue
                gold_label = humor
            elif self.primary_task in {"hate", "hate_speech"}:
                if not hate:
                    continue
                gold_label = hate
            else:
                gold_label = sentiment or "UNKNOWN"

            record = DatasetRecord(
                example_id=f"cmcs_{idx:05d}",
                dataset=self.dataset_id,
                text=text,
                gold_label=gold_label,
                split="test",
                metadata={
                    "sentiment": sentiment,
                    "humor": humor,
                    "hate_speech": hate,
                    "aspects": active_aspects,
                    "script_type": "code_mixed",
                },
            )
            records.append(record)

        return records
