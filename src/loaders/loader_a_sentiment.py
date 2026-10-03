"""
loader_a_sentiment.py — Dataset A: Sinhala News-Comment Sentiment (4-way).

Source: tallip repository / Azri-Muhsin/sinhala-news-comment-sentiment / sinhala-nlp.
Official test split: 15,059 examples with 4 classes:
  - POSITIVE
  - NEGATIVE
  - NEUTRAL
  - CONFLICT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from src.config import DATA_RAW_DIR
from src.loaders.base import BaseLoader, DatasetRecord


class SentimentLoader(BaseLoader):
    """Loader for Dataset A: Sinhala News-Comment Sentiment."""

    DATASET_ID = "dataset_a_sentiment"
    LABELS = ["POSITIVE", "NEGATIVE", "NEUTRAL", "CONFLICT"]

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
        """Load official evaluation/test split."""
        cache_file = self.raw_dir / "test.parquet"

        df: pd.DataFrame
        if cache_file.exists():
            df = pd.read_parquet(cache_file)
        else:
            # 1. Primary: Azri-Muhsin/sinhala-news-comment-sentiment (15,059 test rows with 4 classes)
            token = os.environ.get("HF_TOKEN")
            try:
                ds = load_dataset(
                    "Azri-Muhsin/sinhala-news-comment-sentiment",
                    split="test",
                    token=token,
                )
                df = ds.to_pandas()
                df.to_parquet(cache_file, index=False)
            except Exception:
                # Fallback: sinhala-nlp/sinhala-sentiment-analysis
                p = hf_hub_download(
                    "sinhala-nlp/sinhala-sentiment-analysis",
                    "test.tsv",
                    repo_type="dataset",
                    token=token,
                )
                df = pd.read_csv(p, sep="\t")
                # Normalize column names
                if "comment_phrase" in df.columns and "text" not in df.columns:
                    df["text"] = df["comment_phrase"]
                if "comment_sentiment" in df.columns and "label" not in df.columns:
                    df["label"] = df["comment_sentiment"]
                df.to_parquet(cache_file, index=False)

        # Ensure text and label columns exist
        text_col = "text" if "text" in df.columns else "comment_phrase"
        label_col = "label" if "label" in df.columns else "comment_sentiment"

        records: list[DatasetRecord] = []
        for idx, row in df.iterrows():
            raw_text = str(row[text_col]) if pd.notna(row[text_col]) else ""
            raw_label = str(row[label_col]).strip().upper() if pd.notna(row[label_col]) else ""

            # Standardize label casing (e.g. Positive -> POSITIVE)
            if raw_label not in self.LABELS:
                # Handle numeric labels if present (0: NEGATIVE, 1: NEUTRAL, 2: POSITIVE)
                num_map = {"0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE", 0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
                if row[label_col] in num_map:
                    raw_label = num_map[row[label_col]]

            record = DatasetRecord(
                example_id=f"sentiment_test_{idx:05d}",
                dataset=self.dataset_id,
                text=raw_text,
                gold_label=raw_label,
                split="test",
                metadata={
                    "article_id": row.get("article_id") if "article_id" in row else None,
                    "comment_author": row.get("comment_author") if "comment_author" in row else None,
                },
            )
            records.append(record)

        return records
