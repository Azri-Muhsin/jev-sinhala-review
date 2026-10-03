"""
loader_e_salangabhava.py — Dataset E: SalAngaBhava (Ordinal 1-5 Product Rating).

Source: Hugging Face `lakshani005/SalAngaBhava`.
Reviews from 6 product domains (electronics, fashion, grocery, home, skincare).
Target: 1-5 ordinal rating.
Condition for core probe: `review_type == 'Pure_Sinhala'`.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from src.config import DATA_RAW_DIR
from src.loaders.base import BaseLoader, DatasetRecord


class SalAngaBhavaLoader(BaseLoader):
    """Loader for Dataset E: SalAngaBhava."""

    DATASET_ID = "dataset_e_salangabhava"
    LABELS = ["1", "2", "3", "4", "5"]

    CSV_FILES = [
        "electronic_product_reviews_cleaned_part1.csv",
        "electronic_product_reviews_cleaned_part2.csv",
        "fashion_item_product_reviews_cleaned.csv",
        "grocery_product_reviews_cleaned.csv",
        "home_item_reviews_cleaned.csv",
        "skincare_product_reviews_cleaned.csv",
    ]

    def __init__(
        self,
        raw_dir: Path | None = None,
        filter_pure_sinhala: bool = True,
    ) -> None:
        self.raw_dir = raw_dir or (DATA_RAW_DIR / self.DATASET_ID)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.filter_pure_sinhala = filter_pure_sinhala

    @property
    def dataset_id(self) -> str:
        return self.DATASET_ID

    @property
    def label_set(self) -> list[str]:
        return list(self.LABELS)

    def load(self) -> list[DatasetRecord]:
        """Load and parse product reviews."""
        token = os.environ.get("HF_TOKEN")

        # Download CSV files if not present
        cached_csvs: list[Path] = []
        for fname in self.CSV_FILES:
            dest = self.raw_dir / fname
            if not dest.exists():
                try:
                    p = hf_hub_download(
                        "lakshani005/SalAngaBhava",
                        fname,
                        repo_type="dataset",
                        token=token,
                    )
                    shutil.copy(p, dest)
                except Exception:
                    continue
            if dest.exists():
                cached_csvs.append(dest)

        records: list[DatasetRecord] = []
        for csv_path in sorted(cached_csvs):
            domain = csv_path.stem.replace("_cleaned", "").replace("_product_reviews", "")
            df = pd.read_csv(csv_path)

            for idx, row in df.iterrows():
                review_type = str(row.get("review_type", "")).strip()

                if self.filter_pure_sinhala and review_type != "Pure_Sinhala":
                    continue

                raw_text = str(row.get("review", "")).strip() if pd.notna(row.get("review")) else ""
                raw_rating = row.get("rating")
                if pd.isna(raw_rating) or not raw_text:
                    continue

                try:
                    rating_int = int(float(raw_rating))
                    if rating_int not in {1, 2, 3, 4, 5}:
                        continue
                    rating_str = str(rating_int)
                except (ValueError, TypeError):
                    continue

                ex_id = f"salanga_{domain}_{idx:05d}"
                record = DatasetRecord(
                    example_id=ex_id,
                    dataset=self.dataset_id,
                    text=raw_text,
                    gold_label=rating_str,
                    split="test",
                    metadata={
                        "product_name": row.get("product_name"),
                        "review_title": row.get("review_title"),
                        "review_type": review_type,
                        "rating_numeric": rating_int,
                        "domain": domain,
                        "sentiment": row.get("sentiment"),
                    },
                )
                records.append(record)

        return records
