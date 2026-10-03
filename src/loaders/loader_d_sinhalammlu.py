"""
loader_d_sinhalammlu.py — Dataset D: SinhalaMMLU (4-option QA).

Source: Hugging Face `naist-nlp/SinhalaMMLU`.
Multi-discipline QA across domains: Humanities, Social Sciences, STEM, etc.
Options: A, B, C, D with question and Sinhala choices.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download

from src.config import DATA_RAW_DIR
from src.loaders.base import BaseLoader, DatasetRecord


class SinhalaMMLULoader(BaseLoader):
    """Loader for Dataset D: SinhalaMMLU."""

    DATASET_ID = "dataset_d_sinhalammlu"
    LABELS = ["A", "B", "C", "D"]

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
        """Load all evaluation questions across subjects."""
        token = os.environ.get("HF_TOKEN")
        api = HfApi(token=token)

        # Download and cache json files if needed
        cached_json_files = list(self.raw_dir.glob("*.json"))
        if not cached_json_files:
            remote_files = [
                f for f in api.list_repo_files("naist-nlp/SinhalaMMLU", repo_type="dataset")
                if f.startswith("TEST/") and f.endswith(".json")
            ]
            for rf in remote_files:
                p = hf_hub_download(
                    "naist-nlp/SinhalaMMLU",
                    rf,
                    repo_type="dataset",
                    token=token,
                )
                dest = self.raw_dir / Path(rf).name
                shutil.copy(p, dest)
            cached_json_files = list(self.raw_dir.glob("*.json"))

        records: list[DatasetRecord] = []
        option_letters = ["A", "B", "C", "D"]

        for jf in sorted(cached_json_files):
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)

            for idx, item in enumerate(data):
                q_no = item.get("q_no", str(idx))
                question = item.get("question", "").strip()
                choices = item.get("choices", [])
                answer = str(item.get("answer", "")).strip().upper()
                subject = item.get("subject", "general")
                category = item.get("category", "general")
                meta = item.get("metadata", {})

                # Format formatted state text including choices
                # e.g.:
                # ප්‍රශ්නය: ලංකාවේ ප්‍රථම වරට චිත්‍ර ප්‍රදර්ශනයක් පැවැත්වූ විදේශීය කලාකරුවා කවුද?
                # A. ජෝර්ජ් කීට්
                # B. ඩේවිඩ් පේන්ටර්
                # C. සී. එෆ්. වින්සර්
                # D. ඒ. සී. ජී. එස්. අමරසේකර
                choices_dict: dict[str, str] = {}
                formatted_choices = []
                for i, opt_letter in enumerate(option_letters):
                    if i < len(choices):
                        choice_text = str(choices[i]).strip()
                        choices_dict[opt_letter] = choice_text
                        formatted_choices.append(f"{opt_letter}. {choice_text}")

                full_text = f"ප්‍රශ්නය: {question}\n\nවිකල්ප:\n" + "\n".join(formatted_choices)

                # Map answer if 1-4 numeric
                if answer in {"1", "2", "3", "4"}:
                    answer = option_letters[int(answer) - 1]

                if answer not in self.LABELS or not question:
                    continue

                ex_id = f"mmlu_{subject}_{q_no}_{idx}".replace(" ", "_").lower()

                record = DatasetRecord(
                    example_id=ex_id,
                    dataset=self.dataset_id,
                    text=full_text,
                    gold_label=answer,
                    split="test",
                    metadata={
                        "question": question,
                        "choices": choices_dict,
                        "subject": subject,
                        "category": category,
                        "difficulty": meta.get("difficulty"),
                        "grade": meta.get("grade"),
                    },
                )
                records.append(record)

        return records
