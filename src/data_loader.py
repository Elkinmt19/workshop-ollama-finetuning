"""Data loading and preprocessing utilities for fine-tuning."""

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class TrainingExample(BaseModel):
    """Single training example."""

    instruction: str = Field(..., min_length=1, max_length=500)
    input: str = Field(default="", max_length=2000)
    output: str = Field(..., min_length=1, max_length=2000)

    @validator("instruction", "input", "output", pre=True)
    def normalize_whitespace(cls, v):
        """Normalize whitespace in text fields."""
        if isinstance(v, str):
            return " ".join(v.split())
        return v

    def to_prompt(self) -> str:
        """Format as model prompt."""
        if self.input:
            return f"{self.instruction}\n{self.input}"
        return self.instruction

    def to_chat_messages(self) -> list[dict]:
        """Format as chat messages for tokenizer.apply_chat_template."""
        return [
            {"role": "user", "content": self.to_prompt()},
            {"role": "assistant", "content": self.output},
        ]


class DataLoader:
    """Load and validate training data from various formats."""

    @staticmethod
    def load_jsonl(path: Path) -> Iterator[TrainingExample]:
        """Load JSONL file, yielding validated examples."""
        with open(path) as f:
            for i, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    yield TrainingExample(**data)
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Skipping line {i} in {path}: {e}")

    @staticmethod
    def load_csv(path: Path, delimiter: str = ",") -> Iterator[TrainingExample]:
        """Load CSV file, yielding validated examples."""
        df = pd.read_csv(path, delimiter=delimiter)
        for idx, row in df.iterrows():
            try:
                example = TrainingExample(
                    instruction=row.get("instruction", ""),
                    input=row.get("input", ""),
                    output=row.get("output", ""),
                )
                yield example
            except ValueError as e:
                logger.warning(f"Skipping row {idx} in {path}: {e}")

    @staticmethod
    def load_dataset(path: Path) -> list[TrainingExample]:
        """Load entire dataset into memory."""
        if path.suffix == ".jsonl":
            return list(DataLoader.load_jsonl(path))
        elif path.suffix == ".csv":
            return list(DataLoader.load_csv(path))
        else:
            raise ValueError(f"Unsupported format: {path.suffix}")

    @staticmethod
    def validate_dataset(path: Path, sample_size: Optional[int] = None) -> dict:
        """Validate dataset and return summary statistics."""
        examples = DataLoader.load_dataset(path)
        if sample_size:
            examples = examples[:sample_size]

        if not examples:
            raise ValueError(f"No valid examples found in {path}")

        instructions = [ex.instruction for ex in examples]
        outputs = [ex.output for ex in examples]

        return {
            "total_examples": len(examples),
            "avg_instruction_length": sum(len(i) for i in instructions) / len(instructions),
            "max_instruction_length": max(len(i) for i in instructions),
            "avg_output_length": sum(len(o) for o in outputs) / len(outputs),
            "max_output_length": max(len(o) for o in outputs),
        }
