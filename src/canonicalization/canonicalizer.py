"""Stable canonical representation for the Lab 1 classification path."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.intent.extractor import IntentDraft


@dataclass(frozen=True)
class CanonicalIntent:
    """Validated, stable user intent used to generate a pipeline spec."""

    version: str
    task_type: str
    target_column: str
    missing_marker: str
    numeric_imputation: str
    categorical_imputation: str
    categorical_encoding: str
    scale_numeric: bool
    split_strategy: str

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable canonical intent data."""
        return asdict(self)


def canonicalize_intent(intent: IntentDraft) -> CanonicalIntent:
    """Convert a complete intent draft into the fixed Lab 1 representation."""
    if intent.status != "confirmed":
        raise ValueError("Intent must be explicitly confirmed before canonicalization.")
    if not intent.target_column:
        raise ValueError("A target column is required before canonicalization.")
    if intent.task_type != "classification":
        raise ValueError("Lab 1 supports classification intents only.")

    return CanonicalIntent(
        version="1.0",
        task_type="classification",
        target_column=intent.target_column,
        missing_marker="?",
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        categorical_encoding="one_hot",
        scale_numeric=bool(intent.scale_numeric),
        split_strategy="stratified",
    )
