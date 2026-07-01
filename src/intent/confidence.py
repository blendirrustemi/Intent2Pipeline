"""Deterministic completeness evaluation for tentative and confirmed intent."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.intent.extractor import IntentDraft


@dataclass(frozen=True)
class ConfidenceResult:
    """Confidence score and the information still needed."""

    confidence: float
    missing_information: list[str]
    sufficient: bool
    status: str = "deterministic_validation"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable confidence data."""
        return asdict(self)


def evaluate_confidence(intent: IntentDraft) -> ConfidenceResult:
    """Score observable required fields without trusting model self-confidence."""
    missing: list[str] = []
    completed = 0
    if intent.task_type == "classification":
        completed += 1
    else:
        missing.append("task_type")
    if intent.target_column:
        completed += 1
    else:
        missing.append("target_column")
    if intent.status != "confirmed":
        missing.append("user_confirmation")
    return ConfidenceResult(
        confidence=completed / 2,
        missing_information=missing,
        sufficient=not missing,
    )
