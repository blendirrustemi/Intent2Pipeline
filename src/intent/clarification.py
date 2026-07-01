"""Deterministic clarification for the limited Lab 1 interaction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.intent.confidence import ConfidenceResult


@dataclass(frozen=True)
class ClarificationResult:
    """A single clarification state for the current intent."""

    required: bool
    question: str | None
    status: str = "deterministic_validation"

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable clarification data."""
        return asdict(self)


def request_clarification(confidence: ConfidenceResult) -> ClarificationResult:
    """Request the first user action needed to complete the intent."""
    needs_target = "target_column" in confidence.missing_information
    needs_confirmation = "user_confirmation" in confidence.missing_information
    return ClarificationResult(
        required=needs_target or needs_confirmation,
        question=(
            "Which column should be predicted?"
            if needs_target
            else "Please confirm the interpreted classification intent."
            if needs_confirmation
            else None
        ),
    )
