"""Tests for constrained extraction, validation, and confirmation contracts."""

from __future__ import annotations

from typing import Any

import pytest

from src.canonicalization import canonicalize_intent
from src.intent import (
    IntentExtractionError,
    confirm_intent,
    create_manual_fallback_intent,
    evaluate_confidence,
    extract_intent,
    request_clarification,
)


class FakeGenerator:
    """Return configured structured data and capture extraction constraints."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.prompt = ""
        self.schema: dict[str, Any] | None = None
        self.system: str | None = None

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Capture the request and return the configured object."""
        self.prompt = prompt
        self.schema = schema
        self.system = system
        return self.response


def _extract(response: dict[str, Any]) -> tuple[object, FakeGenerator]:
    """Extract intent against a small Adult-like metadata context."""
    generator = FakeGenerator(response)
    intent = extract_intent(
        "Prepare the data to predict income",
        ["age", "income"],
        {"age": "int64", "income": "object"},
        ["income"],
        generator,
    )
    return intent, generator


def test_valid_ollama_intent_is_tentative_and_metadata_only() -> None:
    """Valid structured output is resolved but still needs user confirmation."""
    intent, generator = _extract(
        {"task_type": "classification", "target_column": "income", "scale_numeric": None}
    )
    confidence = evaluate_confidence(intent)  # type: ignore[arg-type]

    assert intent.target_column == "income"  # type: ignore[union-attr]
    assert intent.source == "ollama"  # type: ignore[union-attr]
    assert intent.status == "tentative"  # type: ignore[union-attr]
    assert confidence.confidence == 1.0
    assert confidence.missing_information == ["user_confirmation"]
    assert generator.schema is not None
    assert "column_names" in generator.prompt
    assert "32,561" not in generator.prompt


def test_case_insensitive_target_is_resolved_to_real_column() -> None:
    """A unique case-only difference maps back to the exact dataset name."""
    intent, _ = _extract(
        {"task_type": "classification", "target_column": "INCOME", "scale_numeric": False}
    )
    assert intent.target_column == "income"  # type: ignore[union-attr]
    assert intent.suggested_target_column == "INCOME"  # type: ignore[union-attr]


def test_unidentified_task_is_reported_as_incomplete() -> None:
    """A schema-valid null task cannot proceed without manual confirmation."""
    intent, _ = _extract(
        {"task_type": None, "target_column": "income", "scale_numeric": None}
    )
    confidence = evaluate_confidence(intent)  # type: ignore[arg-type]
    assert "task_type" in confidence.missing_information
    assert any("classification task" in issue for issue in intent.validation_issues)  # type: ignore[union-attr]


@pytest.mark.parametrize("target", [None, "salary"])
def test_missing_or_unknown_target_requests_clarification(target: str | None) -> None:
    """Unresolved targets remain blocked until the user selects a real column."""
    intent, _ = _extract(
        {"task_type": "classification", "target_column": target, "scale_numeric": None}
    )
    confidence = evaluate_confidence(intent)  # type: ignore[arg-type]
    clarification = request_clarification(confidence)
    assert not confidence.sufficient
    assert "target_column" in confidence.missing_information
    assert clarification.question == "Which column should be predicted?"


@pytest.mark.parametrize(
    "response",
    [
        {"task_type": "classification", "target_column": "income"},
        {
            "task_type": "classification",
            "target_column": "income",
            "scale_numeric": "yes",
        },
        {
            "task_type": "classification",
            "target_column": "income",
            "scale_numeric": False,
            "pipeline_code": "print('unsafe')",
        },
    ],
)
def test_invalid_or_extra_model_fields_are_rejected(response: dict[str, Any]) -> None:
    """Strict validation rejects incomplete, mistyped, and additional fields."""
    with pytest.raises(IntentExtractionError, match="invalid intent object"):
        _extract(response)


def test_confirmation_is_required_before_canonicalization() -> None:
    """Only explicit user confirmation allows deterministic canonicalization."""
    tentative, _ = _extract(
        {"task_type": "classification", "target_column": "income", "scale_numeric": True}
    )
    with pytest.raises(ValueError, match="explicitly confirmed"):
        canonicalize_intent(tentative)  # type: ignore[arg-type]

    confirmed = confirm_intent(tentative, "income", True, ["age", "income"])  # type: ignore[arg-type]
    confidence = evaluate_confidence(confirmed)
    canonical = canonicalize_intent(confirmed)
    assert confidence.sufficient
    assert canonical.target_column == "income"
    assert canonical.scale_numeric is True


def test_manual_fallback_uses_the_same_confirmation_gate() -> None:
    """An unavailable model cannot bypass explicit target confirmation."""
    fallback = create_manual_fallback_intent("Ollama is offline")
    assert fallback.source == "manual_fallback"
    assert not evaluate_confidence(fallback).sufficient
    confirmed = confirm_intent(fallback, "income", False, ["age", "income"])
    assert evaluate_confidence(confirmed).sufficient
