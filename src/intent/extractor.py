"""Constrained Ollama-backed intent extraction and user confirmation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationError


class IntentExtractionError(ValueError):
    """Raised when model output does not satisfy the strict intent schema."""


class StructuredJsonGenerator(Protocol):
    """Provider-independent interface required by the intent extractor."""

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Generate a structured JSON object."""


class ModelIntentResponse(BaseModel):
    """Only semantic fields Ollama may infer during Lab 1."""

    model_config = ConfigDict(extra="forbid", strict=True)

    task_type: Literal["classification"] | None
    target_column: str | None
    scale_numeric: bool | None


@dataclass(frozen=True)
class IntentDraft:
    """Tentative or user-confirmed intent with provenance and validation state."""

    task_type: str | None
    target_column: str | None
    suggested_target_column: str | None
    scale_numeric: bool | None
    requires_cleaning: bool
    requires_preprocessing: bool
    requires_training: bool
    source: Literal["ollama", "manual_fallback"]
    status: Literal["tentative", "confirmed"] = "tentative"
    validation_issues: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable intent data."""
        return asdict(self)


def _resolve_target(target: str | None, columns: list[str]) -> tuple[str | None, tuple[str, ...]]:
    """Resolve exact or unique case-insensitive column names."""
    if target is None or not target.strip():
        return None, ("No target column was identified.",)
    candidate = target.strip()
    if candidate in columns:
        return candidate, ()
    matches = [column for column in columns if column.casefold() == candidate.casefold()]
    if len(matches) == 1:
        return matches[0], ()
    return None, (f"The inferred target column '{candidate}' does not exist in the dataset.",)


def _build_extraction_prompt(
    prompt: str,
    column_names: list[str],
    column_types: dict[str, str],
    possible_targets: list[str],
) -> str:
    """Build a compact metadata-only extraction request."""
    context = {
        "user_request": prompt,
        "dataset": {
            "column_names": column_names,
            "column_types": column_types,
            "possible_targets": possible_targets,
        },
        "instructions": [
            "Infer classification only when the request clearly describes classification.",
            "Use an exact dataset column name for target_column; otherwise return null.",
            "Set scale_numeric only when scaling is explicitly requested; otherwise return null.",
            "Do not propose code, pipeline steps, models, or preprocessing parameters.",
        ],
    }
    return json.dumps(context, ensure_ascii=False)


def extract_intent(
    prompt: str,
    column_names: list[str],
    column_types: dict[str, str],
    possible_targets: list[str],
    generator: StructuredJsonGenerator,
) -> IntentDraft:
    """Ask Ollama for a strict semantic draft and validate it against metadata."""
    if not prompt.strip():
        raise ValueError("A non-empty prompt is required.")
    raw = generator.generate_json(
        _build_extraction_prompt(prompt, column_names, column_types, possible_targets),
        schema=ModelIntentResponse.model_json_schema(),
        system=(
            "You extract tentative data-science intent as structured data. "
            "Return only fields allowed by the supplied schema. Never return executable code."
        ),
    )
    try:
        model_intent = ModelIntentResponse.model_validate(raw)
    except ValidationError as exc:
        raise IntentExtractionError(f"Ollama returned an invalid intent object: {exc}") from exc

    target, target_issues = _resolve_target(model_intent.target_column, column_names)
    issues = list(target_issues)
    if model_intent.task_type != "classification":
        issues.append("A supported classification task was not identified.")
    return IntentDraft(
        task_type=model_intent.task_type,
        target_column=target,
        suggested_target_column=model_intent.target_column,
        scale_numeric=model_intent.scale_numeric,
        requires_cleaning=True,
        requires_preprocessing=True,
        requires_training=False,
        source="ollama",
        validation_issues=tuple(issues),
    )


def create_manual_fallback_intent(reason: str) -> IntentDraft:
    """Create a safe tentative classification draft when Ollama cannot respond."""
    return IntentDraft(
        task_type="classification",
        target_column=None,
        suggested_target_column=None,
        scale_numeric=None,
        requires_cleaning=True,
        requires_preprocessing=True,
        requires_training=False,
        source="manual_fallback",
        validation_issues=(reason,),
    )


def confirm_intent(
    intent: IntentDraft,
    target_column: str,
    scale_numeric: bool,
    available_columns: list[str],
) -> IntentDraft:
    """Apply explicit user confirmation to the limited Lab 1 intent."""
    if target_column not in available_columns:
        raise ValueError("Select a target column that exists in the dataset.")
    return replace(
        intent,
        task_type="classification",
        target_column=target_column,
        scale_numeric=scale_numeric,
        status="confirmed",
        validation_issues=(),
    )
``