"""Placeholder intent-understanding contracts for Lab 1."""

from src.intent.clarification import ClarificationResult, request_clarification
from src.intent.confidence import ConfidenceResult, evaluate_confidence
from src.intent.extractor import (
    IntentDraft,
    IntentExtractionError,
    ModelIntentResponse,
    confirm_intent,
    create_manual_fallback_intent,
    extract_intent,
)

__all__ = [
    "ClarificationResult",
    "ConfidenceResult",
    "IntentDraft",
    "IntentExtractionError",
    "ModelIntentResponse",
    "confirm_intent",
    "create_manual_fallback_intent",
    "evaluate_confidence",
    "extract_intent",
    "request_clarification",
]
