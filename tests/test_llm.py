"""Tests for the standalone Ollama provider adapter."""

from typing import Any

import pytest
import requests

from src.llm import OllamaClient, OllamaConfig, OllamaError


class FakeResponse:
    """Small requests.Response substitute for adapter unit tests."""

    def __init__(self, body: dict[str, Any], error: Exception | None = None) -> None:
        self.body = body
        self.error = error

    def raise_for_status(self) -> None:
        """Raise the configured HTTP failure."""
        if self.error:
            raise self.error

    def json(self) -> dict[str, Any]:
        """Return the configured response body."""
        return self.body


class FakeSession:
    """Capture adapter calls without using a network connection."""

    def __init__(
        self,
        response: FakeResponse,
        fail_get: bool = False,
        fail_post: bool = False,
    ) -> None:
        self.response = response
        self.fail_get = fail_get
        self.fail_post = fail_post
        self.post_payload: dict[str, Any] | None = None

    def get(self, url: str, timeout: float) -> FakeResponse:
        """Return health data or simulate an unavailable service."""
        if self.fail_get:
            raise requests.ConnectionError("offline")
        return self.response

    def post(self, url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        """Capture JSON-mode generation parameters."""
        if self.fail_post:
            raise requests.Timeout("timed out")
        self.post_payload = json
        return self.response


def test_ollama_adapter_requests_and_parses_structured_json() -> None:
    """The provider result is parsed as data and is never evaluated."""
    session = FakeSession(FakeResponse({"response": '{"task_type":"classification"}'}))
    client = OllamaClient(OllamaConfig(), session=session)  # type: ignore[arg-type]
    schema = {"type": "object", "properties": {"task_type": {"type": "string"}}}
    result = client.generate_json("Extract intent", schema=schema, system="Extract safely")
    assert result == {"task_type": "classification"}
    assert session.post_payload is not None
    assert session.post_payload["model"] == "llama3.1:8b"
    assert session.post_payload["format"] == schema
    assert session.post_payload["stream"] is False
    assert session.post_payload["options"] == {"temperature": 0}
    assert session.post_payload["system"] == "Extract safely"


def test_ollama_availability_and_invalid_output_are_handled() -> None:
    """Connection and parsing failures become safe return values or domain errors."""
    unavailable = OllamaClient(OllamaConfig(), session=FakeSession(FakeResponse({}), True))  # type: ignore[arg-type]
    assert unavailable.is_available() is False

    invalid = OllamaClient(
        OllamaConfig(), FakeSession(FakeResponse({"response": "not-json"}))  # type: ignore[arg-type]
    )
    with pytest.raises(OllamaError, match="structured JSON"):
        invalid.generate_json("Extract intent")


def test_ollama_timeout_becomes_a_domain_error() -> None:
    """HTTP timeouts do not leak requests exceptions into the application."""
    client = OllamaClient(
        OllamaConfig(), FakeSession(FakeResponse({}), fail_post=True)  # type: ignore[arg-type]
    )
    with pytest.raises(OllamaError, match="structured JSON"):
        client.generate_json("Extract intent")
