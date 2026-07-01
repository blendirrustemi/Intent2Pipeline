"""Narrow Ollama HTTP adapter returning structured data only."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests


class OllamaError(RuntimeError):
    """Raised when Ollama is unavailable or returns invalid structured output."""


@dataclass(frozen=True)
class OllamaConfig:
    """Connection settings loaded from environment variables."""

    base_url: str = "http://localhost:11434"
    model: str = "llama3.1:8b"
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> "OllamaConfig":
        """Build configuration with safe local defaults."""
        return cls(
            base_url=os.getenv("OLLAMA_BASE_URL", cls.base_url).rstrip("/"),
            model=os.getenv("OLLAMA_MODEL", cls.model),
        )


class OllamaClient:
    """Minimal adapter; it never evaluates or executes model output."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = config or OllamaConfig.from_environment()
        self._session = session or requests.Session()

    def is_available(self) -> bool:
        """Return whether the configured Ollama service answers its tags endpoint."""
        try:
            response = self._session.get(
                f"{self.config.base_url}/api/tags",
                timeout=min(self.config.timeout_seconds, 1.0),
            )
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def generate_json(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        system: str | None = None,
    ) -> dict[str, Any]:
        """Request schema-constrained JSON and parse it as non-executable data."""
        if not prompt.strip():
            raise OllamaError("A non-empty prompt is required.")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "format": schema or "json",
            "stream": False,
            "options": {"temperature": 0},
        }
        if system:
            payload["system"] = system
        try:
            response = self._session.post(
                f"{self.config.base_url}/api/generate",
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            result = json.loads(body["response"])
        except (requests.RequestException, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama did not return valid structured JSON: {exc}") from exc
        if not isinstance(result, dict):
            raise OllamaError("Ollama JSON output must be an object.")
        return result
