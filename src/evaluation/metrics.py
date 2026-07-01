"""Persist Lab 1 structured run artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class ArtifactWriteError(OSError):
    """Raised when run artifacts cannot be persisted."""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write one UTF-8, human-readable JSON document."""
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_run_artifacts(
    canonical_intent: Mapping[str, Any],
    pipeline_spec: Mapping[str, Any],
    metrics: Mapping[str, Any],
    output_directory: str | Path = "outputs",
) -> dict[str, Path]:
    """Write the three required artifacts after a successful execution."""
    directory = Path(output_directory)
    try:
        directory.mkdir(parents=True, exist_ok=True)
        paths = {
            "canonical_intent": directory / "canonical_intent.json",
            "pipeline_spec": directory / "pipeline_spec.json",
            "metrics": directory / "metrics.json",
        }
        _write_json(paths["canonical_intent"], canonical_intent)
        _write_json(paths["pipeline_spec"], pipeline_spec)
        _write_json(paths["metrics"], metrics)
    except (OSError, TypeError, ValueError) as exc:
        raise ArtifactWriteError(f"Could not write output artifacts: {exc}") from exc
    return paths

