# Agent Guide

## Purpose

This repository is the evolving implementation for the Intent2Pipeline Master's thesis. The current milestone is Lab 1. Read `PROJECT.md`, `README.md`, and `ARCHITECTURE.md` before making architectural changes.

## Non-negotiable boundaries

- Never execute LLM-generated code.
- Route all Ollama access through `src/llm`.
- Treat model output as untrusted structured data and validate it before use.
- Send only dataset metadata—not complete rows—to the Lab 1 intent extractor.
- Require explicit user confirmation before canonicalization.
- Keep pipeline execution deterministic and limited to explicitly registered operations.
- Do not use `eval`, `exec`, dynamic imports, or arbitrary callable resolution.
- Do not add model training or later thesis phases unless the active task requests them.
- Preserve the existing UCI Adult source data.

## Engineering conventions

- Keep Streamlit orchestration in `app.py` and domain logic under `src/`.
- Maintain module separation between dataset, intent, canonicalization, pipeline, evaluation, and LLM concerns.
- Add type hints and docstrings to public functions and classes.
- Raise small domain-specific exceptions with user-readable messages.
- Make contracts JSON-serializable and version externally consumed schemas.
- Fit preprocessing only on training data.
- Add or update tests for behavioral changes.

## Verification

Before handing off a change, run:

```bash
pytest
python -m compileall app.py src tests
```

Generated files under `outputs/` are runtime artifacts and should remain untracked.
