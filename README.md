# Intent2Pipeline

Intent2Pipeline is a Master's thesis project exploring how natural-language data science requests can be converted into stable, declarative, and deterministically executable pipelines. This repository currently contains the **Lab 1 foundation**, not the complete thesis system.

Lab 1 provides:

- CSV upload, preview, and profiling in Streamlit
- constrained Ollama intent extraction with deterministic validation and user confirmation
- a versioned, structured pipeline specification
- one deterministic classification preprocessing path
- a local Ollama adapter configured for `llama3.1:8b`
- JSON run artifacts and focused tests

The LLM never generates executable code. Ollama may suggest only the task type, target column, and explicit scaling preference through a strict schema. The user must confirm that suggestion, and only operations implemented in the deterministic executor may run.

## Setup

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install and start [Ollama](https://ollama.com/), then prepare the default model:

```bash
ollama pull llama3.1:8b
ollama serve
```

Optional environment overrides:

```bash
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama3.1:8b
```

Run the application:

```bash
streamlit run app.py
```

Run the tests:

```bash
pytest
```

## Lab 1 workflow

1. Upload a CSV with a header row.
2. Review its preview and deterministic profile.
3. Enter a natural-language request.
4. Review Ollama's tentative structured interpretation.
5. Confirm or correct the target and optional numeric scaling.
6. Review the canonical intent and generated declarative specification.
7. Execute the approved deterministic preprocessing path.

Ollama receives only the request and compact metadata: column names, inferred types, and conservative target candidates. Dataset rows are not included. Its response is validated against a strict schema, checked against actual columns, and cannot proceed without explicit confirmation.

If Ollama is unavailable, times out, or returns invalid data, the app displays the error and switches to the same confirmation controls as a manual fallback. Deterministic preprocessing remains usable.

The executor replaces `?`, separates the target, creates a reproducible stratified split, fits imputers and encoders on training data only, and transforms both partitions. It performs median numeric imputation, mode categorical imputation, one-hot encoding, and optional standard scaling. It does not train a model.

Successful runs create:

- `outputs/canonical_intent.json`
- `outputs/pipeline_spec.json`
- `outputs/metrics.json`

These are runtime artifacts and are intentionally ignored by Git.

## Current limitations

- Ollama intent extraction supports classification, target identification, and explicit scaling requests only.
- Clarification is resolved through confirmation controls rather than a multi-turn conversation.
- Classification is the only supported task.
- The target must be explicitly confirmed and may not contain missing values.
- Split size and random seed are fixed at `0.2` and `42`.
- There is no model training, model evaluation, paraphrase comparison, or multi-agent behavior.
- Existing UCI Adult source files are preserved, but the UI expects an uploaded CSV with headers.

See [ARCHITECTURE.md](ARCHITECTURE.md) for system boundaries and [ROADMAP.md](ROADMAP.md) for future thesis work.
