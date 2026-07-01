# Architecture

## Lab 1 data flow

```text
CSV upload
  -> dataset loader and profiler
  -> prompt + compact dataset metadata
  -> schema-constrained Ollama intent extraction
  -> deterministic validation and explicit user confirmation
  -> canonical intent
  -> validated declarative pipeline specification
  -> deterministic Python executor
  -> transformed train/test data + JSON artifacts
```

Each arrow crosses a typed, JSON-serializable contract. Streamlit coordinates the flow but contains no preprocessing logic.

## Modules

- `src/dataset`: CSV validation and structured profiling
- `src/intent`: constrained extraction, dataset validation, confidence, and confirmation contracts
- `src/canonicalization`: stable classification intent representation
- `src/pipeline`: closed specification vocabulary and deterministic executor
- `src/evaluation`: run metadata artifact persistence; model metrics are future work
- `src/llm`: the only allowed Ollama integration boundary

## Safety and determinism boundary

LLM output is data, never code. Ollama receives only the prompt, column names, inferred types, and possible target names. A strict Pydantic schema permits only `task_type`, `target_column`, and `scale_numeric`; extra or mistyped fields are rejected. The response is checked against real dataset columns and requires explicit user confirmation. It may not provide Python, shell commands, imports, callables, pipeline operations, or preprocessing parameters.

`PipelineSpec` accepts one versioned sequence of approved operations. Validation rejects missing, reordered, or unknown steps; unknown parameters; and unsupported values. The executor maps that fixed vocabulary to locally implemented Pandas and Scikit-Learn functions. It does not use `eval`, `exec`, dynamic imports, or model-produced source code.

The split uses `random_state=42`. Preprocessing is fitted only on the training partition to avoid data leakage. Unknown test categories are ignored by the fitted one-hot encoder.

## Public contracts

- `DatasetProfile`: dimensions, types, missingness, duplicates, cardinalities, and target suggestions
- `ModelIntentResponse`: strict schema for Ollama's three allowed semantic fields
- `IntentDraft`, `ConfidenceResult`, `ClarificationResult`: tentative, validated, and confirmed states with provenance
- `CanonicalIntent`: stable input to pipeline generation
- `PipelineSpec` and `PipelineStep`: declarative executable request
- `ExecutionResult`: transformed partitions, targets, feature names, and run metadata
- `OllamaClient.generate_json`: schema-constrained provider interface with non-streaming JSON and temperature zero

Confidence is deterministic completeness, not a model probability: classification contributes one required field, a valid target contributes the other, and user confirmation is a separate gate. Scaling is optional and defaults to disabled when the prompt does not explicitly request it.

Artifacts are human-readable JSON. `metrics.json` contains operational metadata, not model-quality metrics.

## Repository layout

```text
.
├── app.py
├── data/adult/
├── outputs/
├── src/
│   ├── canonicalization/
│   ├── dataset/
│   ├── evaluation/
│   ├── intent/
│   ├── llm/
│   └── pipeline/
└── tests/
```
