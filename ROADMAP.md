# Roadmap

## Lab 1 — current foundation

- Modular repository and Streamlit interface
- Dataset loading and profiling
- Schema-constrained Ollama intent extraction with explicit confirmation
- Ollama provider boundary with `llama3.1:8b` default
- Canonical intent and closed pipeline specification
- Deterministic classification preprocessing
- Run artifacts and unit tests

## Phase 2 — richer structured intent interaction

- Define strict schemas for provider responses
- Implement iterative clarification and confidence recalculation
- Add field-level evidence and ambiguity reporting
- Preserve deterministic fallbacks and validation

## Phase 3 — stability research

- Generate controlled prompt paraphrases
- Compare canonical intents and pipeline specifications
- Measure operator, parameter, and execution drift
- Compare direct generation with intent-first generation

## Phase 4 — model training and evaluation

- Add explicit model specifications for logistic regression, decision trees, and random forests
- Record accuracy, precision, recall, F1, and ROC AUC
- Keep training and evaluation deterministic and reproducible

## Phase 5 — broader thesis system

- Support regression, clustering, time series, and recommendation tasks
- Add experiment tracking and reproducible datasets
- Evaluate a multi-agent decomposition only after the single-flow baseline is stable

Features beyond the current Lab 1 list must not be added merely because this roadmap mentions them.
