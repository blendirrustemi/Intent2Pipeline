"""Streamlit entry point for the Intent2Pipeline Lab 1 prototype."""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.canonicalization import canonicalize_intent
from src.dataset import DatasetLoadError, load_csv, profile_dataset
from src.evaluation import ArtifactWriteError, write_run_artifacts
from src.intent import (
    IntentExtractionError,
    confirm_intent,
    create_manual_fallback_intent,
    evaluate_confidence,
    extract_intent,
    request_clarification,
)
from src.llm import OllamaClient, OllamaConfig, OllamaError
from src.pipeline import PipelineExecutionError, execute_pipeline, generate_pipeline_spec


st.set_page_config(page_title="Intent2Pipeline", page_icon="🧭", layout="wide")


@st.cache_data(ttl=10, show_spinner=False)
def _ollama_available(base_url: str, model: str) -> bool:
    """Cache the inexpensive local service health check across reruns."""
    return OllamaClient(OllamaConfig(base_url=base_url, model=model)).is_available()


def _initialize_state() -> None:
    """Create stable Streamlit session keys."""
    defaults: dict[str, Any] = {
        "dataset": None,
        "dataset_fingerprint": None,
        "messages": [],
        "latest_prompt": "",
        "intent_draft": None,
        "intent_error": None,
        "confirmed_intent": None,
        "confirmation_signature": None,
        "execution_result": None,
        "artifact_paths": None,
        "execution_signature": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_execution() -> None:
    """Clear execution data when confirmed inputs change."""
    st.session_state.execution_result = None
    st.session_state.artifact_paths = None
    st.session_state.execution_signature = None


def _reset_workflow() -> None:
    """Clear prompt-derived state when a new dataset is loaded."""
    st.session_state.messages = []
    st.session_state.latest_prompt = ""
    st.session_state.intent_draft = None
    st.session_state.intent_error = None
    st.session_state.confirmed_intent = None
    st.session_state.confirmation_signature = None
    st.session_state.pop("target_confirmation", None)
    st.session_state.pop("scale_confirmation", None)
    _reset_execution()


_initialize_state()

st.title("Intent2Pipeline")
st.caption("Lab 1 · constrained local intent extraction and deterministic execution")

config = OllamaConfig.from_environment()
ollama_online = _ollama_available(config.base_url, config.model)
with st.sidebar:
    st.header("Local LLM backend")
    st.code(config.model, language=None)
    st.caption(config.base_url)
    if ollama_online:
        st.success("Ollama server is available")
    else:
        st.warning("Ollama is unavailable. Manual confirmation remains available.")
    st.info(
        "Ollama receives only the prompt and dataset metadata. "
        "Its structured suggestion must be confirmed and is never executed as code."
    )

st.header("1. Dataset upload")
uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
if uploaded_file is not None:
    fingerprint = (uploaded_file.name, uploaded_file.size)
    if fingerprint != st.session_state.dataset_fingerprint:
        try:
            st.session_state.dataset = load_csv(uploaded_file)
            st.session_state.dataset_fingerprint = fingerprint
            _reset_workflow()
        except DatasetLoadError as exc:
            st.session_state.dataset = None
            st.session_state.dataset_fingerprint = None
            st.error(str(exc))

dataframe = st.session_state.dataset
if dataframe is None:
    st.info("Upload a CSV file to begin.")
    st.stop()

st.subheader("Dataset preview")
st.dataframe(dataframe.head(10), use_container_width=True)

st.header("2. Dataset profile")
profile = profile_dataset(dataframe)
summary_columns = st.columns(4)
summary_columns[0].metric("Rows", profile.rows)
summary_columns[1].metric("Columns", profile.columns)
summary_columns[2].metric("Missing values", sum(profile.missing_counts.values()))
summary_columns[3].metric("Duplicate rows", profile.duplicate_rows)
with st.expander("Structured profile", expanded=True):
    st.json(profile.to_dict())

st.header("3. Prompt and chat")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

with st.form("intent_prompt_form", clear_on_submit=True):
    prompt = st.text_input(
        "Describe the classification preparation task",
        placeholder="For example: Prepare this dataset to predict income.",
    )
    prompt_submitted = st.form_submit_button("Interpret prompt", type="primary")

if prompt_submitted and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})
    try:
        if not ollama_online:
            raise OllamaError(
                "The Ollama server is not reachable. Start it with 'ollama serve' and try again."
            )
        with st.spinner(f"Interpreting the request with {config.model}…"):
            draft = extract_intent(
                prompt=prompt,
                column_names=profile.column_names,
                column_types=profile.column_types,
                possible_targets=profile.possible_targets,
                generator=OllamaClient(config),
            )
        error = None
        assistant_message = (
            "Ollama produced a constrained intent suggestion. "
            "Review and confirm it before pipeline generation."
        )
    except (OllamaError, IntentExtractionError) as exc:
        error = str(exc)
        draft = create_manual_fallback_intent(error)
        assistant_message = (
            "Ollama could not produce a valid intent. "
            "Use the manual controls to confirm the classification task."
        )

    st.session_state.messages.append({"role": "assistant", "content": assistant_message})
    st.session_state.latest_prompt = prompt
    st.session_state.intent_draft = draft
    st.session_state.intent_error = error
    st.session_state.confirmed_intent = None
    st.session_state.confirmation_signature = None
    st.session_state.target_confirmation = draft.target_column
    st.session_state.scale_confirmation = bool(draft.scale_numeric)
    _reset_execution()
    st.rerun()

draft = st.session_state.intent_draft
if draft is None:
    st.info("Enter a prompt to ask Ollama for a constrained intent suggestion.")
    st.stop()

st.header("4. Tentative intent understanding")
if st.session_state.intent_error:
    st.warning(
        f"Ollama fallback: {st.session_state.intent_error} "
        "The pipeline remains blocked until you confirm the controls below."
    )
st.json(draft.to_dict())
if draft.validation_issues:
    for issue in draft.validation_issues:
        st.warning(issue)

st.header("5. Confidence evaluation")
confirmed_intent = st.session_state.confirmed_intent
control_target = st.selectbox(
    "Classification target",
    options=[None, *profile.column_names],
    key="target_confirmation",
    format_func=lambda value: "Select a target…" if value is None else value,
    help="Ollama may preselect this value, but your confirmation is the source of truth.",
)
control_scaling = st.checkbox("Scale numeric features", key="scale_confirmation")
control_signature = (st.session_state.latest_prompt, control_target, control_scaling)

if confirmed_intent is not None and st.session_state.confirmation_signature != control_signature:
    st.session_state.confirmed_intent = None
    st.session_state.confirmation_signature = None
    confirmed_intent = None
    _reset_execution()

active_intent = confirmed_intent or draft
confidence = evaluate_confidence(active_intent)
st.json(confidence.to_dict())

st.header("6. Clarification and confirmation")
clarification = request_clarification(confidence)
st.json(clarification.to_dict())
if clarification.required and clarification.question:
    st.warning(clarification.question)

if st.button(
    "Confirm classification intent",
    type="primary",
    disabled=control_target is None,
):
    try:
        st.session_state.confirmed_intent = confirm_intent(
            draft,
            target_column=control_target,
            scale_numeric=control_scaling,
            available_columns=profile.column_names,
        )
        st.session_state.confirmation_signature = control_signature
        _reset_execution()
        st.rerun()
    except ValueError as exc:
        st.error(str(exc))

if confirmed_intent is not None:
    st.success(
        f"Confirmed target '{confirmed_intent.target_column}' "
        f"(source: {confirmed_intent.source})."
    )
    st.json(confirmed_intent.to_dict())

canonical_intent = None
pipeline_spec = None
st.header("7. Canonical intent")
if confirmed_intent is not None and confidence.sufficient:
    canonical_intent = canonicalize_intent(confirmed_intent)
    st.json(canonical_intent.to_dict())
else:
    st.info("Canonicalization is blocked until the intent is explicitly confirmed.")

st.header("8. Pipeline specification")
if canonical_intent is not None:
    pipeline_spec = generate_pipeline_spec(canonical_intent)
    st.json(pipeline_spec.to_dict())
else:
    st.info("A pipeline specification will appear after canonicalization.")

st.header("9. Deterministic execution")
can_execute = confirmed_intent is not None and pipeline_spec is not None
execution_signature = (st.session_state.dataset_fingerprint, control_signature)
if (
    st.session_state.execution_result is not None
    and st.session_state.execution_signature != execution_signature
):
    _reset_execution()

if st.button("Execute preprocessing pipeline", type="primary", disabled=not can_execute):
    try:
        with st.spinner("Running approved deterministic operations…"):
            result = execute_pipeline(dataframe, pipeline_spec)
            metrics = {**result.metadata, "intent_source": confirmed_intent.source}
            paths = write_run_artifacts(
                canonical_intent.to_dict(),
                pipeline_spec.to_dict(),
                metrics,
            )
        st.session_state.execution_result = result
        st.session_state.artifact_paths = paths
        st.session_state.execution_signature = execution_signature
    except (PipelineExecutionError, ArtifactWriteError) as exc:
        _reset_execution()
        st.error(str(exc))

result = st.session_state.execution_result
if result is not None:
    st.success("Preprocessing completed. No model was trained.")
    result_columns = st.columns(3)
    result_columns[0].metric("Train rows", result.metadata["train_rows"])
    result_columns[1].metric("Test rows", result.metadata["test_rows"])
    result_columns[2].metric("Output features", result.metadata["feature_count"])
    with st.expander("Execution metadata", expanded=True):
        st.json({**result.metadata, "intent_source": confirmed_intent.source})
    with st.expander("Output feature names"):
        st.write(result.feature_names)
    artifact_paths = st.session_state.artifact_paths or {}
    st.caption("Artifacts: " + ", ".join(str(path) for path in artifact_paths.values()))
elif not can_execute:
    st.info("Confirm the interpreted intent to enable deterministic execution.")
