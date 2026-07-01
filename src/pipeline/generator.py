"""Generate and validate declarative pipeline specifications."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.canonicalization.canonicalizer import CanonicalIntent


class PipelineSpecError(ValueError):
    """Raised when a specification contains unsupported behavior."""


@dataclass(frozen=True)
class PipelineStep:
    """One declarative operation and its constrained parameters."""

    name: str
    parameters: dict[str, Any]


@dataclass(frozen=True)
class PipelineSpec:
    """Versioned declarative specification accepted by the executor."""

    version: str
    task_type: str
    steps: tuple[PipelineStep, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable specification data."""
        return {
            "version": self.version,
            "task_type": self.task_type,
            "steps": [asdict(step) for step in self.steps],
        }


_ALLOWED_PARAMETERS: dict[str, set[str]] = {
    "replace_missing_markers": {"marker"},
    "separate_target": {"target_column"},
    "impute_numeric": {"strategy"},
    "impute_categorical": {"strategy"},
    "one_hot_encode": {"handle_unknown"},
    "scale_numeric": {"enabled"},
    "train_test_split": {"test_size", "random_state", "stratify"},
}

_REQUIRED_ORDER = tuple(_ALLOWED_PARAMETERS)


def validate_pipeline_spec(specification: PipelineSpec) -> None:
    """Reject unknown operations, parameters, values, and operation order."""
    if specification.version != "1.0" or specification.task_type != "classification":
        raise PipelineSpecError("Only version 1.0 classification specifications are supported.")

    names = tuple(step.name for step in specification.steps)
    if names != _REQUIRED_ORDER:
        raise PipelineSpecError(f"Pipeline steps must be exactly: {list(_REQUIRED_ORDER)}")

    for step in specification.steps:
        allowed = _ALLOWED_PARAMETERS.get(step.name)
        if allowed is None:
            raise PipelineSpecError(f"Unsupported pipeline step: {step.name}")
        supplied = set(step.parameters)
        if supplied != allowed:
            raise PipelineSpecError(
                f"Step '{step.name}' requires parameters {sorted(allowed)}; got {sorted(supplied)}."
            )

    values = {step.name: step.parameters for step in specification.steps}
    if values["replace_missing_markers"]["marker"] != "?":
        raise PipelineSpecError("The only supported missing marker is '?'.")
    target_column = values["separate_target"]["target_column"]
    if not isinstance(target_column, str) or not target_column.strip():
        raise PipelineSpecError("The target column must be a non-empty string.")
    if values["impute_numeric"]["strategy"] != "median":
        raise PipelineSpecError("Numeric imputation must use median.")
    if values["impute_categorical"]["strategy"] != "most_frequent":
        raise PipelineSpecError("Categorical imputation must use most_frequent.")
    if values["one_hot_encode"]["handle_unknown"] != "ignore":
        raise PipelineSpecError("One-hot encoding must ignore unknown categories.")
    if not isinstance(values["scale_numeric"]["enabled"], bool):
        raise PipelineSpecError("scale_numeric.enabled must be a boolean.")

    split = values["train_test_split"]
    if split != {"test_size": 0.2, "random_state": 42, "stratify": True}:
        raise PipelineSpecError("Lab 1 uses a fixed 0.2 split, random_state 42, and stratification.")


def generate_pipeline_spec(intent: CanonicalIntent) -> PipelineSpec:
    """Generate the only executable pipeline shape supported in Lab 1."""
    specification = PipelineSpec(
        version="1.0",
        task_type=intent.task_type,
        steps=(
            PipelineStep("replace_missing_markers", {"marker": intent.missing_marker}),
            PipelineStep("separate_target", {"target_column": intent.target_column}),
            PipelineStep("impute_numeric", {"strategy": intent.numeric_imputation}),
            PipelineStep(
                "impute_categorical", {"strategy": intent.categorical_imputation}
            ),
            PipelineStep("one_hot_encode", {"handle_unknown": "ignore"}),
            PipelineStep("scale_numeric", {"enabled": intent.scale_numeric}),
            PipelineStep(
                "train_test_split",
                {"test_size": 0.2, "random_state": 42, "stratify": True},
            ),
        ),
    )
    validate_pipeline_spec(specification)
    return specification
