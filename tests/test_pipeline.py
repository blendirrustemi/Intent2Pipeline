"""Tests for closed pipeline generation and deterministic execution."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from src.canonicalization import CanonicalIntent
from src.pipeline import PipelineExecutionError, execute_pipeline, generate_pipeline_spec
from src.pipeline.generator import PipelineSpec, PipelineStep


def _canonical(scale_numeric: bool = False, target: str = "target") -> CanonicalIntent:
    """Build the fixed canonical intent used by executor tests."""
    return CanonicalIntent(
        version="1.0",
        task_type="classification",
        target_column=target,
        missing_marker="?",
        numeric_imputation="median",
        categorical_imputation="most_frequent",
        categorical_encoding="one_hot",
        scale_numeric=scale_numeric,
        split_strategy="stratified",
    )


def _dataset() -> pd.DataFrame:
    """Return a mixed table with sufficient class counts and missing values."""
    return pd.DataFrame(
        {
            "number": [1, 2, "?", 4, 5, 6, 7, 8, 9, 10] * 2,
            "category": ["a", "b", "a", np.nan, "b"] * 4,
            "target": [0, 1] * 10,
        }
    )


def _dense(matrix: object) -> np.ndarray:
    """Convert either Scikit-Learn output representation to an array."""
    return matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)


def test_pipeline_executes_all_basic_operations_reproducibly() -> None:
    """Missing values are imputed, categories encoded, and the split is stable."""
    specification = generate_pipeline_spec(_canonical())
    first = execute_pipeline(_dataset(), specification)
    second = execute_pipeline(_dataset(), specification)

    assert first.x_train.shape[0] == 16
    assert first.x_test.shape[0] == 4
    assert first.x_train.shape[1] == len(first.feature_names)
    assert not np.isnan(_dense(first.x_train)).any()
    assert first.y_train.index.tolist() == second.y_train.index.tolist()
    assert first.y_train.value_counts().to_dict() == {0: 8, 1: 8}


def test_optional_scaling_standardizes_training_numeric_feature() -> None:
    """Scaling is fitted on the training numeric column when enabled."""
    result = execute_pipeline(_dataset(), generate_pipeline_spec(_canonical(True)))
    numeric_index = result.feature_names.index("numeric__number")
    assert _dense(result.x_train)[:, numeric_index].mean() == pytest.approx(0.0, abs=1e-9)


def test_preprocessing_is_fitted_on_training_partition_only() -> None:
    """A category present only in test data is not added to fitted feature names."""
    dataframe = _dataset()
    _, test_indices = train_test_split(
        dataframe.index,
        test_size=0.2,
        random_state=42,
        stratify=dataframe["target"],
    )
    dataframe.loc[test_indices[0], "category"] = "test-only-category"

    result = execute_pipeline(dataframe, generate_pipeline_spec(_canonical()))
    assert not any("test-only-category" in name for name in result.feature_names)


def test_executor_rejects_unknown_operation() -> None:
    """Arbitrary operation names cannot cross the executor boundary."""
    valid = generate_pipeline_spec(_canonical())
    steps = list(valid.steps)
    steps[-1] = PipelineStep("execute_python", {"source": "print('unsafe')"})
    unsafe = PipelineSpec(valid.version, valid.task_type, tuple(steps))

    with pytest.raises(PipelineExecutionError, match="steps must be exactly"):
        execute_pipeline(_dataset(), unsafe)


def test_executor_rejects_unknown_target() -> None:
    """The selected target must exist in the uploaded table."""
    specification = generate_pipeline_spec(_canonical(target="missing"))
    with pytest.raises(PipelineExecutionError, match="does not exist"):
        execute_pipeline(_dataset(), specification)


@pytest.mark.parametrize(
    "target",
    [
        [0] * 20,
        [0] * 19 + [1],
    ],
)
def test_executor_reports_invalid_stratification(target: list[int]) -> None:
    """One-class and singleton-class targets fail with useful messages."""
    dataframe = _dataset()
    dataframe["target"] = target
    with pytest.raises(PipelineExecutionError, match="classes|two rows"):
        execute_pipeline(dataframe, generate_pipeline_spec(_canonical()))

