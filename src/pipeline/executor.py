"""Deterministic execution of validated declarative pipeline specs."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.pipeline.generator import PipelineSpec, validate_pipeline_spec


class PipelineExecutionError(ValueError):
    """Raised when valid deterministic execution is impossible for the dataset."""


Matrix = np.ndarray | spmatrix


@dataclass(frozen=True)
class ExecutionResult:
    """Transformed train/test data and metadata; no model is trained."""

    x_train: Matrix
    x_test: Matrix
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    metadata: dict[str, Any]


def _parameters(specification: PipelineSpec, step_name: str) -> dict[str, Any]:
    """Return parameters for a validated step name."""
    return next(step.parameters for step in specification.steps if step.name == step_name)


def _replace_missing_markers(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Replace exact and whitespace-padded question marks without mutating input."""
    with pd.option_context("future.no_silent_downcasting", True):
        return dataframe.replace(to_replace=r"^\s*\?\s*$", value=np.nan, regex=True)


def _coerce_numeric_like_features(features: pd.DataFrame) -> pd.DataFrame:
    """Recover numeric columns whose '?' marker caused CSV object inference."""
    result = features.copy()
    for column in result.select_dtypes(exclude="number").columns:
        non_missing = result[column].notna()
        if not non_missing.any():
            continue
        converted = pd.to_numeric(result[column], errors="coerce")
        if converted[non_missing].notna().all():
            result[column] = converted
    return result


def _validate_stratification(target: pd.Series, test_size: float) -> None:
    """Provide clearer errors than train_test_split for invalid class counts."""
    counts = target.value_counts(dropna=False)
    if len(counts) < 2:
        raise PipelineExecutionError("Classification requires at least two target classes.")
    if int(counts.min()) < 2:
        raise PipelineExecutionError("Every target class needs at least two rows for stratification.")
    test_rows = ceil(len(target) * test_size)
    train_rows = len(target) - test_rows
    if min(test_rows, train_rows) < len(counts):
        raise PipelineExecutionError(
            "The train and test partitions must each have at least one row per target class."
        )


def execute_pipeline(dataframe: pd.DataFrame, specification: PipelineSpec) -> ExecutionResult:
    """Execute the closed Lab 1 operation set using deterministic Python code."""
    started = perf_counter()
    try:
        validate_pipeline_spec(specification)
    except ValueError as exc:
        raise PipelineExecutionError(str(exc)) from exc

    cleaned = _replace_missing_markers(dataframe)
    target_column = _parameters(specification, "separate_target")["target_column"]
    if target_column not in cleaned.columns:
        raise PipelineExecutionError(f"Target column '{target_column}' does not exist.")

    target = cleaned[target_column]
    if target.isna().any():
        raise PipelineExecutionError("Missing target values are not supported in Lab 1.")
    features = _coerce_numeric_like_features(cleaned.drop(columns=[target_column]))
    if features.shape[1] == 0:
        raise PipelineExecutionError("At least one feature column is required.")

    split = _parameters(specification, "train_test_split")
    _validate_stratification(target, float(split["test_size"]))
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        features,
        target,
        test_size=split["test_size"],
        random_state=split["random_state"],
        stratify=target,
    )

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]
    transformers: list[tuple[str, Pipeline, list[Any]]] = []
    scale_numeric = _parameters(specification, "scale_numeric")["enabled"]
    if numeric_columns:
        numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
        if scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    x_train = preprocessor.fit_transform(x_train_raw)
    x_test = preprocessor.transform(x_test_raw)
    feature_names = preprocessor.get_feature_names_out().astype(str).tolist()

    metadata = {
        "input_rows": int(dataframe.shape[0]),
        "input_columns": int(dataframe.shape[1]),
        "train_rows": int(x_train.shape[0]),
        "test_rows": int(x_test.shape[0]),
        "feature_count": int(x_train.shape[1]),
        "target_column": str(target_column),
        "scale_numeric": bool(scale_numeric),
        "stratified": True,
        "random_state": int(split["random_state"]),
        "execution_success": True,
        "elapsed_seconds": round(perf_counter() - started, 6),
        "errors": [],
    }
    return ExecutionResult(x_train, x_test, y_train, y_test, feature_names, metadata)
