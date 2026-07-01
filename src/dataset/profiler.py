"""Deterministic, lightweight dataset profiling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetProfile:
    """Serializable summary of an uploaded tabular dataset."""

    rows: int
    columns: int
    column_names: list[str]
    column_types: dict[str, str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    missing_counts: dict[str, int]
    missing_columns: list[str]
    duplicate_rows: int
    unique_counts: dict[str, int]
    possible_targets: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Return the profile as JSON-serializable primitives."""
        return asdict(self)


def profile_dataset(dataframe: pd.DataFrame) -> DatasetProfile:
    """Create structured metadata without mutating the input data frame."""
    profiled = dataframe.replace(to_replace=r"^\s*\?\s*$", value=pd.NA, regex=True)
    for column in profiled.select_dtypes(exclude="number").columns:
        non_missing = profiled[column].notna()
        if non_missing.any():
            converted = pd.to_numeric(profiled[column], errors="coerce")
            if converted[non_missing].notna().all():
                profiled[column] = converted

    numeric = profiled.select_dtypes(include="number").columns.astype(str).tolist()
    categorical = [str(column) for column in profiled.columns if str(column) not in numeric]
    missing_counts = {str(key): int(value) for key, value in profiled.isna().sum().items()}
    unique_counts = {
        str(key): int(value) for key, value in profiled.nunique(dropna=True).items()
    }

    # A conservative suggestion only; the user remains the source of truth.
    maximum_cardinality = max(2, min(20, int(len(dataframe) ** 0.5)))
    possible_targets = [
        str(column)
        for column in dataframe.columns
        if 2 <= unique_counts[str(column)] <= maximum_cardinality
    ]

    return DatasetProfile(
        rows=int(dataframe.shape[0]),
        columns=int(dataframe.shape[1]),
        column_names=[str(column) for column in dataframe.columns],
        column_types={str(key): str(value) for key, value in dataframe.dtypes.items()},
        numeric_columns=numeric,
        categorical_columns=categorical,
        missing_counts=missing_counts,
        missing_columns=[name for name, count in missing_counts.items() if count > 0],
        duplicate_rows=int(profiled.duplicated().sum()),
        unique_counts=unique_counts,
        possible_targets=possible_targets,
    )
