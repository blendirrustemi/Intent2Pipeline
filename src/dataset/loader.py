"""CSV loading with small, user-facing validation errors."""

from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pandas as pd
from pandas.errors import EmptyDataError, ParserError


class DatasetLoadError(ValueError):
    """Raised when an uploaded CSV cannot be used as a tabular dataset."""


CsvSource = str | Path | BinaryIO


def load_csv(source: CsvSource) -> pd.DataFrame:
    """Load a CSV source and validate that it contains rows and columns.

    Args:
        source: A path or binary file-like object accepted by ``pandas.read_csv``.

    Returns:
        A non-empty data frame.

    Raises:
        DatasetLoadError: If parsing fails or the resulting table is empty.
    """
    try:
        dataframe = pd.read_csv(source)
    except EmptyDataError as exc:
        raise DatasetLoadError("The CSV file is empty.") from exc
    except (ParserError, UnicodeDecodeError, OSError, ValueError) as exc:
        raise DatasetLoadError(f"Could not read the CSV file: {exc}") from exc

    if dataframe.empty:
        raise DatasetLoadError("The CSV must contain at least one data row.")
    if dataframe.columns.empty:
        raise DatasetLoadError("The CSV must contain at least one column.")
    if dataframe.columns.duplicated().any():
        duplicates = dataframe.columns[dataframe.columns.duplicated()].tolist()
        raise DatasetLoadError(f"Duplicate column names are not supported: {duplicates}")

    return dataframe

