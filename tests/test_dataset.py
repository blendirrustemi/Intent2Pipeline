"""Tests for CSV loading and deterministic profiling."""

from io import BytesIO

import pandas as pd
import pytest

from src.dataset import DatasetLoadError, load_csv, profile_dataset


def test_load_csv_returns_non_empty_dataframe() -> None:
    """A regular uploaded CSV is parsed with its header."""
    dataframe = load_csv(BytesIO(b"age,income\n30,low\n40,high\n"))
    assert dataframe.shape == (2, 2)
    assert dataframe.columns.tolist() == ["age", "income"]


@pytest.mark.parametrize("content", [b"", b"age,income\n"])
def test_load_csv_rejects_empty_data(content: bytes) -> None:
    """Both a blank file and a header-only file are invalid datasets."""
    with pytest.raises(DatasetLoadError, match="empty|data row"):
        load_csv(BytesIO(content))


def test_profile_recognizes_question_mark_missingness_and_numeric_data() -> None:
    """Whitespace-padded missing markers count as missing during profiling."""
    dataframe = pd.DataFrame(
        {
            "age": ["10", " ? ", "30", "30"],
            "kind": ["a", "b", "a", "a"],
            "target": [0, 1, 0, 0],
        }
    )
    profile = profile_dataset(dataframe)
    assert profile.rows == 4
    assert profile.missing_counts["age"] == 1
    assert "age" in profile.numeric_columns
    assert profile.unique_counts["kind"] == 2
    assert "target" in profile.possible_targets

