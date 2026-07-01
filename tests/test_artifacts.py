"""Tests for required JSON run artifacts."""

import json

from src.evaluation import write_run_artifacts


def test_write_run_artifacts_creates_three_json_documents(tmp_path) -> None:
    """Successful run data is persisted under the specified output directory."""
    paths = write_run_artifacts(
        {"target_column": "income"},
        {"version": "1.0", "steps": []},
        {"execution_success": True},
        tmp_path,
    )
    assert set(paths) == {"canonical_intent", "pipeline_spec", "metrics"}
    assert json.loads(paths["canonical_intent"].read_text())["target_column"] == "income"
    assert json.loads(paths["metrics"].read_text())["execution_success"] is True

