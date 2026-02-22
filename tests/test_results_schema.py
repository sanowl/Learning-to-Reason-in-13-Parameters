from __future__ import annotations

import json
from pathlib import Path

from ltr13.results import build_result_rows, write_csv, write_json, write_jsonl


def test_results_schema_writers(tmp_path: Path) -> None:
    rows = build_result_rows(
        mode="eval",
        model_name="demo",
        seed=42,
        config_hash="abc123",
        git_commit="deadbeef",
        metrics=[{"dataset": "gsm8k", "num_examples": 10, "exact_match": 0.9}],
    )

    json_path = tmp_path / "result.json"
    jsonl_path = tmp_path / "result.jsonl"
    csv_path = tmp_path / "result.csv"

    write_json(json_path, {"rows": rows})
    write_jsonl(jsonl_path, rows)
    write_csv(csv_path, rows)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["schema_version"] == "1.0"
    assert "gsm8k" in jsonl_path.read_text(encoding="utf-8")
    assert "dataset" in csv_path.read_text(encoding="utf-8")
