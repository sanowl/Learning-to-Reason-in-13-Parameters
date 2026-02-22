from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

RESULT_SCHEMA_VERSION = "1.0"


def build_result_rows(
    *,
    mode: str,
    model_name: str,
    seed: int,
    config_hash: str,
    git_commit: str | None,
    metrics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in metrics:
        rows.append(
            {
                "schema_version": RESULT_SCHEMA_VERSION,
                "mode": mode,
                "model_name": model_name,
                "dataset": metric["dataset"],
                "num_examples": int(metric["num_examples"]),
                "exact_match": float(metric["exact_match"]),
                "seed": int(seed),
                "config_hash": config_hash,
                "git_commit": git_commit or "unknown",
            }
        )
    return rows


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return target


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        target.write_text("", encoding="utf-8")
        return target

    fieldnames = list(rows[0].keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return target
