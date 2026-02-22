#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate eval result JSON files into machine-comparable tables")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input glob(s) for result JSON files")
    parser.add_argument("--output-json", required=True, help="Output JSON path")
    parser.add_argument("--output-csv", required=True, help="Output CSV path")
    return parser.parse_args()


def _expand_inputs(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path).resolve() for path in glob.glob(pattern))
    return sorted(set(paths))


def _rows_from_payload(payload: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    if isinstance(payload.get("rows"), list):
        for row in payload["rows"]:
            item = dict(row)
            item["source_file"] = str(source)
            rows.append(item)
        return rows

    if isinstance(payload.get("results"), list):
        for result in payload["results"]:
            item = {
                "schema_version": payload.get("schema_version", "legacy"),
                "dataset": result.get("dataset"),
                "num_examples": result.get("num_examples"),
                "exact_match": result.get("exact_match"),
                "source_file": str(source),
            }
            rows.append(item)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    input_paths = _expand_inputs(args.inputs)
    if not input_paths:
        raise ValueError("No files matched --inputs patterns")

    all_rows: list[dict[str, Any]] = []
    for path in input_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        all_rows.extend(_rows_from_payload(payload, path))

    summary = {
        "num_files": len(input_paths),
        "num_rows": len(all_rows),
        "rows": all_rows,
    }

    output_json = Path(args.output_json).resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _write_csv(Path(args.output_csv).resolve(), all_rows)

    print("[Done]", output_json)


if __name__ == "__main__":
    main()
