#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full paper sweep with LR grid, multi-seed runs, and best-LR selection"
    )
    parser.add_argument("--config", required=True, help="Sweep YAML path")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--max-points", type=int, default=None, help="Cap update-size points")
    return parser.parse_args()


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _get_nested(mapping: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = mapping
    for part in dotted_key.split("."):
        cursor = cursor[part]
    return cursor


def _resolve_path(path_value: str, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run_command(command: list[str], dry_run: bool) -> None:
    print("[Cmd]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def _read_metric(path: Path, key: str) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    value = payload
    for part in key.split("."):
        value = value[part]
    return float(value)


def _point_name(index: int, point_items: dict[str, Any]) -> str:
    tokens = [f"{key.split('.')[-1]}-{str(value).replace('/', '_')}" for key, value in point_items.items()]
    raw = "__".join(tokens)
    if len(raw) > 140:
        raw = raw[:140]
    return f"point-{index:04d}__{raw}"


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.config).resolve()
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))

    script_dir = Path(__file__).resolve().parent
    run_experiment = script_dir / "run_experiment.py"

    train_base = _resolve_path(str(sweep_cfg["base_train_config"]), sweep_path.parent)
    eval_base = _resolve_path(str(sweep_cfg["base_eval_config"]), sweep_path.parent)
    train_template = yaml.safe_load(train_base.read_text(encoding="utf-8"))
    eval_template = yaml.safe_load(eval_base.read_text(encoding="utf-8"))

    output_root = _resolve_path(str(sweep_cfg.get("output_root", "outputs/sweeps/paper_full")), sweep_path.parent)
    output_root.mkdir(parents=True, exist_ok=True)

    seeds: list[int] = [int(seed) for seed in sweep_cfg.get("seeds", [42, 43, 44])]
    lrs: list[float] = [float(value) for value in sweep_cfg["learning_rates"]]

    grid: dict[str, list[Any]] = sweep_cfg["grid"]
    point_keys = list(grid.keys())
    point_values = list(itertools.product(*(grid[key] for key in point_keys)))
    if args.max_points is not None:
        point_values = point_values[: args.max_points]

    metric_key = str(sweep_cfg.get("metric_key", "mean_exact_match"))
    group_by_keys: list[str] = list(sweep_cfg.get("group_by", point_keys))

    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []

    for point_index, values in enumerate(point_values, start=1):
        point_items = {key: value for key, value in zip(point_keys, values)}
        point_name = _point_name(point_index, point_items)

        for lr in lrs:
            for seed in seeds:
                run_name = f"{point_name}__lr-{lr}__seed-{seed}"
                run_dir = output_root / run_name

                train_cfg = json.loads(json.dumps(train_template))
                for dotted_key, value in point_items.items():
                    _set_nested(train_cfg, dotted_key, value)
                _set_nested(train_cfg, "training.learning_rate", lr)
                _set_nested(train_cfg, "seed", seed)
                _set_nested(train_cfg, "output_dir", str(run_dir / "train"))

                train_cfg_path = generated_dir / f"{run_name}__train.yaml"
                _write_yaml(train_cfg_path, train_cfg)

                train_cmd = [sys.executable, str(run_experiment), "--config", str(train_cfg_path)]
                _run_command(train_cmd, dry_run=args.dry_run)

                eval_cfg = json.loads(json.dumps(eval_template))
                eval_cfg["model_name"] = train_cfg["model_name"]
                eval_cfg["seed"] = seed
                eval_cfg["deterministic"] = train_cfg.get("deterministic", False)
                eval_cfg["deterministic_warn_only"] = train_cfg.get("deterministic_warn_only", True)

                if "adapter" in train_cfg:
                    eval_cfg["adapter"] = train_cfg["adapter"]
                    eval_cfg.pop("tinylora", None)
                elif "tinylora" in train_cfg:
                    eval_cfg["tinylora"] = train_cfg["tinylora"]
                    eval_cfg.pop("adapter", None)

                eval_output_path = run_dir / "eval" / "results.json"
                eval_cfg["trainable_state_path"] = str(run_dir / "train" / "trainable_state.pt")
                eval_cfg["output_path"] = str(eval_output_path)
                eval_cfg["output_jsonl_path"] = str(run_dir / "eval" / "results.jsonl")
                eval_cfg["output_csv_path"] = str(run_dir / "eval" / "results.csv")

                eval_cfg_path = generated_dir / f"{run_name}__eval.yaml"
                _write_yaml(eval_cfg_path, eval_cfg)

                eval_cmd = [sys.executable, str(run_experiment), "--config", str(eval_cfg_path)]
                _run_command(eval_cmd, dry_run=args.dry_run)

                metric_value = float("nan")
                if not args.dry_run and eval_output_path.exists():
                    metric_value = _read_metric(eval_output_path, metric_key)

                row = {
                    "run_name": run_name,
                    "point_name": point_name,
                    "seed": seed,
                    "learning_rate": lr,
                    "metric": metric_value,
                    "metric_key": metric_key,
                    "train_config": str(train_cfg_path),
                    "eval_config": str(eval_cfg_path),
                    "train_output_dir": str(run_dir / "train"),
                    "eval_output_path": str(eval_output_path),
                }
                for key, value in point_items.items():
                    row[key] = value
                all_rows.append(row)

    all_rows_path = output_root / "all_runs.json"
    all_rows_path.write_text(json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8")

    csv_path = output_root / "all_runs.csv"
    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

    grouped: dict[tuple[Any, ...], dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        if row["metric"] != row["metric"]:  # NaN in dry-run
            continue
        group_key = tuple(row.get(key) for key in group_by_keys)
        grouped[group_key][float(row["learning_rate"])].append(float(row["metric"]))

    selection_rows: list[dict[str, Any]] = []
    for group_key, lr_values in grouped.items():
        lr_scores = {lr: mean(values) for lr, values in lr_values.items() if values}
        if not lr_scores:
            continue
        best_lr, best_score = max(lr_scores.items(), key=lambda item: item[1])

        row: dict[str, Any] = {
            "best_learning_rate": best_lr,
            "best_metric": best_score,
            "num_lrs": len(lr_scores),
        }
        for key, value in zip(group_by_keys, group_key):
            row[key] = value
        for lr, score in sorted(lr_scores.items(), key=lambda item: item[0]):
            row[f"lr_{lr}"] = score
        selection_rows.append(row)

    selection_rows.sort(key=lambda item: str([item.get(key) for key in group_by_keys]))

    best_json = output_root / "best_lr_by_group.json"
    best_json.write_text(json.dumps(selection_rows, indent=2, sort_keys=True), encoding="utf-8")

    best_csv = output_root / "best_lr_by_group.csv"
    if selection_rows:
        fieldnames = list(selection_rows[0].keys())
        with best_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(selection_rows)

    print("[Done] all runs:", all_rows_path)
    print("[Done] best lr:", best_json)


if __name__ == "__main__":
    main()
