#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parameter sweep from YAML grid")
    parser.add_argument("--config", required=True, help="Sweep config path")
    parser.add_argument("--dry-run", action="store_true", help="Print runs without executing")
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional override to cap the number of runs from the sweep grid",
    )
    return parser.parse_args()


def _set_nested(mapping: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor = mapping
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def _build_run_name(overrides: dict[str, Any], index: int) -> str:
    short_items = []
    for key, value in overrides.items():
        leaf = key.split(".")[-1]
        safe_value = str(value).replace("/", "_")
        short_items.append(f"{leaf}-{safe_value}")
    joined = "__".join(short_items)
    if len(joined) > 140:
        joined = joined[:140]
    return f"run-{index:04d}__{joined}"


def _resolve_config_path(path_value: str, sweep_dir: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate

    cwd_candidate = candidate.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    sweep_relative = (sweep_dir / candidate).resolve()
    if sweep_relative.exists():
        return sweep_relative

    return cwd_candidate


def main() -> None:
    args = parse_args()
    sweep_path = Path(args.config).resolve()
    script_dir = Path(__file__).resolve().parent
    sweep_cfg = yaml.safe_load(sweep_path.read_text(encoding="utf-8"))

    base_path = _resolve_config_path(sweep_cfg["base_config"], sweep_path.parent)
    base_cfg = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    output_root = Path(sweep_cfg.get("output_root", "outputs/sweeps"))
    if not output_root.is_absolute():
        output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    grid: dict[str, list[Any]] = sweep_cfg["grid"]
    keys = list(grid.keys())
    values_product = list(itertools.product(*(grid[key] for key in keys)))

    max_runs = args.max_runs if args.max_runs is not None else sweep_cfg.get("max_runs")
    if max_runs is not None:
        values_product = values_product[: int(max_runs)]

    generated_configs_dir = output_root / "generated_configs"
    generated_configs_dir.mkdir(parents=True, exist_ok=True)

    for run_index, values in enumerate(values_product, start=1):
        overrides = {key: value for key, value in zip(keys, values)}
        run_cfg = json.loads(json.dumps(base_cfg))

        for dotted_key, value in overrides.items():
            _set_nested(run_cfg, dotted_key, value)

        run_name = _build_run_name(overrides, run_index)
        run_output_dir = output_root / run_name
        run_cfg["output_dir"] = str(run_output_dir)

        config_path = generated_configs_dir / f"{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")

        command = [sys.executable, str(script_dir / "run_experiment.py"), "--config", str(config_path)]
        print("[Sweep]", run_name)
        print("[Cmd]", " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
