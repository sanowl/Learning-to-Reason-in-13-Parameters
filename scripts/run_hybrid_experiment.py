#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid experiment: SFT warmup then GRPO initialized from SFT adapter state"
    )
    parser.add_argument("--config", required=True, help="Path to hybrid YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Print generated commands only")
    return parser.parse_args()


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (base_dir / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    return payload


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _run(command: list[str], dry_run: bool) -> None:
    print("[Cmd]", " ".join(command))
    if dry_run:
        return
    subprocess.run(command, check=True)


def _extract_adapter_block(config: dict[str, Any]) -> dict[str, Any] | None:
    if isinstance(config.get("adapter"), dict):
        return dict(config["adapter"])
    if isinstance(config.get("tinylora"), dict):
        return dict(config["tinylora"])
    return None


def _set_adapter_block(config: dict[str, Any], adapter: dict[str, Any] | None) -> None:
    if adapter is None:
        config.pop("adapter", None)
        config.pop("tinylora", None)
        return
    config["adapter"] = dict(adapter)
    config.pop("tinylora", None)


def _canonical_json(payload: dict[str, Any] | None) -> str:
    return json.dumps(payload or {}, sort_keys=True, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    hybrid_path = Path(args.config).resolve()
    hybrid_cfg = _load_yaml(hybrid_path)

    sft_base_path = _resolve_path(str(hybrid_cfg["sft_config"]), hybrid_path.parent)
    grpo_base_path = _resolve_path(str(hybrid_cfg["grpo_config"]), hybrid_path.parent)
    eval_base_value = hybrid_cfg.get("eval_config")
    eval_base_path = (
        _resolve_path(str(eval_base_value), hybrid_path.parent) if eval_base_value is not None else None
    )

    sft_cfg = _load_yaml(sft_base_path)
    grpo_cfg = _load_yaml(grpo_base_path)
    eval_cfg = _load_yaml(eval_base_path) if eval_base_path is not None else None

    if str(sft_cfg.get("mode", "")).lower() != "sft":
        raise ValueError(f"sft_config must have mode=sft: {sft_base_path}")
    if str(grpo_cfg.get("mode", "")).lower() != "grpo":
        raise ValueError(f"grpo_config must have mode=grpo: {grpo_base_path}")

    output_root = _resolve_path(str(hybrid_cfg.get("output_root", "outputs/hybrid")), hybrid_path.parent)
    generated_dir = output_root / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=True)

    copy_adapter = bool(hybrid_cfg.get("copy_adapter_from_sft", True))
    sft_adapter = _extract_adapter_block(sft_cfg)
    grpo_adapter = _extract_adapter_block(grpo_cfg)

    if copy_adapter:
        _set_adapter_block(grpo_cfg, sft_adapter)
        grpo_adapter = _extract_adapter_block(grpo_cfg)
    elif _canonical_json(sft_adapter) != _canonical_json(grpo_adapter):
        raise ValueError(
            "SFT and GRPO adapter configs differ. Set copy_adapter_from_sft=true "
            "or align adapters manually."
        )

    run_name = str(hybrid_cfg.get("run_name", "hybrid_run"))
    run_root = output_root / run_name
    sft_out = run_root / "sft"
    grpo_out = run_root / "grpo"
    eval_out = run_root / "eval"

    sft_cfg["output_dir"] = str(sft_out)
    sft_generated = generated_dir / f"{run_name}__sft.yaml"
    _write_yaml(sft_generated, sft_cfg)

    script_dir = Path(__file__).resolve().parent
    run_experiment = script_dir / "run_experiment.py"

    sft_cmd = [sys.executable, str(run_experiment), "--config", str(sft_generated)]
    print("[Step] SFT warmup")
    _run(sft_cmd, args.dry_run)

    sft_state = sft_out / "trainable_state.pt"

    grpo_cfg["output_dir"] = str(grpo_out)
    grpo_cfg["init_trainable_state_path"] = str(sft_state)
    grpo_cfg["strict_init_trainable_state"] = bool(hybrid_cfg.get("strict_init_trainable_state", True))
    grpo_generated = generated_dir / f"{run_name}__grpo.yaml"
    _write_yaml(grpo_generated, grpo_cfg)

    grpo_cmd = [sys.executable, str(run_experiment), "--config", str(grpo_generated)]
    print("[Step] GRPO finetune")
    _run(grpo_cmd, args.dry_run)

    if eval_cfg is not None:
        eval_cfg["output_path"] = str(eval_out / "results.json")
        eval_cfg["output_jsonl_path"] = str(eval_out / "results.jsonl")
        eval_cfg["output_csv_path"] = str(eval_out / "results.csv")
        eval_cfg["trainable_state_path"] = str(grpo_out / "trainable_state.pt")
        eval_cfg["model_name"] = grpo_cfg.get("model_name", eval_cfg.get("model_name"))
        _set_adapter_block(eval_cfg, grpo_adapter)

        eval_generated = generated_dir / f"{run_name}__eval.yaml"
        _write_yaml(eval_generated, eval_cfg)

        eval_cmd = [sys.executable, str(run_experiment), "--config", str(eval_generated)]
        print("[Step] Eval")
        _run(eval_cmd, args.dry_run)

    print("[Done]", run_root)


if __name__ == "__main__":
    main()
