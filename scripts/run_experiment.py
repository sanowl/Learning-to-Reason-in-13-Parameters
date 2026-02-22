#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single experiment config")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.config).resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    mode = str(config.get("mode", "grpo")).lower()
    if mode == "grpo":
        target = "ltr13.train_grpo"
    elif mode == "sft":
        target = "ltr13.train_sft"
    elif mode == "eval":
        target = "ltr13.eval"
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    command = [sys.executable, "-m", target, "--config", str(path)]
    print("[Run]", " ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
