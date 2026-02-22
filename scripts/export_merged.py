#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export merged TinyLoRA model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = str(Path(args.config).resolve())
    command = [
        sys.executable,
        "-m",
        "ltr13.export_merged",
        "--config",
        config_path,
        "--output-dir",
        args.output_dir,
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
