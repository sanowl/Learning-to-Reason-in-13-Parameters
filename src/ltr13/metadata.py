from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch

DEFAULT_TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "transformers",
    "datasets",
    "trl",
    "accelerate",
    "numpy",
    "PyYAML",
    "pytest",
    "ruff",
)


def compute_config_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_git_commit(repo_dir: str | Path | None = None) -> str | None:
    cmd = ["git", "rev-parse", "HEAD"]
    try:
        result = subprocess.run(
            cmd,
            cwd=str(repo_dir) if repo_dir is not None else None,
            check=True,
            text=True,
            capture_output=True,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def get_package_versions(packages: tuple[str, ...] = DEFAULT_TRACKED_PACKAGES) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def get_gpu_info() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"cuda_available": False, "device_count": 0, "devices": []}

    devices: list[dict[str, Any]] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "multi_processor_count": int(props.multi_processor_count),
                "capability": f"{props.major}.{props.minor}",
            }
        )
    return {
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "devices": devices,
    }


def build_run_metadata(
    *,
    mode: str,
    config_path: str,
    config: dict[str, Any],
    seed: int,
    deterministic: bool,
    output_dir: str | Path,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    metadata = {
        "timestamp_utc": now,
        "mode": mode,
        "config_path": str(Path(config_path).resolve()),
        "config_hash": compute_config_hash(config),
        "seed": seed,
        "deterministic": deterministic,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "package_versions": get_package_versions(),
        "gpu": get_gpu_info(),
        "output_dir": str(Path(output_dir).resolve()),
    }
    if extra:
        metadata["extra"] = extra
    return metadata


def write_run_metadata(
    *,
    output_dir: str | Path,
    metadata: dict[str, Any],
    filename: str = "run_metadata.json",
) -> Path:
    path = Path(output_dir) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return path
