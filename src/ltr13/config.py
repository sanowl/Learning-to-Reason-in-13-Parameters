from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "up_proj",
    "down_proj",
    "gate_proj",
)


@dataclass(frozen=True)
class TinyLoRAConfig:
    adapter_type: str = "tinylora"
    rank: int = 2
    proj_dim: int = 1
    tie_mode: str = "full"
    tie_factor: int = 1
    scale: float = 1.0
    svd_method: str = "auto"
    svd_niter: int = 2
    projection_mode: str = "random"
    projection_blocks: int = 1
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    seed: int = 0
    vector_dtype: str = "float32"
    compute_dtype: str = "float32"

    def __post_init__(self) -> None:
        checks = (
            (self.rank > 0, "rank must be positive"),
            (self.proj_dim > 0, "proj_dim must be positive"),
            (self.tie_factor > 0, "tie_factor must be positive"),
            (self.scale > 0, "scale must be positive"),
            (self.svd_niter > 0, "svd_niter must be positive"),
            (self.projection_blocks > 0, "projection_blocks must be positive"),
            (self.lora_alpha > 0, "lora_alpha must be positive"),
            (self.lora_dropout >= 0, "lora_dropout must be non-negative"),
            (self.lora_dropout < 1, "lora_dropout must be < 1.0"),
        )
        errors = [message for condition, message in checks if not condition]

        _validate_choice(
            value=self.adapter_type,
            valid_values={"tinylora", "lora", "lora_xs"},
            field_name="adapter_type",
            errors=errors,
        )
        _validate_choice(
            value=self.tie_mode,
            valid_values={"none", "structured", "tiled", "full"},
            field_name="tie_mode",
            errors=errors,
        )
        _validate_choice(
            value=self.svd_method,
            valid_values={"auto", "full", "lowrank"},
            field_name="svd_method",
            errors=errors,
        )
        _validate_choice(
            value=self.projection_mode,
            valid_values={"random", "structured"},
            field_name="projection_mode",
            errors=errors,
        )
        if not self.target_modules:
            errors.append("target_modules must be non-empty")
        elif any((not isinstance(name, str) or not name.strip()) for name in self.target_modules):
            errors.append("target_modules entries must be non-empty strings")
        if (
            self.adapter_type == "tinylora"
            and self.projection_mode == "structured"
            and self.projection_blocks > self.rank
        ):
            errors.append("projection_blocks must be <= rank for tinylora structured projection")

        if errors:
            joined = "; ".join(errors)
            raise ValueError(joined)

    @property
    def vector_torch_dtype(self) -> torch.dtype:
        return _str_to_dtype(self.vector_dtype)

    @property
    def compute_torch_dtype(self) -> torch.dtype:
        return _str_to_dtype(self.compute_dtype)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected top-level mapping in config: {path}")
    return data


def parse_tinylora_config(raw: dict[str, Any] | None) -> TinyLoRAConfig | None:
    if raw is None:
        return None

    targets_tuple = _normalize_target_modules(raw.get("target_modules", DEFAULT_TARGET_MODULES))

    return TinyLoRAConfig(
        adapter_type=str(raw.get("adapter_type", raw.get("type", "tinylora"))),
        rank=int(raw.get("rank", 2)),
        proj_dim=int(raw.get("proj_dim", 1)),
        tie_mode=str(raw.get("tie_mode", "full")),
        tie_factor=int(raw.get("tie_factor", 1)),
        scale=float(raw.get("scale", 1.0)),
        svd_method=str(raw.get("svd_method", "auto")),
        svd_niter=int(raw.get("svd_niter", 2)),
        projection_mode=str(raw.get("projection_mode", "random")),
        projection_blocks=int(raw.get("projection_blocks", 1)),
        lora_alpha=float(raw.get("lora_alpha", 1.0)),
        lora_dropout=float(raw.get("lora_dropout", 0.0)),
        target_modules=targets_tuple,
        seed=int(raw.get("seed", 0)),
        vector_dtype=str(raw.get("vector_dtype", "float32")),
        compute_dtype=str(raw.get("compute_dtype", "float32")),
    )


def parse_adapter_config(config: dict[str, Any]) -> TinyLoRAConfig | None:
    if "adapter" in config and config["adapter"] is not None:
        raw = dict(config["adapter"])
        if "adapter_type" not in raw and "type" not in raw:
            raw["adapter_type"] = "tinylora"
        return parse_tinylora_config(raw)
    if "tinylora" in config and config["tinylora"] is not None:
        raw = dict(config["tinylora"])
        if "adapter_type" not in raw and "type" not in raw:
            raw["adapter_type"] = "tinylora"
        return parse_tinylora_config(raw)
    return None


def _str_to_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype string: {name}")
    return mapping[normalized]


def _normalize_target_modules(raw_targets: Any) -> tuple[str, ...]:
    if raw_targets is None:
        return DEFAULT_TARGET_MODULES
    if isinstance(raw_targets, str):
        return (raw_targets,)
    return tuple(raw_targets)


def _validate_choice(
    *,
    value: str,
    valid_values: set[str],
    field_name: str,
    errors: list[str],
) -> None:
    if value not in valid_values:
        errors.append(f"Unsupported {field_name}: {value}")
