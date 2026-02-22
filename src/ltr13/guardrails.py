from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import nn

from .utils import count_unique_trainable_bytes, count_unique_trainable_parameters


@dataclass(frozen=True)
class TrainableStats:
    parameters: int
    bytes: int


def compute_trainable_stats(model: nn.Module) -> TrainableStats:
    return TrainableStats(
        parameters=count_unique_trainable_parameters(model.parameters()),
        bytes=count_unique_trainable_bytes(model.parameters()),
    )


def enforce_trainable_guardrails(
    stats: TrainableStats,
    guardrails: dict[str, Any] | None,
) -> None:
    guardrails = guardrails or {}

    expected_params = guardrails.get("expected_trainable_params")
    expected_bytes = guardrails.get("expected_trainable_bytes")
    tolerance = int(guardrails.get("tolerance", 0))
    require_trainable = bool(guardrails.get("require_trainable_params", True))

    errors: list[str] = []

    if require_trainable and stats.parameters <= 0:
        errors.append("No trainable parameters found")

    if expected_params is not None:
        if abs(stats.parameters - int(expected_params)) > tolerance:
            errors.append(
                "Trainable parameter count mismatch: "
                f"expected={expected_params}, actual={stats.parameters}, tolerance={tolerance}"
            )

    if expected_bytes is not None:
        if abs(stats.bytes - int(expected_bytes)) > tolerance:
            errors.append(
                "Trainable byte count mismatch: "
                f"expected={expected_bytes}, actual={stats.bytes}, tolerance={tolerance}"
            )

    if errors:
        joined = "\n- " + "\n- ".join(errors)
        raise ValueError(f"Guardrail validation failed:{joined}")
