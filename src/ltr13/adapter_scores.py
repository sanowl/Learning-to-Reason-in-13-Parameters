from __future__ import annotations

from typing import Any

import torch
from torch import nn


def snapshot_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def compute_group_delta_scores(
    *,
    model: nn.Module,
    initial_state: dict[str, torch.Tensor],
    parameter_to_group: dict[str, str],
) -> dict[str, float]:
    scores: dict[str, float] = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name not in initial_state:
            continue

        group = parameter_to_group.get(name)
        if group is None:
            continue

        delta = parameter.detach().cpu() - initial_state[name]
        value = float(torch.linalg.norm(delta).item())
        scores[group] = scores.get(group, 0.0) + value

    return scores


def build_group_score_payload(
    *,
    group_scores: dict[str, float],
    adapter_type: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "adapter_type": adapter_type,
        "seed": seed,
        "group_scores": group_scores,
    }
    if extra:
        payload["extra"] = extra
    return payload
