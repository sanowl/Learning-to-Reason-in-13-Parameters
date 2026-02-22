from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def extract_trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def save_trainable_state(model: nn.Module, output_path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "trainable_state_dict": extract_trainable_state_dict(model),
        "metadata": metadata or {},
    }
    torch.save(payload, output)


def load_trainable_state(model: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> list[str]:
    payload = torch.load(checkpoint_path, map_location="cpu")
    state_dict = payload["trainable_state_dict"]
    unexpected: list[str] = []
    named_parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    missing_in_checkpoint = sorted(set(named_parameters) - set(state_dict))

    with torch.no_grad():
        for name, tensor in state_dict.items():
            if name not in named_parameters:
                unexpected.append(name)
                continue
            named_parameters[name].copy_(tensor.to(named_parameters[name].dtype))

    if strict and (unexpected or missing_in_checkpoint):
        details: list[str] = []
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        if missing_in_checkpoint:
            details.append(f"missing_in_checkpoint: {', '.join(missing_in_checkpoint)}")
        raise KeyError("Trainable state mismatch while loading: " + "; ".join(details))
    return unexpected
