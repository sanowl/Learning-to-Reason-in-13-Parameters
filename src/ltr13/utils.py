from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic_mode(enabled: bool, *, warn_only: bool = True) -> None:
    torch.backends.cudnn.deterministic = enabled
    torch.backends.cudnn.benchmark = not enabled

    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
    except TypeError:
        # Older torch may not support warn_only argument.
        torch.use_deterministic_algorithms(enabled)


def configure_reproducibility(
    seed: int,
    *,
    deterministic: bool,
    deterministic_warn_only: bool = True,
) -> None:
    set_global_seed(seed)
    set_deterministic_mode(deterministic, warn_only=deterministic_warn_only)


def count_unique_trainable_parameters(parameters: Iterable[torch.nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        pid = id(parameter)
        if pid in seen:
            continue
        seen.add(pid)
        total += parameter.numel()
    return total


def count_unique_trainable_bytes(parameters: Iterable[torch.nn.Parameter]) -> int:
    seen: set[int] = set()
    total = 0
    for parameter in parameters:
        if not parameter.requires_grad:
            continue
        pid = id(parameter)
        if pid in seen:
            continue
        seen.add(pid)
        total += parameter.numel() * parameter.element_size()
    return total
