from __future__ import annotations

from contextlib import contextmanager

import torch
from torch import nn

from .adapter_ops import merge_all_adapters, unmerge_all_adapters


@contextmanager
def merged_tinylora(model: nn.Module):
    """Backward-compatible alias for merged_adapters_ctx."""
    merge_all_adapters(model)
    try:
        yield model
    finally:
        unmerge_all_adapters(model)


@contextmanager
def merged_adapters_ctx(model: nn.Module):
    """Temporarily merge adapter weights for inference-only forward passes."""
    merge_all_adapters(model)
    try:
        yield model
    finally:
        unmerge_all_adapters(model)


def truncated_importance_weights(
    train_log_probs: torch.Tensor,
    infer_log_probs: torch.Tensor,
    *,
    clip_max: float,
) -> torch.Tensor:
    """Compute clipped importance weights exp(log p_train - log p_infer)."""
    if clip_max <= 0:
        raise ValueError("clip_max must be positive")
    ratio = torch.exp(train_log_probs - infer_log_probs)
    return torch.clamp(ratio, max=clip_max)


def apply_truncated_is(
    advantages: torch.Tensor,
    train_log_probs: torch.Tensor,
    infer_log_probs: torch.Tensor,
    *,
    clip_max: float,
) -> torch.Tensor:
    weights = truncated_importance_weights(
        train_log_probs=train_log_probs,
        infer_log_probs=infer_log_probs,
        clip_max=clip_max,
    )
    return advantages * weights
