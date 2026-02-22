from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_torch_dtype(name: str | None) -> torch.dtype | None:
    if name is None:
        return None
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


def load_model_and_tokenizer(
    model_name: str,
    *,
    tokenizer_name: str | None = None,
    torch_dtype: str | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> tuple[AutoModelForCausalLM, Any]:
    dtype = resolve_torch_dtype(torch_dtype)
    kwargs = dict(model_kwargs or {})
    if dtype is not None:
        kwargs["torch_dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer
