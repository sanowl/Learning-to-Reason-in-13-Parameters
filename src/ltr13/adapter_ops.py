from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from torch import nn

from .lora import LoRALinear, LoRAXSLinear
from .tinylora import TinyLoRALinear

ADAPTER_MODULE_TYPES = (TinyLoRALinear, LoRALinear, LoRAXSLinear)


def iter_adapter_modules(model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
    for name, module in model.named_modules():
        if isinstance(module, ADAPTER_MODULE_TYPES):
            yield name, module


def merge_all_adapters(model: nn.Module) -> None:
    for _, module in iter_adapter_modules(model):
        module.merge()


def unmerge_all_adapters(model: nn.Module) -> None:
    for _, module in iter_adapter_modules(model):
        module.unmerge()


@contextmanager
def merged_adapters(model: nn.Module):
    merge_all_adapters(model)
    try:
        yield model
    finally:
        unmerge_all_adapters(model)
