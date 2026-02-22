from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .config import TinyLoRAConfig
from .tinylora import TinyLoRALinear
from .utils import count_unique_trainable_parameters

_LAYER_RE = re.compile(r"(?:layers|h|blocks)\.(\d+)\b")


@dataclass(frozen=True)
class InjectionReport:
    adapted_modules: int
    shared_vectors: int
    trainable_parameters: int
    module_to_group: dict[str, str]


def apply_tinylora(model: nn.Module, config: TinyLoRAConfig) -> InjectionReport:
    targets = collect_target_linears(model, config.target_modules)
    if not targets:
        raise ValueError("No target modules matched TinyLoRA target_modules")

    for parameter in model.parameters():
        parameter.requires_grad = False

    module_names = [name for name, _ in targets]
    module_to_group = assign_tie_groups(module_names, mode=config.tie_mode, tie_factor=config.tie_factor)

    shared_vectors: dict[str, nn.Parameter] = {}
    shared_devices: dict[str, torch.device] = {}
    for name, module in targets:
        group = module_to_group[name]
        shared = shared_vectors.get(group)
        if shared is None:
            device = module.weight.device
            shared = nn.Parameter(
                torch.zeros(config.proj_dim, dtype=config.vector_torch_dtype, device=device),
                requires_grad=True,
            )
            shared_vectors[group] = shared
            shared_devices[group] = device
        elif module.weight.device != shared_devices[group]:
            raise ValueError(
                f"Modules in tie group '{group}' span multiple devices "
                f"({shared_devices[group]} vs {module.weight.device}). "
                "Use a tie mode/factor that avoids cross-device sharing."
            )

        module_seed = _stable_seed(config.seed, name)
        wrapper = TinyLoRALinear.from_linear(
            base_linear=module,
            rank=config.rank,
            proj_dim=config.proj_dim,
            seed=module_seed,
            shared_vector=shared,
            vector_dtype=config.vector_torch_dtype,
            compute_dtype=config.compute_torch_dtype,
            scale=config.scale,
            svd_method=config.svd_method,
            svd_niter=config.svd_niter,
        )
        _replace_module(model, name, wrapper)

    trainable_count = count_unique_trainable_parameters(model.parameters())
    return InjectionReport(
        adapted_modules=len(targets),
        shared_vectors=len(shared_vectors),
        trainable_parameters=trainable_count,
        module_to_group=module_to_group,
    )


def collect_target_linears(model: nn.Module, target_modules: Iterable[str]) -> list[tuple[str, nn.Linear]]:
    targets: list[tuple[str, nn.Linear]] = []
    target_list = tuple(target_modules)
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if _matches_target(name, target_list):
            targets.append((name, module))
    return targets


def assign_tie_groups(module_names: list[str], mode: str, tie_factor: int) -> dict[str, str]:
    if tie_factor <= 0:
        raise ValueError("tie_factor must be positive")
    if mode not in {"none", "structured", "tiled", "full"}:
        raise ValueError(f"Unsupported tie mode: {mode}")

    if mode == "none":
        return {name: f"none:{idx}" for idx, name in enumerate(module_names)}
    if mode == "full":
        return {name: "full:0" for name in module_names}

    if mode == "structured":
        by_type: dict[str, list[str]] = defaultdict(list)
        for name in module_names:
            by_type[_module_type(name)].append(name)

        result: dict[str, str] = {}
        for module_type, names in by_type.items():
            ordered = sorted(names, key=_sort_key)
            for idx, name in enumerate(ordered):
                result[name] = f"structured:{module_type}:{idx // tie_factor}"
        return result

    # tiled mode: nearby modules are grouped agnostic of module type.
    ordered_global = sorted(module_names, key=_sort_key)
    return {name: f"tiled:{idx // tie_factor}" for idx, name in enumerate(ordered_global)}


def _matches_target(module_name: str, target_modules: tuple[str, ...]) -> bool:
    leaf = _module_type(module_name)
    for target in target_modules:
        if target.startswith("re:"):
            if re.search(target[3:], module_name):
                return True
        elif leaf == target or module_name.endswith(target):
            return True
    return False


def _module_type(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def _extract_layer_index(module_name: str) -> int:
    match = _LAYER_RE.search(module_name)
    if not match:
        return 10**9
    return int(match.group(1))


def _sort_key(module_name: str) -> tuple[int, str, str]:
    return (_extract_layer_index(module_name), _module_type(module_name), module_name)


def _stable_seed(base_seed: int, text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    value = int(digest[:8], 16)
    return (base_seed ^ value) % (2**31)


def _replace_module(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parent, child_name = _resolve_parent(root, module_name)
    if child_name.isdigit():
        parent[int(child_name)] = new_module  # type: ignore[index]
    else:
        setattr(parent, child_name, new_module)


def _resolve_parent(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]
