from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from torch import nn

from .budget import resolve_group_proj_dims
from .config import TinyLoRAConfig
from .lora import LoRALinear, LoRAXSLinear
from .tinylora import TinyLoRALinear
from .utils import count_unique_trainable_parameters

_LAYER_RE = re.compile(r"(?:layers|h|blocks)\.(\d+)\b")


@dataclass(frozen=True)
class InjectionReport:
    adapted_modules: int
    shared_vectors: int
    trainable_parameters: int
    module_to_group: dict[str, str]
    parameter_to_group: dict[str, str]


def apply_adapter(
    model: nn.Module,
    config: TinyLoRAConfig,
    *,
    budget_cfg: dict[str, Any] | None = None,
) -> InjectionReport:
    targets = collect_target_linears(model, config.target_modules)
    if not targets:
        raise ValueError("No target modules matched adapter target_modules")

    for parameter in model.parameters():
        parameter.requires_grad = False

    module_names = [name for name, _ in targets]
    module_to_group = assign_tie_groups(module_names, mode=config.tie_mode, tie_factor=config.tie_factor)
    group_names = sorted(set(module_to_group.values()))
    group_proj_dims = resolve_group_proj_dims(
        groups=group_names,
        default_proj_dim=config.proj_dim,
        budget_cfg=budget_cfg if config.adapter_type == "tinylora" else None,
    )

    shared_payloads: dict[str, object] = {}
    shared_devices: dict[str, torch.device] = {}
    shared_shapes: dict[str, tuple[int, int]] = {}

    for name, module in targets:
        group = module_to_group[name]
        group_seed = _stable_seed(config.seed, group)
        module_seed = _stable_seed(config.seed, name)

        payload = shared_payloads.get(group)
        device = module.weight.device
        if payload is None:
            shared_devices[group] = device

        if device != shared_devices[group]:
            raise ValueError(
                f"Modules in tie group '{group}' span multiple devices "
                f"({shared_devices[group]} vs {device}). "
                "Use a tie mode/factor that avoids cross-device sharing."
            )

        if config.adapter_type == "tinylora":
            proj_dim = int(group_proj_dims[group])
            if payload is None:
                shared_vector = nn.Parameter(
                    torch.zeros(proj_dim, dtype=config.vector_torch_dtype, device=device),
                    requires_grad=True,
                )
                if config.projection_mode == "structured":
                    block_count = config.projection_blocks * config.projection_blocks
                    shared_scales = nn.Parameter(
                        torch.ones(block_count, dtype=config.vector_torch_dtype, device=device),
                        requires_grad=True,
                    )
                    payload = (shared_vector, shared_scales)
                else:
                    payload = shared_vector
                shared_payloads[group] = payload

            if config.projection_mode == "structured":
                shared_vector, shared_scales = payload  # type: ignore[misc]
            else:
                shared_vector = payload  # type: ignore[assignment]
                shared_scales = None

            wrapper = TinyLoRALinear.from_linear(
                base_linear=module,
                rank=config.rank,
                proj_dim=proj_dim,
                seed=module_seed,
                shared_vector=shared_vector,  # type: ignore[arg-type]
                vector_dtype=config.vector_torch_dtype,
                compute_dtype=config.compute_torch_dtype,
                scale=config.scale,
                svd_method=config.svd_method,
                svd_niter=config.svd_niter,
                projection_mode=config.projection_mode,
                projection_blocks=config.projection_blocks,
                shared_proj_block_scales=shared_scales,  # type: ignore[arg-type]
            )

        elif config.adapter_type == "lora_xs":
            if payload is None:
                payload = nn.Parameter(
                    torch.zeros(config.rank, config.rank, dtype=config.vector_torch_dtype, device=device),
                    requires_grad=True,
                )
                shared_payloads[group] = payload

            wrapper = LoRAXSLinear(
                base_linear=module,
                rank=config.rank,
                seed=module_seed,
                shared_r=payload,  # type: ignore[arg-type]
                vector_dtype=config.vector_torch_dtype,
                compute_dtype=config.compute_torch_dtype,
                scale=config.scale,
                svd_method=config.svd_method,
                svd_niter=config.svd_niter,
            )

        elif config.adapter_type == "lora":
            current_shape = tuple(module.weight.shape)
            if payload is None:
                a = nn.Parameter(
                    torch.empty(
                        module.out_features,
                        config.rank,
                        dtype=config.vector_torch_dtype,
                        device=device,
                    ),
                    requires_grad=True,
                )
                b = nn.Parameter(
                    torch.empty(
                        config.rank,
                        module.in_features,
                        dtype=config.vector_torch_dtype,
                        device=device,
                    ),
                    requires_grad=True,
                )
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(group_seed)
                    nn.init.normal_(a, mean=0.0, std=1.0 / max(1, config.rank) ** 0.5)
                    nn.init.zeros_(b)

                payload = (a, b)
                shared_shapes[group] = current_shape
                shared_payloads[group] = payload

            expected_shape = shared_shapes[group]
            if current_shape != expected_shape:
                raise ValueError(
                    f"LoRA shared group '{group}' mixes incompatible module shapes "
                    f"{expected_shape} and {current_shape}. Use tie_mode=none or a larger tie_factor."
                )

            shared_a, shared_b = payload  # type: ignore[misc]
            wrapper = LoRALinear(
                base_linear=module,
                rank=config.rank,
                seed=module_seed,
                shared_a=shared_a,
                shared_b=shared_b,
                vector_dtype=config.vector_torch_dtype,
                compute_dtype=config.compute_torch_dtype,
                scale=config.scale,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
            )

        else:
            raise ValueError(f"Unsupported adapter type: {config.adapter_type}")

        _replace_module(model, name, wrapper)

    trainable_count = count_unique_trainable_parameters(model.parameters())
    parameter_to_group: dict[str, str] = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "." not in parameter_name:
            continue
        module_name = parameter_name.rsplit(".", 1)[0]
        if module_name in module_to_group:
            parameter_to_group[parameter_name] = module_to_group[module_name]

    return InjectionReport(
        adapted_modules=len(targets),
        shared_vectors=len(shared_payloads),
        trainable_parameters=trainable_count,
        module_to_group=module_to_group,
        parameter_to_group=parameter_to_group,
    )


def apply_tinylora(
    model: nn.Module,
    config: TinyLoRAConfig,
    *,
    budget_cfg: dict[str, Any] | None = None,
) -> InjectionReport:
    """Backward-compatible alias."""
    return apply_adapter(model, config, budget_cfg=budget_cfg)


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
