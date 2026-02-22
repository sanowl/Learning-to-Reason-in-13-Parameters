from __future__ import annotations

import math


def params_full_ft(num_layers: int, modules_per_layer: int, width: int) -> int:
    return num_layers * modules_per_layer * width * width


def params_lora(num_layers: int, modules_per_layer: int, width: int, rank: int) -> int:
    return num_layers * modules_per_layer * (2 * width * rank)


def params_lora_xs(num_layers: int, modules_per_layer: int, rank: int) -> int:
    return num_layers * modules_per_layer * rank * rank


def params_vera(num_layers: int, modules_per_layer: int, width: int, rank: int) -> int:
    return num_layers * modules_per_layer * (width + rank)


def params_tinylora(
    num_layers: int,
    modules_per_layer: int,
    proj_dim: int,
    tie_factor: int,
) -> int:
    if tie_factor <= 0:
        raise ValueError("tie_factor must be positive")
    groups = math.ceil((num_layers * modules_per_layer) / tie_factor)
    return groups * proj_dim


def params_to_bytes(param_count: int, bytes_per_param: int) -> int:
    if param_count < 0:
        raise ValueError("param_count must be non-negative")
    if bytes_per_param <= 0:
        raise ValueError("bytes_per_param must be positive")
    return param_count * bytes_per_param
