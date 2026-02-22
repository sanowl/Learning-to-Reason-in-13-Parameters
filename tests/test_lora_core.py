from __future__ import annotations

import pytest
import torch
from torch import nn

from ltr13.config import TinyLoRAConfig
from ltr13.inject import apply_adapter
from ltr13.lora import LoRALinear, LoRAXSLinear
from ltr13.utils import count_unique_trainable_parameters


class Attn(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.o_proj = nn.Linear(width, width)


class Mlp(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.up_proj = nn.Linear(width, width)
        self.down_proj = nn.Linear(width, width)
        self.gate_proj = nn.Linear(width, width)


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.self_attn = Attn(width)
        self.mlp = Mlp(width)


class ToyTransformer(nn.Module):
    def __init__(self, num_layers: int = 2, width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(width) for _ in range(num_layers)])


def test_lora_linear_merge_unmerge_roundtrip() -> None:
    torch.manual_seed(0)
    base = nn.Linear(6, 6, bias=False)
    module = LoRALinear(base, rank=2, seed=11, lora_alpha=2.0)

    with torch.no_grad():
        module.b.fill_(0.1)

    x = torch.randn(3, 6)
    before = module(x)
    module.merge()
    merged = module(x)
    module.unmerge()
    after = module(x)

    assert torch.allclose(before, merged, atol=1e-6)
    assert torch.allclose(before, after, atol=1e-6)


def test_lora_xs_linear_shape_and_forward() -> None:
    torch.manual_seed(1)
    base = nn.Linear(5, 4, bias=False)
    module = LoRAXSLinear(base, rank=2, seed=3)

    delta = module.delta_weight(dtype=torch.float32)
    assert delta.shape == base.weight.shape

    x = torch.randn(2, 5)
    y = module(x)
    assert y.shape == (2, 4)


def test_apply_adapter_lora_parameter_count() -> None:
    model = ToyTransformer(num_layers=2, width=8)
    cfg = TinyLoRAConfig(
        adapter_type="lora",
        rank=2,
        proj_dim=1,
        tie_mode="none",
        tie_factor=1,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"),
        vector_dtype="float32",
        compute_dtype="float32",
    )
    report = apply_adapter(model, cfg)

    # 14 modules * (out*rank + rank*in) = 14 * (8*2 + 2*8) = 448
    assert report.adapted_modules == 14
    assert count_unique_trainable_parameters(model.parameters()) == 448


def test_apply_adapter_lora_xs_full_tie() -> None:
    model = ToyTransformer(num_layers=2, width=8)
    cfg = TinyLoRAConfig(
        adapter_type="lora_xs",
        rank=2,
        proj_dim=1,
        tie_mode="full",
        tie_factor=1,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"),
        vector_dtype="float32",
        compute_dtype="float32",
    )
    report = apply_adapter(model, cfg)

    assert report.shared_vectors == 1
    assert count_unique_trainable_parameters(model.parameters()) == 4


def test_apply_adapter_lora_rejects_incompatible_full_tie_shapes() -> None:
    class MixedShapes(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                [nn.ModuleDict({"q_proj": nn.Linear(8, 8), "up_proj": nn.Linear(8, 16)})]
            )

    model = MixedShapes()
    cfg = TinyLoRAConfig(
        adapter_type="lora",
        rank=2,
        proj_dim=1,
        tie_mode="full",
        tie_factor=1,
        target_modules=("q_proj", "up_proj"),
        vector_dtype="float32",
        compute_dtype="float32",
    )
    with pytest.raises(ValueError):
        apply_adapter(model, cfg)
