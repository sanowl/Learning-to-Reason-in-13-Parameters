from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ltr13.tinylora import TinyLoRALinear


def test_delta_formula_matches_manual() -> None:
    torch.manual_seed(0)
    base = nn.Linear(5, 4, bias=False)
    module = TinyLoRALinear(base, rank=2, proj_dim=3, seed=7)

    with torch.no_grad():
        module.v.copy_(torch.tensor([0.25, -0.5, 0.75], dtype=module.v.dtype))

    delta = module.delta_weight(dtype=torch.float32)

    r_matrix = torch.tensordot(module.v.to(module.compute_dtype), module.projection, dims=([0], [0]))
    manual = module.svd_u @ (module.svd_s.unsqueeze(1) * r_matrix) @ module.svd_vh
    manual = manual.to(torch.float32)

    assert torch.allclose(delta, manual, atol=1e-6)

    x = torch.randn(2, 5)
    expected = module.base_linear(x) + F.linear(x, manual)
    actual = module(x)
    assert torch.allclose(actual, expected, atol=1e-5)


def test_merge_and_unmerge_roundtrip() -> None:
    torch.manual_seed(1)
    base = nn.Linear(6, 3, bias=False)
    module = TinyLoRALinear(base, rank=2, proj_dim=2, seed=3)

    with torch.no_grad():
        module.v.copy_(torch.tensor([1.2, -0.3], dtype=module.v.dtype))

    x = torch.randn(4, 6)
    premerge = module(x)

    original_weight = module.base_linear.weight.detach().clone()
    module.merge()
    merged = module(x)

    assert module.merged
    assert torch.allclose(premerge, merged, atol=1e-5)

    module.unmerge()
    assert not module.merged
    assert torch.allclose(module.base_linear.weight, original_weight, atol=1e-6)


def test_merge_respects_scale_factor() -> None:
    torch.manual_seed(2)
    base = nn.Linear(4, 4, bias=False)
    module = TinyLoRALinear(base, rank=2, proj_dim=2, seed=9, scale=0.25)

    with torch.no_grad():
        module.v.copy_(torch.tensor([0.8, -0.2], dtype=module.v.dtype))

    x = torch.randn(3, 4)
    unmerged_output = module(x)
    module.merge()
    merged_output = module(x)

    assert torch.allclose(unmerged_output, merged_output, atol=1e-5)


def test_lowrank_svd_initialization_path() -> None:
    torch.manual_seed(3)
    base = nn.Linear(7, 5, bias=False)
    module = TinyLoRALinear(
        base,
        rank=2,
        proj_dim=1,
        seed=5,
        svd_method="lowrank",
        svd_niter=1,
    )
    delta = module.delta_weight(dtype=torch.float32)
    assert delta.shape == base.weight.shape
