from __future__ import annotations

from torch import nn

from ltr13.config import TinyLoRAConfig
from ltr13.inject import apply_tinylora
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


def _base_config(**kwargs) -> TinyLoRAConfig:
    defaults = dict(
        rank=2,
        proj_dim=1,
        tie_mode="full",
        tie_factor=1,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
        ),
        seed=0,
        vector_dtype="float32",
        compute_dtype="float32",
    )
    defaults.update(kwargs)
    return TinyLoRAConfig(**defaults)


def test_full_tie_can_reach_single_trainable_parameter() -> None:
    model = ToyTransformer(num_layers=2)
    report = apply_tinylora(model, _base_config(proj_dim=1, tie_mode="full", tie_factor=1))

    assert report.adapted_modules == 14
    assert report.shared_vectors == 1
    assert count_unique_trainable_parameters(model.parameters()) == 1


def test_no_tie_scales_with_number_of_modules() -> None:
    model = ToyTransformer(num_layers=2)
    report = apply_tinylora(model, _base_config(proj_dim=3, tie_mode="none", tie_factor=1))

    assert report.adapted_modules == 14
    assert report.shared_vectors == 14
    assert count_unique_trainable_parameters(model.parameters()) == 42


def test_structured_tie_groups_by_module_type() -> None:
    model = ToyTransformer(num_layers=2)
    report = apply_tinylora(model, _base_config(proj_dim=2, tie_mode="structured", tie_factor=2))

    # 7 module types; with two layers and tie_factor=2, one shared vector per type.
    assert report.shared_vectors == 7
    assert count_unique_trainable_parameters(model.parameters()) == 14


def test_tiled_tie_groups_nearby_modules() -> None:
    model = ToyTransformer(num_layers=2)
    report = apply_tinylora(model, _base_config(proj_dim=2, tie_mode="tiled", tie_factor=4))

    # 14 modules grouped by fours -> ceil(14/4)=4 groups.
    assert report.shared_vectors == 4
    assert count_unique_trainable_parameters(model.parameters()) == 8


def test_base_model_parameters_are_frozen_after_injection() -> None:
    model = ToyTransformer(num_layers=1)
    apply_tinylora(model, _base_config(proj_dim=2, tie_mode="none", tie_factor=1))

    all_trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert all(name.endswith(".v") for name in all_trainable)


def test_budget_allocation_changes_effective_tinylora_parameter_count() -> None:
    model = ToyTransformer(num_layers=1)
    cfg = _base_config(proj_dim=4, tie_mode="structured", tie_factor=1)
    report = apply_tinylora(
        model,
        cfg,
        budget_cfg={
            "enabled": True,
            "strategy": "uniform",
            "total_proj_dim_budget": 7,
            "min_proj_dim_per_group": 1,
            "max_proj_dim_per_group": 4,
        },
    )

    assert report.shared_vectors == 7
    assert count_unique_trainable_parameters(model.parameters()) == 7


def test_structured_projection_scales_are_shared_per_group() -> None:
    model = ToyTransformer(num_layers=1)
    cfg = _base_config(
        proj_dim=1,
        tie_mode="full",
        tie_factor=1,
        projection_mode="structured",
        projection_blocks=2,
    )
    apply_tinylora(model, cfg)
    # 1 shared v + (2x2)=4 shared block scales.
    assert count_unique_trainable_parameters(model.parameters()) == 5
