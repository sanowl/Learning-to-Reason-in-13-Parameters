from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ltr13.checkpointing import load_trainable_state, save_trainable_state
from ltr13.config import TinyLoRAConfig
from ltr13.inject import apply_tinylora
from ltr13.tinylora import merge_all_tinylora, unmerge_all_tinylora


class TinyNet(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict({"q_proj": nn.Linear(width, width)})])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers[0]["q_proj"](x)


def _build_config() -> TinyLoRAConfig:
    return TinyLoRAConfig(
        rank=2,
        proj_dim=3,
        tie_mode="full",
        tie_factor=1,
        target_modules=("q_proj",),
        seed=7,
        vector_dtype="float32",
        compute_dtype="float32",
    )


def _set_trainable_values(model: nn.Module) -> None:
    with torch.no_grad():
        for index, parameter in enumerate(p for p in model.parameters() if p.requires_grad):
            parameter.copy_(torch.full_like(parameter, float(index + 1) * 0.125))


def test_checkpoint_roundtrip_across_fresh_model(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = TinyNet()
    apply_tinylora(model, _build_config())
    _set_trainable_values(model)

    x = torch.randn(3, 8)
    expected_output = model(x)

    checkpoint = tmp_path / "trainable_state.pt"
    save_trainable_state(model, checkpoint, metadata={"note": "roundtrip"})

    torch.manual_seed(0)
    reloaded = TinyNet()
    apply_tinylora(reloaded, _build_config())
    missing = load_trainable_state(reloaded, checkpoint, strict=True)

    assert missing == []
    actual_output = reloaded(x)
    assert torch.allclose(expected_output, actual_output, atol=1e-6)


def test_merge_unmerge_consistency_after_loading(tmp_path: Path) -> None:
    torch.manual_seed(1)
    model = TinyNet()
    apply_tinylora(model, _build_config())
    _set_trainable_values(model)

    checkpoint = tmp_path / "state.pt"
    save_trainable_state(model, checkpoint)

    loaded = TinyNet()
    apply_tinylora(loaded, _build_config())
    load_trainable_state(loaded, checkpoint, strict=True)

    x = torch.randn(2, 8)
    unmerged_output = loaded(x)

    merge_all_tinylora(loaded)
    merged_output = loaded(x)
    assert torch.allclose(unmerged_output, merged_output, atol=1e-6)

    unmerge_all_tinylora(loaded)
    roundtrip_output = loaded(x)
    assert torch.allclose(unmerged_output, roundtrip_output, atol=1e-6)


def test_strict_load_detects_missing_checkpoint_keys(tmp_path: Path) -> None:
    model = TinyNet()
    apply_tinylora(model, _build_config())
    _set_trainable_values(model)

    checkpoint = tmp_path / "broken.pt"
    save_trainable_state(model, checkpoint)

    payload = torch.load(checkpoint, map_location="cpu")
    key_to_drop = next(iter(payload["trainable_state_dict"]))
    payload["trainable_state_dict"].pop(key_to_drop)
    torch.save(payload, checkpoint)

    fresh = TinyNet()
    apply_tinylora(fresh, _build_config())
    with pytest.raises(KeyError):
        load_trainable_state(fresh, checkpoint, strict=True)
