from __future__ import annotations

import pytest

from ltr13.config import DEFAULT_TARGET_MODULES, TinyLoRAConfig, parse_adapter_config, parse_tinylora_config


def test_parse_tinylora_config_none_targets_uses_defaults() -> None:
    cfg = parse_tinylora_config({"target_modules": None})
    assert cfg is not None
    assert cfg.target_modules == DEFAULT_TARGET_MODULES


def test_tinylora_config_rejects_empty_targets() -> None:
    with pytest.raises(ValueError):
        TinyLoRAConfig(target_modules=())


def test_tinylora_config_rejects_bad_choice() -> None:
    with pytest.raises(ValueError):
        TinyLoRAConfig(tie_mode="invalid")


def test_parse_adapter_config_reads_adapter_block() -> None:
    cfg = parse_adapter_config({"adapter": {"adapter_type": "lora", "rank": 2, "proj_dim": 1}})
    assert cfg is not None
    assert cfg.adapter_type == "lora"
