from __future__ import annotations

from ltr13.analysis import params_lora, params_lora_xs, params_tinylora


def test_parameter_formulas() -> None:
    assert params_lora(num_layers=2, modules_per_layer=7, width=8, rank=1) == 224
    assert params_lora_xs(num_layers=2, modules_per_layer=7, rank=1) == 14
    assert params_tinylora(num_layers=2, modules_per_layer=7, proj_dim=1, tie_factor=14) == 1
