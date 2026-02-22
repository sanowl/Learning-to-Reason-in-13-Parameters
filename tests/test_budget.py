from __future__ import annotations

import json
from pathlib import Path

from ltr13.budget import resolve_group_proj_dims


def test_uniform_budget_allocation() -> None:
    groups = ["g1", "g2", "g3"]
    dims = resolve_group_proj_dims(
        groups=groups,
        default_proj_dim=4,
        budget_cfg={
            "enabled": True,
            "strategy": "uniform",
            "total_proj_dim_budget": 9,
            "min_proj_dim_per_group": 1,
            "max_proj_dim_per_group": 4,
        },
    )
    assert sum(dims.values()) == 9
    assert all(1 <= value <= 4 for value in dims.values())


def test_gradient_budget_allocation(tmp_path: Path) -> None:
    score_path = tmp_path / "scores.json"
    score_path.write_text(json.dumps({"group_scores": {"g1": 10.0, "g2": 1.0}}), encoding="utf-8")

    dims = resolve_group_proj_dims(
        groups=["g1", "g2"],
        default_proj_dim=8,
        budget_cfg={
            "enabled": True,
            "strategy": "gradient",
            "total_proj_dim_budget": 8,
            "min_proj_dim_per_group": 1,
            "max_proj_dim_per_group": 8,
            "group_scores_path": str(score_path),
        },
    )
    assert dims["g1"] > dims["g2"]
    assert sum(dims.values()) == 8
