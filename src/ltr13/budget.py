from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_group_proj_dims(
    *,
    groups: list[str],
    default_proj_dim: int,
    budget_cfg: dict[str, Any] | None,
) -> dict[str, int]:
    if not groups:
        return {}

    if not budget_cfg or not bool(budget_cfg.get("enabled", False)):
        return {group: default_proj_dim for group in groups}

    total_budget = int(budget_cfg.get("total_proj_dim_budget", default_proj_dim * len(groups)))
    min_per_group = int(budget_cfg.get("min_proj_dim_per_group", 1))
    max_per_group = int(budget_cfg.get("max_proj_dim_per_group", default_proj_dim))
    strategy = str(budget_cfg.get("strategy", "uniform")).lower()

    if total_budget <= 0:
        raise ValueError("budget_allocation.total_proj_dim_budget must be positive")
    if min_per_group <= 0:
        raise ValueError("budget_allocation.min_proj_dim_per_group must be positive")
    if max_per_group < min_per_group:
        raise ValueError("budget_allocation.max_proj_dim_per_group must be >= min_proj_dim_per_group")

    group_count = len(groups)
    base_total = min_per_group * group_count
    if base_total > total_budget:
        raise ValueError(
            "Budget too small: total_proj_dim_budget is less than min_proj_dim_per_group * num_groups"
        )

    dims = {group: min_per_group for group in groups}
    remaining = total_budget - base_total
    if remaining == 0:
        return dims

    if strategy == "uniform":
        scores = {group: 1.0 for group in groups}
    elif strategy == "gradient":
        scores = _load_gradient_scores(groups, budget_cfg.get("group_scores_path"))
    else:
        raise ValueError(f"Unsupported budget allocation strategy: {strategy}")

    # Greedy weighted allocation with capacity constraints.
    while remaining > 0:
        eligible = [group for group in groups if dims[group] < max_per_group]
        if not eligible:
            break
        total_score = sum(max(scores[group], 0.0) for group in eligible)
        if total_score <= 0:
            total_score = float(len(eligible))
            weight = {group: 1.0 / total_score for group in eligible}
        else:
            weight = {group: max(scores[group], 0.0) / total_score for group in eligible}

        allocated_this_round = 0
        for group in eligible:
            if remaining <= 0:
                break
            # Round toward at least one unit for high-score groups.
            proposal = max(1, int(round(weight[group] * remaining)))
            room = max_per_group - dims[group]
            delta = min(room, remaining, proposal)
            if delta <= 0:
                continue
            dims[group] += delta
            remaining -= delta
            allocated_this_round += delta

        if allocated_this_round == 0:
            # Fallback to avoid infinite loop under extreme rounding.
            for group in eligible:
                if remaining <= 0:
                    break
                if dims[group] < max_per_group:
                    dims[group] += 1
                    remaining -= 1

    if remaining > 0:
        raise ValueError(
            "Budget allocation cannot satisfy fixed total budget with current "
            "max_proj_dim_per_group constraints"
        )

    return dims


def _load_gradient_scores(groups: list[str], path: Any) -> dict[str, float]:
    if path is None:
        raise ValueError("budget_allocation.group_scores_path is required for strategy=gradient")

    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    if isinstance(payload, dict) and isinstance(payload.get("group_scores"), dict):
        raw_scores = payload["group_scores"]
    elif isinstance(payload, dict):
        raw_scores = payload
    else:
        raise ValueError("Invalid group score file format")

    scores: dict[str, float] = {}
    for group in groups:
        value = raw_scores.get(group, 0.0)
        scores[group] = float(value)
    return scores
