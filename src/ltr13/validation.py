from __future__ import annotations

from typing import Any

from .config import parse_adapter_config
from .data import parse_dataset_config

_VALID_MODES = {"grpo", "sft", "eval"}


def validate_config(config: dict[str, Any]) -> None:
    errors: list[str] = []

    mode = str(config.get("mode", "grpo")).lower()
    if mode not in _VALID_MODES:
        errors.append(f"mode must be one of {_VALID_MODES}, got: {mode}")

    model_name = config.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        errors.append("model_name must be a non-empty string")

    seed = config.get("seed", 42)
    if not isinstance(seed, int):
        errors.append("seed must be an integer")

    deterministic = config.get("deterministic", False)
    if not isinstance(deterministic, bool):
        errors.append("deterministic must be a boolean")
    deterministic_warn_only = config.get("deterministic_warn_only", True)
    if not isinstance(deterministic_warn_only, bool):
        errors.append("deterministic_warn_only must be a boolean")
    init_state_path = config.get("init_trainable_state_path")
    if init_state_path is not None and not isinstance(init_state_path, str):
        errors.append("init_trainable_state_path must be a string when provided")
    strict_init = config.get("strict_init_trainable_state", True)
    if not isinstance(strict_init, bool):
        errors.append("strict_init_trainable_state must be a boolean")

    budget_cfg = config.get("budget_allocation")
    if budget_cfg is not None and not isinstance(budget_cfg, dict):
        errors.append("budget_allocation must be a mapping when provided")
    elif isinstance(budget_cfg, dict):
        _validate_budget_block(budget_cfg, errors)

    try:
        parse_adapter_config(config)
    except Exception as error:  # noqa: BLE001
        errors.append(f"adapter config invalid: {error}")

    if mode in {"grpo", "sft"}:
        if not isinstance(config.get("train_dataset"), dict):
            errors.append("train_dataset must be provided as a mapping")
        else:
            try:
                parse_dataset_config(config["train_dataset"], default_split="train")
            except Exception as error:  # noqa: BLE001
                errors.append(f"train_dataset invalid: {error}")

    if mode == "sft" and config.get("eval_dataset") is not None:
        if not isinstance(config.get("eval_dataset"), dict):
            errors.append("eval_dataset must be a mapping when provided")
        else:
            try:
                parse_dataset_config(config["eval_dataset"], default_split="test")
            except Exception as error:  # noqa: BLE001
                errors.append(f"eval_dataset invalid: {error}")

    training = config.get("training", {})
    if not isinstance(training, dict):
        errors.append("training must be a mapping")
        training = {}

    _validate_training_block(mode=mode, training=training, errors=errors)

    truncated_is = config.get("truncated_importance_sampling", {})
    if truncated_is is not None and not isinstance(truncated_is, dict):
        errors.append("truncated_importance_sampling must be a mapping when provided")
    elif isinstance(truncated_is, dict):
        _validate_truncated_is_block(truncated_is, errors)

    if mode == "eval":
        datasets = config.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            errors.append("eval mode requires non-empty `datasets` list")
        else:
            for index, dataset_cfg in enumerate(datasets):
                if not isinstance(dataset_cfg, dict):
                    errors.append(f"datasets[{index}] must be a mapping")
                    continue
                try:
                    parse_dataset_config(dataset_cfg, default_split="test")
                except Exception as error:  # noqa: BLE001
                    errors.append(f"datasets[{index}] invalid: {error}")

        generation = config.get("generation", {})
        if not isinstance(generation, dict):
            errors.append("generation must be a mapping")
        else:
            _require_positive_int(generation, "batch_size", errors, default=4)
            _require_positive_int(generation, "max_prompt_length", errors, default=1024)
            _require_positive_int(generation, "max_new_tokens", errors, default=1024)
            _require_non_negative_float(generation, "temperature", errors, default=0.0)

    guardrails = config.get("guardrails", {})
    if guardrails is not None and not isinstance(guardrails, dict):
        errors.append("guardrails must be a mapping when provided")
    elif isinstance(guardrails, dict):
        _validate_guardrails_block(guardrails, errors)

    if errors:
        joined = "\n- " + "\n- ".join(errors)
        raise ValueError(f"Configuration validation failed:{joined}")


def _validate_training_block(mode: str, training: dict[str, Any], errors: list[str]) -> None:
    if mode in {"grpo", "sft"}:
        _require_positive_float(training, "learning_rate", errors, default=1e-6)
        _require_positive_int(training, "per_device_train_batch_size", errors, default=1)
        _require_positive_int(training, "gradient_accumulation_steps", errors, default=1)
        _require_positive_float(training, "num_train_epochs", errors, default=3.0)

    if mode == "grpo":
        _require_positive_int(training, "num_generations", errors, default=4)
        _require_positive_int(training, "max_prompt_length", errors, default=1024)
        _require_positive_int(training, "max_completion_length", errors, default=3072)
        _require_non_negative_float(training, "beta", errors, default=0.0)

    if mode == "sft":
        _require_non_negative_float(training, "warmup_ratio", errors, default=0.03)
        if "lr_scheduler_type" in training and not isinstance(training["lr_scheduler_type"], str):
            errors.append("training.lr_scheduler_type must be a string")


def _validate_guardrails_block(guardrails: dict[str, Any], errors: list[str]) -> None:
    if "expected_trainable_params" in guardrails:
        value = guardrails["expected_trainable_params"]
        if not isinstance(value, int) or value < 0:
            errors.append("guardrails.expected_trainable_params must be a non-negative integer")

    if "expected_trainable_bytes" in guardrails:
        value = guardrails["expected_trainable_bytes"]
        if not isinstance(value, int) or value < 0:
            errors.append("guardrails.expected_trainable_bytes must be a non-negative integer")

    if "tolerance" in guardrails:
        value = guardrails["tolerance"]
        if not isinstance(value, int) or value < 0:
            errors.append("guardrails.tolerance must be a non-negative integer")

    if "require_trainable_params" in guardrails and not isinstance(
        guardrails["require_trainable_params"], bool
    ):
        errors.append("guardrails.require_trainable_params must be boolean")


def _validate_truncated_is_block(config: dict[str, Any], errors: list[str]) -> None:
    if "enabled" in config and not isinstance(config["enabled"], bool):
        errors.append("truncated_importance_sampling.enabled must be boolean")
    if "clip_max" in config:
        value = config["clip_max"]
        if not isinstance(value, (int, float)) or float(value) <= 0:
            errors.append("truncated_importance_sampling.clip_max must be a positive number")


def _validate_budget_block(config: dict[str, Any], errors: list[str]) -> None:
    if "enabled" in config and not isinstance(config["enabled"], bool):
        errors.append("budget_allocation.enabled must be boolean")
    if "total_proj_dim_budget" in config:
        value = config["total_proj_dim_budget"]
        if not isinstance(value, int) or value <= 0:
            errors.append("budget_allocation.total_proj_dim_budget must be a positive integer")
    if "min_proj_dim_per_group" in config:
        value = config["min_proj_dim_per_group"]
        if not isinstance(value, int) or value <= 0:
            errors.append("budget_allocation.min_proj_dim_per_group must be a positive integer")
    if "max_proj_dim_per_group" in config:
        value = config["max_proj_dim_per_group"]
        if not isinstance(value, int) or value <= 0:
            errors.append("budget_allocation.max_proj_dim_per_group must be a positive integer")
    if "strategy" in config and str(config["strategy"]) not in {"uniform", "gradient"}:
        errors.append("budget_allocation.strategy must be one of {uniform, gradient}")
    if "group_scores_path" in config and not isinstance(config["group_scores_path"], str):
        errors.append("budget_allocation.group_scores_path must be a string")


def _require_positive_int(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    default: int,
) -> None:
    value = mapping.get(key, default)
    if not isinstance(value, int) or value <= 0:
        errors.append(f"{key} must be a positive integer")


def _require_positive_float(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    default: float,
) -> None:
    value = mapping.get(key, default)
    if not isinstance(value, (int, float)) or float(value) <= 0:
        errors.append(f"{key} must be a positive number")


def _require_non_negative_float(
    mapping: dict[str, Any],
    key: str,
    errors: list[str],
    *,
    default: float,
) -> None:
    value = mapping.get(key, default)
    if not isinstance(value, (int, float)) or float(value) < 0:
        errors.append(f"{key} must be a non-negative number")
