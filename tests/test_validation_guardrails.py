from __future__ import annotations

import pytest

from ltr13.guardrails import TrainableStats, enforce_trainable_guardrails
from ltr13.validation import validate_config


def test_validate_config_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError):
        validate_config({"mode": "bad", "model_name": "x"})


def test_validate_config_accepts_minimal_eval_shape() -> None:
    validate_config(
        {
            "mode": "eval",
            "model_name": "Qwen/Qwen2.5-7B-Instruct",
            "datasets": [
                {
                    "source": "hf",
                    "path": "gsm8k",
                    "name": "main",
                    "split": "test",
                    "prompt_field": "question",
                    "answer_field": "answer",
                }
            ],
            "generation": {
                "batch_size": 1,
                "max_prompt_length": 16,
                "max_new_tokens": 16,
                "temperature": 0.0,
            },
        }
    )


def test_guardrails_enforce_expected_counts() -> None:
    stats = TrainableStats(parameters=13, bytes=52)
    enforce_trainable_guardrails(
        stats,
        {
            "expected_trainable_params": 13,
            "expected_trainable_bytes": 52,
            "tolerance": 0,
        },
    )



def test_guardrails_fail_on_mismatch() -> None:
    stats = TrainableStats(parameters=10, bytes=40)
    with pytest.raises(ValueError):
        enforce_trainable_guardrails(
            stats,
            {
                "expected_trainable_params": 13,
                "expected_trainable_bytes": 52,
                "tolerance": 0,
            },
        )
