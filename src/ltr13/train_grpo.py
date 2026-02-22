from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .checkpointing import save_trainable_state
from .config import load_yaml_config, parse_tinylora_config
from .data import load_reasoning_dataset, parse_dataset_config
from .guardrails import compute_trainable_stats, enforce_trainable_guardrails
from .inject import apply_tinylora
from .metadata import build_run_metadata, write_run_metadata
from .modeling import load_model_and_tokenizer
from .reward import exact_match_reward
from .utils import configure_reproducibility
from .validation import validate_config

try:
    from trl import GRPOConfig, GRPOTrainer
except ImportError as error:  # pragma: no cover - dependency/environment dependent
    raise ImportError(
        "trl with GRPO support is required. Install with `pip install trl` and retry."
    ) from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GRPO with optional TinyLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            if "content" in first:
                return str(first["content"])
            if "text" in first:
                return str(first["text"])
        return str(first)
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion["content"])
        if "text" in completion:
            return str(completion["text"])
    return str(completion)


def build_reward_fn():
    def _normalize_references(reference_answer: Any, target_len: int) -> list[str]:
        if isinstance(reference_answer, str):
            return [reference_answer] * target_len

        refs = [str(item) for item in reference_answer]
        if not refs:
            raise ValueError("reference_answer is empty")
        if len(refs) == target_len:
            return refs
        if len(refs) == 1:
            return refs * target_len
        if target_len % len(refs) == 0:
            repeat = target_len // len(refs)
            return [ref for ref in refs for _ in range(repeat)]
        raise ValueError(
            "Cannot align reference_answer with completions: "
            f"{len(refs)} references vs {target_len} predictions"
        )

    def reward_func(completions: list[Any], reference_answer: list[str], **_: Any) -> list[float]:
        predictions = [_completion_to_text(completion) for completion in completions]
        references = _normalize_references(reference_answer, len(predictions))
        return exact_match_reward(predictions=predictions, references=references)

    return reward_func


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    validate_config(config)

    seed = int(config.get("seed", 42))
    deterministic = bool(config.get("deterministic", False))
    deterministic_warn_only = bool(config.get("deterministic_warn_only", True))
    configure_reproducibility(
        seed=seed,
        deterministic=deterministic,
        deterministic_warn_only=deterministic_warn_only,
    )

    model_name = str(config["model_name"])
    tokenizer_name = config.get("tokenizer_name")
    model_dtype = config.get("model_dtype")

    model, tokenizer = load_model_and_tokenizer(
        model_name=model_name,
        tokenizer_name=tokenizer_name,
        torch_dtype=model_dtype,
        model_kwargs=config.get("model_kwargs"),
    )

    tinylora_cfg = parse_tinylora_config(config.get("tinylora"))
    if tinylora_cfg is not None:
        report = apply_tinylora(model, tinylora_cfg)
        print(
            "[TinyLoRA] adapted_modules=",
            report.adapted_modules,
            " shared_vectors=",
            report.shared_vectors,
            " trainable_params=",
            report.trainable_parameters,
        )

    train_dataset = load_reasoning_dataset(
        parse_dataset_config(config["train_dataset"], default_split="train")
    )

    train_cfg: dict[str, Any] = dict(config.get("training", {}))
    output_dir = str(config.get("output_dir", "outputs/grpo"))

    if bool(train_cfg.get("disable_cache_during_train", True)):
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    if tinylora_cfg is not None and bool(train_cfg.get("enable_input_require_grads", True)):
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(train_cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        bf16=bool(train_cfg.get("bf16", True)),
        fp16=bool(train_cfg.get("fp16", False)),
        max_prompt_length=int(train_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(train_cfg.get("max_completion_length", 3072)),
        num_generations=int(train_cfg.get("num_generations", 4)),
        beta=float(train_cfg.get("beta", 0.0)),
        report_to=list(train_cfg.get("report_to", [])),
        seed=seed,
    )

    stats = compute_trainable_stats(model)
    enforce_trainable_guardrails(stats=stats, guardrails=config.get("guardrails"))
    print(f"[Trainable] parameters={stats.parameters}, bytes={stats.bytes}")

    run_metadata = build_run_metadata(
        mode="grpo",
        config_path=args.config,
        config=config,
        seed=seed,
        deterministic=deterministic,
        output_dir=output_dir,
        extra={
            "trainable_parameters": stats.parameters,
            "trainable_bytes": stats.bytes,
        },
    )
    write_run_metadata(output_dir=output_dir, metadata=run_metadata)

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[build_reward_fn()],
        args=grpo_config,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    save_trainable_state(
        model,
        Path(output_dir) / "trainable_state.pt",
        metadata={
            "config_path": args.config,
            "mode": "grpo",
            "config_hash": run_metadata["config_hash"],
            "git_commit": run_metadata["git_commit"],
            "seed": seed,
        },
    )


if __name__ == "__main__":
    main()
