from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from datasets import Dataset
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from .checkpointing import save_trainable_state
from .config import load_yaml_config, parse_tinylora_config
from .data import build_sft_text, load_reasoning_dataset, parse_dataset_config
from .guardrails import compute_trainable_stats, enforce_trainable_guardrails
from .inject import apply_tinylora
from .metadata import build_run_metadata, write_run_metadata
from .modeling import load_model_and_tokenizer
from .utils import configure_reproducibility
from .validation import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SFT baseline with optional TinyLoRA")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def _prepare_sft_dataset(dataset: Dataset, tokenizer: Any, max_length: int) -> Dataset:
    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        texts = [
            build_sft_text(prompt=prompt, reference_answer=answer)
            for prompt, answer in zip(batch["prompt"], batch["reference_answer"])
        ]
        return tokenizer(texts, truncation=True, max_length=max_length)

    tokenized = dataset.map(_tokenize, batched=True)
    drop_columns = [
        column
        for column in tokenized.column_names
        if column not in {"input_ids", "attention_mask", "token_type_ids"}
    ]
    return tokenized.remove_columns(drop_columns)


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
    eval_dataset = None
    if "eval_dataset" in config and config["eval_dataset"] is not None:
        eval_dataset = load_reasoning_dataset(
            parse_dataset_config(config["eval_dataset"], default_split="test")
        )

    max_length = int(config.get("max_length", 4096))
    train_dataset = _prepare_sft_dataset(train_dataset, tokenizer, max_length)
    if eval_dataset is not None:
        eval_dataset = _prepare_sft_dataset(eval_dataset, tokenizer, max_length)

    output_dir = str(config.get("output_dir", "outputs/sft"))
    train_cfg: dict[str, Any] = dict(config.get("training", {}))
    gradient_checkpointing = bool(train_cfg.get("gradient_checkpointing", True))
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 5e-5)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        evaluation_strategy=str(
            train_cfg.get("evaluation_strategy", train_cfg.get("eval_strategy", "no"))
        ),
        bf16=bool(train_cfg.get("bf16", False)),
        fp16=bool(train_cfg.get("fp16", False)),
        gradient_checkpointing=gradient_checkpointing,
        report_to=list(train_cfg.get("report_to", [])),
        remove_unused_columns=False,
    )

    if gradient_checkpointing:
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
        if tinylora_cfg is not None and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    stats = compute_trainable_stats(model)
    enforce_trainable_guardrails(stats=stats, guardrails=config.get("guardrails"))
    print(f"[Trainable] parameters={stats.parameters}, bytes={stats.bytes}")

    run_metadata = build_run_metadata(
        mode="sft",
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

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    save_trainable_state(
        model,
        Path(output_dir) / "trainable_state.pt",
        metadata={
            "config_path": args.config,
            "mode": "sft",
            "config_hash": run_metadata["config_hash"],
            "git_commit": run_metadata["git_commit"],
            "seed": seed,
        },
    )


if __name__ == "__main__":
    main()
