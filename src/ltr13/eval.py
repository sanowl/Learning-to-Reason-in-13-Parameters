from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .checkpointing import load_trainable_state
from .config import load_yaml_config, parse_adapter_config
from .data import load_reasoning_dataset, parse_dataset_config
from .inject import apply_adapter
from .metadata import build_run_metadata, write_run_metadata
from .modeling import load_model_and_tokenizer
from .reward import exact_match_reward
from .results import build_result_rows, write_csv, write_json, write_jsonl
from .utils import configure_reproducibility
from .validation import validate_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on one or more reasoning datasets")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def _generate_batch(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    *,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> list[str]:
    original_padding_side = tokenizer.padding_side
    try:
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
    finally:
        tokenizer.padding_side = original_padding_side
    model_device = next(model.parameters()).device
    inputs = {key: value.to(model_device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    prompt_length = inputs["input_ids"].shape[1]
    generated = outputs[:, prompt_length:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)


def evaluate_single_dataset(
    model: Any,
    tokenizer: Any,
    dataset_cfg: dict[str, Any],
    *,
    batch_size: int,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    dataset = load_reasoning_dataset(parse_dataset_config(dataset_cfg, default_split="test"))

    predictions: list[str] = []
    references: list[str] = []

    for start in tqdm(range(0, len(dataset), batch_size), desc=f"eval:{dataset_cfg.get('alias', dataset_cfg['path'])}"):
        end = min(start + batch_size, len(dataset))
        batch = dataset.select(range(start, end))
        prompts = batch["prompt"]
        refs = batch["reference_answer"]
        preds = _generate_batch(
            model,
            tokenizer,
            prompts,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        predictions.extend(preds)
        references.extend(refs)

    rewards = exact_match_reward(predictions, references)
    accuracy = float(sum(rewards) / max(1, len(rewards)))
    return {
        "dataset": dataset_cfg.get("alias", dataset_cfg.get("path")),
        "num_examples": len(rewards),
        "exact_match": accuracy,
    }


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

    model, tokenizer = load_model_and_tokenizer(
        model_name=str(config["model_name"]),
        tokenizer_name=config.get("tokenizer_name"),
        torch_dtype=config.get("model_dtype"),
        model_kwargs=config.get("model_kwargs"),
    )

    adapter_cfg = parse_adapter_config(config)
    if adapter_cfg is not None:
        apply_adapter(model, adapter_cfg)

    checkpoint_path = config.get("trainable_state_path")
    if checkpoint_path:
        missing = load_trainable_state(model, checkpoint_path, strict=False)
        if missing:
            print(f"[Warning] Missing parameters from checkpoint: {len(missing)}")

    model.eval()

    generation_cfg = dict(config.get("generation", {}))
    batch_size = int(generation_cfg.get("batch_size", 4))
    max_prompt_length = int(generation_cfg.get("max_prompt_length", 1024))
    max_new_tokens = int(generation_cfg.get("max_new_tokens", 1024))
    temperature = float(generation_cfg.get("temperature", 0.0))
    top_p = float(generation_cfg.get("top_p", 1.0))

    datasets_cfg = config.get("datasets")
    if not isinstance(datasets_cfg, list) or not datasets_cfg:
        raise ValueError("`datasets` must be a non-empty list in eval config")

    results = [
        evaluate_single_dataset(
            model,
            tokenizer,
            dataset_cfg=dataset_cfg,
            batch_size=batch_size,
            max_prompt_length=max_prompt_length,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        for dataset_cfg in datasets_cfg
    ]

    mean_score = sum(row["exact_match"] for row in results) / len(results)
    output = {"results": results, "mean_exact_match": mean_score}

    output_path = config.get("output_path")
    if output_path:
        resolved_output_path = Path(output_path).resolve()
        metadata_dir = resolved_output_path.parent
    else:
        metadata_dir = Path(config.get("output_dir", "outputs/eval")).resolve()

    run_metadata = build_run_metadata(
        mode="eval",
        config_path=args.config,
        config=config,
        seed=seed,
        deterministic=deterministic,
        output_dir=metadata_dir,
        extra={"mean_exact_match": mean_score},
    )
    write_run_metadata(output_dir=metadata_dir, metadata=run_metadata)

    rows = build_result_rows(
        mode="eval",
        model_name=str(config["model_name"]),
        seed=seed,
        config_hash=run_metadata["config_hash"],
        git_commit=run_metadata.get("git_commit"),
        metrics=results,
    )
    output["schema_version"] = "1.0"
    output["rows"] = rows

    print(json.dumps(output, indent=2, sort_keys=True))

    if output_path:
        write_json(output_path, output)

        jsonl_path = config.get("output_jsonl_path")
        if jsonl_path is None:
            jsonl_path = str(Path(output_path).with_suffix(".jsonl"))
        write_jsonl(jsonl_path, rows)

        csv_path = config.get("output_csv_path")
        if csv_path is None:
            csv_path = str(Path(output_path).with_suffix(".csv"))
        write_csv(csv_path, rows)


if __name__ == "__main__":
    main()
