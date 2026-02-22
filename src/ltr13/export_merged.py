from __future__ import annotations

import argparse

from .adapter_ops import merge_all_adapters
from .checkpointing import load_trainable_state
from .config import load_yaml_config, parse_adapter_config
from .inject import apply_adapter
from .modeling import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a merged adapter checkpoint for inference")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", required=True, help="Output folder for merged checkpoint")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)

    model, tokenizer = load_model_and_tokenizer(
        model_name=str(config["model_name"]),
        tokenizer_name=config.get("tokenizer_name"),
        torch_dtype=config.get("model_dtype"),
        model_kwargs=config.get("model_kwargs"),
    )

    adapter_cfg = parse_adapter_config(config)
    if adapter_cfg is None:
        raise ValueError("adapter config is required to export merged checkpoint")

    apply_adapter(model, adapter_cfg)

    checkpoint_path = config.get("trainable_state_path")
    if not checkpoint_path:
        raise ValueError("trainable_state_path must be set in config")

    load_trainable_state(model, checkpoint_path, strict=True)
    merge_all_adapters(model)

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
