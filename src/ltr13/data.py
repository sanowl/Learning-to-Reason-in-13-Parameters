from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset


@dataclass(frozen=True)
class DatasetConfig:
    source: str
    path: str
    name: str | None = None
    prompt_field: str = "question"
    answer_field: str = "answer"
    prompt_template: str = "{question}\n\nThink step by step and then give the final answer."
    split: str = "train"
    difficulty_field: str | None = None
    difficulty_value: str | int | None = None


def parse_dataset_config(raw: dict[str, Any], default_split: str) -> DatasetConfig:
    source = str(raw.get("source", "hf"))
    dataset_id = str(raw.get("path", "gsm8k"))
    dataset_name = raw.get("name")

    if dataset_id == "gsm8k":
        dataset_id = "gsm8k"
        dataset_name = dataset_name or "main"

    return DatasetConfig(
        source=source,
        path=dataset_id,
        name=str(dataset_name) if dataset_name is not None else None,
        prompt_field=str(raw.get("prompt_field", "question")),
        answer_field=str(raw.get("answer_field", "answer")),
        prompt_template=str(
            raw.get(
                "prompt_template",
                "{question}\n\nThink step by step and then give the final answer.",
            )
        ),
        split=str(raw.get("split", default_split)),
        difficulty_field=raw.get("difficulty_field"),
        difficulty_value=raw.get("difficulty_value"),
    )


def load_reasoning_dataset(config: DatasetConfig) -> Dataset:
    if config.source != "hf":
        raise ValueError(f"Unsupported dataset source: {config.source}")

    dataset = load_dataset(config.path, config.name, split=config.split)

    if config.difficulty_field is not None and config.difficulty_value is not None:
        dataset = dataset.filter(lambda row: row[config.difficulty_field] == config.difficulty_value)

    def _map_example(row: dict[str, Any]) -> dict[str, str]:
        question = str(row[config.prompt_field])
        answer = str(row[config.answer_field])
        prompt = config.prompt_template.format(question=question)
        return {
            "prompt": prompt,
            "reference_answer": answer,
            "question": question,
        }

    mapped = dataset.map(_map_example)
    keep_columns = ["prompt", "reference_answer", "question"]
    drop_columns = [column for column in mapped.column_names if column not in keep_columns]
    return mapped.remove_columns(drop_columns)


def build_sft_text(prompt: str, reference_answer: str) -> str:
    return f"{prompt}\n\n{reference_answer}"
