from __future__ import annotations

from pathlib import Path

from ltr13.metadata import build_run_metadata, compute_config_hash, write_run_metadata


def test_config_hash_is_stable_under_key_order() -> None:
    a = {"x": 1, "y": {"z": 2}}
    b = {"y": {"z": 2}, "x": 1}
    assert compute_config_hash(a) == compute_config_hash(b)


def test_write_run_metadata(tmp_path: Path) -> None:
    metadata = build_run_metadata(
        mode="eval",
        config_path="configs/eval.yaml",
        config={"mode": "eval", "model_name": "demo", "datasets": [{"path": "gsm8k"}]},
        seed=123,
        deterministic=False,
        output_dir=tmp_path,
    )
    path = write_run_metadata(output_dir=tmp_path, metadata=metadata)
    assert path.exists()
    assert "config_hash" in path.read_text(encoding="utf-8")
