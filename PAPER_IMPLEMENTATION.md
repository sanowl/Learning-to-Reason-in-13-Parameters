# Paper Implementation Map

This file maps repository components to the paper **Learning to Reason in 13 Parameters**.

## Core method (Section 4)

- TinyLoRA update rule is implemented in `src/ltr13/tinylora.py`.
- Exact structure:
  - frozen SVD factors `(U, Σ, V)` from each target linear weight.
  - fixed random projection tensor `P \in R^{u x r x r}`.
  - trainable vector `v \in R^u`.
  - update: `W' = W + U Σ (sum_i v_i P_i) V^T`.
- Weight tying with `tie_mode` + `tie_factor` is implemented in `src/ltr13/inject.py`.
  - `none`, `structured`, `tiled`, `full`.

## Training algorithms (Section 5)

- GRPO entrypoint: `src/ltr13/train_grpo.py`
  - exact-match reward for verifiable tasks.
  - paper-style defaults in `configs/grpo_gsm8k.yaml` and `configs/grpo_math_hard.yaml`.
- SFT baseline entrypoint: `src/ltr13/train_sft.py`
  - equivalent TinyLoRA injection path for fair capacity comparisons.

## Inference/merge strategy (Section 5.1, vLLM workaround idea)

- Merge/unmerge utilities in `src/ltr13/tinylora.py` (`merge_all_tinylora`, `unmerge_all_tinylora`).
- Inference bridge helpers in `src/ltr13/inference_bridge.py`:
  - merged inference context manager.
  - truncated importance weighting helpers.
- Export merged checkpoint: `src/ltr13/export_merged.py` and `scripts/export_merged.py`.

## Evaluation suite (Section 5 + 6)

- Multi-dataset evaluation runner: `src/ltr13/eval.py`.
- Example benchmark config: `configs/eval_paper_benchmarks.yaml`.
- GSM8K and math answer extraction/reward utilities: `src/ltr13/reward.py`.

## Sweeps and ablations (Section 6 + 7)

- Grid sweep runner: `scripts/run_sweep.py`.
- Sweep configs:
  - `configs/sweeps/paper_repro.yaml`
  - `configs/sweeps/sft_capacity.yaml`
- Parameter-count equations from Table 1 are encoded in `src/ltr13/analysis.py`.

## Tests

- Core formula and merge/unmerge correctness: `tests/test_tinylora_core.py`.
- Tie behavior and trainable parameter counts: `tests/test_inject_tie_modes.py`.
- Reward correctness: `tests/test_reward.py`.
- Parameter formulas: `tests/test_analysis.py`.
- Checkpoint save/load + merge/unmerge compatibility: `tests/test_checkpoint_roundtrip.py`.

## Repro/ops hardening

- Config validation: `src/ltr13/validation.py`.
- Adapter guardrails: `src/ltr13/guardrails.py`.
- Run metadata logging (git/config hash/packages/GPU): `src/ltr13/metadata.py`.
- Result schema writers (`jsonl`/`csv`): `src/ltr13/results.py`.

## Repro reality check

This repository is algorithm-faithful, but published numbers still require:
- the exact model checkpoints used in the paper,
- equivalent compute budget,
- exact dataset mirrors/splits used by the authors,
- and long-running RL training.
