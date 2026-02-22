# Learning to Reason in 13 Parameters

Implementation-focused reproduction starter for:

**Learning to Reason in 13 Parameters**
John X. Morris, Niloofar Mireshghallah, Mark Ibrahim, Saeed Mahloujifar (2026)
`arXiv:2602.04118`

This repository implements the paper's TinyLoRA method and provides runnable scaffolding for:
- RL (GRPO) and SFT training
- Tiny update-size sweeps
- Multi-benchmark evaluation
- Merged checkpoint export for inference

## Scope and goals

This repo is built to match the paper's algorithmic ideas and experiment structure, while remaining easy to modify.

What is included:
- Paper-faithful TinyLoRA module and update equation
- Weight tying modes for extreme parameter compression
- GRPO and SFT training entrypoints with YAML configs
- Reward and answer extraction utilities (math-style EM)
- Evaluation runner for multiple benchmark datasets
- Sweep runner for ablations and parameter/LR grids
- Config validation before runs start
- Adapter guardrails (expected trainable params/bytes)
- Automatic run metadata logging (config hash, git commit, package/GPU info)
- Deterministic mode toggle for reproducibility
- Machine-readable benchmark outputs (`json`, `jsonl`, `csv`)
- Unit tests for correctness and parameter accounting

What is not guaranteed out of the box:
- Exact leaderboard numbers from the paper without equivalent compute, model versions, and dataset mirrors

## TinyLoRA method

For a frozen linear layer `W \in R^{d x k}` with truncated SVD factors `U, Σ, V`, TinyLoRA applies:

`W' = W + U Σ (sum_i v_i P_i) V^T`

Where:
- `v \in R^u` is trainable
- `P \in R^{u x r x r}` is fixed random projection tensor (frozen)
- `U, Σ, V` are frozen SVD factors of the base weight

The implementation is in `src/ltr13/tinylora.py`.

## Repository layout

- `src/ltr13/tinylora.py`: TinyLoRA module, merge/unmerge
- `src/ltr13/inject.py`: inject TinyLoRA into transformer modules, tie strategies
- `src/ltr13/train_grpo.py`: GRPO training entrypoint
- `src/ltr13/train_sft.py`: SFT training entrypoint
- `src/ltr13/eval.py`: multi-dataset evaluation
- `src/ltr13/reward.py`: exact-match reward and answer extraction
- `src/ltr13/data.py`: dataset loading and prompt formatting
- `src/ltr13/export_merged.py`: merged checkpoint export
- `scripts/run_experiment.py`: mode-aware launcher (`grpo`, `sft`, `eval`)
- `scripts/run_sweep.py`: grid sweep runner
- `scripts/export_merged.py`: convenience wrapper for merged export
- `configs/`: example configs for training, evaluation, and sweeps
- `tests/`: unit tests
- `PAPER_IMPLEMENTATION.md`: section-by-section paper mapping

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Use locked dependencies for reproducible environments:

```bash
pip install -r requirements-dev.lock
```

Run tests:

```bash
pytest -q
```

## Quick start

### 1. GRPO training (GSM8K config)

```bash
python scripts/run_experiment.py --config configs/grpo_gsm8k.yaml
```

### 2. SFT baseline

```bash
python scripts/run_experiment.py --config configs/sft_gsm8k.yaml
```

### 3. Multi-benchmark evaluation

```bash
python scripts/run_experiment.py --config configs/eval_paper_benchmarks.yaml
```

### 4. Export merged model for inference

```bash
python scripts/export_merged.py \
  --config configs/export_merged.yaml \
  --output-dir outputs/merged_model
```

### 5. Run sweep (paper-style ablation scaffold)

```bash
python scripts/run_sweep.py --config configs/sweeps/paper_repro.yaml
```

Dry run sweeps without executing:

```bash
python scripts/run_sweep.py --config configs/sweeps/paper_repro.yaml --dry-run
```

## Config system

Experiments are YAML-driven. Main sections:
- `mode`: `grpo`, `sft`, or `eval`
- `model_name`, `model_dtype`
- `train_dataset` / `eval_dataset` / `datasets`
- `tinylora`
- `training`
- `generation` (for eval)
- `guardrails`
- `deterministic` and `deterministic_warn_only`

### TinyLoRA config fields

- `rank`: frozen SVD rank `r` (paper ablations include `r=1` and `r=2`)
- `proj_dim`: trainable vector size `u`
- `tie_mode`: `none`, `structured`, `tiled`, `full`
- `tie_factor`: number of modules sharing one vector (for `structured`/`tiled`)
- `target_modules`: module suffixes to adapt
- `vector_dtype`: storage dtype for trainable `v`
- `compute_dtype`: dtype for adapter math (`U, Σ, V, P` paths)

### Guardrails and reproducibility fields

- `guardrails.expected_trainable_params`: exact expected trainable parameter count
- `guardrails.expected_trainable_bytes`: exact expected trainable byte count
- `guardrails.tolerance`: absolute tolerance for expected counts
- `guardrails.require_trainable_params`: fail if no trainable params are found
- `deterministic`: enable deterministic torch kernels where supported
- `deterministic_warn_only`: warn instead of error when deterministic ops are unavailable

## Weight tying modes

Implemented in `src/ltr13/inject.py`:
- `none`: each adapted module has its own trainable `v`
- `structured`: same module type is grouped locally (for example, `q_proj` with nearby `q_proj`)
- `tiled`: nearby modules grouped regardless of module type
- `full`: all adapted modules share one `v` (can reach a single trainable parameter if `proj_dim=1`)

## Training and evaluation notes

- GRPO reward uses exact match from generated answer vs reference answer.
- Math-style answer extraction supports `####`, `\\boxed{}`, integers, decimals, and simple fractions.
- Evaluation uses generation + exact match scoring for each configured dataset.
- `run_metadata.json` is written to output directories for `grpo`, `sft`, and `eval`.
- Eval writes schema-stable rows for machine comparison:
  - `output_path` (JSON summary)
  - `output_jsonl_path` (one row per dataset)
  - `output_csv_path` (tabular)

## Reproducibility checklist

For closer paper parity, keep these aligned with the paper:
- Model family/checkpoint version
- Dataset version/splits and prompt templates
- RL hyperparameters (num generations, length caps, KL beta, batch schedule)
- Learning-rate sweeps per update size
- Random seeds and number of runs per point

## Known practical constraints

- Some benchmark dataset IDs in `configs/eval_paper_benchmarks.yaml` may differ by Hugging Face mirror; adjust `path`/`name` as needed.
- Full-scale paper runs require substantial multi-GPU compute.
- Results can shift significantly with tokenizer/model revisions.

## Development

Run tests:

```bash
pytest -q
```

Optional compile check:

```bash
python -m compileall src scripts tests
```

Install and use pre-commit hooks:

```bash
pre-commit install
pre-commit run --all-files
```

## Additional docs

- Paper implementation map: `PAPER_IMPLEMENTATION.md`

## Citation

If you use this code, cite the original paper:

```bibtex
@article{morris2026learning,
  title={Learning to Reason in 13 Parameters},
  author={Morris, John X and Mireshghallah, Niloofar and Ibrahim, Mark and Mahloujifar, Saeed},
  journal={arXiv preprint arXiv:2602.04118},
  year={2026}
}
```
