from __future__ import annotations

from typing import Any

import torch

from .adapter_ops import merged_adapters
from .inference_bridge import apply_truncated_is


def maybe_build_truncated_is_trainer(
    *,
    trainer_kwargs: dict[str, Any],
    enabled: bool,
    clip_max: Any,
):
    """Build a GRPO trainer with optional truncated-IS correction.

    When enabled, sampling is done from merged adapter weights and advantages are
    reweighted by clipped importance ratios between train/unmerged and
    infer/merged log-probabilities.
    """

    from trl import GRPOTrainer

    if not enabled:
        return GRPOTrainer(**trainer_kwargs)

    if clip_max is None:
        raise ValueError("truncated_importance_sampling.clip_max is required when enabled=true")
    clip_value = float(clip_max)
    if clip_value <= 0:
        raise ValueError("truncated_importance_sampling.clip_max must be positive")

    class TruncatedISGRPOTrainer(GRPOTrainer):
        def __init__(self, *, truncated_is_clip_max: float, **kwargs: Any) -> None:
            self.truncated_is_clip_max = truncated_is_clip_max
            super().__init__(**kwargs)

        def _generate_and_score_completions(self, inputs: dict[str, Any]) -> dict[str, Any]:
            with merged_adapters(self.model):
                batch = super()._generate_and_score_completions(inputs)

            required = {
                "prompt_ids",
                "prompt_mask",
                "completion_ids",
                "completion_mask",
                "advantages",
            }
            if not isinstance(batch, dict) or not required.issubset(batch):
                return batch

            with torch.no_grad():
                prompt_ids = batch["prompt_ids"]
                prompt_mask = batch["prompt_mask"]
                completion_ids = batch["completion_ids"]
                completion_mask = batch["completion_mask"]
                advantages = batch["advantages"]

                prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
                prompt_completion_mask = torch.cat([prompt_mask, completion_mask], dim=1)
                logits_to_keep = completion_ids.size(1)

                train_log_probs = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    prompt_completion_mask,
                    logits_to_keep,
                )
                with merged_adapters(self.model):
                    infer_log_probs = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        prompt_completion_mask,
                        logits_to_keep,
                    )

                mask = completion_mask.to(train_log_probs.dtype)
                train_seq_log_probs = (train_log_probs * mask).sum(dim=1)
                infer_seq_log_probs = (infer_log_probs * mask).sum(dim=1)

                batch["advantages"] = apply_truncated_is(
                    advantages=advantages,
                    train_log_probs=train_seq_log_probs,
                    infer_log_probs=infer_seq_log_probs,
                    clip_max=self.truncated_is_clip_max,
                )
            return batch

    return TruncatedISGRPOTrainer(truncated_is_clip_max=clip_value, **trainer_kwargs)
