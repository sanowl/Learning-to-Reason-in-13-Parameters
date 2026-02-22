from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from ltr13.config import TinyLoRAConfig
from ltr13.inject import apply_adapter


class DistToyModel(nn.Module):
    def __init__(self, width: int = 8) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "q_proj": nn.Linear(width, width),
                        "k_proj": nn.Linear(width, width),
                    }
                )
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers[0]["q_proj"](x)
        return self.layers[0]["k_proj"](x)


def _worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(0)

    model = DistToyModel(width=8)
    apply_adapter(
        model,
        TinyLoRAConfig(
            adapter_type="tinylora",
            rank=2,
            proj_dim=1,
            tie_mode="full",
            tie_factor=1,
            target_modules=("q_proj", "k_proj"),
            vector_dtype="float32",
            compute_dtype="float32",
        ),
    )

    ddp = DDP(model)
    trainable = [parameter for parameter in ddp.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-2)

    # SFT-like step
    optimizer.zero_grad()
    x = torch.randn(4, 8)
    target = torch.randn(4, 8)
    out = ddp(x)
    sft_loss = F.mse_loss(out, target)
    sft_loss.backward()
    optimizer.step()

    # RL-like policy-gradient step
    optimizer.zero_grad()
    logits = ddp(x)
    log_probs = F.log_softmax(logits, dim=-1)
    actions = torch.argmax(log_probs.detach(), dim=-1)
    chosen_log_probs = torch.gather(log_probs, dim=1, index=actions.unsqueeze(1)).squeeze(1)
    rewards = torch.ones_like(chosen_log_probs)
    rl_loss = -(chosen_log_probs * rewards).mean()
    rl_loss.backward()
    optimizer.step()

    flat = torch.cat([parameter.detach().flatten() for parameter in trainable])
    reduced = flat.clone()
    dist.all_reduce(reduced)
    reduced /= world_size

    if not torch.allclose(flat, reduced, atol=1e-5):
        raise RuntimeError("DDP trainable parameters diverged across ranks")

    if not torch.isfinite(flat).all():
        raise RuntimeError("Non-finite trainable parameters after distributed smoke steps")

    dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get("LTR13_RUN_DISTRIBUTED_SMOKE", "0") != "1",
    reason="Set LTR13_RUN_DISTRIBUTED_SMOKE=1 to run distributed smoke test",
)
def test_distributed_training_smoke() -> None:
    if not dist.is_available():
        pytest.skip("torch.distributed is not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        init_file = Path(tmp_dir) / "dist_init"
        mp.spawn(_worker, args=(2, str(init_file)), nprocs=2, join=True)
