from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Classic LoRA adapter: W' = W + A B."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        *,
        seed: int,
        shared_a: Optional[nn.Parameter] = None,
        shared_b: Optional[nn.Parameter] = None,
        vector_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.float32,
        scale: float = 1.0,
        lora_alpha: float = 1.0,
        lora_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRALinear requires an nn.Linear base module")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive")
        if not (0.0 <= lora_dropout < 1.0):
            raise ValueError("lora_dropout must be in [0, 1)")

        out_features, in_features = base_linear.weight.shape
        max_rank = min(out_features, in_features)
        if rank > max_rank:
            raise ValueError(f"rank={rank} exceeds max rank {max_rank} for shape {base_linear.weight.shape}")

        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.scale = float(scale)
        self.compute_dtype = compute_dtype
        self.lora_alpha = float(lora_alpha)
        self.dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()

        target_device = self.base_linear.weight.device

        if shared_a is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                a_init = torch.randn(
                    out_features,
                    rank,
                    dtype=vector_dtype,
                    device=target_device,
                ) / math.sqrt(rank)
            self.a = nn.Parameter(a_init, requires_grad=True)
        else:
            if shared_a.shape != (out_features, rank):
                raise ValueError(
                    f"Shared A shape {tuple(shared_a.shape)} does not match {(out_features, rank)}"
                )
            self.a = shared_a

        if shared_b is None:
            self.b = nn.Parameter(
                torch.zeros(rank, in_features, dtype=vector_dtype, device=target_device),
                requires_grad=True,
            )
        else:
            if shared_b.shape != (rank, in_features):
                raise ValueError(
                    f"Shared B shape {tuple(shared_b.shape)} does not match {(rank, in_features)}"
                )
            self.b = shared_b

        self.register_buffer("_merged_delta", torch.empty(0), persistent=False)
        self._merged = False

    @property
    def effective_scale(self) -> float:
        return self.scale * (self.lora_alpha / self.rank)

    def delta_weight(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        delta = self.a.to(self.compute_dtype) @ self.b.to(self.compute_dtype)
        if dtype is not None:
            delta = delta.to(dtype)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return self.base_linear(x)

        base = self.base_linear(x)
        dropped = self.dropout(x)
        delta = self.delta_weight(dtype=self.base_linear.weight.dtype)
        update = F.linear(dropped, delta.to(dropped.dtype), bias=None)
        return base + (self.effective_scale * update)

    @torch.no_grad()
    def merge(self) -> None:
        if self._merged:
            return
        delta = self.effective_scale * self.delta_weight(dtype=self.base_linear.weight.dtype)
        self.base_linear.weight.add_(delta)
        self._merged_delta = delta
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        if self._merged_delta.numel() == 0:
            raise RuntimeError("No merged delta stored; cannot unmerge safely")
        self.base_linear.weight.sub_(self._merged_delta)
        self._merged_delta = torch.empty(0, device=self.base_linear.weight.device)
        self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged


class LoRAXSLinear(nn.Module):
    """LoRA-XS adapter: W' = W + U Σ R V^T where R is trainable."""

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        *,
        seed: int,
        shared_r: Optional[nn.Parameter] = None,
        vector_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.float32,
        scale: float = 1.0,
        svd_method: str = "auto",
        svd_niter: int = 2,
    ) -> None:
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("LoRAXSLinear requires an nn.Linear base module")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if svd_method not in {"auto", "full", "lowrank"}:
            raise ValueError(f"Unsupported svd_method: {svd_method}")
        if svd_niter <= 0:
            raise ValueError("svd_niter must be positive")

        out_features, in_features = base_linear.weight.shape
        max_rank = min(out_features, in_features)
        if rank > max_rank:
            raise ValueError(f"rank={rank} exceeds max rank {max_rank} for shape {base_linear.weight.shape}")

        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.scale = float(scale)
        self.compute_dtype = compute_dtype
        self.svd_method = svd_method
        self.svd_niter = svd_niter
        target_device = self.base_linear.weight.device

        with torch.no_grad():
            weight = self.base_linear.weight.detach().to(device="cpu", dtype=torch.float32)
            u, s, vh = _compute_truncated_svd(
                weight=weight,
                rank=rank,
                seed=seed,
                method=svd_method,
                niter=svd_niter,
            )
            self.register_buffer(
                "svd_u",
                u[:, :rank].to(device=target_device, dtype=compute_dtype),
                persistent=True,
            )
            self.register_buffer(
                "svd_s",
                s[:rank].to(device=target_device, dtype=compute_dtype),
                persistent=True,
            )
            self.register_buffer(
                "svd_vh",
                vh[:rank, :].to(device=target_device, dtype=compute_dtype),
                persistent=True,
            )

        if shared_r is None:
            self.r = nn.Parameter(
                torch.zeros(rank, rank, dtype=vector_dtype, device=target_device),
                requires_grad=True,
            )
        else:
            if shared_r.shape != (rank, rank):
                raise ValueError(
                    f"Shared R shape {tuple(shared_r.shape)} does not match {(rank, rank)}"
                )
            if shared_r.device != target_device:
                raise ValueError(
                    "Shared R device mismatch: "
                    f"{shared_r.device} vs base module device {target_device}"
                )
            self.r = shared_r

        self.register_buffer("_merged_delta", torch.empty(0), persistent=False)
        self._merged = False

    def delta_weight(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        scaled_r = self.svd_s.unsqueeze(1) * self.r.to(self.compute_dtype)
        delta = self.svd_u @ scaled_r @ self.svd_vh
        if dtype is not None:
            delta = delta.to(dtype)
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._merged:
            return self.base_linear(x)

        base = self.base_linear(x)
        delta = self.delta_weight(dtype=self.base_linear.weight.dtype)
        update = F.linear(x, delta.to(x.dtype), bias=None)
        return base + (self.scale * update)

    @torch.no_grad()
    def merge(self) -> None:
        if self._merged:
            return
        delta = self.scale * self.delta_weight(dtype=self.base_linear.weight.dtype)
        self.base_linear.weight.add_(delta)
        self._merged_delta = delta
        self._merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self._merged:
            return
        if self._merged_delta.numel() == 0:
            raise RuntimeError("No merged delta stored; cannot unmerge safely")
        self.base_linear.weight.sub_(self._merged_delta)
        self._merged_delta = torch.empty(0, device=self.base_linear.weight.device)
        self._merged = False

    @property
    def merged(self) -> bool:
        return self._merged


def _compute_truncated_svd(
    *,
    weight: torch.Tensor,
    rank: int,
    seed: int,
    method: str,
    niter: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    min_dim = min(weight.shape)
    if method == "auto":
        chosen = "lowrank" if rank < min_dim else "full"
    else:
        chosen = method
    if chosen == "lowrank" and rank >= min_dim:
        chosen = "full"

    if chosen == "full":
        u, s, vh = torch.linalg.svd(weight, full_matrices=False)
        return u[:, :rank], s[:rank], vh[:rank, :]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        u, s, v = torch.svd_lowrank(weight, q=rank, niter=niter)
    vh = v.transpose(0, 1)
    return u[:, :rank], s[:rank], vh[:rank, :]
