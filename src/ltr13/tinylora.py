from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class TinyLoRALinear(nn.Module):
    """TinyLoRA adapter for a single nn.Linear module.

    Implements the paper's update rule:
        W' = W + U Σ (sum_i v_i P_i) V^T
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        rank: int,
        proj_dim: int,
        *,
        seed: int,
        shared_vector: Optional[nn.Parameter] = None,
        vector_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.float32,
        scale: float = 1.0,
        svd_method: str = "auto",
        svd_niter: int = 2,
        projection_mode: str = "random",
        projection_blocks: int = 1,
        shared_proj_block_scales: Optional[nn.Parameter] = None,
    ) -> None:
        super().__init__()

        if not isinstance(base_linear, nn.Linear):
            raise TypeError("TinyLoRALinear requires an nn.Linear base module")
        if rank <= 0:
            raise ValueError("rank must be positive")
        if proj_dim <= 0:
            raise ValueError("proj_dim must be positive")
        if svd_method not in {"auto", "full", "lowrank"}:
            raise ValueError(f"Unsupported svd_method: {svd_method}")
        if svd_niter <= 0:
            raise ValueError("svd_niter must be positive")
        if projection_mode not in {"random", "structured"}:
            raise ValueError(f"Unsupported projection_mode: {projection_mode}")
        if projection_blocks <= 0:
            raise ValueError("projection_blocks must be positive")
        if projection_mode == "structured" and projection_blocks > rank:
            raise ValueError("projection_blocks must be <= rank for structured projection")

        out_features, in_features = base_linear.weight.shape
        max_rank = min(out_features, in_features)
        if rank > max_rank:
            raise ValueError(f"rank={rank} exceeds max rank {max_rank} for shape {base_linear.weight.shape}")

        self.base_linear = base_linear
        for parameter in self.base_linear.parameters():
            parameter.requires_grad = False

        self.rank = rank
        self.proj_dim = proj_dim
        self.scale = float(scale)
        self.compute_dtype = compute_dtype
        self.svd_method = svd_method
        self.svd_niter = svd_niter
        self.projection_mode = projection_mode
        self.projection_blocks = projection_blocks
        target_device = self.base_linear.weight.device

        with torch.no_grad():
            # Compute decomposition on CPU to avoid large transient GPU memory spikes.
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

            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            projection = torch.randn(
                proj_dim,
                rank,
                rank,
                generator=generator,
                dtype=compute_dtype,
            ) / math.sqrt(rank)
            self.register_buffer(
                "projection",
                projection.to(device=target_device, dtype=compute_dtype),
                persistent=True,
            )
            if projection_mode == "structured":
                block = _build_projection_block_mask(
                    rank=rank,
                    num_blocks=projection_blocks,
                    device=target_device,
                )
                self.register_buffer("projection_block_mask", block, persistent=True)

        if shared_vector is None:
            self.v = nn.Parameter(
                torch.zeros(proj_dim, dtype=vector_dtype, device=target_device),
                requires_grad=True,
            )
        else:
            if shared_vector.shape != (proj_dim,):
                raise ValueError(
                    f"Shared vector shape {tuple(shared_vector.shape)} does not match {(proj_dim,)}"
                )
            if shared_vector.device != target_device:
                raise ValueError(
                    "Shared vector device mismatch: "
                    f"{shared_vector.device} vs base module device {target_device}"
                )
            self.v = shared_vector

        if projection_mode == "structured":
            block_count = projection_blocks * projection_blocks
            if shared_proj_block_scales is None:
                self.proj_block_scales = nn.Parameter(
                    torch.ones(block_count, dtype=vector_dtype, device=target_device),
                    requires_grad=True,
                )
            else:
                if shared_proj_block_scales.shape != (block_count,):
                    raise ValueError(
                        "Shared projection scale shape mismatch: "
                        f"{tuple(shared_proj_block_scales.shape)} vs {(block_count,)}"
                    )
                if shared_proj_block_scales.device != target_device:
                    raise ValueError(
                        "Shared projection scale device mismatch: "
                        f"{shared_proj_block_scales.device} vs {target_device}"
                    )
                self.proj_block_scales = shared_proj_block_scales
        else:
            self.register_parameter("proj_block_scales", None)

        self.register_buffer("_merged_delta", torch.empty(0), persistent=False)
        self._merged = False

    @classmethod
    def from_linear(
        cls,
        base_linear: nn.Linear,
        rank: int,
        proj_dim: int,
        *,
        seed: int,
        shared_vector: Optional[nn.Parameter] = None,
        vector_dtype: torch.dtype = torch.float32,
        compute_dtype: torch.dtype = torch.float32,
        scale: float = 1.0,
        svd_method: str = "auto",
        svd_niter: int = 2,
        projection_mode: str = "random",
        projection_blocks: int = 1,
        shared_proj_block_scales: Optional[nn.Parameter] = None,
    ) -> "TinyLoRALinear":
        return cls(
            base_linear=base_linear,
            rank=rank,
            proj_dim=proj_dim,
            seed=seed,
            shared_vector=shared_vector,
            vector_dtype=vector_dtype,
            compute_dtype=compute_dtype,
            scale=scale,
            svd_method=svd_method,
            svd_niter=svd_niter,
            projection_mode=projection_mode,
            projection_blocks=projection_blocks,
            shared_proj_block_scales=shared_proj_block_scales,
        )

    def delta_weight(self, *, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        projection = self.projection
        if self.projection_mode == "structured" and self.proj_block_scales is not None:
            scales = self.proj_block_scales.to(self.compute_dtype)
            scale_matrix = scales[self.projection_block_mask]
            projection = projection * scale_matrix.unsqueeze(0)
        r_matrix = torch.tensordot(self.v.to(self.compute_dtype), projection, dims=([0], [0]))
        scaled_r = self.svd_s.unsqueeze(1) * r_matrix
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

    def extra_repr(self) -> str:
        return (
            f"rank={self.rank}, proj_dim={self.proj_dim}, "
            f"compute_dtype={self.compute_dtype}, svd_method={self.svd_method}, "
            f"projection_mode={self.projection_mode}, merged={self._merged}"
        )


def _build_projection_block_mask(
    *,
    rank: int,
    num_blocks: int,
    device: torch.device,
) -> torch.Tensor:
    edges = torch.linspace(0, rank, steps=num_blocks + 1, device=device)
    block_index = torch.zeros(rank, dtype=torch.long, device=device)
    for idx in range(num_blocks):
        start = int(edges[idx].item())
        end = int(edges[idx + 1].item())
        block_index[start:end] = idx
    row = block_index.unsqueeze(1)
    col = block_index.unsqueeze(0)
    return (row * num_blocks + col).to(dtype=torch.long)


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


def iter_tinylora_modules(model: nn.Module):
    for name, module in model.named_modules():
        if isinstance(module, TinyLoRALinear):
            yield name, module


@torch.no_grad()
def merge_all_tinylora(model: nn.Module) -> None:
    for _, module in iter_tinylora_modules(model):
        module.merge()


@torch.no_grad()
def unmerge_all_tinylora(model: nn.Module) -> None:
    for _, module in iter_tinylora_modules(model):
        module.unmerge()
