from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_quantizer import QuantizedTensor
from .sparsity_tracker import SparsityStats, sparsity_stats


@dataclass
class PruningConfig:
    magnitude_fraction: float = 0.90
    min_keep_values: int = 1


@dataclass
class PrunedTensor:
    values: torch.Tensor
    indices: torch.Tensor
    shape: tuple[int, ...]
    dtype: str
    stats: SparsityStats


def magnitude_prune(tensor: torch.Tensor, config: PruningConfig | None = None) -> PrunedTensor:
    cfg = config or PruningConfig()
    data = tensor.detach()
    flat = data.reshape(-1)
    total = flat.numel()
    keep = max(cfg.min_keep_values, int(round(total * (1.0 - cfg.magnitude_fraction))))
    keep = min(keep, total)
    if keep <= 0:
        indices = torch.empty(0, dtype=torch.int64)
        values = torch.empty(0, dtype=data.dtype)
        mask = torch.zeros_like(flat, dtype=torch.bool)
    else:
        _, indices = torch.topk(flat.float().abs(), k=keep, largest=True, sorted=False)
        indices = indices.to(torch.int64).cpu()
        values = flat[indices.to(flat.device)].cpu()
        mask = torch.zeros_like(flat, dtype=torch.bool)
        mask[indices.to(mask.device)] = True
    return PrunedTensor(values=values, indices=indices, shape=tuple(data.shape), dtype=str(data.dtype), stats=sparsity_stats(mask))


def prune_for_storage(name: str, tensor: torch.Tensor, magnitude_fraction: float = 0.90) -> tuple[QuantizedTensor, SparsityStats]:
    pruned = magnitude_prune(tensor, PruningConfig(magnitude_fraction=magnitude_fraction))
    qt = QuantizedTensor(
        name=name,
        mode="pruned",  # type: ignore[arg-type]
        qvalues=pruned.values,
        scale=None,
        zero_point=pruned.indices,
        original_shape=pruned.shape,
        original_dtype=pruned.dtype,
        bits=None,
        packed=False,
    )
    return qt, pruned.stats


def reconstruct_pruned(values: torch.Tensor, indices: torch.Tensor, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    total = 1
    for dim in shape:
        total *= dim
    out = torch.zeros(total, device=device, dtype=dtype)
    if indices.numel():
        out[indices.to(device).long()] = values.to(device=device, dtype=dtype)
    return out.reshape(shape)
