from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SparsityStats:
    total_values: int
    kept_values: int
    pruned_values: int
    sparsity: float
    density: float

    def to_json(self) -> dict:
        return {
            "total_values": self.total_values,
            "kept_values": self.kept_values,
            "pruned_values": self.pruned_values,
            "sparsity": self.sparsity,
            "density": self.density,
        }


def sparsity_stats(mask: torch.Tensor) -> SparsityStats:
    total = int(mask.numel())
    kept = int(mask.to(torch.bool).sum().item())
    pruned = total - kept
    density = kept / max(total, 1)
    return SparsityStats(total, kept, pruned, pruned / max(total, 1), density)
