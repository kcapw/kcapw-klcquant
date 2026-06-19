from __future__ import annotations

from dataclasses import dataclass

from .streamed_loader import SafetensorIndex


@dataclass
class SchedulerStats:
    layer_loads: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    prefetches: int = 0

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total else 0.0


class TensorScheduler:
    def __init__(self, index: SafetensorIndex, max_layers: int) -> None:
        self.index = index
        self.max_layers = max_layers
        self.stats = SchedulerStats()

    def layer_names(self) -> list[str]:
        names = [f"layer_{idx:03d}" for idx in range(self.max_layers)]
        available = self.index.groups()
        return [name for name in names if name in available]

    def next_layer(self, current_idx: int) -> str | None:
        nxt = current_idx + 1
        if nxt >= self.max_layers:
            return None
        name = f"layer_{nxt:03d}"
        return name if name in self.index.groups() else None

    def record_load(self) -> None:
        self.stats.layer_loads += 1
        self.stats.cache_misses += 1

    def record_hit(self) -> None:
        self.stats.cache_hits += 1

    def record_prefetch(self) -> None:
        self.stats.prefetches += 1
