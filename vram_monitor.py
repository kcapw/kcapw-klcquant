from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import torch

from .utils import cuda_snapshot


@dataclass
class VramEvent:
    label: str
    seconds: float
    snapshot: dict


@dataclass
class VramMonitor:
    events: list[VramEvent] = field(default_factory=list)
    _start: float = field(default_factory=perf_counter)

    def record(self, label: str) -> dict:
        snap = cuda_snapshot()
        self.events.append(VramEvent(label=label, seconds=perf_counter() - self._start, snapshot=snap))
        return snap

    def peak(self) -> dict:
        cuda_events = [e.snapshot for e in self.events if e.snapshot.get("cuda_available")]
        if not cuda_events:
            return {"cuda_available": False}
        return {
            "cuda_available": True,
            "allocated_gb": max(float(e.get("allocated_gb", 0.0)) for e in cuda_events),
            "reserved_gb": max(float(e.get("reserved_gb", 0.0)) for e in cuda_events),
            "free_gb_min": min(float(e.get("free_gb", 0.0)) for e in cuda_events),
            "total_gb": cuda_events[-1].get("total_gb"),
            "device": cuda_events[-1].get("device"),
        }

    def reset_cuda_peak(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def to_json(self) -> list[dict]:
        return [{"label": e.label, "seconds": round(e.seconds, 6), **e.snapshot} for e in self.events]
