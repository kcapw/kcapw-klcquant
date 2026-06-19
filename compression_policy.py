from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .adaptive_quantizer import QuantMode


PROTECTED_TOKENS = ("embed_tokens", "lm_head", "router", "norm", "self_attn")


@dataclass(frozen=True)
class TensorPlan:
    name: str
    mode: QuantMode
    importance_score: float
    rank: int
    percentile: float
    original_bytes: int


@dataclass
class StreamCompressionPolicy:
    """Rank-aware experimental compression policy.

    Scores from static scans are deliberately conservative. This policy uses rank
    percentiles so the bottom tensors are actually exercised by q4/q3 in POC runs.
    """

    keep_top_percent: float = 0.25
    q16_until_percent: float = 0.45
    q8_until_percent: float = 0.75
    q4_until_percent: float = 0.92
    q2_until_percent: float = 0.985
    prune_below_importance: float = 0.02
    protect_core_tensors: bool = True

    def make_plan(self, rankings: Iterable[dict]) -> dict[str, TensorPlan]:
        rows = sorted(rankings, key=lambda x: float(x.get("importance_score", 0.0)), reverse=True)
        total = max(1, len(rows) - 1)
        plan: dict[str, TensorPlan] = {}
        for rank, row in enumerate(rows):
            name = row["name"]
            percentile = rank / total
            mode = self.choose_with_importance(name, percentile, float(row.get("importance_score", 0.0)))
            plan[name] = TensorPlan(
                name=name,
                mode=mode,
                importance_score=float(row.get("importance_score", 0.0)),
                rank=rank,
                percentile=percentile,
                original_bytes=int(row.get("nbytes", 0)),
            )
        return plan

    def choose(self, name: str, percentile: float) -> QuantMode:
        if self.protect_core_tensors and any(token in name for token in PROTECTED_TOKENS):
            return "fp16"
        if percentile <= self.keep_top_percent:
            return "fp16"
        if percentile <= self.q16_until_percent:
            return "q16"
        if percentile <= self.q8_until_percent:
            return "q8"
        if percentile <= self.q4_until_percent:
            return "q4"
        if percentile <= self.q2_until_percent:
            return "q2"
        return "q1"

    def choose_with_importance(self, name: str, percentile: float, importance: float) -> QuantMode:
        if self.protect_core_tensors and any(token in name for token in PROTECTED_TOKENS):
            return "fp16"
        if importance <= self.prune_below_importance:
            return "pruned"
        return self.choose(name, percentile)
