from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .adaptive_quantizer import QuantMode
from .sensitivity_database import SensitivityRecord
from .tensor_criticality_ranker import rank_tensor_criticality, tensor_role


MODE_ORDER = ["fp16", "q8", "q4", "q3", "q2", "q1", "pruned"]


@dataclass(frozen=True)
class PrecisionDecision:
    tensor: str
    recommended_mode: QuantMode
    tier: str
    reason: str
    criticality_score: float
    hardest_stable_mode: str | None
    role: str

    def to_json(self) -> dict:
        return {
            "tensor": self.tensor,
            "recommended_mode": self.recommended_mode,
            "tier": self.tier,
            "reason": self.reason,
            "criticality_score": self.criticality_score,
            "hardest_stable_mode": self.hardest_stable_mode,
            "role": self.role,
        }


class RuntimeCalibratedPrecisionPolicy:
    """Recommend precision tiers from measured autoregressive perturbation sensitivity."""

    def __init__(self, min_safe_sequence_overlap: float = 0.50, max_abs_perplexity_drift: float = 25.0) -> None:
        self.min_safe_sequence_overlap = min_safe_sequence_overlap
        self.max_abs_perplexity_drift = max_abs_perplexity_drift

    def recommend(self, records: Iterable[SensitivityRecord]) -> list[PrecisionDecision]:
        rows = list(records)
        ranking = {item["tensor"]: item for item in rank_tensor_criticality(rows)}
        grouped: dict[str, list[SensitivityRecord]] = {}
        for record in rows:
            grouped.setdefault(record.tensor, []).append(record)

        decisions: list[PrecisionDecision] = []
        for tensor, tensor_records in grouped.items():
            rank_row = ranking[tensor]
            decision = self._decide_tensor(tensor, tensor_records, rank_row)
            decisions.append(decision)
        return sorted(decisions, key=lambda item: item.criticality_score, reverse=True)

    def _decide_tensor(self, tensor: str, rows: list[SensitivityRecord], rank_row: dict) -> PrecisionDecision:
        by_mode: dict[str, list[SensitivityRecord]] = {}
        for row in rows:
            by_mode.setdefault(row.mode, []).append(row)
        unsafe_modes = {mode for mode, items in by_mode.items() if any(not self._is_safe(item) for item in items)}
        safe_modes = [mode for mode, items in by_mode.items() if all(self._is_safe(item) for item in items)]
        score = float(rank_row["criticality_score"])
        role = tensor_role(tensor)

        if "q8" in unsafe_modes or score >= 1.10:
            return PrecisionDecision(tensor, "fp16", "critical", "q8/runtime sensitivity is unstable", score, rank_row["hardest_stable_mode"], role)
        if "q4" in unsafe_modes or score >= 0.80:
            return PrecisionDecision(tensor, "q8", "high", "q4 perturbation is unstable or criticality is high", score, rank_row["hardest_stable_mode"], role)
        if "q2" in unsafe_modes or "q1" in unsafe_modes or "pruned" in unsafe_modes or score >= 0.45:
            return PrecisionDecision(tensor, "q4", "moderate", "ultra-low precision is unstable but q4 is within boundary", score, rank_row["hardest_stable_mode"], role)

        hardest = self._hardest_mode(safe_modes)
        if hardest == "pruned":
            return PrecisionDecision(tensor, "pruned", "near_zero", "pruning stayed stable in calibration", score, hardest, role)
        if hardest in {"q1", "q2"}:
            return PrecisionDecision(tensor, hardest, "cold_stable", f"{hardest} stayed stable in calibration", score, hardest, role)
        return PrecisionDecision(tensor, "q4", "moderate", "insufficient ultra-low evidence; defaulting to q4", score, hardest, role)

    def _is_safe(self, record: SensitivityRecord) -> bool:
        metrics = record.metrics
        return (
            record.stable
            and float(metrics.get("sequence_overlap", 1.0)) >= self.min_safe_sequence_overlap
            and abs(float(metrics.get("perplexity_drift", 0.0))) <= self.max_abs_perplexity_drift
        )

    def _hardest_mode(self, modes: list[str]) -> str | None:
        if not modes:
            return None
        return max(modes, key=lambda mode: MODE_ORDER.index(mode) if mode in MODE_ORDER else -1)

