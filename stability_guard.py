from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StabilityThresholds:
    max_abs_perplexity_drift: float = 25.0
    min_sequence_overlap: float = 0.50
    min_token_jaccard: float = 0.50


@dataclass
class StabilityGuard:
    thresholds: StabilityThresholds = field(default_factory=StabilityThresholds)

    def check(self, comparison: dict) -> dict:
        reasons: list[str] = []
        ppl = abs(float(comparison.get("perplexity_drift", 0.0)))
        if ppl > self.thresholds.max_abs_perplexity_drift:
            reasons.append(f"perplexity_drift>{self.thresholds.max_abs_perplexity_drift}")
        if float(comparison.get("sequence_overlap", 1.0)) < self.thresholds.min_sequence_overlap:
            reasons.append(f"sequence_overlap<{self.thresholds.min_sequence_overlap}")
        if float(comparison.get("token_jaccard", 1.0)) < self.thresholds.min_token_jaccard:
            reasons.append(f"token_jaccard<{self.thresholds.min_token_jaccard}")
        return {"stable": not reasons, "reasons": reasons}
