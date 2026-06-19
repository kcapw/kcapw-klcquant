from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class InstabilityAccumulator:
    """Tracks drift growth across progressive compression attempts."""

    min_sequence_overlap: float = 0.50
    max_abs_perplexity_drift: float = 25.0
    records: list[dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, metrics: dict[str, Any], stable: bool) -> dict[str, Any]:
        step = {
            "label": label,
            "stable": stable,
            "sequence_overlap": float(metrics.get("sequence_overlap", 1.0)),
            "token_jaccard": float(metrics.get("token_jaccard", 1.0)),
            "perplexity_drift": float(metrics.get("perplexity_drift", 0.0)),
            "entropy_drift": float(metrics.get("entropy_drift", 0.0)),
            "sparse_logit_cosine": float(metrics.get("sparse_logit_cosine", 1.0)),
            "first_divergence_token": metrics.get("first_divergence_token"),
        }
        previous = self.records[-1] if self.records else None
        if previous is None:
            step["delta_abs_perplexity_drift"] = abs(step["perplexity_drift"])
            step["delta_sequence_loss"] = 1.0 - step["sequence_overlap"]
        else:
            step["delta_abs_perplexity_drift"] = abs(step["perplexity_drift"]) - abs(float(previous["perplexity_drift"]))
            step["delta_sequence_loss"] = (1.0 - step["sequence_overlap"]) - (1.0 - float(previous["sequence_overlap"]))
        self.records.append(step)
        return step

    def should_stop(self) -> tuple[bool, str | None]:
        if not self.records:
            return False, None
        latest = self.records[-1]
        if not latest["stable"]:
            return True, f"{latest['label']} failed stability guard"
        if abs(float(latest["perplexity_drift"])) > self.max_abs_perplexity_drift:
            return True, f"{latest['label']} exceeded perplexity drift"
        if float(latest["sequence_overlap"]) < self.min_sequence_overlap:
            return True, f"{latest['label']} fell below sequence overlap"
        if len(self.records) >= 2 and float(latest["delta_abs_perplexity_drift"]) > self.max_abs_perplexity_drift * 0.5:
            return True, f"{latest['label']} has rapid drift growth"
        return False, None

    def summary(self) -> dict[str, Any]:
        if not self.records:
            return {"steps": 0, "stable_prefix": 0, "max_abs_perplexity_drift": 0.0}
        stable_prefix = 0
        for record in self.records:
            if not record["stable"]:
                break
            stable_prefix += 1
        return {
            "steps": len(self.records),
            "stable_prefix": stable_prefix,
            "max_abs_perplexity_drift": max(abs(float(item["perplexity_drift"])) for item in self.records),
            "min_sequence_overlap": min(float(item["sequence_overlap"]) for item in self.records),
            "earliest_divergence_token": min(
                (int(item["first_divergence_token"]) for item in self.records if item.get("first_divergence_token") is not None),
                default=None,
            ),
        }

