from __future__ import annotations

from statistics import mean
from typing import Any, Iterable

from .sensitivity_database import SensitivityRecord


def forecast_from_history(records: Iterable[SensitivityRecord]) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        return {"history_count": 0, "risk": "unknown", "predicted_first_divergence_token": None}
    drifts = [abs(float(row.metrics.get("perplexity_drift", 0.0))) for row in rows]
    overlaps = [float(row.metrics.get("sequence_overlap", 1.0)) for row in rows]
    divergences = [int(row.metrics["first_divergence_token"]) for row in rows if row.metrics.get("first_divergence_token") is not None]
    unstable_rate = sum(1 for row in rows if not row.stable) / len(rows)
    avg_drift = mean(drifts)
    avg_overlap = mean(overlaps)
    risk_score = min(1.0, 0.45 * unstable_rate + 0.35 * min(avg_drift / 25.0, 1.0) + 0.20 * (1.0 - avg_overlap))
    if risk_score >= 0.70:
        risk = "high"
    elif risk_score >= 0.35:
        risk = "moderate"
    else:
        risk = "low"
    return {
        "history_count": len(rows),
        "unstable_rate": round(unstable_rate, 6),
        "avg_abs_perplexity_drift": round(avg_drift, 6),
        "avg_sequence_overlap": round(avg_overlap, 6),
        "risk_score": round(risk_score, 6),
        "risk": risk,
        "predicted_first_divergence_token": min(divergences) if divergences else None,
    }


def forecast_combination(singleton_records: Iterable[SensitivityRecord], tensor_count: int) -> dict[str, Any]:
    base = forecast_from_history(singleton_records)
    if base["history_count"] == 0:
        return {**base, "combination_risk_score": 0.5, "combination_risk": "unknown"}
    amplification = min(0.35, max(0, tensor_count - 1) * 0.08)
    combination_score = min(1.0, float(base["risk_score"]) + amplification)
    if combination_score >= 0.70:
        risk = "high"
    elif combination_score >= 0.35:
        risk = "moderate"
    else:
        risk = "low"
    return {
        **base,
        "tensor_count": tensor_count,
        "interaction_amplification_prior": round(amplification, 6),
        "combination_risk_score": round(combination_score, 6),
        "combination_risk": risk,
    }

