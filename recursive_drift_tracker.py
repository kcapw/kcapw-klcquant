from __future__ import annotations

from typing import Any
from math import isfinite


def _gen_logs(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in run.get("token_logs", []) if "generated_token" in item]


def _route_signature(token_log: dict[str, Any]) -> list[tuple[int, tuple[int, ...]]]:
    signature: list[tuple[int, tuple[int, ...]]] = []
    for layer in token_log.get("layers", []):
        route = layer.get("route", {})
        signature.append((int(layer.get("layer", -1)), tuple(int(x) for x in route.get("experts", []))))
    return signature


def _prefix_overlap(base: list[int], quant: list[int], end: int) -> float:
    if end <= 0:
        return 1.0
    matches = sum(1 for idx in range(min(end, len(base), len(quant))) if base[idx] == quant[idx])
    return matches / end


def recursive_drift_series(baseline: dict[str, Any], quantized: dict[str, Any]) -> dict[str, Any]:
    base_ids = [int(x) for x in baseline.get("generated_token_ids", [])]
    quant_ids = [int(x) for x in quantized.get("generated_token_ids", [])]
    base_logs = _gen_logs(baseline)
    quant_logs = _gen_logs(quantized)
    steps = min(len(base_ids), len(quant_ids), len(base_logs), len(quant_logs))
    rows: list[dict[str, Any]] = []
    cumulative = 0.0
    for idx in range(steps):
        token_match = base_ids[idx] == quant_ids[idx]
        overlap = _prefix_overlap(base_ids, quant_ids, idx + 1)
        base_entropy = float(base_logs[idx].get("entropy", 0.0))
        quant_entropy = float(quant_logs[idx].get("entropy", 0.0))
        entropy_drift = quant_entropy - base_entropy if isfinite(base_entropy) and isfinite(quant_entropy) else 0.0
        base_route = _route_signature(base_logs[idx])
        quant_route = _route_signature(quant_logs[idx])
        route_matches = sum(1 for left, right in zip(base_route, quant_route) if left == right)
        route_total = max(len(base_route), len(quant_route), 1)
        routing_divergence = 1.0 - (route_matches / route_total)
        token_instability = (0.0 if token_match else 1.0) + abs(entropy_drift) * 0.05 + routing_divergence
        cumulative += token_instability
        rows.append(
            {
                "token_index": idx,
                "baseline_token": base_ids[idx],
                "quantized_token": quant_ids[idx],
                "token_match": token_match,
                "prefix_overlap": round(overlap, 6),
                "entropy_drift": round(entropy_drift, 6),
                "routing_divergence": round(routing_divergence, 6),
                "cumulative_instability_score": round(cumulative, 6),
            }
        )

    drift_velocity = 0.0
    overlap_decay_rate = 0.0
    instability_acceleration = 0.0
    if len(rows) >= 2:
        drift_velocity = (rows[-1]["cumulative_instability_score"] - rows[0]["cumulative_instability_score"]) / (len(rows) - 1)
        overlap_decay_rate = (rows[0]["prefix_overlap"] - rows[-1]["prefix_overlap"]) / (len(rows) - 1)
    if len(rows) >= 3:
        velocities = [rows[idx]["cumulative_instability_score"] - rows[idx - 1]["cumulative_instability_score"] for idx in range(1, len(rows))]
        instability_acceleration = (velocities[-1] - velocities[0]) / max(len(velocities) - 1, 1)

    return {
        "token_survival_depth": next((idx for idx, row in enumerate(rows) if not row["token_match"]), len(rows)),
        "drift_velocity": round(drift_velocity, 6),
        "overlap_decay_rate": round(overlap_decay_rate, 6),
        "instability_acceleration": round(instability_acceleration, 6),
        "cumulative_instability_score": round(cumulative, 6),
        "routing_divergence_progression": [row["routing_divergence"] for row in rows],
        "series": rows,
    }


def attach_recursive_drift(report: dict[str, Any]) -> dict[str, Any]:
    experiments = report.get("experiments", [])
    if not experiments or not experiments[0].get("results"):
        return {}
    result = experiments[0]["results"][0]
    return recursive_drift_series(result["baseline"], result["quantized"])
