from __future__ import annotations

from typing import Any


def kv_survivability_metrics(report: dict[str, Any], cache_strategy: str) -> dict[str, Any]:
    experiments = report.get("experiments", [])
    if not experiments or not experiments[0].get("results"):
        return {}
    result = experiments[0]["results"][0]
    comparison = result.get("comparison", {})
    kv = result.get("quantized", {}).get("runtime_stats", {}).get("kv_cache", {})
    generated = result.get("quantized", {}).get("generated_token_ids", [])
    first = comparison.get("first_divergence_index")
    survival = len(generated) if first is None else int(first)
    return {
        "cache_strategy": cache_strategy,
        "kv_precision": kv.get("precision", "fp16"),
        "cache_corruption_likelihood": kv.get("cache_corruption_likelihood", 0.0),
        "token_survival_depth": survival,
        "post_recovery_overlap": comparison.get("sequence_overlap", 0.0),
        "stale_cache_divergence_detected": bool(first is not None and kv.get("cache_corruption_likelihood", 0.0) > 0),
        "resident_mb": kv.get("resident_mb", 0.0),
    }


def compare_cache_strategies(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {"best_strategy": None, "items": []}
    best = max(items, key=lambda item: (float(item.get("post_recovery_overlap", 0.0)), int(item.get("token_survival_depth", 0)), -float(item.get("cache_corruption_likelihood", 0.0))))
    return {"best_strategy": best.get("cache_strategy"), "items": items}

