from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


def pressure_from_generation(run: dict[str, Any]) -> dict[str, Any]:
    stats = run.get("runtime_stats", {})
    transfer_events = stats.get("transfer", {}).get("events", [])
    vram_events = stats.get("vram_events", [])
    transfer_by_tensor: Counter[str] = Counter()
    reloads: Counter[str] = Counter()
    for event in transfer_events:
        name = event.get("name", event.get("kind", "unknown"))
        bytes_ = int(event.get("bytes", 0))
        transfer_by_tensor[name] += bytes_
        reloads[name] += 1

    burst_spikes = _transfer_bursts(transfer_events)
    residency = _residency_windows(vram_events)
    churn = [
        {
            "tensor": name,
            "reloads": count,
            "transfer_bytes": transfer_by_tensor[name],
            "transfer_mb": round(transfer_by_tensor[name] / 2**20, 6),
        }
        for name, count in reloads.most_common()
    ]
    kv = stats.get("kv_cache", {})
    expert = stats.get("expert_cache", {})
    return {
        "tensor_residency_duration": residency,
        "reload_churn": churn,
        "transfer_burst_spikes": burst_spikes,
        "cache_residency_pressure": {
            "kv_resident_mb": kv.get("resident_mb", 0.0),
            "kv_precision": kv.get("precision", "fp16"),
            "expert_resident_mb": expert.get("resident_mb", 0.0),
            "expert_evictions": expert.get("evictions", 0),
            "expert_hit_rate": expert.get("hit_rate", 0.0),
        },
        "promotion_triggered_vram_spikes": [],
    }


def _transfer_bursts(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not events:
        return []
    rows = []
    for idx, event in enumerate(events):
        bytes_ = int(event.get("bytes", 0))
        if bytes_ <= 0:
            continue
        rows.append({"event_index": idx, "name": event.get("name"), "kind": event.get("kind"), "bytes": bytes_, "mb": round(bytes_ / 2**20, 6)})
    rows.sort(key=lambda item: item["bytes"], reverse=True)
    return rows[:20]


def _residency_windows(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    starts: dict[str, float] = {}
    totals: defaultdict[str, float] = defaultdict(float)
    for event in events:
        label = str(event.get("label", ""))
        seconds = float(event.get("seconds", 0.0))
        if label.endswith(":start"):
            starts[label[: -len(":start")]] = seconds
        elif label.endswith(":end"):
            name = label[: -len(":end")]
            if name in starts:
                totals[name] += max(0.0, seconds - starts[name])
    return [{"window": name, "seconds": round(value, 6)} for name, value in sorted(totals.items())]


def pressure_from_report(report: dict[str, Any]) -> dict[str, Any]:
    experiments = report.get("experiments", [])
    if not experiments or not experiments[0].get("results"):
        return {}
    result = experiments[0]["results"][0]
    return {
        "baseline": pressure_from_generation(result["baseline"]),
        "quantized": pressure_from_generation(result["quantized"]),
    }

