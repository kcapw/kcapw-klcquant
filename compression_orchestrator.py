from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .adaptive_precision_policy import RuntimeCalibratedPrecisionPolicy
from .drift_forecaster import forecast_combination
from .instability_accumulator import InstabilityAccumulator
from .interaction_sensitivity import build_interaction_graph
from .multi_tensor_calibrator import MultiTensorCalibrator, TensorOverride
from .runtime_sensitivity_probe import _latest_scan, parse_targets, select_candidates
from .sensitivity_database import SensitivityDatabase, SensitivityRecord
from .tensor_criticality_ranker import rank_tensor_criticality, tensor_role
from .utils import cuda_snapshot, now_id, read_json, write_json


PROFILE_STEPS = {
    "conservative": ["q8"],
    "balanced": ["q8", "q4"],
    "aggressive": ["q8", "q4", "q2", "q1"],
    "ultra_low_vram": ["q8", "q4", "q2", "q1", "pruned"],
    "recursive_safe": ["q8", "q4"],
    "recovery_aggressive": ["q8", "q4", "q2", "q1"],
    "cache_preserving": ["q8", "q4"],
}
MODE_ORDER = {"fp16": 0, "q8": 1, "q4": 2, "q3": 3, "q2": 4, "q1": 5, "pruned": 6}
LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


def _mode_min(left: str, right: str) -> str:
    return left if MODE_ORDER.get(left, 99) <= MODE_ORDER.get(right, 99) else right


def _layer_index(tensor: str) -> int | None:
    match = LAYER_RE.search(tensor)
    if not match:
        return None
    return int(match.group(1))


def _static_for(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "importance_score": row.get("importance_score"),
        "rank": row.get("rank"),
        "percentile": row.get("percentile"),
        "nbytes": row.get("nbytes"),
        "role": tensor_role(row["name"]),
    }


def _decisions(records: list[SensitivityRecord]) -> dict[str, dict[str, Any]]:
    if not records:
        return {}
    policy = RuntimeCalibratedPrecisionPolicy()
    return {item["tensor"]: item for item in (decision.to_json() for decision in policy.recommend(records))}


def _safe_mode_for(row: dict[str, Any], profile: str, step_mode: str, decisions: dict[str, dict[str, Any]]) -> str | None:
    tensor = row["name"]
    decision = decisions.get(tensor)
    recommended = decision["recommended_mode"] if decision else None
    if recommended == "fp16":
        return None
    if profile in {"conservative", "recursive_safe", "cache_preserving"}:
        return "q8"
    if profile == "balanced":
        cap = recommended or "q4"
        return _mode_min(step_mode, cap if cap != "fp16" else "q8")
    if profile in {"aggressive", "recovery_aggressive"}:
        if recommended in {"q8", "q4", "q2", "q1", "pruned"}:
            return _mode_min(step_mode, recommended)
        return step_mode if step_mode in {"q8", "q4"} else "q4"
    if profile == "ultra_low_vram":
        if recommended in {"q8", "q4", "q2", "q1", "pruned"}:
            return _mode_min(step_mode, recommended)
        return step_mode
    raise ValueError(f"unknown profile {profile}")


def _overrides_for_step(rows: list[dict[str, Any]], profile: str, step_mode: str, decisions: dict[str, dict[str, Any]]) -> list[TensorOverride]:
    overrides: list[TensorOverride] = []
    for row in rows:
        mode = _safe_mode_for(row, profile, step_mode, decisions)
        if mode is None:
            continue
        overrides.append(TensorOverride(row["name"], mode, _static_for(row)))
    return overrides


def _architecture_maps(candidate_rows: list[dict[str, Any]], records: list[SensitivityRecord], multi_results: list[dict[str, Any]]) -> dict[str, Any]:
    by_tensor_score: dict[str, float] = {}
    for row in rank_tensor_criticality(records):
        by_tensor_score[row["tensor"]] = float(row["criticality_score"])
    for result in multi_results:
        score = float(result.get("score", 0.0))
        for override in result.get("overrides", []):
            by_tensor_score[override["tensor"]] = max(by_tensor_score.get(override["tensor"], 0.0), score)

    layer_scores: dict[int, list[float]] = defaultdict(list)
    role_scores: dict[str, list[float]] = defaultdict(list)
    for row in candidate_rows:
        tensor = row["name"]
        score = by_tensor_score.get(tensor, 0.0)
        layer = _layer_index(tensor)
        if layer is not None:
            layer_scores[layer].append(score)
        role_scores[tensor_role(tensor)].append(score)
    return {
        "layer_criticality": {
            str(layer): {
                "count": len(scores),
                "max_score": round(max(scores), 6),
                "avg_score": round(sum(scores) / max(len(scores), 1), 6),
            }
            for layer, scores in sorted(layer_scores.items())
        },
        "role_criticality": {
            role: {
                "count": len(scores),
                "max_score": round(max(scores), 6),
                "avg_score": round(sum(scores) / max(len(scores), 1), 6),
            }
            for role, scores in sorted(role_scores.items())
        },
    }


def _write_profile_chart(results: list[dict[str, Any]], path: Path) -> str | None:
    if not results:
        return None
    labels = [f"{item['profile']}:{item['step_mode']}@{item['target_layers']}x{item['target_tokens']}" for item in results]
    fig, ax = plt.subplots(figsize=(max(8.0, 0.8 * len(labels)), 4.5))
    ax.plot(labels, [item["metrics"].get("sequence_overlap", 0.0) for item in results], marker="o", label="sequence overlap")
    ax.plot(labels, [item["metrics"].get("token_jaccard", 0.0) for item in results], marker="o", label="token jaccard")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Progressive Compression Stability")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _write_layer_heatmap(architecture_maps: dict[str, Any], path: Path) -> str | None:
    rows = architecture_maps.get("layer_criticality", {})
    if not rows:
        return None
    layers = sorted(rows, key=lambda item: int(item))
    values = [[rows[layer]["avg_score"], rows[layer]["max_score"]] for layer in layers]
    fig, ax = plt.subplots(figsize=(6, max(3.0, 0.35 * len(layers) + 1.5)))
    image = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_xticks([0, 1], labels=["avg", "max"])
    ax.set_yticks(range(len(layers)), labels=[f"L{layer}" for layer in layers])
    ax.set_title("Layer Criticality Overlay")
    fig.colorbar(image, ax=ax, label="criticality")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _compression_for_result(result: dict[str, Any]) -> dict[str, Any]:
    original = 0
    compressed = 0
    for path in result.get("quantization_reports", []):
        if not Path(path).exists():
            continue
        report = read_json(path)
        compression = report.get("compression", {})
        original += int(compression.get("original_bytes", 0))
        compressed += int(compression.get("compressed_payload_bytes", 0))
    ratio = original / max(compressed, 1)
    return {"original_bytes": original, "compressed_bytes": compressed, "compression_ratio": round(ratio, 6)}


def _unsafe_pairs(graph: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for edge in graph.get("edges", []):
        relations = set(edge.get("relations", []))
        if edge.get("unstable_runs", 0) > 0 or "amplifies_instability" in relations or "jointly_unstable" in relations:
            rows.append(edge)
    return sorted(rows, key=lambda item: (item.get("unstable_runs", 0), item.get("max_interaction_score", 0.0)), reverse=True)


def _best_profile(profile_summaries: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any] | None:
    stable = [result for result in results if result.get("stable")]
    if not stable:
        return None
    for result in stable:
        result.setdefault("compression", _compression_for_result(result))
    return max(
        stable,
        key=lambda item: (
            float(item["compression"]["compression_ratio"]),
            float(item.get("metrics", {}).get("sequence_overlap", 0.0)),
            -abs(float(item.get("metrics", {}).get("perplexity_drift", 0.0))),
        ),
    )


def _summary_report(profile_summaries: list[dict[str, Any]], results: list[dict[str, Any]], graph: dict[str, Any]) -> dict[str, Any]:
    for result in results:
        result.setdefault("compression", _compression_for_result(result))
    best = _best_profile(profile_summaries, results)
    worst_drift = max((abs(float(item.get("metrics", {}).get("perplexity_drift", 0.0))) for item in results), default=0.0)
    min_overlap = min((float(item.get("metrics", {}).get("sequence_overlap", 1.0)) for item in results), default=1.0)
    early_stops = [item for item in profile_summaries if item.get("stopped_early")]
    unsafe = _unsafe_pairs(graph)
    return {
        "best_compression_ratio": max((float(item["compression"]["compression_ratio"]) for item in results), default=0.0),
        "worst_abs_perplexity_drift": round(worst_drift, 6),
        "min_sequence_overlap": round(min_overlap, 6),
        "early_stop_triggers": early_stops,
        "unsafe_tensor_pairs": unsafe,
        "final_recommended_profile": None
        if best is None
        else {
            "profile": best["profile"],
            "target": f"{best['target_layers']}x{best['target_tokens']}",
            "step_mode": best["step_mode"],
            "compression_ratio": best["compression"]["compression_ratio"],
            "sequence_overlap": best["metrics"].get("sequence_overlap"),
            "perplexity_drift": best["metrics"].get("perplexity_drift"),
            "active_modes": sorted({override["mode"] for override in best.get("overrides", [])}, key=lambda mode: MODE_ORDER.get(mode, 99)),
            "tensor_count": len(best.get("overrides", [])),
        },
    }


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    targets = parse_targets(args.targets)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    for profile in profiles:
        if profile not in PROFILE_STEPS:
            raise ValueError(f"unknown profile {profile}; choices are {sorted(PROFILE_STEPS)}")

    candidate_rows = select_candidates(importance_report, args.candidate, args.max_candidates)
    db = SensitivityDatabase(args.db_path)
    singleton_records = list(db.latest_by_tensor_mode().values())
    decisions = _decisions(singleton_records)
    calibrator = MultiTensorCalibrator(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
    multi_results: list[dict[str, Any]] = []
    profile_summaries: list[dict[str, Any]] = []

    for target in targets:
        for profile in profiles:
            accumulator = InstabilityAccumulator(args.stability_sequence_overlap, args.stability_perplexity_drift)
            stopped = False
            stop_reason = None
            for step_mode in PROFILE_STEPS[profile]:
                overrides = _overrides_for_step(candidate_rows, profile, step_mode, decisions)
                if not overrides:
                    continue
                forecast = forecast_combination(
                    [record for record in singleton_records if record.tensor in {item.tensor for item in overrides}],
                    len(overrides),
                )
                result = calibrator.run_override_set(
                    overrides,
                    target,
                    profile,
                    prompt_count=args.prompt_count,
                    max_context_tokens=args.max_context_tokens,
                    top_k=args.top_k,
                    lm_head_chunk_rows=args.lm_head_chunk_rows,
                    expert_cache_mb=args.expert_cache_mb,
                    kv_cache_precision=args.kv_cache_precision,
                    prune_fraction=args.prune_fraction,
                    disable_experts=args.disable_experts,
                    stop_perplexity_drift=args.stability_perplexity_drift,
                    stop_sequence_overlap=args.stability_sequence_overlap,
                    overwrite=not args.resume,
                )
                result["step_mode"] = step_mode
                result["forecast"] = forecast
                result["accumulation"] = accumulator.add(f"{profile}:{step_mode}@{target.label}", result["metrics"], result["stable"])
                multi_results.append(result)
                stopped, stop_reason = accumulator.should_stop()
                if stopped:
                    break
            profile_summaries.append(
                {
                    "profile": profile,
                    "target": target.label,
                    "stopped_early": stopped,
                    "stop_reason": stop_reason,
                    "accumulator": accumulator.summary(),
                }
            )

    graph = build_interaction_graph(singleton_records, multi_results)
    architecture_maps = _architecture_maps(candidate_rows, singleton_records, multi_results)
    run_id = now_id()
    profile_chart = _write_profile_chart(multi_results, reports_dir / f"klcquant-orchestration-{run_id}.profiles.png")
    layer_heatmap = _write_layer_heatmap(architecture_maps, reports_dir / f"klcquant-orchestration-{run_id}.layers.png")
    charts = [path for path in [profile_chart, layer_heatmap] if path]
    report = {
        "run_id": run_id,
        "mode": "multi_tensor_adaptive_compression_orchestration",
        "warning": "This is bounded streamed calibration for compression policy discovery, not full-model equivalence.",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "importance_report": str(importance_report),
        "sensitivity_database": str(db.path),
        "cuda": cuda_snapshot(),
        "config": {
            "targets": [target.label for target in targets],
            "profiles": profiles,
            "max_candidates": args.max_candidates,
            "prompt_count": args.prompt_count,
            "max_context_tokens": args.max_context_tokens,
            "stability_perplexity_drift": args.stability_perplexity_drift,
            "stability_sequence_overlap": args.stability_sequence_overlap,
            "progressive_escalation": "stop profile/target after first unstable step",
            "kv_cache_precision": args.kv_cache_precision,
        },
        "candidate_tensors": candidate_rows,
        "runtime_precision_decisions": sorted(decisions.values(), key=lambda item: item["tensor"]),
        "profile_summaries": profile_summaries,
        "multi_tensor_results": multi_results,
        "compression_safety_graph": graph,
        "architecture_maps": architecture_maps,
        "summary_report": _summary_report(profile_summaries, multi_results, graph),
        "recommended_runtime_profiles": _recommended_profiles(profile_summaries, multi_results),
        "charts": charts,
    }
    out = reports_dir / f"klcquant-orchestration-{run_id}.json"
    write_json(out, report)
    return out


def _recommended_profiles(profile_summaries: list[dict[str, Any]], results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    result_lookup = {(item["profile"], f"{item['target_layers']}x{item['target_tokens']}"): item for item in results if item.get("stable")}
    for summary in profile_summaries:
        key = (summary["profile"], summary["target"])
        result = result_lookup.get(key)
        if result is None:
            continue
        modes = {override["mode"] for override in result.get("overrides", [])}
        out.append(
            {
                "profile": summary["profile"],
                "target": summary["target"],
                "stable_prefix": summary["accumulator"]["stable_prefix"],
                "active_modes": sorted(modes, key=lambda mode: MODE_ORDER.get(mode, 99)),
                "tensor_count": len(result.get("overrides", [])),
                "residency": _profile_residency(summary["profile"]),
            }
        )
    return out


def _profile_residency(profile: str) -> dict[str, Any]:
    if profile in {"conservative", "recursive_safe"}:
        return {"expert_cache_mb": 768, "prefetch": "nearest-layer", "eviction": "criticality-aware"}
    if profile == "cache_preserving":
        return {"expert_cache_mb": 768, "prefetch": "kv-preserving", "eviction": "avoid-kv-reset"}
    if profile == "balanced":
        return {"expert_cache_mb": 512, "prefetch": "router-biased", "eviction": "adaptive-lru"}
    if profile in {"aggressive", "recovery_aggressive"}:
        return {"expert_cache_mb": 384, "prefetch": "selected-expert-only", "eviction": "short-window"}
    return {"expert_cache_mb": 256, "prefetch": "minimal", "eviction": "eager-cold-unload"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multi-tensor adaptive compression orchestration")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--db-path", default="reports/sensitivity_database.json")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--work-dir", default="calibration_runs/multi")
    parser.add_argument("--candidate", action="append", help="Exact tensor name to include. Can be repeated.")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--targets", default="2x4", help="Comma targets like 2x4,4x8,8x8.")
    parser.add_argument("--profiles", default="conservative,balanced")
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--kv-cache-precision", default="fp16", choices=["fp16", "bf16", "q8", "q4", "q2", "q1"])
    parser.add_argument("--prune-fraction", type=float, default=0.90)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stability-perplexity-drift", type=float, default=25.0)
    parser.add_argument("--stability-sequence-overlap", type=float, default=0.50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(f"wrote orchestration report to {out}")


if __name__ == "__main__":
    main()
