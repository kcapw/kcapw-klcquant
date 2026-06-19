from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .compression_orchestrator import _static_for
from .runtime_sensitivity_probe import _latest_scan, select_candidates
from .kv_survivability import compare_cache_strategies, kv_survivability_metrics
from .multi_tensor_calibrator import MultiTensorCalibrator, TensorOverride
from .recursive_drift_tracker import attach_recursive_drift
from .runtime_pressure_telemetry import pressure_from_report
from .tensor_perturbation_runner import PerturbationTarget
from .utils import cuda_snapshot, now_id, read_json, write_json


PROMOTION_CHAIN = {"q1": "q2", "q2": "q4", "q4": "q8"}
PROFILE_INITIAL_MODE = {
    "conservative": "q8",
    "balanced": "q4",
    "recursive_safe": "q8",
    "recovery_aggressive": "q2",
    "cache_preserving": "q4",
}
PROFILE_KV_PRECISION = {
    "conservative": "fp16",
    "balanced": "fp16",
    "recursive_safe": "fp16",
    "recovery_aggressive": "q8",
    "cache_preserving": "fp16",
}


def _parse_depths(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _risk_sort_key(override: TensorOverride) -> tuple[float, int]:
    static = override.static
    role_bonus = 1.0 if "scale" in override.tensor or "blocks" in override.tensor else 0.0
    return (role_bonus + float(static.get("importance_score") or 0.0), int(static.get("nbytes") or 0))


def _promote_highest_risk(overrides: list[TensorOverride]) -> tuple[list[TensorOverride], str | None]:
    promotable = [item for item in overrides if item.mode in PROMOTION_CHAIN]
    if not promotable:
        return overrides, None
    chosen = max(promotable, key=_risk_sort_key)
    promoted = []
    for item in overrides:
        if item.tensor == chosen.tensor:
            promoted.append(TensorOverride(item.tensor, PROMOTION_CHAIN[item.mode], item.static))
        else:
            promoted.append(item)
    return promoted, chosen.tensor


def _initial_overrides(rows: list[dict[str, Any]], profile: str) -> list[TensorOverride]:
    mode = PROFILE_INITIAL_MODE[profile]
    return [TensorOverride(row["name"], mode, _static_for(row)) for row in rows]


def _run_once(
    calibrator: MultiTensorCalibrator,
    overrides: list[TensorOverride],
    target: PerturbationTarget,
    profile: str,
    args: argparse.Namespace,
    kv_precision: str,
    cache_strategy: str,
) -> dict[str, Any]:
    result = calibrator.run_override_set(
        overrides,
        target,
        profile,
        prompt_count=args.prompt_count,
        max_context_tokens=args.max_context_tokens,
        top_k=args.top_k,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
        expert_cache_mb=args.expert_cache_mb,
        kv_cache_precision=kv_precision,
        prune_fraction=args.prune_fraction,
        disable_experts=args.disable_experts,
        stop_perplexity_drift=args.stability_perplexity_drift,
        stop_sequence_overlap=args.stability_sequence_overlap,
        overwrite=not args.resume,
    )
    report = read_json(result["autoregressive_report"])
    result["recursive_drift"] = attach_recursive_drift(report)
    result["pressure_telemetry"] = pressure_from_report(report)
    result["kv_survivability"] = kv_survivability_metrics(report, cache_strategy)
    result["cache_strategy"] = cache_strategy
    result["kv_cache_precision"] = kv_precision
    return result


def _recover(
    calibrator: MultiTensorCalibrator,
    initial: dict[str, Any],
    overrides: list[TensorOverride],
    target: PerturbationTarget,
    profile: str,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    current = overrides
    base_survival = int(initial.get("recursive_drift", {}).get("token_survival_depth", 0))
    for depth in range(args.max_promotions):
        current, promoted_tensor = _promote_highest_risk(current)
        if promoted_tensor is None:
            break
        recovery = _run_once(
            calibrator,
            current,
            target,
            profile,
            args,
            "fp16" if profile == "cache_preserving" else PROFILE_KV_PRECISION.get(profile, "fp16"),
            "checkpoint_replay",
        )
        survival = int(recovery.get("recursive_drift", {}).get("token_survival_depth", 0))
        recovery["promotion"] = {
            "promoted_tensor": promoted_tensor,
            "promotion_depth": depth + 1,
            "recovery_depth_extension": survival - base_survival,
        }
        attempts.append(recovery)
        if recovery.get("stable"):
            break
    return attempts, {
        "promotion_success_rate": sum(1 for item in attempts if item.get("stable")) / max(len(attempts), 1),
        "recovered_generations": sum(1 for item in attempts if item.get("stable")),
        "failed_recoveries": sum(1 for item in attempts if not item.get("stable")),
        "max_recovery_depth_extension": max((item.get("promotion", {}).get("recovery_depth_extension", 0) for item in attempts), default=0),
    }


def _frontier(results: list[dict[str, Any]]) -> dict[str, Any]:
    rows = []
    for result in results:
        drift = result.get("recursive_drift", {})
        overlap = float(result.get("metrics", {}).get("sequence_overlap", 0.0))
        recoverable = bool(result.get("recovery_attempts"))
        if result.get("stable"):
            region = "stable"
        elif recoverable and any(item.get("stable") for item in result.get("recovery_attempts", [])):
            region = "degraded_but_recoverable"
        else:
            region = "unrecoverable_divergence"
        rows.append(
            {
                "profile": result["profile"],
                "target": f"{result['target_layers']}x{result['target_tokens']}",
                "region": region,
                "token_survival_depth": drift.get("token_survival_depth"),
                "drift_velocity": drift.get("drift_velocity"),
                "instability_acceleration": drift.get("instability_acceleration"),
                "sequence_overlap": overlap,
                "modes": sorted({item["mode"] for item in result.get("overrides", [])}),
            }
        )
    return {"regions": rows}


def _write_frontier_chart(frontier: dict[str, Any], path: Path) -> str | None:
    rows = frontier.get("regions", [])
    if not rows:
        return None
    colors = {"stable": "green", "degraded_but_recoverable": "orange", "unrecoverable_divergence": "red"}
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for row in rows:
        ax.scatter(row.get("token_survival_depth") or 0, row.get("sequence_overlap") or 0.0, c=colors[row["region"]], label=row["region"])
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys())
    ax.set_xlabel("token survival depth")
    ax.set_ylabel("sequence overlap")
    ax.set_title("Survivability Frontier")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _write_pressure_chart(results: list[dict[str, Any]], path: Path) -> str | None:
    if not results:
        return None
    labels = [f"{r['profile']}@{r['target_tokens']}" for r in results]
    transfers = [r.get("metrics", {}).get("quantized_transfer_mb", 0.0) for r in results]
    kv = [r.get("kv_survivability", {}).get("resident_mb", 0.0) for r in results]
    fig, ax = plt.subplots(figsize=(max(8, 0.5 * len(labels)), 4))
    ax.plot(labels, transfers, marker="o", label="transfer MB")
    ax.plot(labels, kv, marker="o", label="KV resident MB")
    ax.tick_params(axis="x", rotation=45)
    ax.set_title("Transfer And KV Pressure")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    candidate_rows = select_candidates(importance_report, args.candidate, args.max_candidates)
    profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    depths = _parse_depths(args.token_depths)
    calibrator = MultiTensorCalibrator(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
    results: list[dict[str, Any]] = []
    kv_items: list[dict[str, Any]] = []

    for depth in depths:
        target = PerturbationTarget(args.max_layers, depth)
        for profile in profiles:
            if profile not in PROFILE_INITIAL_MODE:
                raise ValueError(f"unknown survivability profile: {profile}")
            overrides = _initial_overrides(candidate_rows, profile)
            result = _run_once(
                calibrator,
                overrides,
                target,
                profile,
                args,
                PROFILE_KV_PRECISION[profile],
                "cache_preserving" if profile == "cache_preserving" else "mixed_kv" if PROFILE_KV_PRECISION[profile] != "fp16" else "cache_reset",
            )
            result["initial_modes"] = sorted({item.mode for item in overrides})
            if (not result.get("stable")) and profile in {"recovery_aggressive", "recursive_safe", "cache_preserving"}:
                attempts, summary = _recover(calibrator, result, overrides, target, profile, args)
                result["recovery_attempts"] = attempts
                result["recovery_summary"] = summary
            else:
                result["recovery_attempts"] = []
                result["recovery_summary"] = {
                    "promotion_success_rate": 0.0,
                    "recovered_generations": 0,
                    "failed_recoveries": 0,
                    "max_recovery_depth_extension": 0,
                }
            kv_items.append(result["kv_survivability"])
            for attempt in result.get("recovery_attempts", []):
                kv_items.append(attempt.get("kv_survivability", {}))
            results.append(result)

    frontier = _frontier(results)
    run_id = now_id()
    charts = [
        path
        for path in [
            _write_frontier_chart(frontier, reports_dir / f"klcquant-survivability-{run_id}.frontier.png"),
            _write_pressure_chart(results, reports_dir / f"klcquant-survivability-{run_id}.pressure.png"),
        ]
        if path
    ]
    report = {
        "run_id": run_id,
        "mode": "adaptive_recursive_survivability",
        "warning": "Experimental streamed survivability control. Recovery uses checkpoint replay/cache reset validation; in-memory mid-token promotion remains future work.",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "importance_report": str(importance_report),
        "cuda": cuda_snapshot(),
        "config": {
            "profiles": profiles,
            "token_depths": depths,
            "max_layers": args.max_layers,
            "prompt_count": args.prompt_count,
            "max_promotions": args.max_promotions,
            "stability_perplexity_drift": args.stability_perplexity_drift,
            "stability_sequence_overlap": args.stability_sequence_overlap,
        },
        "candidate_tensors": candidate_rows,
        "results": results,
        "kv_cache_survivability": compare_cache_strategies(kv_items),
        "survivability_frontier": frontier,
        "summary": _summary(results),
        "charts": charts,
    }
    out = reports_dir / f"klcquant-survivability-{run_id}.json"
    write_json(out, report)
    return out


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    stable = [item for item in results if item.get("stable")]
    recovered = [item for item in results if item.get("recovery_summary", {}).get("recovered_generations", 0) > 0]
    best = max(
        results,
        key=lambda item: (
            int(item.get("recursive_drift", {}).get("token_survival_depth", 0)),
            float(item.get("metrics", {}).get("sequence_overlap", 0.0)),
            -abs(float(item.get("metrics", {}).get("perplexity_drift", 0.0))),
        ),
        default=None,
    )
    return {
        "stable_runs": len(stable),
        "recovered_runs": len(recovered),
        "failed_runs": len(results) - len(stable),
        "max_token_survival_depth": max((int(item.get("recursive_drift", {}).get("token_survival_depth", 0)) for item in results), default=0),
        "best_profile": None if best is None else {"profile": best["profile"], "target_tokens": best["target_tokens"], "stable": best["stable"]},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run adaptive recursive survivability control experiments")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--work-dir", default="calibration_runs/survivability")
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--max-candidates", type=int, default=3)
    parser.add_argument("--profiles", default="conservative,balanced,recursive_safe,recovery_aggressive")
    parser.add_argument("--token-depths", default="256,512,1024")
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--prune-fraction", type=float, default=0.90)
    parser.add_argument("--max-promotions", type=int, default=3)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stability-perplexity-drift", type=float, default=25.0)
    parser.add_argument("--stability-sequence-overlap", type=float, default=0.50)
    return parser


def main() -> None:
    out = run(build_parser().parse_args())
    print(f"wrote survivability report to {out}")


if __name__ == "__main__":
    main()
