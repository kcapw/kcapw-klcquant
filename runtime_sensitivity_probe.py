from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from .adaptive_precision_policy import RuntimeCalibratedPrecisionPolicy
from .sensitivity_database import SensitivityDatabase, SensitivityRecord
from .tensor_criticality_ranker import rank_tensor_criticality, tensor_role
from .tensor_perturbation_runner import PERTURBATION_MODES, PerturbationTarget, TensorPerturbationRunner
from .utils import cuda_snapshot, now_id, read_json, write_json


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")
DEFAULT_TARGETS = "2x4"
RESEARCH_TARGETS = "2x4,4x8,8x8"


def parse_targets(value: str) -> list[PerturbationTarget]:
    targets: list[PerturbationTarget] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        if "x" not in item:
            raise ValueError(f"target must look like 2x4, got {item!r}")
        layers, tokens = item.split("x", 1)
        targets.append(PerturbationTarget(int(layers), int(tokens)))
    return targets


def _latest_scan(reports_dir: str | Path = "reports") -> Path:
    reports = sorted(Path(reports_dir).glob("klcquant-scan-*.json"))
    if not reports:
        raise FileNotFoundError("No klcquant-scan report found. Pass --importance-report.")
    return reports[-1]


def _is_default_candidate(name: str) -> bool:
    executable = (
        "mlp.experts.down_proj_bias" in name
        or "mlp.experts.gate_up_proj_bias" in name
        or name.endswith(".self_attn.o_proj.bias")
    )
    protected = any(token in name for token in ("router", "norm", "embed_tokens", "lm_head"))
    return executable and not protected


def select_candidates(report_path: Path, explicit: list[str] | None, max_candidates: int) -> list[dict]:
    report = read_json(report_path)
    rows = report.get("tensor_importance_rankings", [])
    by_name = {row["name"]: row for row in rows}
    if explicit:
        return [by_name.get(name, {"name": name, "importance_score": None, "rank": None, "nbytes": None}) for name in explicit]
    candidates = [row for row in rows if _is_default_candidate(row["name"])]
    candidates.sort(key=lambda row: (float(row.get("importance_score", 0.0)), -int(row.get("nbytes", 0))))
    return candidates[:max_candidates]


def _layer_index(tensor: str) -> int | None:
    match = LAYER_RE.search(tensor)
    if not match:
        return None
    return int(match.group(1))


def _cluster_summary(ranking: list[dict]) -> dict:
    by_role: dict[str, list[float]] = defaultdict(list)
    by_layer: dict[int, list[float]] = defaultdict(list)
    for row in ranking:
        score = float(row["criticality_score"])
        by_role[row.get("role", tensor_role(row["tensor"]))].append(score)
        layer = _layer_index(row["tensor"])
        if layer is not None:
            by_layer[layer].append(score)
    return {
        "by_role": {
            role: {
                "count": len(scores),
                "avg_criticality_score": round(sum(scores) / max(len(scores), 1), 6),
                "max_criticality_score": round(max(scores), 6),
            }
            for role, scores in sorted(by_role.items())
        },
        "by_layer": {
            str(layer): {
                "count": len(scores),
                "avg_criticality_score": round(sum(scores) / max(len(scores), 1), 6),
                "max_criticality_score": round(max(scores), 6),
            }
            for layer, scores in sorted(by_layer.items())
        },
    }


def _write_heatmap(records: list[SensitivityRecord], path: Path) -> str | None:
    if not records:
        return None
    tensors = sorted({record.tensor for record in records})
    labels = sorted({f"{record.mode}@{record.target_layers}x{record.target_tokens}" for record in records})
    values = [[0.0 for _ in labels] for _ in tensors]
    index = {(tensor, label): (ti, li) for ti, tensor in enumerate(tensors) for li, label in enumerate(labels)}
    for record in records:
        label = f"{record.mode}@{record.target_layers}x{record.target_tokens}"
        ti, li = index[(record.tensor, label)]
        values[ti][li] = float(record.score)

    height = max(3.5, min(12.0, 0.45 * len(tensors) + 2.0))
    width = max(7.0, min(16.0, 0.8 * len(labels) + 3.0))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(values, aspect="auto", cmap="magma")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(tensors)), labels=[_short_tensor(tensor) for tensor in tensors])
    ax.set_title("Runtime Perturbation Sensitivity")
    fig.colorbar(image, ax=ax, label="sensitivity score")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def _short_tensor(tensor: str) -> str:
    if len(tensor) <= 64:
        return tensor
    return tensor.replace("model.layers.", "L").replace(".mlp.experts.", ".experts.")[-64:]


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    targets = parse_targets(args.targets)
    modes = args.modes.split(",") if isinstance(args.modes, str) else args.modes
    modes = [mode.strip() for mode in modes if mode.strip()]
    unsupported = [mode for mode in modes if mode not in PERTURBATION_MODES]
    if unsupported:
        raise ValueError(f"unsupported modes: {unsupported}")

    candidate_rows = select_candidates(importance_report, args.candidate, args.max_candidates)
    runner = TensorPerturbationRunner(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
    db = SensitivityDatabase(args.db_path)
    db.metadata.update(
        {
            "last_importance_report": str(importance_report),
            "last_targets": [target.label for target in targets],
            "last_modes": modes,
            "updated_by": "klcquant-calibrate",
        }
    )

    new_records: list[SensitivityRecord] = []
    for row in candidate_rows:
        tensor = row["name"]
        static = {
            "importance_score": row.get("importance_score"),
            "rank": row.get("rank"),
            "percentile": row.get("percentile"),
            "nbytes": row.get("nbytes"),
            "role": tensor_role(tensor),
        }
        for target in targets:
            for mode in modes:
                record = runner.run_tensor_mode(
                    tensor,
                    mode,
                    target,
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
                    static=static,
                )
                db.add(record)
                db.save()
                new_records.append(record)

    all_latest = list(db.latest_by_tensor_mode().values())
    ranking = rank_tensor_criticality(all_latest)
    policy = RuntimeCalibratedPrecisionPolicy(
        min_safe_sequence_overlap=args.stability_sequence_overlap,
        max_abs_perplexity_drift=args.stability_perplexity_drift,
    )
    decisions = [decision.to_json() for decision in policy.recommend(all_latest)]
    mode_stability = Counter(f"{record.mode}:{'stable' if record.stable else 'unstable'}" for record in new_records)

    run_id = now_id()
    heatmap = _write_heatmap(new_records, reports_dir / f"klcquant-sensitivity-{run_id}.heatmap.png")
    report = {
        "run_id": run_id,
        "mode": "runtime_sensitivity_calibration",
        "warning": "Calibration uses tiny streamed autoregressive perturbation probes. Results identify local stability boundaries, not full-model equivalence.",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "importance_report": str(importance_report),
        "database_path": str(db.path),
        "cuda": cuda_snapshot(),
        "config": {
            "targets": [target.label for target in targets],
            "research_targets": RESEARCH_TARGETS,
            "modes": modes,
            "prompt_count": args.prompt_count,
            "max_context_tokens": args.max_context_tokens,
            "stability_perplexity_drift": args.stability_perplexity_drift,
            "stability_sequence_overlap": args.stability_sequence_overlap,
            "disable_experts": args.disable_experts,
            "expert_cache_mb": args.expert_cache_mb,
            "kv_cache_precision": args.kv_cache_precision,
        },
        "candidate_tensors": candidate_rows,
        "new_record_count": len(new_records),
        "mode_stability": dict(mode_stability),
        "tensor_stability_rankings": ranking,
        "recommended_precision_tiers": decisions,
        "instability_clusters": _cluster_summary(ranking),
        "safety_report": {
            "critical_tensors": [row for row in decisions if row["tier"] == "critical"],
            "q8_only_or_fp16_tensors": [row for row in decisions if row["recommended_mode"] in {"fp16", "q8"}],
            "ultra_low_candidates": [row for row in decisions if row["recommended_mode"] in {"q2", "q1", "pruned"}],
            "policy_note": "Only tensors stable under perturbation are recommended for q2/q1/pruned tiers.",
        },
        "new_records": [record.to_json() for record in new_records],
        "charts": [heatmap] if heatmap else [],
    }
    out = reports_dir / f"klcquant-sensitivity-{run_id}.json"
    write_json(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Runtime sensitivity-calibrated adaptive compression policy probe")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--db-path", default="reports/sensitivity_database.json")
    parser.add_argument("--work-dir", default="calibration_runs")
    parser.add_argument("--candidate", action="append", help="Exact tensor name to perturb. Can be repeated.")
    parser.add_argument("--max-candidates", type=int, default=1)
    parser.add_argument("--modes", default="q8,q4,q2,q1,pruned")
    parser.add_argument("--targets", default=DEFAULT_TARGETS, help=f"Comma targets like 2x4. Full research pass: {RESEARCH_TARGETS}.")
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--max-context-tokens", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--kv-cache-precision", default="fp16", choices=["fp16", "bf16", "q8", "q4", "q2", "q1"])
    parser.add_argument("--prune-fraction", type=float, default=0.90)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep existing perturbation artifacts instead of regenerating them.")
    parser.add_argument("--stability-perplexity-drift", type=float, default=25.0)
    parser.add_argument("--stability-sequence-overlap", type=float, default=0.50)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out = run(args)
    print(f"wrote sensitivity calibration report to {out}")


if __name__ == "__main__":
    main()
