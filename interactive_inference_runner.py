from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, GenerationRuntime, compare_generations
from .kv_survivability import kv_survivability_metrics
from .multi_tensor_calibrator import MultiTensorCalibrator, TensorOverride
from .recursive_drift_tracker import recursive_drift_series
from .runtime_pressure_telemetry import pressure_from_generation
from .runtime_sensitivity_probe import _latest_scan
from .streamed_transformer_executor import ExecutorConfig
from .tensor_perturbation_runner import PerturbationTarget
from .utils import cuda_snapshot, now_id, read_json, write_json


DEFAULT_RECURSIVE_SAFE_TENSORS = [
    "model.layers.1.mlp.experts.down_proj_bias",
    "model.layers.1.mlp.experts.gate_up_proj_bias",
]


def _static_rows(importance_report: Path, names: list[str]) -> dict[str, dict[str, Any]]:
    report = read_json(importance_report)
    rows = {row["name"]: row for row in report.get("tensor_importance_rankings", [])}
    return {
        name: {
            "importance_score": rows.get(name, {}).get("importance_score"),
            "rank": rows.get(name, {}).get("rank"),
            "nbytes": rows.get(name, {}).get("nbytes"),
        }
        for name in names
    }


def _token_stream(tokenizer, baseline: dict[str, Any], quantized: dict[str, Any], drift: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    series = {int(item["token_index"]): item for item in drift.get("series", [])}
    gen_logs = [item for item in quantized.get("token_logs", []) if "generated_token" in item]
    for idx, token_id in enumerate(quantized.get("generated_token_ids", [])):
        item = series.get(idx, {})
        gen_log = gen_logs[idx] if idx < len(gen_logs) else {}
        rows.append(
            {
                "index": idx,
                "token_id": int(token_id),
                "text": tokenizer.decode([int(token_id)]),
                "baseline_token_id": baseline.get("generated_token_ids", [None] * (idx + 1))[idx]
                if idx < len(baseline.get("generated_token_ids", []))
                else None,
                "token_match": item.get("token_match"),
                "prefix_overlap": item.get("prefix_overlap"),
                "routing_divergence": item.get("routing_divergence"),
                "cumulative_instability_score": item.get("cumulative_instability_score"),
                "token_latency_s": gen_log.get("token_latency_s"),
                "logits_finite": gen_log.get("logits_finite"),
                "nonfinite_logit_count": gen_log.get("nonfinite_logit_count"),
            }
        )
    return rows


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    candidates = args.candidate or DEFAULT_RECURSIVE_SAFE_TENSORS
    static = _static_rows(importance_report, candidates)
    overrides = [TensorOverride(name, "q8", static.get(name, {})) for name in candidates]
    target = PerturbationTarget(args.max_layers, args.max_new_tokens)
    calibrator = MultiTensorCalibrator(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
    quantized_dir, quant_reports = calibrator.build_override_dir(
        overrides,
        target,
        "interactive_recursive_safe",
        prune_fraction=args.prune_fraction,
        overwrite=not args.resume,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    executor_config = ExecutorConfig(
        max_layers=args.max_layers,
        expert_cache_mb=args.expert_cache_mb,
        offload_kv_cache=True,
        use_quantized_overrides=False,
        execute_experts=not args.disable_experts,
        kv_cache_precision="fp16",
        dtype=torch.bfloat16,
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        top_k=args.top_k,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
    )
    runtime = GenerationRuntime(args.model_dir, args.support_dir, quantized_dir, executor_config, generation_config)
    baseline = runtime.generate(args.prompt, tokenizer, quantized=False)
    compressed = runtime.generate(args.prompt, tokenizer, quantized=True)
    comparison = compare_generations(baseline, compressed)
    drift = recursive_drift_series(baseline, compressed)
    token_stream = _token_stream(tokenizer, baseline, compressed, drift)
    pseudo_report = {"experiments": [{"results": [{"comparison": comparison, "quantized": compressed}]}]}
    kv_metrics = kv_survivability_metrics(pseudo_report, "recursive_safe_fp16_kv")
    pressure = pressure_from_generation(compressed)
    stable = (
        abs(float(comparison.get("perplexity_drift", 0.0))) <= args.stability_perplexity_drift
        and float(comparison.get("sequence_overlap", 0.0)) >= args.stability_sequence_overlap
    )
    promotions = []
    recoveries = []
    report = {
        "run_id": now_id(),
        "mode": "interactive_recursive_safe_inference",
        "warning": "Experimental streamed generation. Full-depth runs validate streamed runtime mechanics and local output quality, but this is still research software.",
        "prompt": args.prompt,
        "profile": "recursive_safe",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "quantized_dir": str(quantized_dir),
        "quantization_reports": quant_reports,
        "candidate_tensors": [item.to_json() for item in overrides],
        "config": {
            "max_layers": args.max_layers,
            "max_context_tokens": args.max_context_tokens,
            "max_new_tokens": args.max_new_tokens,
            "kv_cache_precision": "fp16",
            "expert_cache_mb": args.expert_cache_mb,
            "stability_perplexity_drift": args.stability_perplexity_drift,
            "stability_sequence_overlap": args.stability_sequence_overlap,
        },
        "cuda": cuda_snapshot(),
        "baseline": baseline,
        "compressed": compressed,
        "generated_text": compressed.get("generated_text", ""),
        "token_stream": token_stream,
        "comparison": comparison,
        "recursive_drift": drift,
        "kv_survivability": kv_metrics,
        "runtime_pressure": pressure,
        "recovery_events": recoveries,
        "tensor_promotion_events": promotions,
        "stable_without_recovery": stable and not recoveries and not promotions,
        "summary": {
            "coherence_note": _coherence_note(compressed.get("generated_text", "")),
            "first_token_latency_s": _first_token_latency(compressed),
            "nonfinite_logit_counts": [
                item.get("nonfinite_logit_count", 0)
                for item in compressed.get("token_logs", [])
                if "generated_token" in item
            ],
            "sequence_overlap": comparison.get("sequence_overlap"),
            "drift_velocity": drift.get("drift_velocity"),
            "instability_acceleration": drift.get("instability_acceleration"),
            "recovery_or_promotion_activity": bool(recoveries or promotions),
            "peak_vram_residency": compressed.get("runtime_stats", {}).get("vram_peak", {}),
            "tokens_per_second": round(len(compressed.get("generated_token_ids", [])) / max(float(compressed.get("latency_s", 0.0)), 1e-9), 6),
            "expert_routing_summary": compressed.get("runtime_stats", {}).get("router", {}),
            "transfer_bursts": pressure.get("transfer_burst_spikes", [])[:10],
        },
    }
    out = reports_dir / f"klcquant-interactive-{report['run_id']}.json"
    write_json(out, report)
    text_path = reports_dir / f"klcquant-interactive-{report['run_id']}.txt"
    text_path.write_text(compressed.get("generated_text", ""), encoding="utf-8")
    report["generated_text_path"] = str(text_path)
    write_json(out, report)
    return out


def _coherence_note(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return "empty output; not coherent"
    if len(stripped.split()) >= 2 or any(char in stripped for char in ".!?"):
        return "non-empty text fragment; inspect manually because this is a bounded layer-prefix runtime"
    return "very short fragment; bounded prefix may not produce assistant-like prose"


def _first_token_latency(run: dict[str, Any]) -> float | None:
    for item in run.get("token_logs", []):
        if "generated_token" in item:
            return item.get("token_latency_s")
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one interactive recursive-safe compressed inference probe")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--work-dir", default="calibration_runs/interactive")
    parser.add_argument("--prompt", default="Hello.")
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--max-layers", type=int, default=2)
    parser.add_argument("--max-context-tokens", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--prune-fraction", type=float, default=0.90)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stability-perplexity-drift", type=float, default=25.0)
    parser.add_argument("--stability-sequence-overlap", type=float, default=0.50)
    return parser


def main() -> None:
    out = run(build_parser().parse_args())
    print(f"wrote interactive inference report to {out}")


if __name__ == "__main__":
    main()
