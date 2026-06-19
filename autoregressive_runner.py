from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, GenerationRuntime, compare_generations
from .prompt_generator import generate_prompts
from .stability_guard import StabilityGuard, StabilityThresholds
from .streamed_transformer_executor import ExecutorConfig
from .utils import cuda_snapshot, now_id, read_json, write_json


def _load_prompts(path: str | None, count: int) -> list[dict]:
    if path and Path(path).exists():
        return read_json(path)[:count]
    wanted = {"coding", "reasoning", "math", "memory"}
    prompts = [p.__dict__ for p in generate_prompts(max(count * 3, 12)) if p.domain in wanted]
    return prompts[:count]


def _parse_int_list(value: str | None, default: list[int]) -> list[int]:
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _run_single(args: argparse.Namespace, tokenizer, prompts: list[dict], max_layers: int, max_new_tokens: int) -> dict:
    executor_config = ExecutorConfig(
        max_layers=max_layers,
        expert_cache_mb=args.expert_cache_mb,
        offload_kv_cache=args.offload_kv_cache,
        use_quantized_overrides=False,
        execute_experts=not args.disable_experts,
        kv_cache_precision=args.kv_cache_precision,
        dtype=torch.bfloat16,
    )
    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        top_k=args.top_k,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
    )
    runtime = GenerationRuntime(args.model_dir, args.support_dir, args.quantized_dir, executor_config, gen_config)
    guard = StabilityGuard(
        StabilityThresholds(
            max_abs_perplexity_drift=args.stop_perplexity_drift,
            min_sequence_overlap=args.stop_sequence_overlap,
            min_token_jaccard=args.stop_sequence_overlap,
        )
    )
    results = []
    for item in prompts:
        baseline = runtime.generate(item["prompt"], tokenizer, quantized=False)
        quantized = runtime.generate(item["prompt"], tokenizer, quantized=True)
        comparison = compare_generations(baseline, quantized)
        results.append(
            {
                "id": item.get("id"),
                "domain": item.get("domain"),
                "prompt": item["prompt"],
                "baseline": baseline,
                "quantized": quantized,
                "comparison": comparison,
                "stability": guard.check(comparison),
            }
        )

    summary = {
        "prompt_count": len(results),
        "max_layers": max_layers,
        "max_new_tokens": max_new_tokens,
        "avg_token_jaccard": sum(r["comparison"]["token_jaccard"] for r in results) / max(len(results), 1),
        "avg_sequence_overlap": sum(r["comparison"]["sequence_overlap"] for r in results) / max(len(results), 1),
        "avg_latency_baseline_s": sum(r["baseline"]["latency_s"] for r in results) / max(len(results), 1),
        "avg_latency_quantized_s": sum(r["quantized"]["latency_s"] for r in results) / max(len(results), 1),
        "avg_perplexity_drift": sum(r["comparison"]["perplexity_drift"] for r in results) / max(len(results), 1),
        "avg_baseline_transfer_mb": sum(r["baseline"]["runtime_stats"]["transfer"]["mb"] for r in results) / max(len(results), 1),
        "avg_quantized_transfer_mb": sum(r["quantized"]["runtime_stats"]["transfer"]["mb"] for r in results) / max(len(results), 1),
        "avg_baseline_transfer_bandwidth_mb_s": sum(
            r["baseline"]["runtime_stats"]["transfer"]["mb"] / max(r["baseline"]["latency_s"], 1e-9) for r in results
        )
        / max(len(results), 1),
        "avg_quantized_transfer_bandwidth_mb_s": sum(
            r["quantized"]["runtime_stats"]["transfer"]["mb"] / max(r["quantized"]["latency_s"], 1e-9) for r in results
        )
        / max(len(results), 1),
        "avg_quantized_expert_cache_hit_rate": sum(
            r["quantized"]["runtime_stats"]["expert_cache"]["hit_rate"] for r in results
        )
        / max(len(results), 1),
        "avg_quantized_expert_evictions": sum(r["quantized"]["runtime_stats"]["expert_cache"]["evictions"] for r in results)
        / max(len(results), 1),
        "avg_quantized_expert_loaded_mb": sum(r["quantized"]["runtime_stats"]["expert_cache"]["loaded_mb"] for r in results)
        / max(len(results), 1),
        "max_baseline_vram_gb": max((r["baseline"]["runtime_stats"]["vram_peak"].get("allocated_gb", 0.0) for r in results), default=0.0),
        "max_quantized_vram_gb": max((r["quantized"]["runtime_stats"]["vram_peak"].get("allocated_gb", 0.0) for r in results), default=0.0),
    }
    return {"summary": summary, "results": results}


def _sensitivity_ranking(experiments: list[dict]) -> list[dict]:
    scores: dict[str, dict] = {}
    for experiment in experiments:
        for result in experiment.get("results", []):
            divergence = result["comparison"].get("first_divergence_index") is not None
            drift = abs(float(result["comparison"].get("perplexity_drift", 0.0)))
            for token_log in result["quantized"].get("token_logs", []):
                for layer in token_log.get("layers", []):
                    for name in layer.get("quantized_overrides", []):
                        row = scores.setdefault(name, {"tensor": name, "uses": 0, "divergent_uses": 0, "drift_sum": 0.0})
                        row["uses"] += 1
                        row["drift_sum"] += drift
                        if divergence:
                            row["divergent_uses"] += 1
    ranked = []
    for row in scores.values():
        row["avg_abs_perplexity_drift_when_used"] = row["drift_sum"] / max(row["uses"], 1)
        row["sensitivity_score"] = row["divergent_uses"] + row["avg_abs_perplexity_drift_when_used"]
        ranked.append(row)
    return sorted(ranked, key=lambda x: x["sensitivity_score"], reverse=True)


def _write_progression_charts(report: dict, out_prefix: Path) -> list[str]:
    experiments = report.get("experiments", [])
    if not experiments:
        return []
    labels = [f"L{s['summary']['max_layers']}/T{s['summary']['max_new_tokens']}" for s in experiments]
    paths: list[str] = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, [e["summary"]["avg_sequence_overlap"] for e in experiments], marker="o", label="sequence overlap")
    ax.plot(labels, [e["summary"]["avg_token_jaccard"] for e in experiments], marker="o", label="token jaccard")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Generation Agreement")
    ax.set_xlabel("experiment")
    ax.legend()
    fig.tight_layout()
    path = out_prefix.with_suffix(".agreement.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, [e["summary"]["avg_perplexity_drift"] for e in experiments], marker="o")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Perplexity Drift")
    ax.set_xlabel("experiment")
    fig.tight_layout()
    path = out_prefix.with_suffix(".perplexity.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, [e["summary"]["max_quantized_vram_gb"] for e in experiments], marker="o", label="peak VRAM GB")
    ax.plot(labels, [e["summary"]["avg_quantized_transfer_mb"] / 1024 for e in experiments], marker="o", label="transfer GB")
    ax.set_title("Residency And Transfer")
    ax.set_xlabel("experiment")
    ax.legend()
    fig.tight_layout()
    path = out_prefix.with_suffix(".residency.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(labels, [e["summary"]["avg_quantized_expert_cache_hit_rate"] for e in experiments], marker="o", label="expert hit rate")
    ax.plot(labels, [e["summary"]["avg_quantized_expert_evictions"] for e in experiments], marker="o", label="expert evictions")
    ax.set_title("Expert Cache Efficiency")
    ax.set_xlabel("experiment")
    ax.legend()
    fig.tight_layout()
    path = out_prefix.with_suffix(".expert_cache.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def run(args: argparse.Namespace) -> Path:
    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    prompts = _load_prompts(args.prompts, args.prompt_count)
    layer_depths = _parse_int_list(args.layer_depths, [args.max_layers])
    token_counts = _parse_int_list(args.token_counts, [args.max_new_tokens])
    experiments = []
    stopped = False
    stop_reason = None
    for depth in layer_depths:
        for tokens in token_counts:
            item = _run_single(args, tokenizer, prompts, depth, tokens)
            experiments.append(item)
            if abs(item["summary"]["avg_perplexity_drift"]) > args.stop_perplexity_drift:
                stopped = True
                stop_reason = f"perplexity drift exceeded {args.stop_perplexity_drift} at layers={depth}, tokens={tokens}"
                break
            if item["summary"]["avg_sequence_overlap"] < args.stop_sequence_overlap:
                stopped = True
                stop_reason = f"sequence overlap fell below {args.stop_sequence_overlap} at layers={depth}, tokens={tokens}"
                break
        if stopped:
            break

    latest = experiments[-1] if experiments else {"summary": {}, "results": []}
    report = {
        "run_id": f"{now_id()}-{os.getpid()}",
        "mode": "streamed_autoregressive_generation_progression",
        "warning": "Executes real token-by-token streamed attention/router/MXFP4 expert blocks for a bounded layer prefix. This is still not full 120B equivalence until full depth and longer generations are validated.",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "quantized_dir": args.quantized_dir,
        "cuda": cuda_snapshot(),
        "config": {
            "layer_depths": layer_depths,
            "token_counts": token_counts,
            "max_context_tokens": args.max_context_tokens,
            "lm_head_chunk_rows": args.lm_head_chunk_rows,
            "offload_kv_cache": args.offload_kv_cache,
            "expert_cache_mb": args.expert_cache_mb,
            "execute_experts": not args.disable_experts,
            "kv_cache_precision": args.kv_cache_precision,
            "stop_perplexity_drift": args.stop_perplexity_drift,
            "stop_sequence_overlap": args.stop_sequence_overlap,
        },
        "stopped_early": stopped,
        "stop_reason": stop_reason,
        "summary": latest["summary"],
        "tensor_sensitivity_ranking": _sensitivity_ranking(experiments),
        "experiments": experiments,
    }
    out = Path(args.out_dir) / f"klcquant-autoreg-{report['run_id']}.json"
    charts = _write_progression_charts(report, out.with_suffix(""))
    report["charts"] = charts
    write_json(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run minimal streamed autoregressive generation")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--quantized-dir", default="quantized_model")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--prompts")
    parser.add_argument("--prompt-count", type=int, default=1)
    parser.add_argument("--max-layers", type=int, default=1)
    parser.add_argument("--layer-depths", help="Comma-separated progression, e.g. 2,4,8,16.")
    parser.add_argument("--max-context-tokens", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--token-counts", help="Comma-separated progression, e.g. 1,4,8,16.")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--offload-kv-cache", action="store_true")
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--kv-cache-precision", default="fp16", choices=["fp16", "bf16", "q8", "q4", "q2", "q1"])
    parser.add_argument("--stop-perplexity-drift", type=float, default=100.0)
    parser.add_argument("--stop-sequence-overlap", type=float, default=0.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = run(args)
    print(f"wrote autoregressive report to {out}")


if __name__ == "__main__":
    main()
