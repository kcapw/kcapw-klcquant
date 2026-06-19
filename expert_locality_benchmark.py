from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, GenerationRuntime
from .prompt_formatting import format_prompt
from .streamed_transformer_executor import ExecutorConfig
from .utils import cuda_snapshot, now_id, read_json, write_json


DEFAULT_PROMPTS = [
    "Hello.",
    "What is 2+2?",
    "Write one sentence about space.",
]


def _make_runtime(args: argparse.Namespace, locality: bool) -> GenerationRuntime:
    model_config = read_json(Path(args.support_dir) / "config.json")
    executor_config = ExecutorConfig(
        max_layers=args.max_layers or int(model_config["num_hidden_layers"]),
        expert_cache_mb=args.expert_cache_mb,
        offload_kv_cache=True,
        use_quantized_overrides=False,
        execute_experts=not args.disable_experts,
        kv_cache_precision=args.kv_cache_precision,
        dtype=torch.bfloat16,
        hot_residency=True,
        hot_vram_budget_mb=args.hot_vram_budget_mb,
        pin_lm_head=args.pin_lm_head,
        pin_layer_tensors=args.pin_layer_tensors,
        routing_locality=locality,
        sticky_routing_strength=args.sticky_routing_strength,
        sticky_routing_decay=args.sticky_routing_decay,
        max_hot_experts_per_layer=args.max_hot_experts_per_layer,
        active_experts_per_token_cap=args.active_experts_per_token_cap if locality else 0,
        routing_exploration_margin=args.routing_exploration_margin,
        cache_aware_routing_strength=args.cache_aware_routing_strength if locality else 0.0,
        predictive_expert_prefetch=args.predictive_expert_prefetch if locality else False,
        expert_prefetch_limit=args.expert_prefetch_limit,
        expert_async_prefetch=args.expert_async_prefetch if locality else False,
        routing_prediction_window=args.routing_prediction_window,
        routing_workload_window=args.routing_workload_window,
        dynamic_active_expert_cap=args.dynamic_active_expert_cap if locality else False,
        min_active_experts_per_token=args.min_active_experts_per_token,
        max_active_experts_per_token=args.max_active_experts_per_token,
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        max_context_tokens=args.max_context_tokens,
        top_k=args.top_k,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
        stop_token_ids=args.stop_token_id,
    )
    return GenerationRuntime(args.model_dir, args.support_dir, args.quantized_dir, executor_config, generation_config)


def _run_profile(args: argparse.Namespace, tokenizer, locality: bool) -> list[dict[str, Any]]:
    runtime = _make_runtime(args, locality)
    rows = []
    for prompt in args.prompt:
        rendered = format_prompt(tokenizer, prompt, args.prompt_format, args.system_prompt)
        generation = runtime.generate(rendered, tokenizer, quantized=False)
        stats = generation["runtime_stats"]
        router = stats.get("router", {})
        locality_stats = router.get("locality", {})
        expert = stats.get("expert_cache", {})
        tokens = len(generation.get("generated_token_ids", []))
        latency = float(generation.get("latency_s", 0.0))
        rows.append(
            {
                "prompt": prompt,
                "rendered_prompt": rendered,
                "generated_text": generation.get("generated_text", ""),
                "assistant_final_text": generation.get("assistant_final_text", ""),
                "generated_tokens": tokens,
                "latency_s": latency,
                "tokens_per_second": tokens / max(latency, 1e-9),
                "transfer_mb": stats.get("transfer", {}).get("mb", 0.0),
                "peak_vram": stats.get("vram_peak", {}),
                "expert_cache": expert,
                "routing_locality": locality_stats,
                "hot_residency": stats.get("hot_residency", {}),
                "expert_store": stats.get("expert_store", {}),
                "route_events_sample": router.get("route_events", [])[:100],
            }
        )
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "avg_tokens_per_second": sum(float(row["tokens_per_second"]) for row in rows) / max(len(rows), 1),
        "avg_transfer_mb": sum(float(row["transfer_mb"]) for row in rows) / max(len(rows), 1),
        "avg_expert_hit_rate": sum(float(row["expert_cache"].get("hit_rate", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_evictions": sum(int(row["expert_cache"].get("evictions", 0)) for row in rows) / max(len(rows), 1),
        "avg_materialization_ms": sum(float(row["expert_cache"].get("avg_materialization_time_ms", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_dequant_ms": sum(float(row["expert_cache"].get("avg_dequant_time_ms", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_blocked_on_prefetch_s": sum(float(row["expert_cache"].get("blocked_on_prefetch_s", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_prefetch_use_rate": sum(float(row["expert_cache"].get("prefetch_use_rate", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_evictions_per_1k_routed_tokens": sum(
            float(row["routing_locality"].get("evictions_per_1k_routed_tokens", 0.0)) for row in rows
        )
        / max(len(rows), 1),
        "avg_reuse_density": sum(float(row["routing_locality"].get("reuse_density", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_routing_entropy": sum(float(row["routing_locality"].get("routing_entropy_mean", 0.0)) for row in rows) / max(len(rows), 1),
        "avg_prediction_recall": sum(
            float((row["routing_locality"].get("prediction_accuracy") or {}).get("mean_recall") or 0.0) for row in rows
        )
        / max(len(rows), 1),
        "thrash_prompts": [row["prompt"] for row in rows if row["routing_locality"].get("expert_thrash_mode")],
    }


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    before = report["profiles"]["baseline_lru"]["summary"]
    after = report["profiles"]["sticky_locality"]["summary"]
    lines = [
        "# KLCQUANT Expert Locality Benchmark",
        "",
        f"Run: `{path.name}`",
        "",
        "## Aggregate",
        "",
        f"- baseline tokens/sec: `{before['avg_tokens_per_second']}`",
        f"- sticky tokens/sec: `{after['avg_tokens_per_second']}`",
        f"- baseline avg reuse density: `{before['avg_reuse_density']}`",
        f"- sticky avg reuse density: `{after['avg_reuse_density']}`",
        f"- baseline avg evictions/1k routed tokens: `{before['avg_evictions_per_1k_routed_tokens']}`",
        f"- sticky avg evictions/1k routed tokens: `{after['avg_evictions_per_1k_routed_tokens']}`",
        f"- baseline avg routing entropy: `{before['avg_routing_entropy']}`",
        f"- sticky avg routing entropy: `{after['avg_routing_entropy']}`",
        f"- sticky prediction recall: `{after['avg_prediction_recall']}`",
        f"- sticky prefetch use rate: `{after['avg_prefetch_use_rate']}`",
        f"- sticky avg materialization ms: `{after['avg_materialization_ms']}`",
        f"- sticky blocked-on-prefetch seconds: `{after['avg_blocked_on_prefetch_s']}`",
        f"- baseline thrash prompts: `{before['thrash_prompts']}`",
        f"- sticky thrash prompts: `{after['thrash_prompts']}`",
        "",
        "## Prompt Results",
        "",
    ]
    for base, sticky in zip(report["profiles"]["baseline_lru"]["runs"], report["profiles"]["sticky_locality"]["runs"]):
        lines.extend(
            [
                f"### `{base['prompt']}`",
                "",
                f"- baseline tokens/sec: `{base['tokens_per_second']}`",
                f"- sticky tokens/sec: `{sticky['tokens_per_second']}`",
                f"- baseline expert hit rate: `{base['expert_cache'].get('hit_rate')}`",
                f"- sticky expert hit rate: `{sticky['expert_cache'].get('hit_rate')}`",
                f"- sticky prediction accuracy: `{sticky['routing_locality'].get('prediction_accuracy')}`",
                f"- sticky prefetch use rate: `{sticky['expert_cache'].get('prefetch_use_rate')}`",
                f"- sticky blocked on expert availability: `{sticky['expert_cache'].get('blocked_on_prefetch_s')}` s",
                f"- sticky avg materialization: `{sticky['expert_cache'].get('avg_materialization_time_ms')}` ms",
                f"- baseline reuse distance: `{base['routing_locality'].get('reuse_distance')}`",
                f"- sticky reuse distance: `{sticky['routing_locality'].get('reuse_distance')}`",
                f"- baseline active set final: `{(base['routing_locality'].get('active_expert_set_size_over_time') or [{}])[-1].get('active_expert_count')}`",
                f"- sticky active set final: `{(sticky['routing_locality'].get('active_expert_set_size_over_time') or [{}])[-1].get('active_expert_count')}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    baseline = _run_profile(args, tokenizer, locality=False)
    sticky = _run_profile(args, tokenizer, locality=True)
    report = {
        "run_id": now_id(),
        "mode": "expert_locality_before_after_benchmark",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "cuda": cuda_snapshot(),
        "config": vars(args),
        "profiles": {
            "baseline_lru": {"summary": _aggregate(baseline), "runs": baseline},
            "sticky_locality": {"summary": _aggregate(sticky), "runs": sticky},
        },
    }
    out = reports_dir / f"klcquant-expert-locality-{report['run_id']}.json"
    write_json(out, report)
    _write_summary(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark expert routing locality before/after sticky routing")
    parser.add_argument("--model-dir", default="/mnt/ramdisk")
    parser.add_argument("--support-dir", default="/mnt/ramdisk")
    parser.add_argument("--quantized-dir", default="quantized_model")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--max-layers", type=int, default=0)
    parser.add_argument("--max-context-tokens", type=int, default=192)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--hot-vram-budget-mb", type=int, default=8192)
    parser.add_argument("--expert-cache-mb", type=int, default=4096)
    parser.add_argument("--kv-cache-precision", default="fp16")
    parser.add_argument("--stop-token-id", action="append", type=int, default=[200002])
    parser.add_argument("--prompt-format", default="chat", choices=["raw", "chat"])
    parser.add_argument("--system-prompt", default="You are a helpful assistant. Answer directly and concisely.")
    parser.add_argument("--pin-lm-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-layer-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--sticky-routing-strength", type=float, default=0.35)
    parser.add_argument("--sticky-routing-decay", type=float, default=0.92)
    parser.add_argument("--max-hot-experts-per-layer", type=int, default=8)
    parser.add_argument("--active-experts-per-token-cap", type=int, default=0)
    parser.add_argument("--routing-exploration-margin", type=float, default=0.25)
    parser.add_argument("--cache-aware-routing-strength", type=float, default=0.08)
    parser.add_argument("--predictive-expert-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expert-prefetch-limit", type=int, default=4)
    parser.add_argument("--expert-async-prefetch", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--routing-prediction-window", type=int, default=16)
    parser.add_argument("--routing-workload-window", type=int, default=64)
    parser.add_argument("--dynamic-active-expert-cap", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-active-experts-per-token", type=int, default=2)
    parser.add_argument("--max-active-experts-per-token", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS
    out = run(args)
    print(f"wrote expert locality benchmark to {out}")


if __name__ == "__main__":
    main()
