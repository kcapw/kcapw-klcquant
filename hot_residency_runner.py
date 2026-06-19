from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, GenerationRuntime, compare_generations
from .interactive_inference_runner import DEFAULT_RECURSIVE_SAFE_TENSORS, _static_rows
from .multi_tensor_calibrator import MultiTensorCalibrator, TensorOverride
from .prompt_formatting import format_prompt, prompt_format_reference
from .recursive_drift_tracker import recursive_drift_series
from .rope_validation_runner import _norm_summary, _runtime_summary, _sparse_logit_metrics
from .runtime_pressure_telemetry import pressure_from_generation
from .runtime_sensitivity_probe import _latest_scan
from .streamed_transformer_executor import ExecutorConfig
from .tensor_perturbation_runner import PerturbationTarget
from .utils import cuda_snapshot, now_id, write_json


DEFAULT_PROMPTS = [
    "Hello.",
    "What is 2+2?",
    "Write one sentence about space.",
    "Hi, I'm testing a streamed AI runtime. Please answer conversationally in two short sentences.",
]


def _make_runtime(args: argparse.Namespace, quantized_dir: str | Path) -> GenerationRuntime:
    executor_config = ExecutorConfig(
        max_layers=args.max_layers,
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
        routing_locality=args.routing_locality,
        sticky_routing_strength=args.sticky_routing_strength,
        sticky_routing_decay=args.sticky_routing_decay,
        routing_semantic_guard=args.routing_semantic_guard,
        sticky_candidate_margin=args.sticky_candidate_margin,
        max_sticky_bonus=args.max_sticky_bonus,
        min_raw_route_overlap=args.min_raw_route_overlap,
        max_hot_experts_per_layer=args.max_hot_experts_per_layer,
        active_experts_per_token_cap=args.active_experts_per_token_cap,
        routing_exploration_margin=args.routing_exploration_margin,
        cache_aware_routing_strength=args.cache_aware_routing_strength,
        predictive_expert_prefetch=args.predictive_expert_prefetch,
        expert_prefetch_limit=args.expert_prefetch_limit,
        expert_async_prefetch=args.expert_async_prefetch,
        routing_prediction_window=args.routing_prediction_window,
        routing_workload_window=args.routing_workload_window,
        dynamic_active_expert_cap=args.dynamic_active_expert_cap,
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
    return GenerationRuntime(args.model_dir, args.support_dir, quantized_dir, executor_config, generation_config)


def _assistant_style_score(text: str) -> dict[str, Any]:
    stripped = text.strip()
    lower = stripped.lower()
    conversational = any(piece in lower for piece in ("hello", "hi", "sure", "2+2", "4", "space"))
    completion_drift = any(piece in lower for piece in ("30-year-old", "persistent cough", "capital of france"))
    return {
        "non_empty": bool(stripped),
        "conversational_signal": conversational,
        "known_completion_drift_signal": completion_drift,
        "score": float(bool(stripped)) + float(conversational) - float(completion_drift),
    }


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# KLCQUANT Hot Residency Validation Summary",
        "",
        f"Run: `{path.name}`",
        "",
        "Configuration:",
        "",
        f"- Hot VRAM budget: `{report['config']['hot_vram_budget_mb']} MB`",
        f"- Expert cache: `{report['config']['expert_cache_mb']} MB`",
        f"- Prompt format: `{report['config']['prompt_format']}`",
        f"- Max new tokens: `{report['config']['max_new_tokens']}`",
        "",
        "## Results",
        "",
    ]
    for item in report["prompt_results"]:
        compressed_summary = item["compressed_summary"]
        hot = item["compressed"]["runtime_stats"]["hot_residency"]
        expert = compressed_summary["expert_cache"]
        comparison = item["comparison"]
        lines.extend(
            [
                f"### `{item['prompt']}`",
                "",
                "```text",
                compressed_summary.get("assistant_final_text") or compressed_summary["generated_text"],
                "```",
                "",
                f"- sequence overlap: `{comparison['sequence_overlap']}`",
                f"- first divergence token: `{comparison['first_divergence_index']}`",
                f"- perplexity drift: `{comparison['perplexity_drift']}`",
                f"- q8 tokens/sec: `{compressed_summary['tokens_per_second']}`",
                f"- q8 transfer: `{compressed_summary['transfer_mb']}` MB",
                f"- peak VRAM: `{compressed_summary['peak_vram'].get('allocated_gb')}` GB",
                f"- layer cache hit rate: `{hot['layer_tensor_cache_hit_rate']}`",
                f"- lm_head resident: `{hot['lm_head_resident']}`",
                f"- expert hit rate: `{expert.get('hit_rate')}`",
                f"- expert evictions: `{expert.get('evictions')}`",
                f"- assistant style score: `{item['assistant_style']['score']}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    candidates = args.candidate or DEFAULT_RECURSIVE_SAFE_TENSORS
    static = _static_rows(importance_report, candidates)
    overrides = [TensorOverride(name, "q8", static.get(name, {})) for name in candidates]
    calibrator = MultiTensorCalibrator(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
    quantized_dir, quant_reports = calibrator.build_override_dir(
        overrides,
        PerturbationTarget(args.max_layers, args.max_new_tokens),
        "hot_residency_q8",
        overwrite=not args.resume,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    runtime = _make_runtime(args, quantized_dir)
    prompt_results = []
    for prompt in args.prompt:
        rendered = format_prompt(tokenizer, prompt, args.prompt_format, args.system_prompt)
        baseline = runtime.generate(rendered, tokenizer, quantized=False)
        compressed = runtime.generate(rendered, tokenizer, quantized=True)
        comparison = compare_generations(baseline, compressed)
        drift = recursive_drift_series(baseline, compressed)
        prompt_results.append(
            {
                "prompt": prompt,
                "rendered_prompt": rendered,
                "format_reference": prompt_format_reference(tokenizer, prompt, args.system_prompt),
                "transformers_reference_generation": {
                    "status": "skipped",
                    "reason": "Full reference generate would require materializing the full GPT-OSS model; this streamed runtime intentionally avoids that on 16GB VRAM.",
                },
                "baseline": baseline,
                "compressed": compressed,
                "comparison": comparison,
                "recursive_drift": drift,
                "logit_divergence": _sparse_logit_metrics(baseline, compressed),
                "baseline_norm_growth": _norm_summary(baseline),
                "compressed_norm_growth": _norm_summary(compressed),
                "baseline_summary": _runtime_summary(baseline),
                "compressed_summary": _runtime_summary(compressed),
                "runtime_pressure": pressure_from_generation(compressed),
                "assistant_style": _assistant_style_score(compressed.get("assistant_final_text") or compressed.get("generated_text", "")),
            }
        )

    run_id = now_id()
    report = {
        "run_id": run_id,
        "mode": "hot_residency_streamed_generation_validation",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "quantized_dir": str(quantized_dir),
        "quantization_reports": quant_reports,
        "candidate_tensors": [item.to_json() for item in overrides],
        "config": {
            "max_layers": args.max_layers,
            "max_new_tokens": args.max_new_tokens,
            "max_context_tokens": args.max_context_tokens,
            "hot_vram_budget_mb": args.hot_vram_budget_mb,
            "expert_cache_mb": args.expert_cache_mb,
            "pin_lm_head": args.pin_lm_head,
            "pin_layer_tensors": args.pin_layer_tensors,
            "prompt_format": args.prompt_format,
            "kv_cache_precision": args.kv_cache_precision,
            "stop_token_ids": args.stop_token_id,
            "routing_locality": args.routing_locality,
            "sticky_routing_strength": args.sticky_routing_strength,
            "sticky_routing_decay": args.sticky_routing_decay,
            "routing_semantic_guard": args.routing_semantic_guard,
            "sticky_candidate_margin": args.sticky_candidate_margin,
            "max_sticky_bonus": args.max_sticky_bonus,
            "min_raw_route_overlap": args.min_raw_route_overlap,
            "max_hot_experts_per_layer": args.max_hot_experts_per_layer,
            "active_experts_per_token_cap": args.active_experts_per_token_cap,
            "routing_exploration_margin": args.routing_exploration_margin,
            "cache_aware_routing_strength": args.cache_aware_routing_strength,
            "predictive_expert_prefetch": args.predictive_expert_prefetch,
            "expert_prefetch_limit": args.expert_prefetch_limit,
            "expert_async_prefetch": args.expert_async_prefetch,
            "routing_prediction_window": args.routing_prediction_window,
            "routing_workload_window": args.routing_workload_window,
            "dynamic_active_expert_cap": args.dynamic_active_expert_cap,
            "min_active_experts_per_token": args.min_active_experts_per_token,
            "max_active_experts_per_token": args.max_active_experts_per_token,
        },
        "cuda": cuda_snapshot(),
        "prompt_results": prompt_results,
    }
    out = reports_dir / f"klcquant-hot-residency-{run_id}.json"
    write_json(out, report)
    _write_summary(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hot-residency streamed generation validation")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--work-dir", default="calibration_runs/hot_residency")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--max-layers", type=int, default=36)
    parser.add_argument("--max-context-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--hot-vram-budget-mb", type=int, default=8192)
    parser.add_argument("--expert-cache-mb", type=int, default=5120)
    parser.add_argument("--kv-cache-precision", default="fp16")
    parser.add_argument("--stop-token-id", action="append", type=int, default=[200002])
    parser.add_argument("--prompt-format", default="chat", choices=["raw", "chat"])
    parser.add_argument("--system-prompt", default="You are a helpful assistant. Answer directly and concisely.")
    parser.add_argument("--pin-lm-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-layer-tensors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--routing-locality", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sticky-routing-strength", type=float, default=0.35)
    parser.add_argument("--sticky-routing-decay", type=float, default=0.92)
    parser.add_argument("--routing-semantic-guard", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sticky-candidate-margin", type=float, default=0.50)
    parser.add_argument("--max-sticky-bonus", type=float, default=0.25)
    parser.add_argument("--min-raw-route-overlap", type=int, default=2)
    parser.add_argument("--max-hot-experts-per-layer", type=int, default=8)
    parser.add_argument("--active-experts-per-token-cap", type=int, default=0)
    parser.add_argument("--routing-exploration-margin", type=float, default=0.25)
    parser.add_argument("--cache-aware-routing-strength", type=float, default=0.08)
    parser.add_argument("--predictive-expert-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expert-prefetch-limit", type=int, default=4)
    parser.add_argument("--expert-async-prefetch", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--routing-prediction-window", type=int, default=16)
    parser.add_argument("--routing-workload-window", type=int, default=64)
    parser.add_argument("--dynamic-active-expert-cap", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--min-active-experts-per-token", type=int, default=2)
    parser.add_argument("--max-active-experts-per-token", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS
    out = run(args)
    print(f"wrote hot residency report to {out}")


if __name__ == "__main__":
    main()
