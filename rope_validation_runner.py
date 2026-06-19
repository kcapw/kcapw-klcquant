from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, GenerationRuntime, compare_generations
from .interactive_inference_runner import DEFAULT_RECURSIVE_SAFE_TENSORS, _static_rows
from .multi_tensor_calibrator import MultiTensorCalibrator, TensorOverride
from .recursive_drift_tracker import recursive_drift_series
from .rope_parity import rope_reference_parity
from .runtime_pressure_telemetry import pressure_from_generation
from .runtime_sensitivity_probe import _latest_scan
from .streamed_transformer_executor import ExecutorConfig
from .tensor_perturbation_runner import PerturbationTarget
from .utils import cuda_snapshot, now_id, read_json, write_json


DEFAULT_PROMPTS = [
    "Hello.",
    "What is 2+2?",
    "Write one sentence about space.",
]


def _gen_logs(run: dict[str, Any]) -> list[dict[str, Any]]:
    return [item for item in run.get("token_logs", []) if "generated_token" in item]


def _sparse_logit_metrics(baseline: dict[str, Any], quantized: dict[str, Any]) -> dict[str, float]:
    base_logs = _gen_logs(baseline)
    quant_logs = _gen_logs(quantized)
    cosines: list[float] = []
    max_abs: list[float] = []
    rel_l2s: list[float] = []
    for base_log, quant_log in zip(base_logs, quant_logs):
        base = {int(token): float(logit) for token, logit in zip(base_log.get("top_token_ids", []), base_log.get("top_logits", []))}
        quant = {int(token): float(logit) for token, logit in zip(quant_log.get("top_token_ids", []), quant_log.get("top_logits", []))}
        keys = sorted(set(base) | set(quant))
        if not keys:
            continue
        left = torch.tensor([base.get(key, 0.0) for key in keys], dtype=torch.float32)
        right = torch.tensor([quant.get(key, 0.0) for key in keys], dtype=torch.float32)
        denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
        cosines.append(float((torch.dot(left, right) / denom.clamp_min(1e-12)).item()))
        diff = left - right
        max_abs.append(float(diff.abs().max().item()))
        rel_l2s.append(float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(left).clamp_min(1e-12)).item()))
    return {
        "sparse_top_logit_cosine_mean": sum(cosines) / max(len(cosines), 1),
        "sparse_top_logit_max_abs_delta": max(max_abs, default=0.0),
        "sparse_top_logit_relative_l2_mean": sum(rel_l2s) / max(len(rel_l2s), 1),
    }


def _norm_summary(run: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for token_index, token_log in enumerate(_gen_logs(run)):
        for layer in token_log.get("layers", []):
            output_norm = float(layer.get("output_norm", 0.0))
            input_norm = float(layer.get("input_norm", 0.0))
            rows.append(
                {
                    "token_index": token_index,
                    "layer": int(layer.get("layer", -1)),
                    "input_norm": input_norm,
                    "output_norm": output_norm,
                    "residual_delta_norm": float(layer.get("residual_delta_norm", 0.0)),
                    "attention_norm": float(layer.get("attention_norm", 0.0)),
                    "moe_norm": float(layer.get("moe_norm", 0.0)),
                    "rope": layer.get("attention_diagnostics", {}),
                }
            )
    first_bad = next(
        (
            row
            for row in rows
            if row["output_norm"] < 1e-6 or row["output_norm"] > 1e5 or not torch.isfinite(torch.tensor(row["output_norm"]))
        ),
        None,
    )
    return {
        "max_output_norm": max((row["output_norm"] for row in rows), default=0.0),
        "min_output_norm": min((row["output_norm"] for row in rows), default=0.0),
        "first_layer_where_norm_explodes_or_collapses": first_bad,
        "sample_rows": rows[:72],
    }


def _runtime_summary(run: dict[str, Any]) -> dict[str, Any]:
    stats = run.get("runtime_stats", {})
    tokens = len(run.get("generated_token_ids", []))
    latency = float(run.get("latency_s", 0.0))
    return {
        "generated_text": run.get("generated_text", ""),
        "assistant_final_text": run.get("assistant_final_text", ""),
        "latency_s": latency,
        "tokens_per_second": tokens / max(latency, 1e-9),
        "peak_vram": stats.get("vram_peak", {}),
        "transfer_mb": stats.get("transfer", {}).get("mb", 0.0),
        "expert_cache": stats.get("expert_cache", {}),
    }


def _make_runtime(args: argparse.Namespace, quantized_dir: str | Path) -> GenerationRuntime:
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
    return GenerationRuntime(args.model_dir, args.support_dir, quantized_dir, executor_config, generation_config)


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# KLCQUANT YaRN RoPE Validation Summary",
        "",
        f"Run: `{path.name}`",
        "",
        "## RoPE Parity",
        "",
        f"- Passed: `{report['rope_parity']['passed']}`",
        f"- inv_freq max abs diff: `{report['rope_parity']['inv_freq_max_abs_diff']}`",
        f"- attention scaling abs diff: `{report['rope_parity']['attention_scaling_abs_diff']}`",
        "",
        "## Prompt Results",
        "",
    ]
    for item in report["prompt_results"]:
        summary = item["comparison"]
        compressed = item["compressed_summary"]
        drift = item["recursive_drift"]
        lines.extend(
            [
                f"### `{item['prompt']}`",
                "",
                "```text",
                compressed["generated_text"],
                "```",
                "",
                f"- sequence overlap: `{summary['sequence_overlap']}`",
                f"- perplexity drift: `{summary['perplexity_drift']}`",
                f"- first divergence token: `{summary['first_divergence_index']}`",
                f"- drift velocity: `{drift['drift_velocity']}`",
                f"- instability acceleration: `{drift['instability_acceleration']}`",
                f"- sparse top-logit relative L2 mean: `{item['logit_divergence']['sparse_top_logit_relative_l2_mean']}`",
                f"- baseline tokens/sec: `{item['baseline_summary']['tokens_per_second']}`",
                f"- q8 tokens/sec: `{compressed['tokens_per_second']}`",
                f"- q8 peak VRAM: `{compressed['peak_vram'].get('allocated_gb')}` GB",
                f"- q8 transfer: `{compressed['transfer_mb']}` MB",
                f"- q8 expert hit rate: `{compressed['expert_cache'].get('hit_rate')}`",
                "",
            ]
        )
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    model_config = read_json(Path(args.support_dir) / "config.json")
    rope_parity = rope_reference_parity(model_config)

    candidates = args.candidate or DEFAULT_RECURSIVE_SAFE_TENSORS
    static = _static_rows(importance_report, candidates)
    overrides = [TensorOverride(name, "q8", static.get(name, {})) for name in candidates]
    calibrator = MultiTensorCalibrator(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
    quantized_dir, quant_reports = calibrator.build_override_dir(
        overrides,
        PerturbationTarget(args.max_layers, args.max_new_tokens),
        "rope_yarn_q8",
        overwrite=not args.resume,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    runtime = _make_runtime(args, quantized_dir)
    prompt_results = []
    for prompt in args.prompt:
        baseline = runtime.generate(prompt, tokenizer, quantized=False)
        compressed = runtime.generate(prompt, tokenizer, quantized=True)
        comparison = compare_generations(baseline, compressed)
        drift = recursive_drift_series(baseline, compressed)
        prompt_results.append(
            {
                "prompt": prompt,
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
            }
        )

    run_id = now_id()
    report = {
        "run_id": run_id,
        "mode": "gpt_oss_yarn_rope_parity_and_generation_validation",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "quantized_dir": str(quantized_dir),
        "quantization_reports": quant_reports,
        "candidate_tensors": [item.to_json() for item in overrides],
        "config": {
            "max_layers": args.max_layers,
            "max_new_tokens": args.max_new_tokens,
            "max_context_tokens": args.max_context_tokens,
            "top_k": args.top_k,
            "expert_cache_mb": args.expert_cache_mb,
        },
        "cuda": cuda_snapshot(),
        "rope_parity": rope_parity,
        "prompt_results": prompt_results,
    }
    out = reports_dir / f"klcquant-rope-validation-{run_id}.json"
    write_json(out, report)
    _write_summary(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate GPT-OSS YaRN RoPE parity and run streamed q8 generation probes")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--work-dir", default="calibration_runs/rope_validation")
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--max-layers", type=int, default=36)
    parser.add_argument("--max-context-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.prompt is None:
        args.prompt = DEFAULT_PROMPTS
    out = run(args)
    print(f"wrote RoPE validation report to {out}")


if __name__ == "__main__":
    main()
