from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, GenerationRuntime, compare_generations
from .interactive_inference_runner import DEFAULT_RECURSIVE_SAFE_TENSORS, _static_rows, _token_stream
from .kv_survivability import kv_survivability_metrics
from .multi_tensor_calibrator import MultiTensorCalibrator, TensorOverride
from .recursive_drift_tracker import recursive_drift_series
from .runtime_pressure_telemetry import pressure_from_generation
from .runtime_sensitivity_probe import _latest_scan
from .streamed_transformer_executor import ExecutorConfig
from .tensor_perturbation_runner import PerturbationTarget
from .utils import cuda_snapshot, now_id, read_json, write_json


def _format_prompt(tokenizer, prompt: str, prompt_format: str) -> str:
    if prompt_format == "raw":
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _decoded_top_tokens(tokenizer, run: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for idx, log in enumerate(item for item in run.get("token_logs", []) if "generated_token" in item):
        top = []
        for token_id, logit, logprob in zip(log.get("top_token_ids", [])[:10], log.get("top_logits", [])[:10], log.get("top_logprobs", [])[:10]):
            top.append(
                {
                    "token_id": int(token_id),
                    "token": tokenizer.decode([int(token_id)]),
                    "logit": float(logit),
                    "logprob": float(logprob),
                }
            )
        margin = None
        if len(top) >= 2:
            margin = top[0]["logit"] - top[1]["logit"]
        rows.append(
            {
                "token_index": idx,
                "generated_token": tokenizer.decode([int(log["generated_token"])]),
                "generated_token_id": int(log["generated_token"]),
                "final_hidden_norm_before_lm_head": log.get("final_hidden_norm_before_lm_head"),
                "final_hidden_mean_before_lm_head": log.get("final_hidden_mean_before_lm_head"),
                "final_hidden_std_before_lm_head": log.get("final_hidden_std_before_lm_head"),
                "final_hidden_max_abs_before_lm_head": log.get("final_hidden_max_abs_before_lm_head"),
                "final_hidden_finite": log.get("final_hidden_finite"),
                "final_hidden_nonfinite_count": log.get("final_hidden_nonfinite_count"),
                "top10": top,
                "logits_entropy": log.get("entropy"),
                "max_logit_margin": margin,
                "nonfinite_logit_count": log.get("nonfinite_logit_count"),
                "raw_nonfinite_logit_count": log.get("raw_nonfinite_logit_count"),
                "logits_finite": log.get("logits_finite"),
            }
        )
    return rows


def _norm_diagnostics(run: dict[str, Any]) -> dict[str, Any]:
    rows = []
    first_bad = None
    token_idx = -1
    for token_log in run.get("token_logs", []):
        if "generated_token" not in token_log:
            continue
        token_idx += 1
        for layer in token_log.get("layers", []):
            row = {
                "token_index": token_idx,
                "layer": layer.get("layer"),
                "input_norm": layer.get("input_norm"),
                "attention_norm": layer.get("attention_norm"),
                "post_attention_residual_norm": layer.get("post_attention_residual_norm"),
                "moe_norm": layer.get("moe_norm"),
                "output_norm": layer.get("output_norm"),
                "residual_delta_norm": layer.get("residual_delta_norm"),
            }
            rows.append(row)
            out = float(layer.get("output_norm", 0.0))
            if first_bad is None and (out < 1e-6 or out > 1e5 or not torch.isfinite(torch.tensor(out))):
                first_bad = row
    return {
        "first_layer_where_norm_explodes_or_collapses": first_bad,
        "max_output_norm": max((float(row["output_norm"]) for row in rows if row.get("output_norm") is not None), default=0.0),
        "min_output_norm": min((float(row["output_norm"]) for row in rows if row.get("output_norm") is not None), default=0.0),
        "layer_norm_rows": rows[:500],
    }


def _baseline_is_sane(run: dict[str, Any], decoded_top: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    reasons = []
    text = run.get("generated_text", "")
    ids = run.get("generated_token_ids", [])
    if len(set(ids)) <= 1 and len(ids) > 2:
        reasons.append("single-token repetition collapse")
    if any(row.get("nonfinite_logit_count", 0) or row.get("raw_nonfinite_logit_count", 0) for row in decoded_top):
        reasons.append("non-finite logits")
    if text.strip() in {"!" * len(text.strip()), "." * len(text.strip())} and len(text.strip()) > 2:
        reasons.append("punctuation-only collapse")
    return not reasons, reasons


def _make_runtime(args: argparse.Namespace, quantized_dir: str | Path) -> GenerationRuntime:
    return GenerationRuntime(
        args.model_dir,
        args.support_dir,
        quantized_dir,
        ExecutorConfig(
            max_layers=args.max_layers,
            expert_cache_mb=args.expert_cache_mb,
            offload_kv_cache=True,
            use_quantized_overrides=False,
            execute_experts=not args.disable_experts,
            kv_cache_precision="fp16",
            dtype=torch.bfloat16,
        ),
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            max_context_tokens=args.max_context_tokens,
            top_k=max(args.top_k, 10),
            lm_head_chunk_rows=args.lm_head_chunk_rows,
        ),
    )


def run(args: argparse.Namespace) -> Path:
    reports_dir = Path(args.out_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    importance_report = Path(args.importance_report) if args.importance_report else _latest_scan(reports_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    prompt_text = _format_prompt(tokenizer, args.prompt, args.prompt_format)
    runtime = _make_runtime(args, "quantized_model")
    baseline = runtime.generate(prompt_text, tokenizer, quantized=False)
    baseline_top = _decoded_top_tokens(tokenizer, baseline)
    baseline_norms = _norm_diagnostics(baseline)
    baseline_sane, sanity_reasons = _baseline_is_sane(baseline, baseline_top)

    comparison = None
    compressed = None
    compressed_top = None
    drift = None
    token_stream = None
    kv = None
    pressure = None
    q8_skipped_reason = None
    quantized_dir = None
    quant_reports: list[str] = []
    if baseline_sane or args.force_q8:
        candidates = args.candidate or DEFAULT_RECURSIVE_SAFE_TENSORS
        static = _static_rows(importance_report, candidates)
        overrides = [TensorOverride(name, "q8", static.get(name, {})) for name in candidates]
        calibrator = MultiTensorCalibrator(args.model_dir, args.support_dir, importance_report, args.work_dir, reports_dir)
        quantized_dir, quant_reports = calibrator.build_override_dir(
            overrides,
            PerturbationTarget(args.max_layers, args.max_new_tokens),
            "diagnostic_recursive_safe_q8",
            overwrite=not args.resume,
        )
        runtime_q = _make_runtime(args, quantized_dir)
        compressed = runtime_q.generate(prompt_text, tokenizer, quantized=True)
        comparison = compare_generations(baseline, compressed)
        drift = recursive_drift_series(baseline, compressed)
        token_stream = _token_stream(tokenizer, baseline, compressed, drift)
        pseudo = {"experiments": [{"results": [{"comparison": comparison, "quantized": compressed}]}]}
        kv = kv_survivability_metrics(pseudo, "diagnostic_recursive_safe_fp16_kv")
        pressure = pressure_from_generation(compressed)
        compressed_top = _decoded_top_tokens(tokenizer, compressed)
    else:
        q8_skipped_reason = "baseline failed sanity gate"

    report = {
        "run_id": now_id(),
        "mode": "full_depth_streamed_executor_diagnostic",
        "prompt": args.prompt,
        "prompt_format": args.prompt_format,
        "rendered_prompt": prompt_text,
        "prompt_token_ids": tokenizer(prompt_text, add_special_tokens=False)["input_ids"],
        "config": {
            "max_layers": args.max_layers,
            "max_new_tokens": args.max_new_tokens,
            "max_context_tokens": args.max_context_tokens,
            "baseline_first": True,
            "q8_comparison_requires_sane_baseline": not args.force_q8,
        },
        "cuda": cuda_snapshot(),
        "baseline_sane": baseline_sane,
        "baseline_sanity_reasons": sanity_reasons,
        "baseline": baseline,
        "baseline_top10_per_token": baseline_top,
        "baseline_norm_diagnostics": baseline_norms,
        "q8_skipped_reason": q8_skipped_reason,
        "quantized_dir": str(quantized_dir) if quantized_dir else None,
        "quantization_reports": quant_reports,
        "compressed": compressed,
        "compressed_top10_per_token": compressed_top,
        "comparison": comparison,
        "recursive_drift": drift,
        "token_stream": token_stream,
        "kv_survivability": kv,
        "runtime_pressure": pressure,
        "investigation_notes": {
            "final_norm_before_lm_head": "covered by baseline layer/final logits diagnostics; final norm tensor loaded from model.norm.weight",
            "attention_mask": "token-by-token causal cache; no future tokens are visible",
            "rope": "uses GPT-OSS split-half rotary form; YaRN scaling still simplified",
            "attention_sinks": "enabled in this diagnostic run",
            "residual_order": "pre-norm attention residual, then pre-norm MoE residual, matching Transformers GptOssDecoderLayer",
            "lm_head": "loaded from lm_head.weight; config tie_word_embeddings=false",
        },
    }
    out = reports_dir / f"klcquant-diagnostic-{report['run_id']}.json"
    write_json(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose full-depth streamed executor semantic collapse")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--importance-report")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--work-dir", default="calibration_runs/diagnostic")
    parser.add_argument("--prompt", default="Hello.")
    parser.add_argument("--prompt-format", default="raw", choices=["raw", "chat"])
    parser.add_argument("--candidate", action="append")
    parser.add_argument("--max-layers", type=int, default=36)
    parser.add_argument("--max-context-tokens", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--expert-cache-mb", type=int, default=512)
    parser.add_argument("--disable-experts", action="store_true")
    parser.add_argument("--force-q8", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    out = run(build_parser().parse_args())
    print(f"wrote executor diagnostic report to {out}")


if __name__ == "__main__":
    main()
