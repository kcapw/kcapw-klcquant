from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .prompt_generator import generate_prompts, save_prompts
from .utils import cuda_snapshot, now_id, read_json, write_json


def _load_prompts(path: str | None, count: int) -> list[dict]:
    if path and Path(path).exists():
        data = read_json(path)
        return data[:count]
    records = generate_prompts(count)
    return [r.__dict__ for r in records]


def run_scan(args: argparse.Namespace) -> Path:
    from .adaptive_quantizer import AdaptiveQuantizer
    from .streamed_loader import StreamedTensorLoader
    from .tensor_profiler import StaticScanConfig, TensorProfiler

    profiler = TensorProfiler()
    if args.cuda_stream:
        loader = StreamedTensorLoader(args.model_dir, args.support_dir, device="cuda", dtype=torch.float16)
        ranked, vram = profiler.cuda_stream_scan(loader)
    else:
        ranked = profiler.scan_safetensors(
            args.model_dir,
            args.support_dir,
            StaticScanConfig(sample_values=args.sample_values, load_to_cuda=False),
        )
        vram = [cuda_snapshot()]

    decisions = AdaptiveQuantizer().plan(ranked)
    original = sum(d.original_bytes for d in decisions)
    estimated = sum(d.estimated_bytes for d in decisions)
    report = {
        "run_id": now_id(),
        "mode": "cuda_stream_scan" if args.cuda_stream else "static_scan",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "cuda": cuda_snapshot(),
        "summary": {
            "tensor_count": len(ranked),
            "original_bytes": original,
            "estimated_quantized_bytes": estimated,
            "estimated_compression_ratio": round(original / max(estimated, 1), 4),
        },
        "vram_usage": vram,
        "tensor_importance_rankings": [s.to_json() for s in ranked],
        "least_used_tensors": [s.to_json() for s in sorted(ranked, key=lambda s: s.importance_score)[: args.top_k]],
        "universally_important_tensors": [s.to_json() for s in ranked[: args.top_k]],
        "quantization_plan": [d.to_json() for d in decisions],
    }
    out = Path(args.out_dir) / f"klcquant-scan-{report['run_id']}.json"
    write_json(out, report)
    return out


def run_hf_profile(args: argparse.Namespace) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .adaptive_quantizer import AdaptiveQuantizer
    from .tensor_profiler import TensorProfiler, measure_generation

    prompts = _load_prompts(args.prompts, args.prompt_count)
    profiler = TensorProfiler()
    model_path = args.hf_model_path or args.support_dir
    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    profiler.attach_hooks(model)
    generation = measure_generation(model, tokenizer, prompts, args.max_new_tokens)
    profiler.detach_hooks()
    ranked = profiler.merge_hook_stats(len(prompts))
    decisions = AdaptiveQuantizer().plan(ranked)
    report = {
        "run_id": now_id(),
        "mode": "hf_profile",
        "prompt_count": len(prompts),
        "generation": generation,
        "cuda": cuda_snapshot(),
        "tensor_importance_rankings": [s.to_json() for s in ranked],
        "quantization_plan": [d.to_json() for d in decisions],
        "note": "HF profile requires a transformers build that supports this architecture. Use scan mode for shard-only analysis.",
    }
    out = Path(args.out_dir) / f"klcquant-hf-profile-{report['run_id']}.json"
    write_json(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="KLCQUANT adaptive tensor importance benchmark runner")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--prompts")
    parser.add_argument("--prompt-count", type=int, default=1000)
    parser.add_argument("--top-k", type=int, default=50)
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate-prompts")
    gen.add_argument("--count", type=int, default=1000)
    gen.add_argument("--out", default="prompts/generated_prompts.json")

    scan = sub.add_parser("scan")
    scan.add_argument("--cuda-stream", action="store_true", help="Load one tensor group to CUDA at a time.")
    scan.add_argument("--sample-values", type=int, default=4096)

    hf = sub.add_parser("hf-profile")
    hf.add_argument("--hf-model-path", help="Optional full HF model path. Defaults to support dir.")
    hf.add_argument("--max-new-tokens", type=int, default=32)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    if args.command == "generate-prompts":
        save_prompts(args.out, args.count)
        print(f"wrote prompts to {args.out}")
        return
    if args.command == "scan":
        out = run_scan(args)
    elif args.command == "hf-profile":
        out = run_hf_profile(args)
    else:
        raise SystemExit(f"unknown command {args.command}")
    print(f"wrote report to {out}")


if __name__ == "__main__":
    main()
