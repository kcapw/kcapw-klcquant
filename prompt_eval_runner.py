from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from transformers import AutoTokenizer

from .prompt_generator import generate_prompts
from .streamed_inference import StreamedInferenceEngine, StreamedProbeConfig
from .utils import cuda_snapshot, now_id, read_json, write_json


DEFAULT_DOMAINS = {"coding", "reasoning", "multilingual", "long_context", "math", "memory"}


def load_eval_prompts(path: str | None, count: int, domains: set[str]) -> list[dict]:
    if path and Path(path).exists():
        data = read_json(path)
    else:
        data = [item.__dict__ for item in generate_prompts(max(count * 3, 24))]
    filtered = [item for item in data if item.get("domain") in domains]
    by_domain: dict[str, list[dict]] = defaultdict(list)
    for item in filtered:
        by_domain[item.get("domain")].append(item)
    selected: list[dict] = []
    for domain in sorted(domains):
        if by_domain.get(domain) and len(selected) < count:
            selected.append(by_domain[domain].pop(0))
    for item in filtered:
        if len(selected) >= count:
            break
        if item not in selected:
            selected.append(item)
    return selected[:count]


def summarize(results: list[dict]) -> dict:
    if not results:
        return {}
    sims = [item["similarity"] for item in results]
    drift = [item["similarity"]["logit_drift"] for item in results]
    cache = results[-1]["cache"]
    return {
        "prompt_count": len(results),
        "avg_token_jaccard": sum(s["token_jaccard"] for s in sims) / len(sims),
        "avg_sequence_overlap": sum(s["sequence_overlap"] for s in sims) / len(sims),
        "avg_logit_relative_l2": sum(d["relative_l2"] for d in drift) / len(drift),
        "avg_logit_cosine": sum(d["cosine"] for d in drift) / len(drift),
        "avg_perplexity_drift": sum(s["perplexity_drift"] for s in sims) / len(sims),
        "avg_latency_s": sum(item["latency_s"] for item in results) / len(results),
        "cache_hit_rate": cache["hit_rate"],
    }


def write_charts(report: dict, out_prefix: Path) -> list[str]:
    paths: list[str] = []
    results = report["results"]

    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(range(len(results)))
    ax.plot(xs, [r["similarity"]["logit_drift"]["relative_l2"] for r in results], label="logit relative L2")
    ax.plot(xs, [1.0 - r["similarity"]["token_jaccard"] for r in results], label="1 - token jaccard")
    ax.set_xlabel("prompt")
    ax.set_title("Compression vs Probe Quality")
    ax.legend()
    fig.tight_layout()
    path = out_prefix.with_suffix(".quality.png")
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    events = report.get("vram_events", [])
    if events:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([e["seconds"] for e in events], [e.get("allocated_gb", 0.0) for e in events], label="allocated GB")
        ax.plot([e["seconds"] for e in events], [e.get("reserved_gb", 0.0) for e in events], label="reserved GB")
        ax.set_xlabel("seconds")
        ax.set_title("VRAM Residency")
        ax.legend()
        fig.tight_layout()
        path = out_prefix.with_suffix(".vram.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))

    hotness: dict[str, float] = defaultdict(float)
    for result in results:
        for row in result["tensor_hotness"]:
            hotness[row["name"]] += row["quantized_delta_norm"]
    top = sorted(hotness.items(), key=lambda x: x[1], reverse=True)[:20]
    if top:
        fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.28)))
        labels = [name.split("model.layers.")[-1] for name, _ in top]
        ax.barh(labels, [value for _, value in top])
        ax.invert_yaxis()
        ax.set_title("Runtime Tensor Hotness Map")
        fig.tight_layout()
        path = out_prefix.with_suffix(".hotness.png")
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def run_eval(args: argparse.Namespace) -> Path:
    domains = set(args.domain or DEFAULT_DOMAINS)
    prompts = load_eval_prompts(args.prompts, args.prompt_count, domains)
    tokenizer = AutoTokenizer.from_pretrained(args.support_dir, trust_remote_code=True)
    config = StreamedProbeConfig(
        top_k=args.top_k,
        lm_head_chunk_rows=args.lm_head_chunk_rows,
        max_runtime_tensors=args.max_runtime_tensors,
        cache_mb=args.cache_mb,
        contribution_scale=args.contribution_scale,
    )
    engine = StreamedInferenceEngine(args.model_dir, args.support_dir, args.quantized_dir, "cuda", config)
    results = []
    for item in prompts:
        result = engine.run_prompt(item["prompt"], tokenizer)
        result["id"] = item.get("id")
        result["domain"] = item.get("domain")
        results.append(result)

    run_id = now_id()
    report = {
        "run_id": run_id,
        "mode": "streamed_logit_probe_eval",
        "warning": "This is a tiny streamed inference-quality probe, not full 120B autoregressive generation.",
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "quantized_dir": args.quantized_dir,
        "cuda": cuda_snapshot(),
        "config": config.__dict__,
        "summary": summarize(results),
        "vram_peak": engine.monitor.peak(),
        "vram_events": engine.monitor.to_json(),
        "results": results,
    }
    out = Path(args.out_dir) / f"klcquant-eval-{run_id}.json"
    write_json(out, report)
    charts = write_charts(report, out.with_suffix(""))
    report["charts"] = charts
    write_json(out, report)
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run streamed quantized runtime prompt comparisons")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--quantized-dir", default="quantized_model")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--prompts")
    parser.add_argument("--prompt-count", type=int, default=6)
    parser.add_argument("--domain", action="append")
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--max-runtime-tensors", type=int, default=16)
    parser.add_argument("--cache-mb", type=int, default=8)
    parser.add_argument("--lm-head-chunk-rows", type=int, default=4096)
    parser.add_argument("--contribution-scale", type=float, default=0.10)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = run_eval(args)
    print(f"wrote evaluation report to {out}")


if __name__ == "__main__":
    main()
