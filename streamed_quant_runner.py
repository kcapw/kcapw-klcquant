from __future__ import annotations

import argparse
import fnmatch
import os
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .adaptive_quantizer import QuantizedTensor, quantize_for_storage
from .compression_policy import StreamCompressionPolicy, TensorPlan
from .streamed_loader import SafetensorIndex, TensorRef
from .tensor_similarity import tensor_quality
from .tensor_pruner import prune_for_storage
from .utils import cuda_snapshot, now_id, read_json, write_json
from .vram_monitor import VramMonitor


def _latest_report(reports_dir: str | Path = "reports") -> Path:
    reports = sorted(Path(reports_dir).glob("klcquant-scan-*.json"))
    if not reports:
        raise FileNotFoundError("No scan reports found. Run `klcquant ... scan` first or pass --importance-report.")
    return reports[-1]


def _safe_stem(group: str) -> str:
    return group.replace("/", "_")


def _storage_keys(name: str) -> tuple[str, str, str]:
    return f"{name}.__data__", f"{name}.__scale__", f"{name}.__zero__"


def _quantized_to_tensors(item: QuantizedTensor) -> dict[str, torch.Tensor]:
    data_key, scale_key, _ = _storage_keys(item.name)
    _, _, zero_key = _storage_keys(item.name)
    tensors = {data_key: item.qvalues.detach().cpu().contiguous()}
    if item.scale is not None:
        tensors[scale_key] = item.scale.detach().cpu().contiguous()
    if item.zero_point is not None:
        tensors[zero_key] = item.zero_point.detach().cpu().contiguous()
    return tensors


def _read_completed(manifest_path: Path) -> set[str]:
    if not manifest_path.exists():
        return set()
    manifest = read_json(manifest_path)
    return {item["name"] for item in manifest.get("tensors", [])}


def _select_refs(
    index: SafetensorIndex,
    plan: dict[str, TensorPlan],
    max_tensors: int,
    max_original_bytes: int,
    groups: list[str] | None,
    pattern: str | None,
    modes: set[str] | None,
    force_mode: str | None,
    completed: set[str],
) -> list[tuple[TensorRef, TensorPlan]]:
    refs_by_name = {ref.name: ref for ref in index.refs}
    candidates = sorted(plan.values(), key=lambda item: (item.importance_score, item.original_bytes))
    selected: list[tuple[TensorRef, TensorPlan]] = []
    total = 0
    group_set = set(groups or [])
    for item in candidates:
        ref = refs_by_name.get(item.name)
        if ref is None:
            continue
        if ref.name in completed:
            continue
        if group_set and ref.group not in group_set:
            continue
        if pattern and not fnmatch.fnmatch(item.name, pattern):
            continue
        if modes and item.mode not in modes:
            continue
        if force_mode is not None:
            item = TensorPlan(
                name=item.name,
                mode=force_mode,  # type: ignore[arg-type]
                importance_score=item.importance_score,
                rank=item.rank,
                percentile=item.percentile,
                original_bytes=item.original_bytes,
            )
        if item.original_bytes <= 0:
            continue
        if selected and total + item.original_bytes > max_original_bytes:
            break
        selected.append((ref, item))
        total += item.original_bytes
        if len(selected) >= max_tensors:
            break
    return selected


def _load_tensor(ref: TensorRef, device: torch.device) -> torch.Tensor:
    with safe_open(ref.path, framework="pt", device="cpu") as handle:
        tensor = handle.get_tensor(ref.name)
    if tensor.is_floating_point():
        tensor = tensor.to(torch.float16)
    if device.type == "cuda":
        tensor = tensor.pin_memory().to(device, non_blocking=True)
        torch.cuda.synchronize()
    return tensor


def compress_subset(args: argparse.Namespace) -> Path:
    report_path = Path(args.importance_report) if args.importance_report else _latest_report(args.out_dir)
    report = read_json(report_path)
    rankings = report["tensor_importance_rankings"]
    policy = StreamCompressionPolicy(
        keep_top_percent=args.keep_top_percent,
        q16_until_percent=args.q16_until_percent,
        q8_until_percent=args.q8_until_percent,
        q4_until_percent=args.q4_until_percent,
        protect_core_tensors=not args.allow_core_tensors,
    )
    plan = policy.make_plan(rankings)
    index = SafetensorIndex(args.model_dir, args.support_dir)
    modes = set(args.mode or []) or None
    out_dir = Path(args.quantized_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"
    completed = set() if args.overwrite else _read_completed(manifest_path)
    selected = _select_refs(
        index=index,
        plan=plan,
        max_tensors=args.max_tensors,
        max_original_bytes=args.max_original_mb * 2**20,
        groups=args.group,
        pattern=args.pattern,
        modes=modes,
        force_mode=args.force_mode,
        completed=completed,
    )
    if not selected:
        raise RuntimeError("No tensors matched the quantization selection constraints.")

    run_id = f"{now_id()}-{os.getpid()}"
    monitor = VramMonitor()
    monitor.reset_cuda_peak()
    monitor.record("start")
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    by_group: dict[str, list[tuple[TensorRef, TensorPlan]]] = defaultdict(list)
    for ref, item in selected:
        if ref.name not in completed:
            by_group[ref.group].append((ref, item))

    tensor_reports: list[dict] = []
    stored_files: list[str] = []
    original_total = 0
    compressed_total = 0

    for group, items in by_group.items():
        group_tensors: dict[str, torch.Tensor] = {}
        group_meta: dict[str, str] = {
            "klcquant_format": "experimental-streamed-v1",
            "group": group,
            "run_id": run_id,
        }
        for ref, item in items:
            monitor.record(f"before:{ref.name}")
            start = time.perf_counter()
            original = _load_tensor(ref, device)
            loaded_bytes = original.numel() * original.element_size()
            sparsity = None
            if item.mode == "pruned":
                quantized, sparsity = prune_for_storage(ref.name, original, args.prune_fraction)
            else:
                quantized = quantize_for_storage(ref.name, original, item.mode)
            reconstructed = quantized.dequantize(dtype=original.dtype).to(original.device)
            quality = tensor_quality(original, reconstructed)
            elapsed = time.perf_counter() - start
            storage = _quantized_to_tensors(quantized)
            group_tensors.update(storage)
            stored_bytes = sum(t.numel() * t.element_size() for t in storage.values())
            original_total += loaded_bytes
            compressed_total += stored_bytes
            group_meta[f"{ref.name}.mode"] = item.mode
            group_meta[f"{ref.name}.shape"] = ",".join(str(dim) for dim in quantized.original_shape)
            group_meta[f"{ref.name}.dtype"] = quantized.original_dtype
            group_meta[f"{ref.name}.bits"] = str(quantized.bits or "")
            group_meta[f"{ref.name}.packed"] = str(quantized.packed).lower()
            if sparsity is not None:
                group_meta[f"{ref.name}.sparsity"] = str(sparsity.sparsity)
            tensor_reports.append(
                {
                    "name": ref.name,
                    "group": ref.group,
                    "shard": ref.shard,
                    "mode": item.mode,
                    "importance_score": item.importance_score,
                    "rank": item.rank,
                    "percentile": item.percentile,
                    "shape": list(quantized.original_shape),
                    "original_dtype": quantized.original_dtype,
                    "original_bytes": loaded_bytes,
                    "compressed_payload_bytes": stored_bytes,
                    "payload_compression_ratio": round(loaded_bytes / max(stored_bytes, 1), 6),
                    "quality": quality,
                    "sparsity": sparsity.to_json() if sparsity is not None else None,
                    "seconds": round(elapsed, 6),
                    "vram_after": monitor.record(f"after:{ref.name}"),
                }
            )
            del original, reconstructed, quantized
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if group_tensors:
            group_path = out_dir / f"{_safe_stem(group)}.safetensors"
            if group_path.exists() and not args.overwrite:
                group_path = out_dir / f"{_safe_stem(group)}-{run_id}.safetensors"
            save_file(group_tensors, group_path, metadata=group_meta)
            stored_files.append(str(group_path))
            compressed_total += max(0, group_path.stat().st_size - sum(t.numel() * t.element_size() for t in group_tensors.values()))
            del group_tensors

    summary = {
        "run_id": run_id,
        "mode": "partial_streamed_quantization",
        "source_report": str(report_path),
        "model_dir": args.model_dir,
        "support_dir": args.support_dir,
        "quantized_dir": str(out_dir),
        "device": str(device),
        "cuda": cuda_snapshot(),
        "selection": {
            "requested_max_tensors": args.max_tensors,
            "selected_tensors": len(selected),
            "processed_tensors": len(tensor_reports),
            "groups": sorted(by_group),
            "pattern": args.pattern,
            "modes": sorted(modes) if modes else None,
        },
        "compression": {
            "original_bytes": original_total,
            "compressed_payload_bytes": compressed_total,
            "compression_ratio": round(original_total / max(compressed_total, 1), 6),
        },
        "vram_peak": monitor.peak(),
        "vram_events": monitor.to_json(),
        "stored_files": stored_files,
        "policy": asdict(policy),
        "tensors": tensor_reports,
    }

    report_out = Path(args.out_dir) / f"klcquant-quant-{run_id}.json"
    write_json(report_out, summary)
    write_json(manifest_path, summary)
    return report_out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Partial streamed adaptive quantization prototype")
    parser.add_argument("--model-dir", default="model")
    parser.add_argument("--support-dir", default="model_support")
    parser.add_argument("--out-dir", default="reports")
    parser.add_argument("--quantized-dir", default="quantized_model")
    parser.add_argument("--importance-report")
    parser.add_argument("--max-tensors", type=int, default=12)
    parser.add_argument("--max-original-mb", type=int, default=256)
    parser.add_argument("--group", action="append")
    parser.add_argument("--pattern")
    modes = ["fp16", "q16", "q8", "q4", "q3", "q2", "q1", "pruned"]
    parser.add_argument("--mode", action="append", choices=modes)
    parser.add_argument("--force-mode", choices=modes, help="Override policy mode for selected tensors.")
    parser.add_argument("--prune-fraction", type=float, default=0.90)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-core-tensors", action="store_true")
    parser.add_argument("--keep-top-percent", type=float, default=0.25)
    parser.add_argument("--q16-until-percent", type=float, default=0.45)
    parser.add_argument("--q8-until-percent", type=float, default=0.75)
    parser.add_argument("--q4-until-percent", type=float, default=0.92)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out = compress_subset(args)
    print(f"wrote quantization report to {out}")


if __name__ == "__main__":
    main()
