from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from safetensors import safe_open
from tqdm import tqdm

from .importance_ranker import ImportanceRanker, TensorStats
from .streamed_loader import SafetensorIndex, StreamedTensorLoader
from .utils import cuda_snapshot


def _norm(value: float, high: float) -> float:
    if high <= 0:
        return 0.0
    return max(0.0, min(1.0, value / high))


@dataclass
class StaticScanConfig:
    sample_values: int = 4096
    load_to_cuda: bool = True


class TensorProfiler:
    """Collect tensor statistics from static shard scans and optional module hooks."""

    def __init__(self) -> None:
        self.stats: dict[str, TensorStats] = {}
        self._module_calls: defaultdict[str, int] = defaultdict(int)
        self._module_magnitudes: defaultdict[str, float] = defaultdict(float)
        self._handles = []

    def scan_safetensors(
        self,
        model_dir: str | Path = "model",
        support_dir: str | Path = "model_support",
        config: StaticScanConfig | None = None,
    ) -> list[TensorStats]:
        cfg = config or StaticScanConfig()
        index = SafetensorIndex(model_dir, support_dir)
        max_magnitude = 0.0
        collected: list[TensorStats] = []

        for group, refs in tqdm(index.groups().items(), desc="static tensor scan"):
            by_shard: dict[Path, list] = defaultdict(list)
            for ref in refs:
                by_shard[ref.path].append(ref)
            for shard_path, shard_refs in by_shard.items():
                with safe_open(shard_path, framework="pt", device="cpu") as f:
                    for ref in shard_refs:
                        tensor, shape = self._sample_tensor(f, ref.name, cfg.sample_values)
                        flat = tensor.reshape(-1)
                        sample = flat[: min(cfg.sample_values, flat.numel())]
                        magnitude = float(sample.float().abs().mean().item()) if sample.numel() else 0.0
                        numel = 1
                        for dim in shape:
                            numel *= dim
                        nbytes = numel * tensor.element_size()
                        stat = TensorStats(
                            name=ref.name,
                            shard=ref.shard,
                            shape=list(shape),
                            dtype=str(tensor.dtype),
                            nbytes=nbytes,
                            activation_count=0,
                            contribution_magnitude=magnitude,
                            reuse_frequency=self._static_reuse_prior(ref.name),
                            attention_relevance=self._static_attention_prior(ref.name),
                        )
                        max_magnitude = max(max_magnitude, magnitude)
                        self.stats[ref.name] = stat
                        collected.append(stat)
                        del tensor, flat, sample

        for stat in collected:
            stat.contribution_magnitude = _norm(stat.contribution_magnitude, max_magnitude)
            stat.activation_frequency = self._static_activation_prior(stat.name)
        return ImportanceRanker().rank(collected)

    def cuda_stream_scan(self, loader: StreamedTensorLoader) -> tuple[list[TensorStats], list[dict]]:
        snapshots: list[dict] = []
        max_magnitude = 0.0
        collected: list[TensorStats] = []

        for group in tqdm(loader.stream(), desc="cuda streamed scan"):
            before = cuda_snapshot()
            for name, tensor in group.tensors.items():
                with torch.no_grad():
                    sample = tensor.reshape(-1)[: min(4096, tensor.numel())]
                    magnitude = float(sample.float().abs().mean().item()) if sample.numel() else 0.0
                    max_magnitude = max(max_magnitude, magnitude)
                    stat = TensorStats(
                        name=name,
                        shard=loader.index.weight_map.get(name),
                        shape=list(tensor.shape),
                        dtype=str(tensor.dtype),
                        nbytes=tensor.numel() * tensor.element_size(),
                        contribution_magnitude=magnitude,
                        activation_frequency=self._static_activation_prior(name),
                        attention_relevance=self._static_attention_prior(name),
                        reuse_frequency=self._static_reuse_prior(name),
                    )
                    self.stats[name] = stat
                    collected.append(stat)
            after = cuda_snapshot()
            snapshots.append({"group": group.group, "loaded_bytes": group.nbytes, "before": before, "after": after})
            group.unload()

        for stat in collected:
            stat.contribution_magnitude = _norm(stat.contribution_magnitude, max_magnitude)
        return ImportanceRanker().rank(collected), snapshots

    def attach_hooks(self, model: torch.nn.Module) -> None:
        for name, module in model.named_modules():
            if not name:
                continue
            if any(part in name for part in ("self_attn", "mlp", "router", "experts")):
                self._handles.append(module.register_forward_hook(self._hook(name)))

    def detach_hooks(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def inner(_module, _inputs, output):
            self._module_calls[name] += 1
            tensor = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(tensor):
                self._module_magnitudes[name] += float(tensor.detach().float().abs().mean().item())
        return inner

    def merge_hook_stats(self, prompt_count: int) -> list[TensorStats]:
        max_calls = max(self._module_calls.values(), default=1)
        max_mag = max(self._module_magnitudes.values(), default=1.0)
        merged = list(self.stats.values())
        for stat in merged:
            module_name = self._best_module_match(stat.name)
            calls = self._module_calls.get(module_name, 0)
            mag = self._module_magnitudes.get(module_name, 0.0) / max(calls, 1)
            stat.activation_count = calls
            stat.activation_frequency = _norm(calls, max(max_calls, prompt_count))
            stat.contribution_magnitude = max(stat.contribution_magnitude, _norm(mag, max_mag))
        return ImportanceRanker().rank(merged)

    @staticmethod
    def _best_module_match(tensor_name: str) -> str:
        parts = tensor_name.split(".")
        if "weight" in parts:
            parts = parts[: parts.index("weight")]
        if "bias" in parts:
            parts = parts[: parts.index("bias")]
        return ".".join(parts)

    @staticmethod
    def _sample_tensor(handle, name: str, sample_values: int) -> tuple[torch.Tensor, list[int]]:
        if hasattr(handle, "get_slice"):
            try:
                sliced = handle.get_slice(name)
                shape = list(sliced.get_shape())
                if len(shape) == 0:
                    tensor = sliced[:]
                elif len(shape) == 1:
                    tensor = sliced[: min(sample_values, shape[0])]
                else:
                    row_width = 1
                    for dim in shape[1:]:
                        row_width *= max(1, dim)
                    rows = max(1, min(shape[0], max(1, sample_values // row_width)))
                    tensor = sliced[:rows]
                return tensor, shape
            except Exception:
                pass
        tensor = handle.get_tensor(name)
        return tensor.reshape(-1)[: min(sample_values, tensor.numel())], list(tensor.shape)

    @staticmethod
    def _static_activation_prior(name: str) -> float:
        if any(token in name for token in ("embed_tokens", "lm_head", "norm", "self_attn")):
            return 1.0
        if ".router." in name:
            return 0.95
        if ".experts." in name:
            return 0.20
        return 0.50

    @staticmethod
    def _static_attention_prior(name: str) -> float:
        if ".self_attn." in name:
            return 1.0
        if any(token in name for token in ("embed_tokens", "lm_head")):
            return 0.75
        return 0.05

    @staticmethod
    def _static_reuse_prior(name: str) -> float:
        if any(token in name for token in ("embed_tokens", "lm_head", "norm", "self_attn", "router")):
            return 1.0
        if ".experts." in name:
            return 0.25
        return 0.50


def measure_generation(model, tokenizer, prompts: Iterable[dict], max_new_tokens: int = 32) -> dict:
    latencies: list[float] = []
    generated_tokens = 0
    outputs: list[str] = []
    for item in tqdm(list(prompts), desc="hf generation profile"):
        prompt = item["prompt"] if isinstance(item, dict) else str(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        start = time.perf_counter()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        new_tokens = max(0, out.shape[-1] - inputs["input_ids"].shape[-1])
        generated_tokens += new_tokens
        latencies.append(elapsed)
        outputs.append(tokenizer.decode(out[0], skip_special_tokens=True))
    total = sum(latencies)
    return {
        "prompt_count": len(latencies),
        "generated_tokens": generated_tokens,
        "latency_avg_s": total / max(len(latencies), 1),
        "latency_p95_s": sorted(latencies)[int(0.95 * (len(latencies) - 1))] if latencies else 0.0,
        "tokens_per_sec": generated_tokens / total if total > 0 else 0.0,
        "sample_outputs": outputs[:5],
    }
