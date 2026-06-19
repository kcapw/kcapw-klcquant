from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from .output_similarity import compare_outputs
from .quantized_runtime_loader import LoadedRuntimeGroup, QuantizedRuntimeLoader
from .runtime_cache import RuntimeTensorCache
from .runtime_prefetcher import RuntimePrefetcher
from .streamed_loader import SafetensorIndex
from .vram_monitor import VramMonitor


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


@dataclass
class StreamedProbeConfig:
    top_k: int = 16
    lm_head_chunk_rows: int = 4096
    max_runtime_tensors: int = 32
    cache_mb: int = 8
    contribution_scale: float = 0.10
    resident_core_tensors: list[str] = field(default_factory=lambda: ["model.embed_tokens.weight", "lm_head.weight"])


class OriginalTensorProvider:
    def __init__(self, model_dir: str | Path, support_dir: str | Path, device: torch.device) -> None:
        self.index = SafetensorIndex(model_dir, support_dir)
        self.device = device

    def load_tensor(self, name: str, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        shard = self.index.weight_map[name]
        with safe_open(self.index.model_dir / shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        if tensor.is_floating_point():
            tensor = tensor.to(dtype)
        if self.device.type == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        return tensor.to(self.device)

    def load_rows(self, name: str, row_ids: list[int], dtype: torch.dtype = torch.float16) -> torch.Tensor:
        shard = self.index.weight_map[name]
        rows: list[torch.Tensor] = []
        with safe_open(self.index.model_dir / shard, framework="pt", device="cpu") as handle:
            sliced = handle.get_slice(name)
            for row_id in row_ids:
                rows.append(sliced[row_id : row_id + 1])
        tensor = torch.cat(rows, dim=0).to(dtype)
        if self.device.type == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        return tensor

    def iter_matrix_logits(
        self,
        name: str,
        hidden: torch.Tensor,
        chunk_rows: int,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        shard = self.index.weight_map[name]
        logits: list[torch.Tensor] = []
        hidden = hidden.to(self.device)
        with safe_open(self.index.model_dir / shard, framework="pt", device="cpu") as handle:
            sliced = handle.get_slice(name)
            shape = sliced.get_shape()
            rows = int(shape[0])
            for start in range(0, rows, chunk_rows):
                end = min(rows, start + chunk_rows)
                chunk = sliced[start:end].to(dtype)
                if self.device.type == "cuda":
                    chunk = chunk.pin_memory().to(self.device, non_blocking=True)
                else:
                    chunk = chunk.to(self.device)
                part = torch.matmul(chunk.float(), hidden.float())
                logits.append(part.detach().cpu())
                del chunk, part
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return torch.cat(logits, dim=0)


class StreamedInferenceEngine:
    """Prompt-conditioned streamed logit probe.

    This is a correctness bridge between tensor reconstruction and full streamed
    transformer execution. It uses real tokenizer ids, real embedding rows, real
    original-vs-quantized runtime tensors, and real chunked lm_head logits while
    avoiding full-model materialization.
    """

    def __init__(
        self,
        model_dir: str | Path = "model",
        support_dir: str | Path = "model_support",
        quantized_dir: str | Path = "quantized_model",
        device: str = "cuda",
        config: StreamedProbeConfig | None = None,
    ) -> None:
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.config = config or StreamedProbeConfig()
        self.quant_loader = QuantizedRuntimeLoader(quantized_dir, device=str(self.device))
        self.originals = OriginalTensorProvider(model_dir, support_dir, self.device)
        self.prefetcher = RuntimePrefetcher(self.quant_loader)
        self.cache = RuntimeTensorCache(self.config.cache_mb * 2**20) if self.config.cache_mb > 0 else None
        self.cache_stats = CacheStats()
        self.monitor = VramMonitor()

    def run_prompt(self, prompt: str, tokenizer) -> dict:
        start = time.perf_counter()
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or 0]
        token_ids = [int(t) for t in token_ids]
        self.monitor.record("prompt:start")

        embedding_rows = self.originals.load_rows("model.embed_tokens.weight", token_ids)
        baseline_hidden = embedding_rows.float().mean(dim=0)
        quantized_hidden = baseline_hidden.clone()
        del embedding_rows

        tensor_hotness: list[dict] = []
        processed = 0
        groups = list(self.quant_loader.iter_group_names())
        if groups:
            self.prefetcher.prefetch(groups[0])
        for idx, group_name in enumerate(groups):
            if processed >= self.config.max_runtime_tensors:
                break
            group = self._get_group(group_name)
            if idx + 1 < len(groups) and self.prefetcher.pending_group is None:
                self.prefetcher.prefetch(groups[idx + 1])
            for name, quant_tensor in group.tensors.items():
                if processed >= self.config.max_runtime_tensors:
                    break
                original_tensor = self.originals.load_tensor(name)
                base_delta = self._tensor_contribution(original_tensor, token_ids, baseline_hidden.numel())
                quant_delta = self._tensor_contribution(quant_tensor, token_ids, baseline_hidden.numel())
                baseline_hidden = baseline_hidden + base_delta * self.config.contribution_scale
                quantized_hidden = quantized_hidden + quant_delta * self.config.contribution_scale
                tensor_hotness.append(
                    {
                        "name": name,
                        "group": group_name,
                        "mode": group.metas[name].mode,
                        "baseline_delta_norm": float(torch.linalg.vector_norm(base_delta.float()).item()),
                        "quantized_delta_norm": float(torch.linalg.vector_norm(quant_delta.float()).item()),
                        "delta_drift_l2": float(torch.linalg.vector_norm((base_delta - quant_delta).float()).item()),
                    }
                )
                processed += 1
                del original_tensor, base_delta, quant_delta
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if self.cache is None:
                group.unload()

        baseline_logits = self.originals.iter_matrix_logits("lm_head.weight", baseline_hidden, self.config.lm_head_chunk_rows)
        quantized_logits = self.originals.iter_matrix_logits("lm_head.weight", quantized_hidden, self.config.lm_head_chunk_rows)
        base_top = torch.topk(baseline_logits, k=min(self.config.top_k, baseline_logits.numel())).indices.tolist()
        quant_top = torch.topk(quantized_logits, k=min(self.config.top_k, quantized_logits.numel())).indices.tolist()
        comparison = compare_outputs(baseline_logits, quantized_logits, base_top, quant_top, token_ids[: min(len(token_ids), 64)])
        elapsed = time.perf_counter() - start
        self.monitor.record("prompt:end")

        return {
            "prompt": prompt,
            "prompt_tokens": len(token_ids),
            "runtime_mode": "streamed_logit_probe",
            "note": "Prompt-conditioned logit probe over streamed original and quantized tensors; not a full autoregressive transformer forward.",
            "processed_runtime_tensors": processed,
            "baseline_top_token_ids": base_top,
            "quantized_top_token_ids": quant_top,
            "baseline_top_tokens": [tokenizer.decode([tok]) for tok in base_top],
            "quantized_top_tokens": [tokenizer.decode([tok]) for tok in quant_top],
            "similarity": comparison,
            "tensor_hotness": tensor_hotness,
            "latency_s": round(elapsed, 6),
            "cache": {
                "hits": self.cache_stats.hits,
                "misses": self.cache_stats.misses,
                "hit_rate": self.cache_stats.hit_rate,
            },
        }

    def _get_group(self, group_name: str) -> LoadedRuntimeGroup:
        if self.cache is not None:
            cached = self.cache.get(group_name)
            if cached is not None:
                self.cache_stats.hits += 1
                return cached
        self.cache_stats.misses += 1
        prefetched = self.prefetcher.consume()
        if prefetched is None or prefetched.group != group_name:
            prefetched = self.quant_loader.load_group(group_name)
        if self.cache is not None:
            self.cache.put(prefetched)
        return prefetched

    def _tensor_contribution(self, tensor: torch.Tensor, token_ids: list[int], hidden_size: int) -> torch.Tensor:
        prompt_hash = (sum(token_ids) + 131 * len(token_ids)) & 0x7FFFFFFF
        if tensor.ndim == 0:
            vec = tensor.reshape(1)
        elif tensor.ndim == 1:
            vec = tensor
        else:
            row = prompt_hash % max(1, tensor.shape[0])
            vec = tensor[row].reshape(-1)
        if vec.numel() >= hidden_size:
            vec = vec[:hidden_size]
        else:
            padded = torch.zeros(hidden_size, device=tensor.device, dtype=tensor.dtype)
            padded[: vec.numel()] = vec
            vec = padded
        return torch.tanh(vec.float()).to(tensor.device)
