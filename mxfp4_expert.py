from __future__ import annotations

import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

import torch
from safetensors import safe_open

from .quantized_runtime_loader import QuantizedRuntimeLoader
from .streamed_loader import SafetensorIndex


FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


@dataclass
class ExpertCacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    loaded_bytes: int = 0
    dequantized_bytes: int = 0
    hit_bytes: int = 0
    miss_loaded_bytes: int = 0
    pinned_hits: int = 0
    lifetime_accesses: int = 0
    materializations: int = 0
    materialization_time_s: float = 0.0
    transfer_time_s: float = 0.0
    dequant_time_s: float = 0.0
    blocked_on_prefetch_s: float = 0.0
    futures_waited: int = 0
    prefetch_submitted: int = 0
    prefetch_completed: int = 0
    prefetch_failed: int = 0
    prefetch_used: int = 0
    prefetch_skipped_resident: int = 0
    prefetch_skipped_pending: int = 0
    protected_eviction_skips: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class ExpertWeightCache:
    def __init__(
        self,
        max_bytes: int,
        layer_count: int = 0,
        layer_aware: bool = False,
        adaptive_layer_quota: bool = False,
        adaptive_layer_quota_strength: float = 1.5,
    ) -> None:
        self.max_bytes = max_bytes
        self.layer_count = max(int(layer_count), 0)
        self.layer_aware = layer_aware
        self.adaptive_layer_quota = adaptive_layer_quota
        self.adaptive_layer_quota_strength = max(float(adaptive_layer_quota_strength), 0.0)
        self.current_bytes = 0
        self.items: dict[tuple[int, int, str], torch.Tensor] = {}
        self.item_bytes: dict[tuple[int, int, str], int] = {}
        self.layer_bytes: Counter[int] = Counter()
        self.frequency: dict[tuple[int, int, str], int] = {}
        self.last_access: dict[tuple[int, int, str], int] = {}
        self.birth_access: dict[tuple[int, int, str], int] = {}
        self.future_score: dict[tuple[int, int, str], float] = {}
        self.protected_until: dict[tuple[int, int, str], int] = {}
        self.prefetched: set[tuple[int, int, str]] = set()
        self.evicted_lifetimes: list[int] = []
        self.eviction_reasons: Counter[str] = Counter()
        self.layer_hits: Counter[int] = Counter()
        self.layer_misses: Counter[int] = Counter()
        self.layer_evictions: Counter[int] = Counter()
        self.layer_hit_bytes: Counter[int] = Counter()
        self.layer_loaded_bytes: Counter[int] = Counter()
        self.pinned: set[tuple[int, int, str]] = set()
        self.access_index = 0
        self.stats = ExpertCacheStats()
        self.lock = RLock()

    def get(self, key: tuple[int, int, str]) -> torch.Tensor | None:
        with self.lock:
            value = self.items.get(key)
            self.access_index += 1
            self.stats.lifetime_accesses += 1
            if value is not None:
                self.stats.hits += 1
                nbytes = value.numel() * value.element_size()
                self.stats.hit_bytes += nbytes
                self.layer_hits[key[0]] += 1
                self.layer_hit_bytes[key[0]] += nbytes
                self.frequency[key] = self.frequency.get(key, 0) + 1
                self.last_access[key] = self.access_index
                self.future_score[key] = self.future_score.get(key, 0.0) * 0.80
                if key in self.pinned:
                    self.stats.pinned_hits += 1
                if key in self.prefetched:
                    self.prefetched.discard(key)
                    self.stats.prefetch_used += 1
            else:
                self.stats.misses += 1
                self.layer_misses[key[0]] += 1
            return value

    def put(self, key: tuple[int, int, str], value: torch.Tensor, pin: bool = False, prefetched: bool = False) -> None:
        nbytes = value.numel() * value.element_size()
        if self.max_bytes <= 0:
            return
        with self.lock:
            if key in self.items:
                old_bytes = self.item_bytes.get(key, self.items[key].numel() * self.items[key].element_size())
                self.current_bytes -= old_bytes
                self.layer_bytes[key[0]] -= old_bytes
            else:
                self.birth_access[key] = self.access_index
            self.items[key] = value
            self.item_bytes[key] = nbytes
            self.access_index += 1
            self.frequency[key] = self.frequency.get(key, 0) + 1
            self.last_access[key] = self.access_index
            if prefetched:
                self.prefetched.add(key)
            if pin:
                self.pinned.add(key)
            self.current_bytes += nbytes
            self.layer_bytes[key[0]] += nbytes
            self.stats.dequantized_bytes += nbytes
            self._evict_to_budget()

    def pin(self, key: tuple[int, int, str]) -> None:
        with self.lock:
            if key in self.items:
                self.pinned.add(key)

    def contains(self, key: tuple[int, int, str]) -> bool:
        with self.lock:
            return key in self.items

    def resident_experts(self, layer_idx: int) -> set[int]:
        with self.lock:
            return {expert for layer, expert, _proj in self.items if layer == layer_idx}

    def set_future_scores(
        self,
        keys: list[tuple[int, int, str]],
        scores: dict[tuple[int, int, str], float] | None = None,
        protect_window: int = 64,
    ) -> None:
        with self.lock:
            for key in list(self.future_score):
                self.future_score[key] *= 0.85
                if self.future_score[key] < 1e-4:
                    del self.future_score[key]
            horizon = self.access_index + max(int(protect_window), 1)
            for key in keys:
                score = float(scores.get(key, 1.0)) if scores else 1.0
                self.future_score[key] = max(self.future_score.get(key, 0.0), score)
                self.protected_until[key] = max(self.protected_until.get(key, 0), horizon)

    def _evict_to_budget(self) -> None:
        while self.current_bytes > self.max_bytes and self.items:
            candidates, reason = self._eviction_candidates()
            if not candidates:
                break
            unprotected = [key for key in candidates if self.protected_until.get(key, 0) <= self.access_index]
            if unprotected:
                candidates = unprotected
            else:
                self.stats.protected_eviction_skips += 1
            victim = min(
                candidates,
                key=lambda key: (
                    self.future_score.get(key, 0.0),
                    self.frequency.get(key, 0),
                    self.last_access.get(key, 0),
                ),
            )
            old = self.items.pop(victim)
            old_bytes = self.item_bytes.pop(victim, old.numel() * old.element_size())
            born = self.birth_access.pop(victim, self.access_index)
            self.evicted_lifetimes.append(max(self.access_index - born, 0))
            self.frequency.pop(victim, None)
            self.last_access.pop(victim, None)
            self.future_score.pop(victim, None)
            self.protected_until.pop(victim, None)
            self.prefetched.discard(victim)
            self.current_bytes -= old_bytes
            self.layer_bytes[victim[0]] -= old_bytes
            if self.layer_bytes[victim[0]] <= 0:
                self.layer_bytes.pop(victim[0], None)
            self.eviction_reasons[reason] += 1
            self.layer_evictions[victim[0]] += 1
            self.stats.evictions += 1

    def _eviction_candidates(self) -> tuple[list[tuple[int, int, str]], str]:
        candidates = [key for key in self.items if key not in self.pinned]
        if not candidates or not self.layer_aware or self.layer_count <= 0:
            return candidates, "global_pressure"
        layer_quota = self.max_bytes / max(self.layer_count, 1)
        overfull_layers = [
            layer
            for layer, nbytes in self.layer_bytes.items()
            if nbytes > self._layer_quota(layer, layer_quota)
            and any(key[0] == layer and key not in self.pinned for key in self.items)
        ]
        if not overfull_layers:
            return candidates, "global_pressure"
        worst_layer = max(overfull_layers, key=lambda layer: self.layer_bytes[layer] / max(self._layer_quota(layer, layer_quota), 1.0))
        layer_candidates = [key for key in candidates if key[0] == worst_layer]
        return layer_candidates or candidates, "adaptive_layer_quota_pressure" if self.adaptive_layer_quota else "layer_quota_pressure"

    def _layer_quota(self, layer: int, base_quota: float) -> float:
        if not self.adaptive_layer_quota:
            return base_quota
        hits = self.layer_hits.get(layer, 0)
        misses = self.layer_misses.get(layer, 0)
        total = hits + misses
        if total < 32:
            return base_quota
        hit_rate = hits / max(total, 1)
        weight = 0.5 + self.adaptive_layer_quota_strength * hit_rate
        return base_quota * max(weight, 0.25)

    def to_json(self) -> dict:
        with self.lock:
            hot = sorted(
                (
                    {
                        "layer": key[0],
                        "expert": key[1],
                        "proj": key[2],
                        "frequency": self.frequency.get(key, 0),
                        "future_score": round(self.future_score.get(key, 0.0), 6),
                        "age_accesses": self.access_index - self.birth_access.get(key, self.access_index),
                        "pinned": key in self.pinned,
                        "prefetched": key in self.prefetched,
                    }
                    for key in self.items
                ),
                key=lambda row: (row["future_score"], row["frequency"], row["pinned"]),
                reverse=True,
            )
            lifetimes = sorted(self.evicted_lifetimes)
            lifetime_summary = {
                "count": len(lifetimes),
                "mean_accesses": sum(lifetimes) / len(lifetimes) if lifetimes else None,
                "p50_accesses": lifetimes[int(0.50 * (len(lifetimes) - 1))] if lifetimes else None,
                "p90_accesses": lifetimes[int(0.90 * (len(lifetimes) - 1))] if lifetimes else None,
            }
            materializations = max(self.stats.materializations, 1)
            total_prefetches = self.stats.prefetch_completed + self.stats.prefetch_failed
            prefetch_use_rate = self.stats.prefetch_used / max(total_prefetches, 1)
            layer_residency = [
                {
                    "layer": int(layer),
                    "resident_mb": round(nbytes / 2**20, 6),
                    "resident_items": sum(1 for key in self.items if key[0] == layer),
                    "resident_experts": len({expert for key_layer, expert, _proj in self.items if key_layer == layer}),
                }
                for layer, nbytes in sorted(self.layer_bytes.items())
            ]
            layers = sorted(
                set(self.layer_hits)
                | set(self.layer_misses)
                | set(self.layer_evictions)
                | set(self.layer_loaded_bytes)
                | set(self.layer_hit_bytes)
            )
            layer_activity = []
            for layer in layers:
                hits = int(self.layer_hits.get(layer, 0))
                misses = int(self.layer_misses.get(layer, 0))
                evictions = int(self.layer_evictions.get(layer, 0))
                loaded = int(self.layer_loaded_bytes.get(layer, 0))
                hit_bytes = int(self.layer_hit_bytes.get(layer, 0))
                layer_activity.append(
                    {
                        "layer": int(layer),
                        "hits": hits,
                        "misses": misses,
                        "hit_rate": hits / max(hits + misses, 1),
                        "evictions": evictions,
                        "loaded_mb": round(loaded / 2**20, 6),
                        "hit_mb": round(hit_bytes / 2**20, 6),
                    }
                )
            return {
                "hits": self.stats.hits,
                "misses": self.stats.misses,
                "hit_rate": self.stats.hit_rate,
                "pinned_hits": self.stats.pinned_hits,
                "evictions": self.stats.evictions,
                "resident_bytes": self.current_bytes,
                "resident_mb": round(self.current_bytes / 2**20, 6),
                "resident_items": len(self.items),
                "pinned_items": len(self.pinned),
                "loaded_bytes": self.stats.loaded_bytes,
                "loaded_mb": round(self.stats.loaded_bytes / 2**20, 6),
                "dequantized_bytes": self.stats.dequantized_bytes,
                "dequantized_mb": round(self.stats.dequantized_bytes / 2**20, 6),
                "hit_bytes": self.stats.hit_bytes,
                "hit_mb": round(self.stats.hit_bytes / 2**20, 6),
                "miss_loaded_bytes": self.stats.miss_loaded_bytes,
                "miss_loaded_mb": round(self.stats.miss_loaded_bytes / 2**20, 6),
                "effective_useful_bandwidth": self.stats.hit_bytes / max(self.stats.loaded_bytes, 1),
                "materializations": self.stats.materializations,
                "materialization_time_s": round(self.stats.materialization_time_s, 6),
                "avg_materialization_time_ms": round(1000.0 * self.stats.materialization_time_s / materializations, 6),
                "transfer_time_s": round(self.stats.transfer_time_s, 6),
                "dequant_time_s": round(self.stats.dequant_time_s, 6),
                "avg_dequant_time_ms": round(1000.0 * self.stats.dequant_time_s / materializations, 6),
                "blocked_on_prefetch_s": round(self.stats.blocked_on_prefetch_s, 6),
                "futures_waited": self.stats.futures_waited,
                "prefetch_submitted": self.stats.prefetch_submitted,
                "prefetch_completed": self.stats.prefetch_completed,
                "prefetch_failed": self.stats.prefetch_failed,
                "prefetch_used": self.stats.prefetch_used,
                "prefetch_use_rate": prefetch_use_rate,
                "prefetch_skipped_resident": self.stats.prefetch_skipped_resident,
                "prefetch_skipped_pending": self.stats.prefetch_skipped_pending,
                "protected_eviction_skips": self.stats.protected_eviction_skips,
                "evicted_lifetime": lifetime_summary,
                "layer_aware": self.layer_aware,
                "adaptive_layer_quota": self.adaptive_layer_quota,
                "layer_count": self.layer_count,
                "layer_quota_mb": round((self.max_bytes / max(self.layer_count, 1)) / 2**20, 6) if self.layer_count else None,
                "layer_residency": layer_residency,
                "layer_activity": layer_activity,
                "eviction_reasons": dict(self.eviction_reasons),
                "hot_items": hot[:50],
            }


def dequantize_mxfp4_matrix(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """Dequantize one GPT-OSS MXFP4 expert matrix.

    Input block layout is `[out_dim, groups, 16 packed bytes]`, where each byte
    stores two FP4 values. The returned matrix is `[in_dim, out_dim]`, matching
    the GPT-OSS expert matmul layout used by `hidden @ weight`.
    """

    device = blocks.device
    blocks = blocks.to(torch.uint8)
    exponents = scales.to(torch.int32) - 127
    if blocks.shape[:-1] != scales.shape:
        raise ValueError(f"MXFP4 block/scale shape mismatch: {blocks.shape=} {scales.shape=}")
    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=device)
    out_dim, groups, packed = blocks.shape
    flat_blocks = blocks.reshape(out_dim * groups, packed)
    flat_exp = exponents.reshape(out_dim * groups, 1)
    unpacked = torch.empty(out_dim * groups, packed * 2, dtype=dtype, device=device)
    unpacked[:, 0::2] = lut[(flat_blocks & 0x0F).to(torch.long)]
    unpacked[:, 1::2] = lut[(flat_blocks >> 4).to(torch.long)]
    torch.ldexp(unpacked, flat_exp, out=unpacked)
    matrix_out_in = unpacked.reshape(out_dim, groups * packed * 2)
    return matrix_out_in.transpose(0, 1).contiguous()


class Mxfp4ExpertStore:
    def __init__(
        self,
        model_dir: str | Path,
        support_dir: str | Path,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
        cache_bytes: int = 0,
        quant_loader: QuantizedRuntimeLoader | None = None,
        async_prefetch: bool = False,
        max_prefetch_workers: int = 1,
        layer_count: int = 0,
        layer_aware_cache: bool = False,
        adaptive_layer_quota: bool = False,
        adaptive_layer_quota_strength: float = 1.5,
        default_protect_window: int = 64,
    ) -> None:
        self.index = SafetensorIndex(model_dir, support_dir)
        self.device = device
        self.dtype = dtype
        self.cache = ExpertWeightCache(
            cache_bytes,
            layer_count=layer_count,
            layer_aware=layer_aware_cache,
            adaptive_layer_quota=adaptive_layer_quota,
            adaptive_layer_quota_strength=adaptive_layer_quota_strength,
        )
        self.quant_loader = quant_loader
        self.async_prefetch = async_prefetch
        self.default_protect_window = max(int(default_protect_window), 1)
        self.executor = ThreadPoolExecutor(max_workers=max_prefetch_workers) if async_prefetch and max_prefetch_workers > 0 else None
        self.pending: dict[tuple[int, int, str], Future] = {}
        self.pending_lock = RLock()

    def load_expert_matrix(self, layer_idx: int, expert_idx: int, proj: str) -> torch.Tensor:
        key = (layer_idx, expert_idx, proj)
        self._consume_pending_if_needed(key)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        return self._materialize_expert_matrix(key, prefetched=False)

    def protect_experts(self, layer_idx: int, expert_ids: list[int], protect_window: int | None = None) -> None:
        keys = self._matrix_keys(layer_idx, expert_ids)
        self.cache.set_future_scores(keys, protect_window=protect_window or self.default_protect_window)

    def prefetch_experts(self, layer_idx: int, expert_ids: list[int], protect_window: int | None = None) -> None:
        if not expert_ids:
            return
        keys = self._matrix_keys(layer_idx, expert_ids)
        self.cache.set_future_scores(keys, protect_window=protect_window or self.default_protect_window)
        if self.executor is None:
            return
        with self.pending_lock:
            for key in keys:
                if self.cache.contains(key):
                    self.cache.stats.prefetch_skipped_resident += 1
                    continue
                if key in self.pending:
                    self.cache.stats.prefetch_skipped_pending += 1
                    continue
                self.pending[key] = self.executor.submit(self._prefetch_worker, key)
                self.cache.stats.prefetch_submitted += 1

    def resident_experts(self, layer_idx: int) -> set[int]:
        return self.cache.resident_experts(layer_idx)

    def _matrix_keys(self, layer_idx: int, expert_ids: list[int]) -> list[tuple[int, int, str]]:
        return [(int(layer_idx), int(expert_id), "gate_up_proj") for expert_id in expert_ids] + [
            (int(layer_idx), int(expert_id), "down_proj") for expert_id in expert_ids
        ]

    def _consume_pending_if_needed(self, key: tuple[int, int, str]) -> None:
        with self.pending_lock:
            future = self.pending.pop(key, None)
        if future is None:
            return
        start = time.perf_counter()
        self.cache.stats.futures_waited += 1
        try:
            future.result()
        finally:
            self.cache.stats.blocked_on_prefetch_s += time.perf_counter() - start

    def _prefetch_worker(self, key: tuple[int, int, str]) -> None:
        try:
            if not self.cache.contains(key):
                self._materialize_expert_matrix(key, prefetched=True)
            self.cache.stats.prefetch_completed += 1
        except Exception:
            self.cache.stats.prefetch_failed += 1
            raise
        finally:
            with self.pending_lock:
                self.pending.pop(key, None)

    def _materialize_expert_matrix(self, key: tuple[int, int, str], prefetched: bool) -> torch.Tensor:
        layer_idx, expert_idx, proj = key
        start = time.perf_counter()
        base = f"model.layers.{layer_idx}.mlp.experts.{proj}"
        block_name = f"{base}_blocks"
        scale_name = f"{base}_scales"
        transfer_start = time.perf_counter()
        blocks = self._load_expert_tensor(block_name, expert_idx)
        scales = self._load_expert_tensor(scale_name, expert_idx)
        packed_bytes = blocks.numel() * blocks.element_size() + scales.numel() * scales.element_size()
        self.cache.stats.loaded_bytes += packed_bytes
        self.cache.stats.miss_loaded_bytes += packed_bytes
        self.cache.layer_loaded_bytes[layer_idx] += packed_bytes
        if self.device.type == "cuda":
            blocks = blocks.pin_memory().to(self.device, non_blocking=True)
            scales = scales.pin_memory().to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            blocks = blocks.to(self.device)
            scales = scales.to(self.device)
        self.cache.stats.transfer_time_s += time.perf_counter() - transfer_start
        dequant_start = time.perf_counter()
        matrix = dequantize_mxfp4_matrix(blocks, scales, self.dtype)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.cache.stats.dequant_time_s += time.perf_counter() - dequant_start
        self.cache.stats.materialization_time_s += time.perf_counter() - start
        self.cache.stats.materializations += 1
        self.cache.put(key, matrix, prefetched=prefetched)
        return matrix

    def pin_experts(self, keys: list[tuple[int, int, str]]) -> None:
        for layer_idx, expert_idx, proj in keys:
            matrix = self.load_expert_matrix(layer_idx, expert_idx, proj)
            self.cache.pin((layer_idx, expert_idx, proj))
            del matrix

    def _load_expert_tensor(self, name: str, expert_idx: int) -> torch.Tensor:
        if self.quant_loader is not None and name in self.quant_loader.tensor_index:
            full, meta = self.quant_loader.load_tensor(name, dtype=self.dtype)
            selected = full[expert_idx].detach().cpu()
            del full
            if meta.dtype == "torch.uint8":
                return torch.clamp(torch.round(selected.float()), 0, 255).to(torch.uint8)
            return selected
        shard = self.index.weight_map[name]
        with safe_open(self.index.model_dir / shard, framework="pt", device="cpu") as handle:
            return handle.get_slice(name)[expert_idx]

    def to_json(self) -> dict:
        payload = self.cache.to_json()
        with self.pending_lock:
            payload["pending_prefetches"] = len(self.pending)
        return payload
