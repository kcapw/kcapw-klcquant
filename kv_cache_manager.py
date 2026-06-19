from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .adaptive_quantizer import quantize_for_storage


@dataclass
class LayerKV:
    keys: list[torch.Tensor] = field(default_factory=list)
    values: list[torch.Tensor] = field(default_factory=list)


@dataclass
class KVCacheStats:
    tokens: int = 0
    resident_bytes: int = 0
    device: str = "cpu"


class KVCacheManager:
    def __init__(
        self,
        device: torch.device,
        offload_to_cpu: bool = False,
        max_tokens: int | None = None,
        precision: str = "fp16",
    ) -> None:
        self.device = torch.device("cpu") if offload_to_cpu else device
        self.compute_device = device
        self.max_tokens = max_tokens
        self.precision = precision
        self.layers: dict[int, LayerKV] = {}
        self.corruption_events: list[dict] = []

    def append(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor) -> None:
        cache = self.layers.setdefault(layer_idx, LayerKV())
        stored_key = self._store_tensor(key, layer_idx, "key")
        stored_value = self._store_tensor(value, layer_idx, "value")
        cache.keys.append(stored_key)
        cache.values.append(stored_value)
        if self.max_tokens and len(cache.keys) > self.max_tokens:
            cache.keys = cache.keys[-self.max_tokens :]
            cache.values = cache.values[-self.max_tokens :]

    def get(self, layer_idx: int, window: int | None = None) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        cache = self.layers.get(layer_idx)
        if cache is None or not cache.keys:
            return None, None
        keys = cache.keys[-window:] if window else cache.keys
        values = cache.values[-window:] if window else cache.values
        return torch.stack(keys, dim=0).to(self.compute_device), torch.stack(values, dim=0).to(self.compute_device)

    def stats(self) -> KVCacheStats:
        resident = 0
        tokens = 0
        for cache in self.layers.values():
            tokens = max(tokens, len(cache.keys))
            for tensor in cache.keys + cache.values:
                resident += tensor.numel() * tensor.element_size()
        return KVCacheStats(tokens=tokens, resident_bytes=resident, device=str(self.device))

    def extended_stats(self) -> dict:
        stats = self.stats()
        likelihood = 0.0
        if self.corruption_events:
            likelihood = sum(float(item["relative_l2"]) for item in self.corruption_events) / len(self.corruption_events)
        return {
            "tokens": stats.tokens,
            "resident_bytes": stats.resident_bytes,
            "resident_mb": round(stats.resident_bytes / 2**20, 6),
            "device": stats.device,
            "precision": self.precision,
            "corruption_events": self.corruption_events[:200],
            "cache_corruption_likelihood": round(likelihood, 8),
        }

    def snapshot(self) -> dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]]:
        return {
            layer: ([item.detach().clone() for item in cache.keys], [item.detach().clone() for item in cache.values])
            for layer, cache in self.layers.items()
        }

    def restore(self, snapshot: dict[int, tuple[list[torch.Tensor], list[torch.Tensor]]]) -> None:
        self.layers = {
            layer: LayerKV(keys=[item.detach().clone() for item in keys], values=[item.detach().clone() for item in values])
            for layer, (keys, values) in snapshot.items()
        }

    def clear(self) -> None:
        self.layers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _store_tensor(self, tensor: torch.Tensor, layer_idx: int, kind: str) -> torch.Tensor:
        original = tensor.detach()
        if self.precision in {"fp16", "bf16"}:
            dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
            stored = original.to(dtype)
        elif self.precision in {"q8", "q4", "q2", "q1"}:
            quantized = quantize_for_storage(f"kv.{layer_idx}.{kind}", original, self.precision)  # type: ignore[arg-type]
            stored = quantized.dequantize(dtype=torch.float16)
        else:
            raise ValueError(f"unsupported KV cache precision: {self.precision}")
        relative = float(
            torch.linalg.vector_norm((original.float().cpu() - stored.float().cpu())).item()
            / max(torch.linalg.vector_norm(original.float().cpu()).item(), 1e-9)
        )
        if self.precision not in {"fp16", "bf16"}:
            self.corruption_events.append({"layer": layer_idx, "kind": kind, "precision": self.precision, "relative_l2": relative})
        return stored.to(self.device)
