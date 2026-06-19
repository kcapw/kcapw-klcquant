from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

from .kv_cache_manager import KVCacheManager
from .mxfp4_expert import Mxfp4ExpertStore
from .quantized_runtime_loader import QuantizedRuntimeLoader
from .streamed_attention import AttentionConfig, StreamedAttention, rms_norm
from .streamed_loader import SafetensorIndex
from .streamed_moe_router import RoutingLocalityConfig, StreamedMoERouter
from .tensor_scheduler import TensorScheduler
from .utils import read_json
from .vram_monitor import VramMonitor


@dataclass
class ExecutorConfig:
    max_layers: int = 2
    cache_layers_mb: int = 0
    expert_cache_mb: int = 128
    offload_kv_cache: bool = False
    use_quantized_overrides: bool = False
    execute_experts: bool = True
    kv_cache_precision: str = "fp16"
    dtype: torch.dtype = torch.bfloat16
    hot_residency: bool = False
    hot_vram_budget_mb: int = 0
    pin_lm_head: bool = False
    pin_layer_tensors: bool = False
    routing_locality: bool = False
    sticky_routing_strength: float = 0.0
    sticky_routing_decay: float = 0.92
    routing_semantic_guard: bool = False
    sticky_candidate_margin: float = 0.50
    max_sticky_bonus: float = 0.25
    min_raw_route_overlap: int = 2
    max_hot_experts_per_layer: int = 0
    active_experts_per_token_cap: int = 0
    routing_exploration_margin: float = 0.25
    cache_aware_routing_strength: float = 0.0
    predictive_expert_prefetch: bool = False
    expert_prefetch_limit: int = 4
    expert_async_prefetch: bool = False
    routing_prediction_window: int = 16
    routing_workload_window: int = 64
    dynamic_active_expert_cap: bool = False
    min_active_experts_per_token: int = 1
    max_active_experts_per_token: int = 0
    layer_aware_expert_cache: bool = False
    expert_matmul_precision: str = "fp32"
    expert_protect_tokens: int = 2
    adaptive_layer_quota: bool = False
    adaptive_layer_quota_strength: float = 1.5


class StreamedTransformerExecutor:
    def __init__(
        self,
        model_dir: str | Path = "model",
        support_dir: str | Path = "model_support",
        quantized_dir: str | Path = "quantized_model",
        device: str = "cuda",
        config: ExecutorConfig | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.support_dir = Path(support_dir)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.config = config or ExecutorConfig()
        self.model_config = read_json(self.support_dir / "config.json")
        self.index = SafetensorIndex(self.model_dir, self.support_dir)
        self.scheduler = TensorScheduler(self.index, self.config.max_layers)
        self.quant_loader = QuantizedRuntimeLoader(quantized_dir, str(self.device)) if self.config.use_quantized_overrides else None
        rope_params = self.model_config.get("rope_parameters") or self.model_config.get("rope_scaling") or {}
        self.attn = StreamedAttention(
            AttentionConfig(
                hidden_size=self.model_config["hidden_size"],
                num_heads=self.model_config["num_attention_heads"],
                num_kv_heads=self.model_config["num_key_value_heads"],
                head_dim=self.model_config["head_dim"],
                rope_theta=float(self.model_config.get("rope_theta", 150000.0)),
                rope_factor=float(rope_params.get("factor", 1.0)),
                rope_beta_fast=float(rope_params.get("beta_fast", 32.0)),
                rope_beta_slow=float(rope_params.get("beta_slow", 1.0)),
                rope_original_max_position_embeddings=int(rope_params.get("original_max_position_embeddings", 4096)),
                rope_truncate=bool(rope_params.get("truncate", False)),
                sliding_window=None,
                rms_norm_eps=float(self.model_config.get("rms_norm_eps", 1e-5)),
            )
        )
        self.router = StreamedMoERouter(
            self.model_config.get("experts_per_token", 4),
            self.model_config.get("swiglu_limit", 7.0),
            RoutingLocalityConfig(
                enabled=self.config.routing_locality,
                sticky_strength=self.config.sticky_routing_strength,
                sticky_decay=self.config.sticky_routing_decay,
                semantic_guard=self.config.routing_semantic_guard,
                sticky_candidate_margin=self.config.sticky_candidate_margin,
                max_sticky_bonus=self.config.max_sticky_bonus,
                min_raw_overlap=self.config.min_raw_route_overlap,
                max_hot_experts_per_layer=self.config.max_hot_experts_per_layer,
                active_experts_per_token_cap=self.config.active_experts_per_token_cap,
                exploration_margin=self.config.routing_exploration_margin,
                cache_aware_strength=self.config.cache_aware_routing_strength,
                predictive_prefetch=self.config.predictive_expert_prefetch,
                prediction_window=self.config.routing_prediction_window,
                workload_window=self.config.routing_workload_window,
                dynamic_active_expert_cap=self.config.dynamic_active_expert_cap,
                min_active_experts_per_token=self.config.min_active_experts_per_token,
                max_active_experts_per_token=self.config.max_active_experts_per_token,
            ),
            matmul_precision=self.config.expert_matmul_precision,
        )
        self.expert_store = Mxfp4ExpertStore(
            self.model_dir,
            self.support_dir,
            self.device,
            dtype=self.config.dtype,
            cache_bytes=self.config.expert_cache_mb * 2**20,
            quant_loader=self.quant_loader,
            async_prefetch=self.config.expert_async_prefetch,
            layer_count=self.config.max_layers,
            layer_aware_cache=self.config.layer_aware_expert_cache,
            adaptive_layer_quota=self.config.adaptive_layer_quota,
            adaptive_layer_quota_strength=self.config.adaptive_layer_quota_strength,
            default_protect_window=self._expert_protect_window(),
        )
        self.kv_cache = KVCacheManager(
            self.device,
            offload_to_cpu=self.config.offload_kv_cache,
            precision=self.config.kv_cache_precision,
        )
        self.monitor = VramMonitor()
        self.transfer_bytes = 0
        self.transfer_events: list[dict] = []
        self.layer_tensor_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.layer_tensor_cache_bytes = 0
        self.layer_tensor_cache_hits = 0
        self.layer_tensor_cache_misses = 0
        self.lm_head_cache: torch.Tensor | None = None
        self.lm_head_cache_bytes = 0
        self.lm_head_cache_hits = 0
        self.lm_head_cache_misses = 0
        if self.config.hot_residency:
            self._initialize_hot_residency()

    @property
    def hidden_size(self) -> int:
        return int(self.model_config["hidden_size"])

    def _expert_protect_window(self) -> int:
        experts_per_token = int(self.model_config.get("experts_per_token") or self.model_config.get("num_experts_per_tok") or 4)
        matrices_per_token = max(int(self.config.max_layers), 1) * max(experts_per_token, 1) * 2
        return max(matrices_per_token * max(int(self.config.expert_protect_tokens), 1), 64)

    def embed_token(self, token_id: int) -> torch.Tensor:
        with self._open_tensor("model.embed_tokens.weight") as tensor:
            row = tensor[token_id : token_id + 1].squeeze(0)
        if row.is_floating_point():
            row = row.to(self.config.dtype)
        if self.device.type == "cuda":
            row = row.pin_memory().to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        return row.to(self.device)

    def final_norm(self, hidden: torch.Tensor) -> torch.Tensor:
        with self._open_tensor("model.norm.weight") as weight:
            w = weight[:].to(self.config.dtype)
        if self.device.type == "cuda":
            w = w.pin_memory().to(self.device, non_blocking=True)
        return rms_norm(hidden, w, float(self.model_config.get("rms_norm_eps", 1e-5)))

    def logits(self, hidden: torch.Tensor, chunk_rows: int = 4096) -> torch.Tensor:
        if self.lm_head_cache is not None:
            self.lm_head_cache_hits += 1
            out = []
            rows = self.lm_head_cache.shape[0]
            for start in range(0, rows, chunk_rows):
                end = min(rows, start + chunk_rows)
                chunk = self.lm_head_cache[start:end]
                part = torch.matmul(chunk.float(), hidden.float())
                out.append(part.detach().cpu())
                del part
            return torch.cat(out, dim=0)
        self.lm_head_cache_misses += 1
        shard = self.index.weight_map["lm_head.weight"]
        from safetensors import safe_open

        out = []
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
            sliced = handle.get_slice("lm_head.weight")
            rows = sliced.get_shape()[0]
            for start in range(0, rows, chunk_rows):
                end = min(rows, start + chunk_rows)
                chunk = sliced[start:end].to(self.config.dtype)
                if self.device.type == "cuda":
                    chunk = chunk.pin_memory().to(self.device, non_blocking=True)
                else:
                    chunk = chunk.to(self.device)
                part = torch.matmul(chunk.float(), hidden.float())
                out.append(part.detach().cpu())
                del chunk, part
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return torch.cat(out, dim=0)

    def step_token(self, token_id: int, position: int) -> tuple[torch.Tensor, list[dict]]:
        hidden = self.embed_token(token_id)
        layer_logs: list[dict] = []
        for group_name in self.scheduler.layer_names():
            layer_idx = int(group_name.split("_")[-1])
            self.monitor.record(f"layer:{layer_idx}:start")
            cache_hit_before_load = layer_idx in self.layer_tensor_cache
            tensors = self._load_executable_layer_tensors(layer_idx)
            overrides = self._load_quantized_overrides(group_name)
            tensors.update(overrides)
            log = self._run_layer(layer_idx, hidden, tensors, position)
            layer_logs.append(log)
            tensors.clear()
            for tensor in overrides.values():
                del tensor
            if not cache_hit_before_load:
                self.scheduler.record_load()
            self.monitor.record(f"layer:{layer_idx}:end")
        return self.final_norm(hidden), layer_logs

    def _run_layer(self, layer_idx: int, hidden: torch.Tensor, tensors: dict[str, torch.Tensor], position: int) -> dict:
        prefix = f"model.layers.{layer_idx}"
        predicted_experts = self._prefetch_predicted_experts(layer_idx)
        input_norm = float(torch.linalg.vector_norm(hidden.float()).item())
        normed = rms_norm(hidden, tensors[f"{prefix}.input_layernorm.weight"], float(self.model_config.get("rms_norm_eps", 1e-5)))
        normed_norm = float(torch.linalg.vector_norm(normed.float()).item())
        attn_out, attention_diagnostics = self.attn.forward(
            layer_idx,
            normed,
            tensors,
            position,
            self.kv_cache,
            self._sliding_window_for_layer(layer_idx),
        )
        hidden += attn_out
        post_attention_residual_norm = float(torch.linalg.vector_norm(hidden.float()).item())
        post = rms_norm(hidden, tensors[f"{prefix}.post_attention_layernorm.weight"], float(self.model_config.get("rms_norm_eps", 1e-5)))
        post_norm = float(torch.linalg.vector_norm(post.float()).item())
        resident_experts = self.expert_store.resident_experts(layer_idx)
        moe_out, route = self.router.route(
            layer_idx,
            post,
            tensors,
            self.expert_store,
            self.config.execute_experts,
            position,
            predicted_experts=predicted_experts,
            resident_experts=resident_experts,
        )
        hidden += moe_out
        output_norm = float(torch.linalg.vector_norm(hidden.float()).item())
        return {
            "layer": layer_idx,
            "route": route,
            "input_norm": input_norm,
            "input_layernorm_output_norm": normed_norm,
            "attention_norm": float(torch.linalg.vector_norm(attn_out.float()).item()),
            "attention_diagnostics": attention_diagnostics,
            "post_attention_residual_norm": post_attention_residual_norm,
            "post_attention_layernorm_output_norm": post_norm,
            "moe_norm": float(torch.linalg.vector_norm(moe_out.float()).item()),
            "output_norm": output_norm,
            "residual_delta_norm": output_norm - input_norm,
            "quantized_overrides": sorted(name for name in tensors if self.quant_loader and name in self.quant_loader.tensor_index),
        }

    def _prefetch_predicted_experts(self, layer_idx: int) -> list[int]:
        if not (self.config.execute_experts and self.config.predictive_expert_prefetch):
            return []
        base_cap = int(self.config.active_experts_per_token_cap or self.model_config.get("experts_per_token", 4))
        current_cap = int(self.router.dynamic_caps.get(layer_idx, base_cap))
        configured_limit = int(self.config.expert_prefetch_limit or current_cap)
        limit = max(min(configured_limit, current_cap), 1)
        predicted = self.router.predict_experts(layer_idx, limit=limit)
        if predicted:
            self.expert_store.prefetch_experts(layer_idx, predicted)
        return predicted

    def _sliding_window_for_layer(self, layer_idx: int) -> int | None:
        layer_types = self.model_config.get("layer_types") or []
        if layer_idx < len(layer_types) and layer_types[layer_idx] == "sliding_attention":
            return int(self.model_config.get("sliding_window") or 0) or None
        return None

    def _load_quantized_overrides(self, group_name: str) -> dict[str, torch.Tensor]:
        if self.quant_loader is None:
            return {}
        overrides: dict[str, torch.Tensor] = {}
        for name in self.quant_loader.groups().get(group_name, []):
            if name.endswith("_blocks") or name.endswith("_scales"):
                continue
            tensor = self.quant_loader.load_tensor(name, dtype=self.config.dtype)[0]
            overrides[name] = tensor
            payload_bytes = self.quant_loader.payload_nbytes(name)
            resident_bytes = tensor.numel() * tensor.element_size()
            self.transfer_bytes += payload_bytes
            self.transfer_events.append(
                {
                    "kind": "quantized_override",
                    "group": group_name,
                    "name": name,
                    "bytes": payload_bytes,
                    "resident_bytes": resident_bytes,
                }
            )
        return overrides

    def _load_executable_layer_tensors(self, layer_idx: int) -> dict[str, torch.Tensor]:
        if layer_idx in self.layer_tensor_cache:
            self.layer_tensor_cache_hits += 1
            self.scheduler.record_hit()
            return dict(self.layer_tensor_cache[layer_idx])
        self.layer_tensor_cache_misses += 1
        prefix = f"model.layers.{layer_idx}"
        names = [
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.self_attn.o_proj.bias",
            f"{prefix}.self_attn.o_proj.weight",
            f"{prefix}.self_attn.q_proj.bias",
            f"{prefix}.self_attn.k_proj.bias",
            f"{prefix}.self_attn.v_proj.bias",
            f"{prefix}.self_attn.q_proj.weight",
            f"{prefix}.self_attn.k_proj.weight",
            f"{prefix}.self_attn.v_proj.weight",
            f"{prefix}.self_attn.sinks",
            f"{prefix}.mlp.router.bias",
            f"{prefix}.mlp.router.weight",
            f"{prefix}.mlp.experts.gate_up_proj_bias",
            f"{prefix}.mlp.experts.down_proj_bias",
            f"{prefix}.post_attention_layernorm.weight",
        ]
        by_shard: dict[str, list[str]] = {}
        for name in names:
            if self.config.use_quantized_overrides and self.quant_loader is not None and name in self.quant_loader.tensor_index:
                continue
            shard = self.index.weight_map.get(name)
            if shard:
                by_shard.setdefault(shard, []).append(name)
        tensors: dict[str, torch.Tensor] = {}
        for shard, shard_names in by_shard.items():
            with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    tensor = handle.get_tensor(name)
                    if tensor.is_floating_point():
                        tensor = tensor.to(self.config.dtype)
                    if self.device.type == "cuda":
                        tensor = tensor.pin_memory().to(self.device, non_blocking=True)
                    else:
                        tensor = tensor.to(self.device)
                    tensors[name] = tensor
                    nbytes = tensor.numel() * tensor.element_size()
                    self.transfer_bytes += nbytes
                    self.transfer_events.append({"kind": "original_tensor", "layer": layer_idx, "name": name, "bytes": nbytes})
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        if self.config.pin_layer_tensors:
            self.layer_tensor_cache[layer_idx] = tensors
            self.layer_tensor_cache_bytes += sum(t.numel() * t.element_size() for t in tensors.values())
            return dict(tensors)
        return tensors

    def reset(self) -> None:
        self.kv_cache.clear()
        self.router.stats.expert_hits.clear()
        self.router.stats.routed_tokens = 0

    def runtime_stats(self) -> dict:
        kv = self.kv_cache.extended_stats()
        return {
            "scheduler": {
                "layer_loads": self.scheduler.stats.layer_loads,
                "cache_hits": self.scheduler.stats.cache_hits,
                "cache_misses": self.scheduler.stats.cache_misses,
                "cache_hit_rate": self.scheduler.stats.cache_hit_rate,
                "prefetches": self.scheduler.stats.prefetches,
            },
            "kv_cache": kv,
            "router": self.router.stats.to_json(),
            "expert_cache": self.expert_store.to_json(),
            "transfer": {
                "bytes": self.transfer_bytes,
                "mb": round(self.transfer_bytes / 2**20, 6),
                "events": self.transfer_events[:200],
            },
            "hot_residency": {
                "enabled": self.config.hot_residency,
                "budget_mb": self.config.hot_vram_budget_mb,
                "layer_tensor_cache_bytes": self.layer_tensor_cache_bytes,
                "layer_tensor_cache_mb": round(self.layer_tensor_cache_bytes / 2**20, 6),
                "layer_tensor_cache_layers": len(self.layer_tensor_cache),
                "layer_tensor_cache_hits": self.layer_tensor_cache_hits,
                "layer_tensor_cache_misses": self.layer_tensor_cache_misses,
                "layer_tensor_cache_hit_rate": self.layer_tensor_cache_hits
                / max(self.layer_tensor_cache_hits + self.layer_tensor_cache_misses, 1),
                "lm_head_cache_bytes": self.lm_head_cache_bytes,
                "lm_head_cache_mb": round(self.lm_head_cache_bytes / 2**20, 6),
                "lm_head_cache_hits": self.lm_head_cache_hits,
                "lm_head_cache_misses": self.lm_head_cache_misses,
                "lm_head_resident": self.lm_head_cache is not None,
            },
            "vram_peak": self.monitor.peak(),
            "vram_events": self.monitor.to_json(),
        }

    def _open_tensor(self, name: str):
        from contextlib import contextmanager
        from safetensors import safe_open

        @contextmanager
        def manager():
            shard = self.index.weight_map[name]
            with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
                yield handle.get_slice(name)

        return manager()

    def _initialize_hot_residency(self) -> None:
        if self.config.hot_vram_budget_mb <= 0:
            return
        if self.config.pin_lm_head:
            self._pin_lm_head_if_budget_allows()

    def _pin_lm_head_if_budget_allows(self) -> None:
        shard = self.index.weight_map.get("lm_head.weight")
        if shard is None:
            return
        with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor("lm_head.weight")
        if tensor.is_floating_point():
            tensor = tensor.to(self.config.dtype)
        estimated = tensor.numel() * tensor.element_size()
        if estimated + self.layer_tensor_cache_bytes > self.config.hot_vram_budget_mb * 2**20:
            return
        if self.device.type == "cuda":
            tensor = tensor.pin_memory().to(self.device, non_blocking=True)
            torch.cuda.synchronize()
        else:
            tensor = tensor.to(self.device)
        self.lm_head_cache = tensor
        self.lm_head_cache_bytes = estimated
        self.transfer_bytes += estimated
        self.transfer_events.append({"kind": "hot_lm_head", "name": "lm_head.weight", "bytes": estimated})
