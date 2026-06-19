from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .kv_cache_manager import KVCacheManager


@dataclass
class AttentionConfig:
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float = 150000.0
    rope_factor: float = 1.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_original_max_position_embeddings: int = 4096
    rope_truncate: bool = False
    sliding_window: int | None = None
    rms_norm_eps: float = 1e-5


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(torch.mean(x.float() * x.float(), dim=-1, keepdim=True) + eps).to(x.dtype) * weight


def _yarn_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def yarn_attention_scaling(config: AttentionConfig) -> float:
    return _yarn_mscale(float(config.rope_factor))


def yarn_inv_freq(config: AttentionConfig, device: torch.device) -> torch.Tensor:
    """Match Transformers GPT-OSS YaRN RoPE inverse-frequency construction."""

    base = float(config.rope_theta)
    dim = int(config.head_dim)
    factor = float(config.rope_factor)
    original_max = int(config.rope_original_max_position_embeddings)
    beta_fast = float(config.rope_beta_fast)
    beta_slow = float(config.rope_beta_slow)
    truncate = bool(config.rope_truncate)

    def find_correction_dim(num_rotations: float) -> float:
        return (dim * math.log(original_max / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range() -> tuple[float, float]:
        low = find_correction_dim(beta_fast)
        high = find_correction_dim(beta_slow)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0.0), min(high, float(dim - 1))

    def linear_ramp_factor(low: float, high: float) -> torch.Tensor:
        if low == high:
            high += 0.001
        linear = (torch.arange(dim // 2, dtype=torch.float32, device=device) - low) / (high - low)
        return torch.clamp(linear, 0, 1)

    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
    inv_freq_extrapolation = 1.0 / pos_freqs
    inv_freq_interpolation = 1.0 / (factor * pos_freqs)
    low, high = find_correction_range()
    inv_freq_extrapolation_factor = 1 - linear_ramp_factor(low, high)
    return inv_freq_interpolation * (1 - inv_freq_extrapolation_factor) + inv_freq_extrapolation * inv_freq_extrapolation_factor


def rope_cos_sin(config: AttentionConfig, position: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = yarn_inv_freq(config, device)
    angles = inv_freq.float() * float(position)
    attention_scaling = yarn_attention_scaling(config)
    return (torch.cos(angles) * attention_scaling).to(dtype), (torch.sin(angles) * attention_scaling).to(dtype)


def apply_rope(q: torch.Tensor, k: torch.Tensor, position: int, config: AttentionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = rope_cos_sin(config, position, q.device, q.dtype)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        first_half, second_half = torch.chunk(x, 2, dim=-1)
        first = first_half * cos - second_half * sin
        second = second_half * cos + first_half * sin
        return torch.cat((first, second), dim=-1)

    return rotate(q), rotate(k)


class StreamedAttention:
    def __init__(self, config: AttentionConfig) -> None:
        self.config = config

    def forward(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        tensors: dict[str, torch.Tensor],
        position: int,
        kv_cache: KVCacheManager,
        sliding_window: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        prefix = f"model.layers.{layer_idx}.self_attn"
        x = hidden
        q = torch.nn.functional.linear(x, tensors[f"{prefix}.q_proj.weight"], tensors[f"{prefix}.q_proj.bias"])
        k = torch.nn.functional.linear(x, tensors[f"{prefix}.k_proj.weight"], tensors[f"{prefix}.k_proj.bias"])
        v = torch.nn.functional.linear(x, tensors[f"{prefix}.v_proj.weight"], tensors[f"{prefix}.v_proj.bias"])
        q = q.view(self.config.num_heads, self.config.head_dim)
        k = k.view(self.config.num_kv_heads, self.config.head_dim)
        v = v.view(self.config.num_kv_heads, self.config.head_dim)
        q_pre = q
        k_pre = k
        q, k = apply_rope(q, k, position, self.config)
        kv_cache.append(layer_idx, k, v)
        keys, values = kv_cache.get(layer_idx, sliding_window if sliding_window is not None else self.config.sliding_window)
        assert keys is not None and values is not None
        repeat = self.config.num_heads // self.config.num_kv_heads
        keys_h = keys.repeat_interleave(repeat, dim=1).transpose(0, 1)
        values_h = values.repeat_interleave(repeat, dim=1).transpose(0, 1)
        scores = torch.einsum("hd,htd->ht", q.float(), keys_h.float()) / math.sqrt(self.config.head_dim)
        sinks = tensors.get(f"{prefix}.sinks")
        if sinks is not None:
            sink_logits = sinks.float().reshape(-1, 1)
            combined = torch.cat([scores, sink_logits], dim=-1)
            combined = combined - combined.max(dim=-1, keepdim=True).values
            probs = torch.softmax(combined, dim=-1)[..., :-1].to(values_h.dtype)
        else:
            probs = torch.softmax(scores, dim=-1).to(values_h.dtype)
        context = torch.einsum("ht,htd->hd", probs, values_h).reshape(-1)
        o_weight = tensors[f"{prefix}.o_proj.weight"]
        context = context.to(o_weight.dtype)
        out = torch.nn.functional.linear(context, o_weight, tensors[f"{prefix}.o_proj.bias"])
        diagnostics = {
            "rope_position": int(position),
            "rope_attention_scaling": float(yarn_attention_scaling(self.config)),
            "sliding_window": sliding_window,
            "q_pre_rope_norm": float(torch.linalg.vector_norm(q_pre.float()).item()),
            "k_pre_rope_norm": float(torch.linalg.vector_norm(k_pre.float()).item()),
            "q_post_rope_norm": float(torch.linalg.vector_norm(q.float()).item()),
            "k_post_rope_norm": float(torch.linalg.vector_norm(k.float()).item()),
            "q_rope_delta_norm": float(torch.linalg.vector_norm((q.float() - q_pre.float())).item()),
            "k_rope_delta_norm": float(torch.linalg.vector_norm((k.float() - k_pre.float())).item()),
            "q_post_rope_nonfinite_count": int((~torch.isfinite(q)).sum().item()),
            "k_post_rope_nonfinite_count": int((~torch.isfinite(k)).sum().item()),
        }
        return out, diagnostics
