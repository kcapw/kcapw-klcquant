from __future__ import annotations

from typing import Any

import torch

from .streamed_attention import AttentionConfig, apply_rope, rope_cos_sin, yarn_attention_scaling, yarn_inv_freq


def attention_config_from_model_config(model_config: dict[str, Any]) -> AttentionConfig:
    rope_params = model_config.get("rope_parameters") or model_config.get("rope_scaling") or {}
    return AttentionConfig(
        hidden_size=int(model_config["hidden_size"]),
        num_heads=int(model_config["num_attention_heads"]),
        num_kv_heads=int(model_config["num_key_value_heads"]),
        head_dim=int(model_config["head_dim"]),
        rope_theta=float(model_config.get("rope_theta", rope_params.get("rope_theta", 150000.0))),
        rope_factor=float(rope_params.get("factor", 1.0)),
        rope_beta_fast=float(rope_params.get("beta_fast", 32.0)),
        rope_beta_slow=float(rope_params.get("beta_slow", 1.0)),
        rope_original_max_position_embeddings=int(rope_params.get("original_max_position_embeddings", 4096)),
        rope_truncate=bool(rope_params.get("truncate", False)),
        sliding_window=None,
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-5)),
    )


def _hf_config(model_config: dict[str, Any]):
    from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

    rope_params = dict(model_config.get("rope_parameters") or model_config.get("rope_scaling") or {})
    rope_params.setdefault("rope_type", "yarn")
    rope_params.setdefault("rope_theta", float(model_config.get("rope_theta", 150000.0)))
    return GptOssConfig(
        num_hidden_layers=int(model_config["num_hidden_layers"]),
        num_local_experts=int(model_config["num_local_experts"]),
        vocab_size=int(model_config["vocab_size"]),
        hidden_size=int(model_config["hidden_size"]),
        intermediate_size=int(model_config["intermediate_size"]),
        head_dim=int(model_config["head_dim"]),
        num_attention_heads=int(model_config["num_attention_heads"]),
        num_key_value_heads=int(model_config["num_key_value_heads"]),
        sliding_window=model_config.get("sliding_window"),
        tie_word_embeddings=bool(model_config.get("tie_word_embeddings", False)),
        max_position_embeddings=int(model_config["max_position_embeddings"]),
        rms_norm_eps=float(model_config.get("rms_norm_eps", 1e-5)),
        rope_parameters=rope_params,
        attention_dropout=float(model_config.get("attention_dropout", 0.0)),
        num_experts_per_tok=int(model_config.get("num_experts_per_tok", model_config.get("experts_per_token", 4))),
        output_router_logits=bool(model_config.get("output_router_logits", False)),
        use_cache=bool(model_config.get("use_cache", True)),
        layer_types=model_config.get("layer_types"),
        pad_token_id=model_config.get("pad_token_id"),
        eos_token_id=model_config.get("eos_token_id"),
        attention_bias=bool(model_config.get("attention_bias", True)),
    )


def rope_reference_parity(model_config: dict[str, Any], layers: list[int] | None = None, positions: list[int] | None = None) -> dict[str, Any]:
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.gpt_oss.modeling_gpt_oss import apply_rotary_pos_emb

    device = torch.device("cpu")
    local_cfg = attention_config_from_model_config(model_config)
    hf_cfg = _hf_config(model_config)
    local_inv = yarn_inv_freq(local_cfg, device)
    hf_inv, hf_scaling = ROPE_INIT_FUNCTIONS["yarn"](hf_cfg, device)
    inv_diff = (local_inv - hf_inv).abs()
    scaling_diff = abs(yarn_attention_scaling(local_cfg) - float(hf_scaling))
    layers = layers if layers is not None else list(range(int(model_config["num_hidden_layers"])))
    positions = positions if positions is not None else [0, 1, 7, 15, 127, 128]
    rows: list[dict[str, Any]] = []
    for layer_idx in layers:
        for position in positions:
            generator = torch.Generator(device="cpu").manual_seed(10_000 + layer_idx * 257 + position)
            q = torch.randn(1, local_cfg.num_heads, 1, local_cfg.head_dim, generator=generator, dtype=torch.float32)
            k = torch.randn(1, local_cfg.num_kv_heads, 1, local_cfg.head_dim, generator=generator, dtype=torch.float32)
            cos, sin = rope_cos_sin(local_cfg, position, device, torch.float32)
            hf_q, hf_k = apply_rotary_pos_emb(q, k, cos.reshape(1, 1, -1), sin.reshape(1, 1, -1))
            local_q, local_k = apply_rope(q.squeeze(0).squeeze(1), k.squeeze(0).squeeze(1), position, local_cfg)
            q_diff = (local_q.reshape_as(hf_q) - hf_q).abs()
            k_diff = (local_k.reshape_as(hf_k) - hf_k).abs()
            rows.append(
                {
                    "layer": int(layer_idx),
                    "position": int(position),
                    "q_max_abs_diff": float(q_diff.max().item()),
                    "q_mean_abs_diff": float(q_diff.mean().item()),
                    "k_max_abs_diff": float(k_diff.max().item()),
                    "k_mean_abs_diff": float(k_diff.mean().item()),
                }
            )
    return {
        "local_attention_scaling": float(yarn_attention_scaling(local_cfg)),
        "reference_attention_scaling": float(hf_scaling),
        "attention_scaling_abs_diff": float(scaling_diff),
        "inv_freq_max_abs_diff": float(inv_diff.max().item()),
        "inv_freq_mean_abs_diff": float(inv_diff.mean().item()),
        "layer_position_rows": rows,
        "passed": bool(inv_diff.max().item() <= 1e-8 and scaling_diff <= 1e-12 and max((r["q_max_abs_diff"] for r in rows), default=0.0) <= 1e-7),
    }
