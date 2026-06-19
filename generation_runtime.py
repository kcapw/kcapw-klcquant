from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch

from .output_similarity import hallucination_indicators, sequence_overlap, token_jaccard
from .streamed_transformer_executor import ExecutorConfig, StreamedTransformerExecutor


@dataclass
class GenerationConfig:
    max_new_tokens: int = 2
    max_context_tokens: int = 4
    top_k: int = 8
    temperature: float = 0.0
    lm_head_chunk_rows: int = 4096
    stop_token_ids: list[int] | None = None


def _logit_summary(logits: torch.Tensor, logprobs: torch.Tensor, top_k: int) -> dict:
    k = min(top_k, logits.numel())
    top_logits, top_ids = torch.topk(logits.float(), k=k)
    top_logprobs = logprobs[top_ids].float()
    probs = torch.softmax(logits.float(), dim=0)
    entropy = float((-(probs * logprobs.float()).sum()).item())
    finite = torch.isfinite(logits)
    return {
        "top_token_ids": [int(x) for x in top_ids.tolist()],
        "top_logits": [float(x) for x in top_logits.tolist()],
        "top_logprobs": [float(x) for x in top_logprobs.tolist()],
        "entropy": entropy,
        "logits_finite": bool(finite.all().item()),
        "nonfinite_logit_count": int((~finite).sum().item()),
    }


class GenerationRuntime:
    def __init__(
        self,
        model_dir: str | Path,
        support_dir: str | Path,
        quantized_dir: str | Path,
        executor_config: ExecutorConfig,
        generation_config: GenerationConfig,
        device: str = "cuda",
    ) -> None:
        self.model_dir = model_dir
        self.support_dir = support_dir
        self.quantized_dir = quantized_dir
        self.executor_config = executor_config
        self.generation_config = generation_config
        self.device = device

    def generate(self, prompt: str, tokenizer, quantized: bool) -> dict:
        config = ExecutorConfig(
            max_layers=self.executor_config.max_layers,
            cache_layers_mb=self.executor_config.cache_layers_mb,
            expert_cache_mb=self.executor_config.expert_cache_mb,
            offload_kv_cache=self.executor_config.offload_kv_cache,
            use_quantized_overrides=quantized,
            execute_experts=self.executor_config.execute_experts,
            kv_cache_precision=self.executor_config.kv_cache_precision,
            dtype=self.executor_config.dtype,
            hot_residency=self.executor_config.hot_residency,
            hot_vram_budget_mb=self.executor_config.hot_vram_budget_mb,
            pin_lm_head=self.executor_config.pin_lm_head,
            pin_layer_tensors=self.executor_config.pin_layer_tensors,
            routing_locality=self.executor_config.routing_locality,
            sticky_routing_strength=self.executor_config.sticky_routing_strength,
            sticky_routing_decay=self.executor_config.sticky_routing_decay,
            routing_semantic_guard=self.executor_config.routing_semantic_guard,
            sticky_candidate_margin=self.executor_config.sticky_candidate_margin,
            max_sticky_bonus=self.executor_config.max_sticky_bonus,
            min_raw_route_overlap=self.executor_config.min_raw_route_overlap,
            max_hot_experts_per_layer=self.executor_config.max_hot_experts_per_layer,
            active_experts_per_token_cap=self.executor_config.active_experts_per_token_cap,
            routing_exploration_margin=self.executor_config.routing_exploration_margin,
            cache_aware_routing_strength=self.executor_config.cache_aware_routing_strength,
            predictive_expert_prefetch=self.executor_config.predictive_expert_prefetch,
            expert_prefetch_limit=self.executor_config.expert_prefetch_limit,
            expert_async_prefetch=self.executor_config.expert_async_prefetch,
            routing_prediction_window=self.executor_config.routing_prediction_window,
            routing_workload_window=self.executor_config.routing_workload_window,
            dynamic_active_expert_cap=self.executor_config.dynamic_active_expert_cap,
            min_active_experts_per_token=self.executor_config.min_active_experts_per_token,
            max_active_experts_per_token=self.executor_config.max_active_experts_per_token,
            layer_aware_expert_cache=self.executor_config.layer_aware_expert_cache,
            expert_matmul_precision=self.executor_config.expert_matmul_precision,
            expert_protect_tokens=self.executor_config.expert_protect_tokens,
            adaptive_layer_quota=self.executor_config.adaptive_layer_quota,
            adaptive_layer_quota_strength=self.executor_config.adaptive_layer_quota_strength,
        )
        executor = StreamedTransformerExecutor(self.model_dir, self.support_dir, self.quantized_dir, self.device, config)
        token_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][: self.generation_config.max_context_tokens]
        if not token_ids:
            token_ids = [tokenizer.eos_token_id or 0]
        generated: list[int] = []
        token_nlls: list[float] = []
        token_logs: list[dict] = []
        start = time.perf_counter()
        position = 0
        current = int(token_ids[0])
        context = [int(t) for t in token_ids]
        with torch.no_grad():
            for token in context[:-1]:
                _hidden, logs = executor.step_token(int(token), position)
                token_logs.append({"prefill_token": int(token), "layers": logs})
                position += 1
            current = int(context[-1])
            for _ in range(self.generation_config.max_new_tokens):
                token_start = time.perf_counter()
                hidden, logs = executor.step_token(current, position)
                final_hidden_float = hidden.float()
                final_hidden_finite = torch.isfinite(final_hidden_float)
                raw_logits = executor.logits(hidden, self.generation_config.lm_head_chunk_rows).float()
                raw_logit_finite = torch.isfinite(raw_logits)
                logits = torch.nan_to_num(raw_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                logprobs = torch.log_softmax(logits, dim=0)
                next_id = int(torch.argmax(logits).item())
                token_nlls.append(float((-logprobs[next_id]).item()))
                logit_summary = _logit_summary(logits, logprobs, self.generation_config.top_k)
                token_latency = time.perf_counter() - token_start
                generated.append(next_id)
                token_logs.append(
                    {
                        "input_token": current,
                        "generated_token": next_id,
                        "token_latency_s": round(token_latency, 6),
                        "final_hidden_norm_before_lm_head": float(torch.linalg.vector_norm(final_hidden_float).item()),
                        "final_hidden_mean_before_lm_head": float(final_hidden_float.mean().item()),
                        "final_hidden_std_before_lm_head": float(final_hidden_float.std(unbiased=False).item()),
                        "final_hidden_max_abs_before_lm_head": float(final_hidden_float.abs().max().item()),
                        "final_hidden_finite": bool(final_hidden_finite.all().item()),
                        "final_hidden_nonfinite_count": int((~final_hidden_finite).sum().item()),
                        "raw_nonfinite_logit_count": int((~raw_logit_finite).sum().item()),
                        **logit_summary,
                        "layers": logs,
                    }
                )
                current = next_id
                position += 1
                if self.generation_config.stop_token_ids and next_id in self.generation_config.stop_token_ids:
                    break
        elapsed = time.perf_counter() - start
        generated_text = tokenizer.decode(generated)
        return {
            "quantized": quantized,
            "prompt": prompt,
            "context_token_ids": context,
            "generated_token_ids": generated,
            "generated_text": generated_text,
            "assistant_final_text": extract_assistant_final_text(generated_text),
            "mean_token_nll": sum(token_nlls) / max(len(token_nlls), 1),
            "perplexity": float(torch.exp(torch.tensor(sum(token_nlls) / max(len(token_nlls), 1))).item()) if token_nlls else 0.0,
            "latency_s": round(elapsed, 6),
            "token_logs": token_logs,
            "runtime_stats": executor.runtime_stats(),
        }


def extract_assistant_final_text(text: str) -> str:
    marker = "<|channel|>final<|message|>"
    if marker in text:
        text = text.split(marker)[-1]
    if "<|return|>" in text:
        text = text.split("<|return|>")[0]
    if "<|end|>" in text:
        text = text.split("<|end|>")[0]
    if "<|start|>" in text:
        text = text.split("<|start|>")[0]
    return text.strip()


def compare_generations(baseline: dict, quantized: dict) -> dict:
    base = baseline["generated_token_ids"]
    quant = quantized["generated_token_ids"]
    limit = min(len(base), len(quant))
    divergence_index = next((idx for idx in range(limit) if base[idx] != quant[idx]), None)
    if divergence_index is None and len(base) != len(quant):
        divergence_index = limit
    return {
        "token_jaccard": token_jaccard(base, quant),
        "sequence_overlap": sequence_overlap(base, quant),
        "per_token_agreement": [base[idx] == quant[idx] for idx in range(limit)],
        "first_divergence_index": divergence_index,
        "response_length_delta": len(quant) - len(base),
        "perplexity_baseline": baseline.get("perplexity", 0.0),
        "perplexity_quantized": quantized.get("perplexity", 0.0),
        "perplexity_drift": quantized.get("perplexity", 0.0) - baseline.get("perplexity", 0.0),
        "hallucination_indicators": hallucination_indicators(base, quant, {"relative_l2": 0.0}),
        "latency_delta_s": quantized["latency_s"] - baseline["latency_s"],
    }
