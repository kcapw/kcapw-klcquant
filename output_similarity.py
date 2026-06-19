from __future__ import annotations

import math
from collections import Counter

import torch


def token_jaccard(a: list[int], b: list[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def sequence_overlap(a: list[int], b: list[int]) -> float:
    if not a and not b:
        return 1.0
    limit = min(len(a), len(b))
    if limit == 0:
        return 0.0
    return sum(1 for i in range(limit) if a[i] == b[i]) / max(len(a), len(b))


def cosine_logits(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=0, eps=1e-12).item())


def logit_drift(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    diff = a.float() - b.float()
    return {
        "mae": float(diff.abs().mean().item()),
        "max_abs": float(diff.abs().max().item()),
        "relative_l2": float((torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(a.float()).clamp_min(1e-12)).item()),
        "cosine": cosine_logits(a, b),
    }


def perplexity_from_logprobs(logprobs: torch.Tensor, token_ids: list[int]) -> float:
    if not token_ids:
        return 0.0
    idx = torch.tensor(token_ids, dtype=torch.long, device=logprobs.device)
    nll = -logprobs[idx].mean()
    return float(torch.exp(nll.clamp_max(30)).item())


def hallucination_indicators(baseline_tokens: list[int], quantized_tokens: list[int], drift: dict[str, float]) -> dict[str, float | bool]:
    base_counts = Counter(baseline_tokens)
    quant_counts = Counter(quantized_tokens)
    repeated = sum(max(0, count - 1) for count in quant_counts.values())
    new_token_fraction = len(set(quantized_tokens) - set(baseline_tokens)) / max(len(set(quantized_tokens)), 1)
    length_delta = len(quantized_tokens) - len(baseline_tokens)
    risk = drift.get("relative_l2", 0.0) > 0.20 or new_token_fraction > 0.50 or repeated > max(2, len(quantized_tokens) // 3)
    return {
        "length_delta": length_delta,
        "new_token_fraction": new_token_fraction,
        "repeat_excess": repeated,
        "drift_risk": bool(risk),
        "baseline_unique": len(base_counts),
        "quantized_unique": len(quant_counts),
    }


def compare_outputs(
    baseline_logits: torch.Tensor,
    quantized_logits: torch.Tensor,
    baseline_tokens: list[int],
    quantized_tokens: list[int],
    target_token_ids: list[int],
) -> dict:
    base_logprobs = torch.log_softmax(baseline_logits.float(), dim=0)
    quant_logprobs = torch.log_softmax(quantized_logits.float(), dim=0)
    base_ppl = perplexity_from_logprobs(base_logprobs, target_token_ids)
    quant_ppl = perplexity_from_logprobs(quant_logprobs, target_token_ids)
    drift = logit_drift(baseline_logits, quantized_logits)
    return {
        "token_jaccard": token_jaccard(baseline_tokens, quantized_tokens),
        "sequence_overlap": sequence_overlap(baseline_tokens, quantized_tokens),
        "response_length_baseline": len(baseline_tokens),
        "response_length_quantized": len(quantized_tokens),
        "logit_drift": drift,
        "perplexity_baseline": base_ppl,
        "perplexity_quantized": quant_ppl,
        "perplexity_drift": quant_ppl - base_ppl if math.isfinite(base_ppl) and math.isfinite(quant_ppl) else 0.0,
        "hallucination_indicators": hallucination_indicators(baseline_tokens, quantized_tokens, drift),
    }
