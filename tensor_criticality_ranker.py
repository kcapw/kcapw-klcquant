from __future__ import annotations

from collections import defaultdict
from math import isfinite
from typing import Iterable

from .sensitivity_database import SensitivityRecord


PROTECTED_HINTS = ("router", "self_attn", "norm", "embed_tokens", "lm_head")
EXPERT_HINTS = ("mlp.experts", "gate_up_proj", "down_proj")
KV_HINTS = ("k_proj", "v_proj")


def tensor_role(name: str) -> str:
    if "router" in name:
        return "router"
    if "self_attn" in name:
        return "attention_kv" if any(token in name for token in KV_HINTS) else "attention"
    if "norm" in name:
        return "normalization"
    if "mlp.experts" in name:
        if "down_proj" in name:
            return "expert_down_projection"
        if "gate_up_proj" in name:
            return "expert_gate_up_projection"
        return "expert"
    if "embed_tokens" in name or "lm_head" in name:
        return "io_core"
    return "other"


def structural_prior(name: str) -> float:
    score = 0.0
    if any(token in name for token in PROTECTED_HINTS):
        score += 0.35
    if any(token in name for token in EXPERT_HINTS):
        score += 0.15
    if any(token in name for token in KV_HINTS):
        score += 0.15
    return min(score, 0.6)


def score_metrics(metrics: dict) -> float:
    seq_loss = 1.0 - float(metrics.get("sequence_overlap", 1.0))
    token_loss = 1.0 - float(metrics.get("token_jaccard", 1.0))
    agreement = float(metrics.get("token_agreement_rate", 1.0))
    agreement_loss = 1.0 - agreement
    ppl = min(abs(float(metrics.get("perplexity_drift", 0.0))) / 25.0, 2.0)
    entropy = min(abs(float(metrics.get("entropy_drift", 0.0))) / 2.0, 1.0)
    cosine_loss = 1.0 - float(metrics.get("sparse_logit_cosine", 1.0))
    if not isfinite(cosine_loss):
        cosine_loss = 1.0
    divergence = 0.0
    first = metrics.get("first_divergence_token")
    generated = max(int(metrics.get("generated_tokens", 1)), 1)
    if first is not None:
        divergence = 1.0 - (int(first) / generated)
    collapse = 0.35 if metrics.get("generation_collapse") else 0.0
    repetition = 0.25 if metrics.get("repetition_instability") else 0.0
    return max(
        0.0,
        0.22 * seq_loss
        + 0.18 * token_loss
        + 0.12 * agreement_loss
        + 0.16 * ppl
        + 0.12 * divergence
        + 0.08 * entropy
        + 0.07 * cosine_loss
        + collapse
        + repetition,
    )


def score_record(record: SensitivityRecord) -> float:
    penalty = 0.45 if not record.stable else 0.0
    return min(2.0, penalty + score_metrics(record.metrics) + structural_prior(record.tensor))


def rank_tensor_criticality(records: Iterable[SensitivityRecord]) -> list[dict]:
    grouped: dict[str, list[SensitivityRecord]] = defaultdict(list)
    for record in records:
        grouped[record.tensor].append(record)

    ranking: list[dict] = []
    for tensor, rows in grouped.items():
        scored = [(score_record(row), row) for row in rows]
        max_score = max(score for score, _ in scored)
        avg_score = sum(score for score, _ in scored) / max(len(scored), 1)
        unstable = sum(1 for _, row in scored if not row.stable)
        hardest_stable_mode = _hardest_stable_mode(rows)
        earliest_divergence = min(
            (int(row.metrics["first_divergence_token"]) for row in rows if row.metrics.get("first_divergence_token") is not None),
            default=None,
        )
        ranking.append(
            {
                "tensor": tensor,
                "role": tensor_role(tensor),
                "criticality_score": round(max_score, 6),
                "avg_score": round(avg_score, 6),
                "unstable_runs": unstable,
                "total_runs": len(rows),
                "hardest_stable_mode": hardest_stable_mode,
                "earliest_divergence_token": earliest_divergence,
                "structural_prior": structural_prior(tensor),
                "static": rows[-1].static,
            }
        )
    return sorted(ranking, key=lambda item: (item["criticality_score"], item["unstable_runs"]), reverse=True)


def _hardest_stable_mode(rows: list[SensitivityRecord]) -> str | None:
    order = {"fp16": 0, "q8": 1, "q4": 2, "q3": 3, "q2": 4, "q1": 5, "pruned": 6}
    stable = [row.mode for row in rows if row.stable]
    if not stable:
        return None
    return max(stable, key=lambda mode: order.get(mode, -1))

