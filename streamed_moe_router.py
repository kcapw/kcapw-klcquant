from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math

import torch

from .mxfp4_expert import Mxfp4ExpertStore


@dataclass
class RoutingLocalityConfig:
    enabled: bool = False
    sticky_strength: float = 0.0
    sticky_decay: float = 0.92
    semantic_guard: bool = False
    sticky_candidate_margin: float = 0.50
    max_sticky_bonus: float = 0.25
    min_raw_overlap: int = 2
    max_hot_experts_per_layer: int = 0
    active_experts_per_token_cap: int = 0
    exploration_margin: float = 0.25
    cache_aware_strength: float = 0.0
    predictive_prefetch: bool = False
    prediction_window: int = 16
    workload_window: int = 64
    dynamic_active_expert_cap: bool = False
    min_active_experts_per_token: int = 1
    max_active_experts_per_token: int = 0
    cap_eviction_pressure_threshold: float = 0.50
    cap_hit_rate_expand_threshold: float = 0.65
    thrash_evictions_per_activation: float = 0.35
    thrash_reuse_density: float = 0.15
    thrash_active_growth: float = 0.25


@dataclass
class RouterStats:
    expert_hits: Counter = field(default_factory=Counter)
    routed_tokens: int = 0
    route_events: list[dict] = field(default_factory=list)
    last_seen_position: dict[tuple[int, int], int] = field(default_factory=dict)
    reuse_distances: list[int] = field(default_factory=list)
    active_by_position: dict[int, set[tuple[int, int]]] = field(default_factory=dict)
    route_entropies: list[float] = field(default_factory=list)
    cache_delta_events: list[dict] = field(default_factory=list)
    cap_events: list[dict] = field(default_factory=list)
    guarded_adjustments: int = 0
    blocked_adjustments: int = 0
    raw_overlap_total: int = 0
    route_count: int = 0

    def to_json(self) -> dict:
        activations = sum(self.expert_hits.values())
        unique_active = len(self.expert_hits)
        reused = len(self.reuse_distances)
        eviction_delta = sum(int(item.get("evictions", 0)) for item in self.cache_delta_events)
        active_rows = [
            {"position": int(position), "active_expert_count": len(experts)}
            for position, experts in sorted(self.active_by_position.items())
        ]
        active_counts = [row["active_expert_count"] for row in active_rows]
        active_growth = 0.0
        if len(active_counts) >= 2:
            active_growth = (active_counts[-1] - active_counts[0]) / max(active_counts[0], 1)
        reuse_density = reused / max(activations, 1)
        evictions_per_activation = eviction_delta / max(activations, 1)
        entropy = self._distribution_entropy()
        evictions_per_1k = 1000.0 * eviction_delta / max(self.routed_tokens, 1)
        expert_thrash_mode = evictions_per_activation >= 0.35 and (
            reuse_density <= 0.15
            or active_growth >= 0.25
            or evictions_per_1k >= 1000.0
            or unique_active >= max(activations * 0.50, 1)
        )
        return {
            "routed_tokens": self.routed_tokens,
            "expert_activation_frequency": {str(k): v for k, v in self.expert_hits.most_common()},
            "locality": {
                "total_expert_activations": activations,
                "unique_expert_activations": unique_active,
                "reuse_density": reuse_density,
                "reuse_distance": self._reuse_distance_summary(),
                "routing_entropy_mean": sum(self.route_entropies) / max(len(self.route_entropies), 1),
                "routing_entropy_distribution": entropy,
                "active_expert_set_size_over_time": active_rows,
                "active_expert_set_growth": active_growth,
                "cache_evictions_observed": eviction_delta,
                "evictions_per_activation": evictions_per_activation,
                "evictions_per_1k_routed_tokens": evictions_per_1k,
                "expert_thrash_mode": expert_thrash_mode,
                "cache_hit_eviction_correlation": self._cache_hit_eviction_correlation(),
                "prediction_accuracy": self._prediction_accuracy(),
                "dynamic_cap_events": self.cap_events[:500],
                "guarded_adjustments": int(self.guarded_adjustments),
                "blocked_adjustments": int(self.blocked_adjustments),
                "mean_raw_route_overlap": self.raw_overlap_total / max(self.route_count, 1),
            },
            "route_events": self.route_events[:500],
            "cache_delta_events": self.cache_delta_events[:500],
        }

    def record_route(
        self,
        layer_idx: int,
        position: int,
        experts: list[int],
        weights: list[float],
        raw_experts: list[int],
        adjusted: bool,
        predicted_experts: list[int] | None = None,
        resident_experts: list[int] | None = None,
        active_cap: int | None = None,
        blocked_adjustment: bool = False,
    ) -> None:
        self.routed_tokens += 1
        self.route_count += 1
        route_entropy = -sum(float(w) * math.log(max(float(w), 1e-12)) for w in weights)
        self.route_entropies.append(route_entropy)
        raw_overlap = len(set(int(x) for x in experts) & set(int(x) for x in raw_experts))
        self.raw_overlap_total += raw_overlap
        if adjusted:
            self.guarded_adjustments += 1
        if blocked_adjustment:
            self.blocked_adjustments += 1
        active = self.active_by_position.setdefault(int(position), set())
        reuse = []
        for expert_id in experts:
            key = (int(layer_idx), int(expert_id))
            self.expert_hits[key] += 1
            active.add(key)
            previous = self.last_seen_position.get(key)
            if previous is not None:
                distance = int(position) - previous
                self.reuse_distances.append(distance)
                reuse.append(distance)
            self.last_seen_position[key] = int(position)
        self.route_events.append(
            {
                "position": int(position),
                "layer": int(layer_idx),
                "experts": [int(x) for x in experts],
                "raw_experts": [int(x) for x in raw_experts],
                "predicted_experts": [int(x) for x in (predicted_experts or [])],
                "resident_experts": [int(x) for x in (resident_experts or [])],
                "weights": [float(x) for x in weights],
                "route_entropy": route_entropy,
                "reuse_distances": reuse,
                "sticky_adjusted": adjusted,
                "blocked_locality_adjustment": blocked_adjustment,
                "raw_route_overlap": raw_overlap,
                "active_expert_cap": active_cap,
            }
        )

    def record_cache_delta(self, layer_idx: int, position: int, experts: list[int], before: dict, after: dict) -> None:
        self.cache_delta_events.append(
            {
                "position": int(position),
                "layer": int(layer_idx),
                "experts": [int(x) for x in experts],
                "hits": int(after.get("hits", 0)) - int(before.get("hits", 0)),
                "misses": int(after.get("misses", 0)) - int(before.get("misses", 0)),
                "evictions": int(after.get("evictions", 0)) - int(before.get("evictions", 0)),
            }
        )

    def record_cap_event(self, layer_idx: int, position: int, old_cap: int, new_cap: int, reason: str, hits: int, misses: int, evictions: int) -> None:
        if old_cap == new_cap:
            return
        self.cap_events.append(
            {
                "position": int(position),
                "layer": int(layer_idx),
                "old_cap": int(old_cap),
                "new_cap": int(new_cap),
                "reason": reason,
                "hits": int(hits),
                "misses": int(misses),
                "evictions": int(evictions),
            }
        )

    def _reuse_distance_summary(self) -> dict:
        if not self.reuse_distances:
            return {"count": 0, "mean": None, "p50": None, "p90": None, "max": None}
        values = sorted(self.reuse_distances)
        return {
            "count": len(values),
            "mean": sum(values) / len(values),
            "p50": values[int(0.50 * (len(values) - 1))],
            "p90": values[int(0.90 * (len(values) - 1))],
            "max": values[-1],
        }

    def _distribution_entropy(self) -> float:
        total = sum(self.expert_hits.values())
        if total == 0:
            return 0.0
        return -sum((count / total) * math.log(max(count / total, 1e-12)) for count in self.expert_hits.values())

    def _cache_hit_eviction_correlation(self) -> dict:
        if not self.cache_delta_events:
            return {"sample_count": 0, "correlation": None}
        hits = [float(item.get("hits", 0)) for item in self.cache_delta_events]
        evictions = [float(item.get("evictions", 0)) for item in self.cache_delta_events]
        mean_h = sum(hits) / len(hits)
        mean_e = sum(evictions) / len(evictions)
        cov = sum((h - mean_h) * (e - mean_e) for h, e in zip(hits, evictions))
        var_h = sum((h - mean_h) ** 2 for h in hits)
        var_e = sum((e - mean_e) ** 2 for e in evictions)
        denom = math.sqrt(var_h * var_e)
        return {"sample_count": len(hits), "correlation": cov / denom if denom > 0 else None}

    def _prediction_accuracy(self) -> dict:
        events = [item for item in self.route_events if item.get("predicted_experts")]
        if not events:
            return {"sample_count": 0, "mean_precision": None, "mean_recall": None, "exact_match_rate": None}
        precisions = []
        recalls = []
        exact = 0
        for item in events:
            actual = set(int(x) for x in item.get("experts", []))
            predicted = set(int(x) for x in item.get("predicted_experts", []))
            overlap = len(actual & predicted)
            precisions.append(overlap / max(len(predicted), 1))
            recalls.append(overlap / max(len(actual), 1))
            exact += int(actual == predicted)
        return {
            "sample_count": len(events),
            "mean_precision": sum(precisions) / len(precisions),
            "mean_recall": sum(recalls) / len(recalls),
            "exact_match_rate": exact / len(events),
        }


class StreamedMoERouter:
    def __init__(
        self,
        experts_per_token: int = 4,
        swiglu_limit: float = 7.0,
        locality_config: RoutingLocalityConfig | None = None,
        matmul_precision: str = "fp32",
    ) -> None:
        self.experts_per_token = experts_per_token
        self.swiglu_limit = swiglu_limit
        self.alpha = 1.702
        self.stats = RouterStats()
        self.locality_config = locality_config or RoutingLocalityConfig()
        self.matmul_precision = matmul_precision
        self.sticky_scores: dict[int, dict[int, float]] = {}
        self.workload_scores: dict[int, dict[int, float]] = {}
        self.recent_history: dict[int, deque[int]] = {}
        self.dynamic_caps: dict[int, int] = {}

    def route(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        tensors: dict[str, torch.Tensor],
        expert_store: Mxfp4ExpertStore | None = None,
        execute_experts: bool = True,
        position: int = 0,
        predicted_experts: list[int] | None = None,
        resident_experts: set[int] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        prefix = f"model.layers.{layer_idx}.mlp"
        logits = torch.nn.functional.linear(hidden, tensors[f"{prefix}.router.weight"], tensors[f"{prefix}.router.bias"])
        k = self._effective_expert_count(layer_idx, logits.numel())
        raw_top = torch.topk(logits.float(), k=k)
        adjusted_logits = self._apply_locality_bias(layer_idx, logits.float(), resident_experts or set())
        top = torch.topk(adjusted_logits, k=k)
        experts, score_values, blocked_adjustment = self._guarded_expert_selection(
            logits.float(),
            adjusted_logits,
            raw_top.indices.tolist(),
            top.indices.tolist(),
            k,
        )
        score_tensor = torch.tensor(score_values, device=logits.device, dtype=torch.float32)
        weights = torch.softmax(score_tensor, dim=-1).to(hidden.dtype)
        weights_list = [float(w) for w in weights]
        raw_experts = raw_top.indices.tolist()
        adjusted = experts != raw_experts
        predicted = predicted_experts if predicted_experts is not None else self.predict_experts(layer_idx, k)
        resident = sorted(int(x) for x in (resident_experts or set()))
        self.stats.record_route(
            layer_idx,
            position,
            [int(x) for x in experts],
            weights_list,
            [int(x) for x in raw_experts],
            adjusted,
            predicted_experts=[int(x) for x in predicted],
            resident_experts=resident,
            active_cap=k,
            blocked_adjustment=blocked_adjustment,
        )
        self._update_sticky_scores(layer_idx, [int(x) for x in experts], weights_list)
        self._update_workload_scores(layer_idx, [int(x) for x in experts], weights_list)

        if execute_experts and expert_store is not None:
            before = self._cache_snapshot(expert_store)
            expert_store.protect_experts(layer_idx, [int(x) for x in experts])
            output, route = self._execute_mxfp4_experts(layer_idx, hidden, tensors, experts, weights, expert_store)
            after = self._cache_snapshot(expert_store)
            self.stats.record_cache_delta(layer_idx, position, [int(x) for x in experts], before, after)
            self._adapt_active_cap(layer_idx, position, before, after)
            route["raw_experts"] = [int(x) for x in raw_experts]
            route["predicted_experts"] = [int(x) for x in predicted]
            route["resident_experts"] = resident
            route["sticky_adjusted"] = adjusted
            route["active_expert_cap"] = k
            return output, route

        down_bias = tensors.get(f"{prefix}.experts.down_proj_bias")
        if down_bias is None:
            return torch.zeros_like(hidden), {"experts": experts, "weights": [float(w) for w in weights], "mode": "router_only"}
        expert_index = torch.tensor([int(x) for x in experts], device=down_bias.device, dtype=torch.long)
        selected = down_bias[expert_index].to(hidden.dtype)
        contribution = torch.sum(selected * weights[:, None], dim=0)
        return contribution, {
            "experts": experts,
            "raw_experts": [int(x) for x in raw_experts],
            "weights": weights_list,
            "mode": "bias_only_moe",
            "sticky_adjusted": adjusted,
        }

    def _apply_locality_bias(self, layer_idx: int, logits: torch.Tensor, resident_experts: set[int]) -> torch.Tensor:
        cfg = self.locality_config
        if not cfg.enabled:
            return logits
        adjusted = logits.clone()
        raw_k = min(self.experts_per_token, logits.numel())
        raw_top = torch.topk(logits, k=raw_k)
        candidate_floor = raw_top.values[-1] - abs(float(cfg.sticky_candidate_margin))
        scores = self.sticky_scores.get(layer_idx, {})
        if scores and cfg.sticky_strength > 0:
            eligible_ids = [int(expert_id) for expert_id in scores if logits[int(expert_id)] >= candidate_floor]
            if eligible_ids:
                ids = torch.tensor(eligible_ids, device=logits.device, dtype=torch.long)
                vals = torch.tensor([scores[int(idx)] for idx in ids.tolist()], device=logits.device, dtype=logits.dtype)
                bonus = vals * float(cfg.sticky_strength)
                if cfg.semantic_guard and cfg.max_sticky_bonus > 0:
                    bonus = torch.clamp(bonus, max=float(cfg.max_sticky_bonus))
                adjusted[ids] += bonus
        if cfg.max_hot_experts_per_layer > 0 and len(scores) >= cfg.max_hot_experts_per_layer:
            hot = sorted(scores, key=scores.get, reverse=True)[: cfg.max_hot_experts_per_layer]
            hot_set = set(int(x) for x in hot)
            raw_top = torch.topk(logits, k=min(self.experts_per_token, logits.numel())).indices.tolist()
            allowed = hot_set | {int(x) for x in raw_top[: max(1, self.experts_per_token // 2)]}
            mask = torch.ones_like(adjusted, dtype=torch.bool)
            mask[list(allowed)] = False
            adjusted[mask] -= abs(float(cfg.exploration_margin))
        if resident_experts and cfg.cache_aware_strength > 0:
            ids = torch.tensor(sorted(resident_experts), device=logits.device, dtype=torch.long)
            threshold = torch.max(logits) - abs(float(cfg.exploration_margin))
            eligible = ids[logits[ids] >= threshold]
            if eligible.numel() > 0:
                adjusted[eligible] += float(cfg.cache_aware_strength)
        return adjusted

    def _guarded_expert_selection(
        self,
        logits: torch.Tensor,
        adjusted_logits: torch.Tensor,
        raw_experts: list[int],
        adjusted_experts: list[int],
        k: int,
    ) -> tuple[list[int], list[float], bool]:
        cfg = self.locality_config
        if not (cfg.enabled and cfg.semantic_guard):
            return [int(x) for x in adjusted_experts], [float(adjusted_logits[int(x)].item()) for x in adjusted_experts], False
        raw = [int(x) for x in raw_experts[:k]]
        selected = [int(x) for x in adjusted_experts[:k]]
        raw_set = set(raw)
        selected_set = set(selected)
        min_overlap = min(max(int(cfg.min_raw_overlap), 0), k)
        blocked = False
        while len(raw_set & selected_set) < min_overlap:
            missing_raw = [expert_id for expert_id in raw if expert_id not in selected_set]
            replaceable = [expert_id for expert_id in selected if expert_id not in raw_set]
            if not missing_raw or not replaceable:
                break
            bring_back = missing_raw[0]
            evict = min(replaceable, key=lambda expert_id: float(adjusted_logits[expert_id].item()))
            selected[selected.index(evict)] = bring_back
            selected_set = set(selected)
            blocked = True
        selected = sorted(selected, key=lambda expert_id: float(adjusted_logits[expert_id].item()), reverse=True)
        values = [float(logits[int(expert_id)].item()) for expert_id in selected]
        return selected, values, blocked

    def _update_sticky_scores(self, layer_idx: int, experts: list[int], weights: list[float]) -> None:
        cfg = self.locality_config
        if not cfg.enabled:
            return
        scores = self.sticky_scores.setdefault(layer_idx, {})
        for expert_id in list(scores):
            scores[expert_id] *= float(cfg.sticky_decay)
            if scores[expert_id] < 1e-4:
                del scores[expert_id]
        for expert_id, weight in zip(experts, weights):
            scores[int(expert_id)] = scores.get(int(expert_id), 0.0) + float(weight)
        if cfg.max_hot_experts_per_layer > 0 and len(scores) > cfg.max_hot_experts_per_layer * 2:
            keep = set(sorted(scores, key=scores.get, reverse=True)[: cfg.max_hot_experts_per_layer])
            for expert_id in list(scores):
                if expert_id not in keep:
                    del scores[expert_id]

    def _update_workload_scores(self, layer_idx: int, experts: list[int], weights: list[float]) -> None:
        cfg = self.locality_config
        scores = self.workload_scores.setdefault(layer_idx, {})
        decay = 1.0 - (1.0 / max(float(cfg.workload_window), 1.0))
        for expert_id in list(scores):
            scores[expert_id] *= decay
            if scores[expert_id] < 1e-4:
                del scores[expert_id]
        history = self.recent_history.setdefault(layer_idx, deque(maxlen=max(int(cfg.prediction_window), 1)))
        for expert_id, weight in zip(experts, weights):
            scores[int(expert_id)] = scores.get(int(expert_id), 0.0) + float(weight)
            history.append(int(expert_id))

    def predict_experts(self, layer_idx: int, limit: int | None = None) -> list[int]:
        cfg = self.locality_config
        if not cfg.enabled or not cfg.predictive_prefetch:
            return []
        scores: Counter[int] = Counter()
        for expert_id, value in self.workload_scores.get(layer_idx, {}).items():
            scores[int(expert_id)] += float(value)
        for expert_id, value in self.sticky_scores.get(layer_idx, {}).items():
            scores[int(expert_id)] += float(value) * 0.75
        for expert_id in self.recent_history.get(layer_idx, []):
            scores[int(expert_id)] += 0.50
        if not scores:
            return []
        n = max(int(limit or self.experts_per_token), 1)
        return [int(expert_id) for expert_id, _score in scores.most_common(n)]

    def _effective_expert_count(self, layer_idx: int, expert_count: int) -> int:
        cfg = self.locality_config
        k = min(self.experts_per_token, expert_count)
        if cfg.enabled and cfg.active_experts_per_token_cap > 0:
            k = min(k, int(cfg.active_experts_per_token_cap))
        if cfg.enabled and cfg.dynamic_active_expert_cap:
            base = cfg.active_experts_per_token_cap or self.experts_per_token
            current = self.dynamic_caps.setdefault(layer_idx, min(max(base, cfg.min_active_experts_per_token), cfg.max_active_experts_per_token or self.experts_per_token))
            k = min(k, current)
        return max(k, 1)

    def _adapt_active_cap(self, layer_idx: int, position: int, before: dict, after: dict) -> None:
        cfg = self.locality_config
        if not (cfg.enabled and cfg.dynamic_active_expert_cap):
            return
        old = self.dynamic_caps.setdefault(layer_idx, cfg.active_experts_per_token_cap or self.experts_per_token)
        hits = int(after.get("hits", 0)) - int(before.get("hits", 0))
        misses = int(after.get("misses", 0)) - int(before.get("misses", 0))
        evictions = int(after.get("evictions", 0)) - int(before.get("evictions", 0))
        total = hits + misses
        hit_rate = hits / max(total, 1)
        eviction_pressure = evictions / max(total, 1)
        min_cap = max(int(cfg.min_active_experts_per_token), 1)
        max_cap = int(cfg.max_active_experts_per_token or self.experts_per_token)
        new = old
        reason = ""
        if eviction_pressure >= float(cfg.cap_eviction_pressure_threshold) and old > min_cap:
            new = old - 1
            reason = "eviction_pressure"
        elif evictions == 0 and hit_rate >= float(cfg.cap_hit_rate_expand_threshold) and old < max_cap:
            new = old + 1
            reason = "stable_high_hit_rate"
        if new != old:
            self.dynamic_caps[layer_idx] = new
            self.stats.record_cap_event(layer_idx, position, old, new, reason, hits, misses, evictions)

    @staticmethod
    def _cache_snapshot(expert_store: Mxfp4ExpertStore) -> dict:
        stats = expert_store.cache.stats
        return {"hits": stats.hits, "misses": stats.misses, "evictions": stats.evictions}

    def _execute_mxfp4_experts(
        self,
        layer_idx: int,
        hidden: torch.Tensor,
        tensors: dict[str, torch.Tensor],
        experts: list[int],
        weights: torch.Tensor,
        expert_store: Mxfp4ExpertStore,
    ) -> tuple[torch.Tensor, dict]:
        prefix = f"model.layers.{layer_idx}.mlp"
        gate_up_bias = tensors.get(f"{prefix}.experts.gate_up_proj_bias")
        down_bias = tensors.get(f"{prefix}.experts.down_proj_bias")
        if gate_up_bias is None or down_bias is None:
            return torch.zeros_like(hidden), {"experts": experts, "weights": [float(w) for w in weights], "mode": "mxfp4_missing_bias"}
        output = torch.zeros_like(hidden)
        expert_logs = []
        for pos, expert_id in enumerate(experts):
            gate_up = expert_store.load_expert_matrix(layer_idx, int(expert_id), "gate_up_proj")
            down = expert_store.load_expert_matrix(layer_idx, int(expert_id), "down_proj")
            if self.matmul_precision == "bf16":
                gu = (hidden.to(torch.bfloat16) @ gate_up.to(torch.bfloat16)).float() + gate_up_bias[int(expert_id)].float()
            else:
                gu = hidden.float() @ gate_up.float() + gate_up_bias[int(expert_id)].float()
            gated = self._apply_gate(gu)
            if self.matmul_precision == "bf16":
                out = gated.to(torch.bfloat16) @ down.to(torch.bfloat16)
                out = out.float() + down_bias[int(expert_id)].float()
            else:
                out = gated @ down.float() + down_bias[int(expert_id)].float()
            weighted = out.to(hidden.dtype) * weights[pos]
            output += weighted
            expert_logs.append(
                {
                    "expert": int(expert_id),
                    "routing_weight": float(weights[pos]),
                    "gate_up_norm": float(torch.linalg.vector_norm(gated.float()).item()),
                    "output_norm": float(torch.linalg.vector_norm(out.float()).item()),
                }
            )
        return output, {"experts": experts, "weights": [float(w) for w in weights], "mode": "mxfp4_expert_matmul", "expert_logs": expert_logs}

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(max=self.swiglu_limit)
        up = up.clamp(min=-self.swiglu_limit, max=self.swiglu_limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        return (up + 1.0) * glu
