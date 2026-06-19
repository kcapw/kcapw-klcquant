from __future__ import annotations

from itertools import combinations
from typing import Any

from .sensitivity_database import SensitivityRecord
from .tensor_criticality_ranker import score_metrics, tensor_role


def _record_score(record: SensitivityRecord) -> float:
    return float(record.score)


def _multi_score(result: dict[str, Any]) -> float:
    metrics = result.get("metrics", {})
    penalty = 0.45 if not result.get("stable", False) else 0.0
    return penalty + score_metrics(metrics)


class CompressionSafetyGraph:
    """Graph of tensors and measured co-compression interactions."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], dict[str, Any]] = {}

    def add_singleton_records(self, records: list[SensitivityRecord]) -> None:
        for record in records:
            node = self.nodes.setdefault(
                record.tensor,
                {
                    "tensor": record.tensor,
                    "role": tensor_role(record.tensor),
                    "singleton_runs": 0,
                    "unstable_singleton_runs": 0,
                    "max_singleton_score": 0.0,
                    "safe_modes": [],
                    "unsafe_modes": [],
                },
            )
            node["singleton_runs"] += 1
            node["max_singleton_score"] = max(float(node["max_singleton_score"]), _record_score(record))
            key = "safe_modes" if record.stable else "unsafe_modes"
            if record.mode not in node[key]:
                node[key].append(record.mode)
            if not record.stable:
                node["unstable_singleton_runs"] += 1

    def add_multi_result(self, result: dict[str, Any], singleton_lookup: dict[str, list[SensitivityRecord]]) -> None:
        tensors = [item["tensor"] for item in result.get("overrides", [])]
        if len(tensors) < 2:
            return
        combo_score = _multi_score(result)
        singleton_scores = []
        for tensor in tensors:
            rows = singleton_lookup.get(tensor, [])
            singleton_scores.append(max((_record_score(row) for row in rows), default=0.0))
            self.nodes.setdefault(
                tensor,
                {
                    "tensor": tensor,
                    "role": tensor_role(tensor),
                    "singleton_runs": len(rows),
                    "unstable_singleton_runs": sum(1 for row in rows if not row.stable),
                    "max_singleton_score": max((_record_score(row) for row in rows), default=0.0),
                    "safe_modes": sorted({row.mode for row in rows if row.stable}),
                    "unsafe_modes": sorted({row.mode for row in rows if not row.stable}),
                },
            )
        expected = sum(singleton_scores) / max(len(singleton_scores), 1)
        interaction_score = combo_score - expected
        if interaction_score > 0.20:
            relation = "amplifies_instability"
        elif interaction_score < -0.15:
            relation = "compensates_or_tolerates"
        elif result.get("stable", False):
            relation = "co_compressible"
        else:
            relation = "jointly_unstable"
        for left, right in combinations(sorted(tensors), 2):
            edge = self.edges.setdefault(
                (left, right),
                {
                    "source": left,
                    "target": right,
                    "runs": 0,
                    "unstable_runs": 0,
                    "max_interaction_score": -999.0,
                    "relations": [],
                    "profiles": [],
                },
            )
            edge["runs"] += 1
            edge["unstable_runs"] += 0 if result.get("stable", False) else 1
            edge["max_interaction_score"] = max(float(edge["max_interaction_score"]), interaction_score)
            if relation not in edge["relations"]:
                edge["relations"].append(relation)
            profile = result.get("profile")
            if profile and profile not in edge["profiles"]:
                edge["profiles"].append(profile)

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": sorted(self.nodes.values(), key=lambda item: item["tensor"]),
            "edges": sorted(self.edges.values(), key=lambda item: (item["source"], item["target"])),
        }


def build_interaction_graph(singleton_records: list[SensitivityRecord], multi_results: list[dict[str, Any]]) -> dict[str, Any]:
    graph = CompressionSafetyGraph()
    graph.add_singleton_records(singleton_records)
    lookup: dict[str, list[SensitivityRecord]] = {}
    for record in singleton_records:
        lookup.setdefault(record.tensor, []).append(record)
    for result in multi_results:
        graph.add_multi_result(result, lookup)
    return graph.to_json()

