from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


@dataclass
class TensorStats:
    name: str
    shard: str | None = None
    shape: list[int] | None = None
    dtype: str | None = None
    nbytes: int = 0
    activation_count: int = 0
    activation_frequency: float = 0.0
    contribution_magnitude: float = 0.0
    attention_relevance: float = 0.0
    reuse_frequency: float = 0.0
    domain_hits: dict[str, int] | None = None
    importance_score: float = 0.0

    def to_json(self) -> dict:
        data = asdict(self)
        data["nbytes_mb"] = round(self.nbytes / 2**20, 4)
        return data


@dataclass
class ImportanceWeights:
    activation_frequency: float = 0.30
    contribution_magnitude: float = 0.35
    attention_relevance: float = 0.20
    reuse_frequency: float = 0.15


class ImportanceRanker:
    def __init__(self, weights: ImportanceWeights | None = None) -> None:
        self.weights = weights or ImportanceWeights()

    def score(self, stats: TensorStats) -> float:
        w = self.weights
        return (
            w.activation_frequency * stats.activation_frequency
            + w.contribution_magnitude * stats.contribution_magnitude
            + w.attention_relevance * stats.attention_relevance
            + w.reuse_frequency * stats.reuse_frequency
        )

    def rank(self, stats: Iterable[TensorStats]) -> list[TensorStats]:
        ranked = []
        for item in stats:
            item.importance_score = self.score(item)
            ranked.append(item)
        return sorted(ranked, key=lambda s: s.importance_score, reverse=True)

    @staticmethod
    def least_used(stats: Iterable[TensorStats], limit: int = 50) -> list[TensorStats]:
        return sorted(stats, key=lambda s: (s.importance_score, s.activation_count, s.nbytes))[:limit]

    @staticmethod
    def universally_important(stats: Iterable[TensorStats], limit: int = 50) -> list[TensorStats]:
        return sorted(stats, key=lambda s: (s.importance_score, s.reuse_frequency), reverse=True)[:limit]
