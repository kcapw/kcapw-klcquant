from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .utils import now_id, read_json, write_json


SCHEMA_VERSION = 1


@dataclass
class SensitivityRecord:
    tensor: str
    mode: str
    target_layers: int
    target_tokens: int
    stable: bool
    score: float
    metrics: dict[str, Any]
    quantized_dir: str
    report_path: str
    static: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=now_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "tensor": self.tensor,
            "mode": self.mode,
            "target_layers": self.target_layers,
            "target_tokens": self.target_tokens,
            "stable": self.stable,
            "score": self.score,
            "metrics": self.metrics,
            "quantized_dir": self.quantized_dir,
            "report_path": self.report_path,
            "static": self.static,
            "created_at": self.created_at,
        }

    @classmethod
    def from_json(cls, item: dict[str, Any]) -> "SensitivityRecord":
        return cls(
            tensor=str(item["tensor"]),
            mode=str(item["mode"]),
            target_layers=int(item["target_layers"]),
            target_tokens=int(item["target_tokens"]),
            stable=bool(item["stable"]),
            score=float(item.get("score", 0.0)),
            metrics=dict(item.get("metrics", {})),
            quantized_dir=str(item.get("quantized_dir", "")),
            report_path=str(item.get("report_path", "")),
            static=dict(item.get("static", {})),
            created_at=str(item.get("created_at", now_id())),
        )


class SensitivityDatabase:
    """Small append-only JSON database for runtime tensor perturbation results."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.records: list[SensitivityRecord] = []
        self.metadata: dict[str, Any] = {}
        if self.path.exists():
            payload = read_json(self.path)
            self.metadata = dict(payload.get("metadata", {}))
            self.records = [SensitivityRecord.from_json(item) for item in payload.get("records", [])]

    def add(self, record: SensitivityRecord) -> None:
        self.records.append(record)

    def add_many(self, records: list[SensitivityRecord]) -> None:
        self.records.extend(records)

    def by_tensor(self) -> dict[str, list[SensitivityRecord]]:
        grouped: dict[str, list[SensitivityRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.tensor].append(record)
        return dict(grouped)

    def latest_by_tensor_mode(self) -> dict[tuple[str, str, int, int], SensitivityRecord]:
        latest: dict[tuple[str, str, int, int], SensitivityRecord] = {}
        for record in self.records:
            key = (record.tensor, record.mode, record.target_layers, record.target_tokens)
            if key not in latest or record.created_at >= latest[key].created_at:
                latest[key] = record
        return latest

    def save(self) -> None:
        payload = {
            "version": SCHEMA_VERSION,
            "updated_at": now_id(),
            "metadata": self.metadata,
            "records": [record.to_json() for record in self.records],
        }
        write_json(self.path, payload)

