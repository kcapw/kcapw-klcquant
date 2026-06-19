from __future__ import annotations

import re
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from .utils import read_json


LAYER_RE = re.compile(r"model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class TensorRef:
    name: str
    shard: str
    path: Path
    group: str


@dataclass
class LoadedTensorGroup:
    group: str
    tensors: dict[str, torch.Tensor]
    device: str

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def unload(self) -> None:
        self.tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class SafetensorIndex:
    def __init__(self, model_dir: str | Path, support_dir: str | Path) -> None:
        self.model_dir = Path(model_dir)
        self.support_dir = Path(support_dir)
        self.index_path = self.support_dir / "model.safetensors.index.json"
        self.index = read_json(self.index_path)
        self.weight_map: dict[str, str] = self.index["weight_map"]
        self.refs = [TensorRef(name, shard, self.model_dir / shard, self.group_for(name)) for name, shard in self.weight_map.items()]

    @staticmethod
    def group_for(name: str) -> str:
        match = LAYER_RE.search(name)
        if match:
            return f"layer_{int(match.group(1)):03d}"
        if name.startswith("model.embed_tokens"):
            return "embeddings"
        if name.startswith("lm_head"):
            return "lm_head"
        return "global"

    def groups(self) -> dict[str, list[TensorRef]]:
        grouped: dict[str, list[TensorRef]] = defaultdict(list)
        for ref in self.refs:
            grouped[ref.group].append(ref)
        return dict(sorted(grouped.items()))

    def shards_for_group(self, group: str) -> list[Path]:
        return sorted({ref.path for ref in self.groups()[group]})

    def tensor_names(self, group: str | None = None) -> list[str]:
        refs = self.refs if group is None else self.groups().get(group, [])
        return [ref.name for ref in refs]

    def metadata(self) -> dict:
        return {
            "total_size": self.index.get("metadata", {}).get("total_size"),
            "tensor_count": len(self.weight_map),
            "shard_count": len(set(self.weight_map.values())),
            "group_count": len(self.groups()),
        }


class StreamedTensorLoader:
    def __init__(
        self,
        model_dir: str | Path = "model",
        support_dir: str | Path = "model_support",
        device: str = "cuda",
        dtype: torch.dtype | None = torch.float16,
        pin_memory: bool = True,
    ) -> None:
        self.index = SafetensorIndex(model_dir, support_dir)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.pin_memory = pin_memory
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._prefetch: Future[LoadedTensorGroup] | None = None

    def iter_group_names(self) -> Iterator[str]:
        yield from self.index.groups().keys()

    def load_group(self, group: str, non_blocking: bool = True) -> LoadedTensorGroup:
        tensors: dict[str, torch.Tensor] = {}
        refs = self.index.groups()[group]
        by_shard: dict[Path, list[TensorRef]] = defaultdict(list)
        for ref in refs:
            by_shard[ref.path].append(ref)

        for shard_path, shard_refs in by_shard.items():
            with safe_open(shard_path, framework="pt", device="cpu") as f:
                for ref in shard_refs:
                    tensor = f.get_tensor(ref.name)
                    if self.dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(self.dtype)
                    if self.pin_memory and self.device.type == "cuda":
                        tensor = tensor.pin_memory()
                    tensors[ref.name] = tensor.to(self.device, non_blocking=non_blocking)

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return LoadedTensorGroup(group, tensors, str(self.device))

    def prefetch_group(self, group: str) -> None:
        self._prefetch = self._executor.submit(self.load_group, group)

    def consume_prefetch(self) -> LoadedTensorGroup | None:
        if self._prefetch is None:
            return None
        result = self._prefetch.result()
        self._prefetch = None
        return result

    def stream(self) -> Iterator[LoadedTensorGroup]:
        names = list(self.iter_group_names())
        if not names:
            return
        self.prefetch_group(names[0])
        for idx, name in enumerate(names):
            loaded = self.consume_prefetch()
            if idx + 1 < len(names):
                self.prefetch_group(names[idx + 1])
            if loaded is None or loaded.group != name:
                loaded = self.load_group(name)
            yield loaded
