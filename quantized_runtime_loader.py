from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open

from .runtime_dequantizer import QuantizedTensorMeta, RuntimeDequantizer, parse_bool


@dataclass(frozen=True)
class QuantizedArtifact:
    path: Path
    group: str
    tensors: dict[str, QuantizedTensorMeta]
    mtime: float


@dataclass
class LoadedRuntimeGroup:
    group: str
    tensors: dict[str, torch.Tensor]
    metas: dict[str, QuantizedTensorMeta]
    source_files: list[str]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def unload(self) -> None:
        self.tensors.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class QuantizedRuntimeLoader:
    def __init__(self, quantized_dir: str | Path = "quantized_model", device: str = "cuda") -> None:
        self.quantized_dir = Path(quantized_dir)
        self.device = torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        self.dequantizer = RuntimeDequantizer()
        self.artifacts = self._scan_artifacts()
        self.tensor_index = self._build_tensor_index()

    def _scan_artifacts(self) -> list[QuantizedArtifact]:
        artifacts: list[QuantizedArtifact] = []
        for path in sorted(self.quantized_dir.glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
            group = metadata.get("group", path.stem.split("-")[0])
            tensors: dict[str, QuantizedTensorMeta] = {}
            for key, value in metadata.items():
                if not key.endswith(".mode"):
                    continue
                name = key[: -len(".mode")]
                shape = tuple(int(x) for x in metadata.get(f"{name}.shape", "").split(",") if x)
                bits_raw = metadata.get(f"{name}.bits", "")
                tensors[name] = QuantizedTensorMeta(
                    name=name,
                    mode=value,  # type: ignore[arg-type]
                    shape=shape,
                    dtype=metadata.get(f"{name}.dtype", "torch.float16"),
                    bits=int(bits_raw) if bits_raw else 16,
                    packed=parse_bool(metadata.get(f"{name}.packed")),
                    group=group,
                    file=str(path),
                )
            if tensors:
                artifacts.append(QuantizedArtifact(path=path, group=group, tensors=tensors, mtime=os.path.getmtime(path)))
        return artifacts

    def _build_tensor_index(self) -> dict[str, tuple[QuantizedArtifact, QuantizedTensorMeta]]:
        index: dict[str, tuple[QuantizedArtifact, QuantizedTensorMeta]] = {}
        for artifact in sorted(self.artifacts, key=lambda item: item.mtime):
            for name, meta in artifact.tensors.items():
                index[name] = (artifact, meta)
        return index

    def groups(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for name, (_artifact, meta) in self.tensor_index.items():
            grouped[meta.group].append(name)
        return dict(sorted((k, sorted(v)) for k, v in grouped.items()))

    def tensor_names(self) -> list[str]:
        return sorted(self.tensor_index)

    def iter_group_names(self) -> Iterator[str]:
        yield from self.groups().keys()

    def load_tensor(self, name: str, dtype: torch.dtype = torch.float16) -> tuple[torch.Tensor, QuantizedTensorMeta]:
        artifact, meta = self.tensor_index[name]
        data_key = f"{name}.__data__"
        scale_key = f"{name}.__scale__"
        zero_key = f"{name}.__zero__"
        with safe_open(artifact.path, framework="pt", device="cpu") as handle:
            data = handle.get_tensor(data_key)
            if meta.mode == "pruned" and zero_key in handle.keys():
                scale = handle.get_tensor(zero_key)
            else:
                scale = handle.get_tensor(scale_key) if scale_key in handle.keys() else None
        tensor = self.dequantizer.dequantize(meta, data, scale, self.device, dtype=dtype)
        return tensor, meta

    def payload_nbytes(self, name: str) -> int:
        artifact, meta = self.tensor_index[name]
        keys = [f"{name}.__data__", f"{name}.__scale__", f"{name}.__zero__"]
        total = 0
        with safe_open(artifact.path, framework="pt", device="cpu") as handle:
            available = set(handle.keys())
            for key in keys:
                if key in available:
                    tensor = handle.get_tensor(key)
                    total += tensor.numel() * tensor.element_size()
        return total

    def load_group(self, group: str, dtype: torch.dtype = torch.float16) -> LoadedRuntimeGroup:
        tensors: dict[str, torch.Tensor] = {}
        metas: dict[str, QuantizedTensorMeta] = {}
        sources: set[str] = set()
        for name in self.groups().get(group, []):
            tensor, meta = self.load_tensor(name, dtype=dtype)
            tensors[name] = tensor
            metas[name] = meta
            sources.add(meta.file)
        return LoadedRuntimeGroup(group=group, tensors=tensors, metas=metas, source_files=sorted(sources))
