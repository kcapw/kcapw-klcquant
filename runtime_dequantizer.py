from __future__ import annotations

from dataclasses import dataclass

import torch

from .adaptive_quantizer import QuantMode, unpack_lowbit
from .tensor_pruner import reconstruct_pruned


@dataclass(frozen=True)
class QuantizedTensorMeta:
    name: str
    mode: QuantMode
    shape: tuple[int, ...]
    dtype: str
    bits: int
    packed: bool
    group: str
    file: str


def parse_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


class RuntimeDequantizer:
    def dequantize(
        self,
        meta: QuantizedTensorMeta,
        data: torch.Tensor,
        scale: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        if meta.mode == "fp16":
            return data.to(device=device, dtype=dtype, non_blocking=True).reshape(meta.shape)
        if meta.mode == "pruned":
            if scale is None:
                raise ValueError(f"{meta.name} is pruned but has no sparse index tensor")
            return reconstruct_pruned(data, scale, meta.shape, device, dtype)

        if scale is None:
            raise ValueError(f"{meta.name} is quantized but has no scale tensor")

        if meta.packed:
            q = unpack_lowbit(data.cpu(), meta.bits, self._numel(meta.shape)).to(device)
        else:
            q = data.to(device=device, non_blocking=True)
        return (q.float() * scale.to(device=device).float()).reshape(meta.shape).to(dtype)

    @staticmethod
    def _numel(shape: tuple[int, ...]) -> int:
        total = 1
        for dim in shape:
            total *= dim
        return total
