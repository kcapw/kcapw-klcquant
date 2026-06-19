from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from .importance_ranker import TensorStats
from .q1_quantizer import quantize_q1
from .q2_quantizer import quantize_q2

QuantMode = Literal["fp16", "q16", "q8", "q4", "q3", "q2", "q1", "pruned"]


@dataclass
class QuantizedTensor:
    name: str
    mode: QuantMode
    qvalues: torch.Tensor
    scale: torch.Tensor | None
    zero_point: torch.Tensor | None
    original_shape: tuple[int, ...]
    original_dtype: str
    bits: int | None = None
    packed: bool = False

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        if self.mode == "fp16":
            return self.qvalues.to(dtype)
        if self.mode == "pruned":
            from .tensor_pruner import reconstruct_pruned

            assert self.zero_point is not None
            return reconstruct_pruned(self.qvalues, self.zero_point, self.original_shape, self.qvalues.device, dtype)
        if self.packed:
            assert self.scale is not None and self.bits is not None
            unpacked = unpack_lowbit(self.qvalues.cpu(), self.bits, int(torch.prod(torch.tensor(self.original_shape)).item()))
            return (unpacked.to(self.scale.device).float() * self.scale.float()).reshape(self.original_shape).to(dtype)
        assert self.scale is not None
        return (self.qvalues.float() * self.scale.float()).reshape(self.original_shape).to(dtype)


@dataclass
class QuantizationDecision:
    name: str
    mode: QuantMode
    importance_score: float
    original_bytes: int
    estimated_bytes: int

    @property
    def compression_ratio(self) -> float:
        if self.estimated_bytes == 0:
            return 1.0
        return self.original_bytes / self.estimated_bytes

    def to_json(self) -> dict:
        data = asdict(self)
        data["compression_ratio"] = round(self.compression_ratio, 4)
        return data


@dataclass
class AdaptiveQuantizationPolicy:
    critical_threshold: float = 0.78
    medium_threshold: float = 0.40
    low_threshold: float = 0.16
    prefer_q3_below: float = 0.06

    def choose(self, stats: TensorStats) -> QuantMode:
        name = stats.name
        if any(token in name for token in ("embed_tokens", "lm_head", "router", "norm", "self_attn")):
            return "fp16"
        if stats.importance_score >= self.critical_threshold:
            return "fp16"
        if stats.importance_score >= self.medium_threshold:
            return "q16"
        if stats.importance_score >= self.low_threshold:
            return "q8"
        if stats.importance_score < self.prefer_q3_below:
            return "q3"
        return "q4"


class AdaptiveQuantizer:
    def __init__(self, policy: AdaptiveQuantizationPolicy | None = None) -> None:
        self.policy = policy or AdaptiveQuantizationPolicy()

    def decide(self, stats: TensorStats) -> QuantizationDecision:
        mode = self.policy.choose(stats)
        bits = {"fp16": 16, "q16": 16, "q8": 8, "q4": 4, "q3": 3, "q2": 2, "q1": 1, "pruned": 0}[mode]
        estimated = max(1, int((stats.nbytes / 16) * bits))
        if mode != "fp16":
            estimated += 8
        return QuantizationDecision(stats.name, mode, stats.importance_score, stats.nbytes, estimated)

    def plan(self, ranked: list[TensorStats]) -> list[QuantizationDecision]:
        return [self.decide(item) for item in ranked]

    @staticmethod
    def quantize_tensor(name: str, tensor: torch.Tensor, mode: QuantMode) -> QuantizedTensor:
        original_dtype = str(tensor.dtype)
        original_shape = tuple(tensor.shape)
        if mode == "fp16":
            return QuantizedTensor(name, mode, tensor.to(torch.float16), None, None, original_shape, original_dtype, bits=16)

        bits = {"q16": 16, "q8": 8, "q4": 4, "q3": 3, "q2": 2, "q1": 1}[mode]
        if mode == "q1":
            q, scale = quantize_q1(tensor)
            return QuantizedTensor(name, mode, q, scale, None, original_shape, original_dtype, bits=1)
        if mode == "q2":
            q, scale = quantize_q2(tensor)
            return QuantizedTensor(name, mode, q, scale, None, original_shape, original_dtype, bits=2)
        qmax = (2 ** (bits - 1)) - 1
        data = tensor.detach().float()
        max_abs = data.abs().max().clamp_min(1e-12)
        scale = max_abs / qmax
        q = torch.clamp(torch.round(data / scale), -qmax - 1, qmax)
        storage_dtype = torch.int16 if mode == "q16" else torch.int8
        return QuantizedTensor(name, mode, q.to(storage_dtype), scale.cpu(), None, original_shape, original_dtype, bits=bits)


def quantize_for_storage(name: str, tensor: torch.Tensor, mode: QuantMode) -> QuantizedTensor:
    original_dtype = str(tensor.dtype)
    original_shape = tuple(tensor.shape)
    if mode == "fp16":
        return QuantizedTensor(name, mode, tensor.detach().to(torch.float16).cpu(), None, None, original_shape, original_dtype, bits=16)

    bits = {"q16": 16, "q8": 8, "q4": 4, "q3": 3, "q2": 2, "q1": 1}[mode]
    if mode == "q1":
        q, scale = quantize_q1(tensor)
        return QuantizedTensor(name, mode, pack_lowbit(q, 1), scale, None, original_shape, original_dtype, bits=1, packed=True)
    if mode == "q2":
        q, scale = quantize_q2(tensor)
        return QuantizedTensor(name, mode, pack_lowbit(q, 2), scale, None, original_shape, original_dtype, bits=2, packed=True)
    qmax = (2 ** (bits - 1)) - 1
    data = tensor.detach().float()
    max_abs = data.abs().max().clamp_min(1e-12)
    scale = (max_abs / qmax).detach().cpu().reshape(1)
    q = torch.clamp(torch.round(data / scale.to(data.device)), -qmax - 1, qmax).to(torch.int16 if mode == "q16" else torch.int8)
    if mode in {"q4", "q3"}:
        return QuantizedTensor(name, mode, pack_lowbit(q.cpu(), bits), scale, None, original_shape, original_dtype, bits=bits, packed=True)
    return QuantizedTensor(name, mode, q.cpu(), scale, None, original_shape, original_dtype, bits=bits)


def pack_lowbit(qvalues: torch.Tensor, bits: int) -> torch.Tensor:
    if bits not in {1, 2, 3, 4}:
        raise ValueError("pack_lowbit only supports q1, q2, q3 and q4")
    q = qvalues.reshape(-1).to(torch.int16)
    if bits == 1:
        unsigned = (q > 0).to(torch.int64)
    else:
        offset = 2 ** (bits - 1)
        unsigned = torch.clamp(q + offset, 0, (2**bits) - 1).to(torch.int64)
    values_per_word = 8 // bits if bits in {1, 2, 4} else 8
    if bits == 4:
        if unsigned.numel() % 2:
            unsigned = torch.cat([unsigned, torch.zeros(1, dtype=unsigned.dtype)])
        packed = (unsigned[0::2] | (unsigned[1::2] << 4)).to(torch.uint8)
        return packed.contiguous()

    if bits in {1, 2}:
        pad = (-unsigned.numel()) % values_per_word
        if pad:
            unsigned = torch.cat([unsigned, torch.zeros(pad, dtype=unsigned.dtype)])
        words = unsigned.reshape(-1, values_per_word)
        packed = torch.zeros(words.shape[0], dtype=torch.int64)
        for idx in range(values_per_word):
            packed |= words[:, idx] << (idx * bits)
        return packed.to(torch.uint8).contiguous()

    pad = (-unsigned.numel()) % values_per_word
    if pad:
        unsigned = torch.cat([unsigned, torch.zeros(pad, dtype=unsigned.dtype)])
    words = unsigned.reshape(-1, values_per_word)
    packed32 = torch.zeros(words.shape[0], dtype=torch.int64)
    for idx in range(values_per_word):
        packed32 |= words[:, idx] << (idx * bits)
    byte_rows = torch.stack([(packed32 >> shift) & 0xFF for shift in (0, 8, 16)], dim=1)
    return byte_rows.reshape(-1).to(torch.uint8).contiguous()


def unpack_lowbit(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    if bits not in {1, 2, 3, 4}:
        raise ValueError("unpack_lowbit only supports q1, q2, q3 and q4")
    p = packed.reshape(-1).to(torch.int64)
    offset = 2 ** (bits - 1)
    if bits in {1, 2}:
        values_per_word = 8 // bits
        values = []
        mask = (2**bits) - 1
        for idx in range(values_per_word):
            values.append((p >> (idx * bits)) & mask)
        unsigned = torch.stack(values, dim=1).reshape(-1)[:numel]
        if bits == 1:
            return torch.where(unsigned > 0, torch.ones_like(unsigned), -torch.ones_like(unsigned)).to(torch.int8)
        return (unsigned.to(torch.int16) - offset).to(torch.int8)
    if bits == 4:
        low = p & 0x0F
        high = (p >> 4) & 0x0F
        unsigned = torch.stack([low, high], dim=1).reshape(-1)[:numel]
        return (unsigned.to(torch.int16) - offset).to(torch.int8)

    if p.numel() % 3:
        pad = 3 - (p.numel() % 3)
        p = torch.cat([p, torch.zeros(pad, dtype=p.dtype)])
    words = p.reshape(-1, 3)
    packed32 = words[:, 0] | (words[:, 1] << 8) | (words[:, 2] << 16)
    values = []
    for idx in range(8):
        values.append((packed32 >> (idx * bits)) & ((2**bits) - 1))
    unsigned = torch.stack(values, dim=1).reshape(-1)[:numel]
    return (unsigned.to(torch.int16) - offset).to(torch.int8)
