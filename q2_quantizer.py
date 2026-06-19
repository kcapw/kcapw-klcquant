from __future__ import annotations

import torch


def quantize_q2(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data = tensor.detach().float()
    scale = data.abs().max().clamp_min(1e-12).reshape(1).cpu()
    q = torch.clamp(torch.round(data / scale.to(data.device)), -2, 1).to(torch.int8)
    return q.cpu(), scale


def dequantize_q2(qvalues: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (qvalues.to(device).float() * scale.to(device).float()).reshape(shape).to(dtype)
