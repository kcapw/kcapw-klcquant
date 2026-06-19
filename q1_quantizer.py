from __future__ import annotations

import torch


def quantize_q1(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    data = tensor.detach().float()
    scale = data.abs().mean().clamp_min(1e-12).reshape(1).cpu()
    signs = torch.where(data >= 0, torch.ones_like(data, dtype=torch.int8), -torch.ones_like(data, dtype=torch.int8))
    return signs.cpu(), scale


def dequantize_q1(qvalues: torch.Tensor, scale: torch.Tensor, shape: tuple[int, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    signs = torch.where(qvalues.to(device).reshape(-1) > 0, 1.0, -1.0)
    return (signs * scale.to(device).float()).reshape(shape).to(dtype)
