from __future__ import annotations

import torch


def tensor_quality(original: torch.Tensor, reconstructed: torch.Tensor) -> dict[str, float]:
    a = original.detach().float().reshape(-1)
    b = reconstructed.detach().float().reshape(-1)
    if a.numel() == 0:
        return {
            "mse": 0.0,
            "mae": 0.0,
            "max_abs_error": 0.0,
            "relative_l2": 0.0,
            "cosine_similarity": 1.0,
            "sqnr_db": 99.0,
        }

    diff = a - b
    mse = torch.mean(diff * diff)
    mae = torch.mean(diff.abs())
    max_abs = torch.max(diff.abs())
    denom = torch.linalg.vector_norm(a).clamp_min(1e-12)
    relative_l2 = torch.linalg.vector_norm(diff) / denom
    cosine = torch.nn.functional.cosine_similarity(a, b, dim=0, eps=1e-12)
    signal = torch.mean(a * a).clamp_min(1e-24)
    sqnr_db = 10.0 * torch.log10(signal / mse.clamp_min(1e-24))
    return {
        "mse": float(mse.item()),
        "mae": float(mae.item()),
        "max_abs_error": float(max_abs.item()),
        "relative_l2": float(relative_l2.item()),
        "cosine_similarity": float(cosine.item()),
        "sqnr_db": float(sqnr_db.item()),
    }
