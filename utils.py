from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, data: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def now_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def cuda_snapshot() -> dict[str, float | str | bool]:
    if not torch.cuda.is_available():
        return {"cuda_available": False}
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    return {
        "cuda_available": True,
        "device": torch.cuda.get_device_name(device),
        "allocated_gb": round(torch.cuda.memory_allocated(device) / 2**30, 4),
        "reserved_gb": round(torch.cuda.memory_reserved(device) / 2**30, 4),
        "free_gb": round(free / 2**30, 4),
        "total_gb": round(total / 2**30, 4),
    }


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this operation, but torch.cuda.is_available() is false.")
    return torch.device("cuda")


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}
