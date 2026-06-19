from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .autoregressive_runner import run as run_autoregressive
from .streamed_loader import SafetensorIndex
from .streamed_quant_runner import compress_subset
from .tensor_criticality_ranker import score_metrics, tensor_role
from .tensor_perturbation_runner import PerturbationTarget, extract_sensitivity_metrics, safe_tensor_dir_name
from .utils import read_json


@dataclass(frozen=True)
class TensorOverride:
    tensor: str
    mode: str
    static: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {"tensor": self.tensor, "mode": self.mode, "static": self.static}


def override_set_stem(overrides: list[TensorOverride]) -> str:
    pieces = [f"{safe_tensor_dir_name(item.tensor)}-{item.mode}" for item in overrides]
    joined = "__".join(pieces)
    return joined[:180] if len(joined) > 180 else joined


class MultiTensorCalibrator:
    """Builds mixed-precision override sets and runs streamed autoregressive validation."""

    def __init__(
        self,
        model_dir: str | Path,
        support_dir: str | Path,
        importance_report: str | Path,
        work_dir: str | Path = "calibration_runs/multi",
        reports_dir: str | Path = "reports",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.support_dir = Path(support_dir)
        self.importance_report = Path(importance_report)
        self.work_dir = Path(work_dir)
        self.reports_dir = Path(reports_dir)
        self.index = SafetensorIndex(self.model_dir, self.support_dir)

    def build_override_dir(
        self,
        overrides: list[TensorOverride],
        target: PerturbationTarget,
        profile: str,
        *,
        prune_fraction: float = 0.90,
        overwrite: bool = True,
    ) -> tuple[Path, list[str]]:
        qdir = self.work_dir / profile / target.label / override_set_stem(overrides)
        if overwrite and qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True, exist_ok=True)
        quant_reports: list[str] = []
        first = True
        for override in overrides:
            if override.tensor not in self.index.weight_map:
                raise KeyError(f"tensor not found in index: {override.tensor}")
            group = SafetensorIndex.group_for(override.tensor)
            quant_args = argparse.Namespace(
                model_dir=str(self.model_dir),
                support_dir=str(self.support_dir),
                out_dir=str(self.reports_dir),
                quantized_dir=str(qdir),
                importance_report=str(self.importance_report),
                max_tensors=1,
                max_original_mb=1024,
                group=[group],
                pattern=override.tensor,
                mode=None,
                force_mode=override.mode,
                prune_fraction=prune_fraction,
                cpu=False,
                overwrite=first,
                allow_core_tensors=False,
                keep_top_percent=0.0,
                q16_until_percent=0.0,
                q8_until_percent=0.0,
                q4_until_percent=1.0,
            )
            quant_reports.append(str(compress_subset(quant_args)))
            first = False
        return qdir, quant_reports

    def run_override_set(
        self,
        overrides: list[TensorOverride],
        target: PerturbationTarget,
        profile: str,
        *,
        prompt_count: int = 1,
        max_context_tokens: int = 2,
        top_k: int = 16,
        lm_head_chunk_rows: int = 4096,
        expert_cache_mb: int = 512,
        kv_cache_precision: str = "fp16",
        prune_fraction: float = 0.90,
        disable_experts: bool = False,
        stop_perplexity_drift: float = 25.0,
        stop_sequence_overlap: float = 0.50,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        qdir, quant_reports = self.build_override_dir(
            overrides,
            target,
            profile,
            prune_fraction=prune_fraction,
            overwrite=overwrite,
        )
        autoreg_args = argparse.Namespace(
            model_dir=str(self.model_dir),
            support_dir=str(self.support_dir),
            quantized_dir=str(qdir),
            out_dir=str(self.reports_dir),
            prompts=None,
            prompt_count=prompt_count,
            max_layers=target.max_layers,
            layer_depths=str(target.max_layers),
            max_context_tokens=max_context_tokens,
            max_new_tokens=target.max_new_tokens,
            token_counts=str(target.max_new_tokens),
            top_k=top_k,
            lm_head_chunk_rows=lm_head_chunk_rows,
            offload_kv_cache=True,
            expert_cache_mb=expert_cache_mb,
            kv_cache_precision=kv_cache_precision,
            disable_experts=disable_experts,
            stop_perplexity_drift=stop_perplexity_drift,
            stop_sequence_overlap=stop_sequence_overlap,
        )
        report_path = run_autoregressive(autoreg_args)
        report = read_json(report_path)
        metrics = extract_sensitivity_metrics(report)
        stable = bool(report.get("experiments", [{}])[0].get("results", [{}])[0].get("stability", {}).get("stable", False))
        score = score_metrics(metrics) + (0.45 if not stable else 0.0)
        return {
            "profile": profile,
            "target_layers": target.max_layers,
            "target_tokens": target.max_new_tokens,
            "stable": stable,
            "score": round(score, 6),
            "metrics": metrics,
            "overrides": [item.to_json() for item in overrides],
            "quantized_dir": str(qdir),
            "quantization_reports": quant_reports,
            "autoregressive_report": str(report_path),
            "roles": sorted({tensor_role(item.tensor) for item in overrides}),
        }
