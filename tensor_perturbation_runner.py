from __future__ import annotations

import argparse
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .autoregressive_runner import run as run_autoregressive
from .sensitivity_database import SensitivityRecord
from .streamed_loader import SafetensorIndex
from .streamed_quant_runner import compress_subset
from .tensor_criticality_ranker import score_metrics
from .utils import read_json


PERTURBATION_MODES = ("q8", "q4", "q2", "q1", "pruned")


@dataclass(frozen=True)
class PerturbationTarget:
    max_layers: int
    max_new_tokens: int

    @property
    def label(self) -> str:
        return f"{self.max_layers}x{self.max_new_tokens}"


def safe_tensor_dir_name(tensor: str) -> str:
    return tensor.replace("/", "_").replace(".", "__")


def _sparse_logit_cosine(base_log: dict, quant_log: dict) -> float:
    base = {int(token): float(logit) for token, logit in zip(base_log.get("top_token_ids", []), base_log.get("top_logits", []))}
    quant = {int(token): float(logit) for token, logit in zip(quant_log.get("top_token_ids", []), quant_log.get("top_logits", []))}
    keys = sorted(set(base) | set(quant))
    if not keys:
        return 1.0
    dot = sum(base.get(key, 0.0) * quant.get(key, 0.0) for key in keys)
    base_norm = math.sqrt(sum(base.get(key, 0.0) ** 2 for key in keys))
    quant_norm = math.sqrt(sum(quant.get(key, 0.0) ** 2 for key in keys))
    if base_norm == 0.0 or quant_norm == 0.0:
        return 0.0
    return dot / (base_norm * quant_norm)


def _token_agreement_rate(comparison: dict) -> float:
    flags = comparison.get("per_token_agreement", [])
    if not flags:
        return 1.0
    return sum(1 for item in flags if item) / len(flags)


def _detect_repetition_instability(base_ids: list[int], quant_ids: list[int]) -> bool:
    if len(quant_ids) < 4:
        return False
    quant_repeats = sum(1 for prev, cur in zip(quant_ids, quant_ids[1:]) if prev == cur)
    base_repeats = sum(1 for prev, cur in zip(base_ids, base_ids[1:]) if prev == cur)
    return quant_repeats >= base_repeats + 2


def extract_sensitivity_metrics(report: dict) -> dict[str, Any]:
    experiments = report.get("experiments", [])
    if not experiments or not experiments[0].get("results"):
        return {}
    result = experiments[0]["results"][0]
    comparison = result["comparison"]
    base = result["baseline"]
    quant = result["quantized"]
    base_gen_logs = [log for log in base.get("token_logs", []) if "generated_token" in log]
    quant_gen_logs = [log for log in quant.get("token_logs", []) if "generated_token" in log]
    entropy_pairs = list(zip(base_gen_logs, quant_gen_logs))
    entropy_drift = 0.0
    sparse_cosine = 1.0
    if entropy_pairs:
        entropy_drift = sum(float(q.get("entropy", 0.0)) - float(b.get("entropy", 0.0)) for b, q in entropy_pairs) / len(entropy_pairs)
        sparse_cosine = sum(_sparse_logit_cosine(b, q) for b, q in entropy_pairs) / len(entropy_pairs)

    base_ids = [int(x) for x in base.get("generated_token_ids", [])]
    quant_ids = [int(x) for x in quant.get("generated_token_ids", [])]
    hallucination = comparison.get("hallucination_indicators", {})
    metrics = {
        "token_jaccard": float(comparison.get("token_jaccard", 0.0)),
        "sequence_overlap": float(comparison.get("sequence_overlap", 0.0)),
        "token_agreement_rate": _token_agreement_rate(comparison),
        "first_divergence_token": comparison.get("first_divergence_index"),
        "perplexity_drift": float(comparison.get("perplexity_drift", 0.0)),
        "perplexity_baseline": float(comparison.get("perplexity_baseline", 0.0)),
        "perplexity_quantized": float(comparison.get("perplexity_quantized", 0.0)),
        "entropy_drift": entropy_drift,
        "sparse_logit_cosine": sparse_cosine,
        "generation_collapse": bool(hallucination.get("empty_or_short", False)),
        "repetition_instability": _detect_repetition_instability(base_ids, quant_ids),
        "generated_tokens": len(quant_ids),
        "baseline_text": base.get("generated_text", ""),
        "quantized_text": quant.get("generated_text", ""),
        "latency_baseline_s": float(base.get("latency_s", 0.0)),
        "latency_quantized_s": float(quant.get("latency_s", 0.0)),
        "baseline_transfer_mb": float(base.get("runtime_stats", {}).get("transfer", {}).get("mb", 0.0)),
        "quantized_transfer_mb": float(quant.get("runtime_stats", {}).get("transfer", {}).get("mb", 0.0)),
        "quantized_peak_vram_gb": float(quant.get("runtime_stats", {}).get("vram_peak", {}).get("allocated_gb", 0.0)),
    }
    return metrics


class TensorPerturbationRunner:
    def __init__(
        self,
        model_dir: str | Path,
        support_dir: str | Path,
        importance_report: str | Path,
        work_dir: str | Path = "calibration_runs",
        reports_dir: str | Path = "reports",
    ) -> None:
        self.model_dir = Path(model_dir)
        self.support_dir = Path(support_dir)
        self.importance_report = Path(importance_report)
        self.work_dir = Path(work_dir)
        self.reports_dir = Path(reports_dir)
        self.index = SafetensorIndex(self.model_dir, self.support_dir)

    def run_tensor_mode(
        self,
        tensor: str,
        mode: str,
        target: PerturbationTarget,
        *,
        prompt_count: int = 1,
        max_context_tokens: int = 2,
        top_k: int = 16,
        lm_head_chunk_rows: int = 4096,
        expert_cache_mb: int = 512,
        kv_cache_precision: str = "fp16",
        prune_fraction: float = 0.90,
        disable_experts: bool = False,
        stop_perplexity_drift: float = 1000.0,
        stop_sequence_overlap: float = -1.0,
        overwrite: bool = True,
        static: dict[str, Any] | None = None,
    ) -> SensitivityRecord:
        if mode not in PERTURBATION_MODES:
            raise ValueError(f"unsupported perturbation mode {mode}")
        if tensor not in self.index.weight_map:
            raise KeyError(f"tensor not found in index: {tensor}")

        group = SafetensorIndex.group_for(tensor)
        qdir = self.work_dir / safe_tensor_dir_name(tensor) / target.label / mode
        if overwrite and qdir.exists():
            shutil.rmtree(qdir)
        qdir.mkdir(parents=True, exist_ok=True)

        quant_args = argparse.Namespace(
            model_dir=str(self.model_dir),
            support_dir=str(self.support_dir),
            out_dir=str(self.reports_dir),
            quantized_dir=str(qdir),
            importance_report=str(self.importance_report),
            max_tensors=1,
            max_original_mb=512,
            group=[group],
            pattern=tensor,
            mode=None,
            force_mode=mode,
            prune_fraction=prune_fraction,
            cpu=False,
            overwrite=True,
            allow_core_tensors=False,
            keep_top_percent=0.0,
            q16_until_percent=0.0,
            q8_until_percent=0.0,
            q4_until_percent=1.0,
        )
        quant_report = compress_subset(quant_args)

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
        metrics["quantization_report"] = str(quant_report)
        score = score_metrics(metrics) + (0.45 if not stable else 0.0)
        return SensitivityRecord(
            tensor=tensor,
            mode=mode,
            target_layers=target.max_layers,
            target_tokens=target.max_new_tokens,
            stable=stable,
            score=round(score, 6),
            metrics=metrics,
            quantized_dir=str(qdir),
            report_path=str(report_path),
            static=static or {},
        )
