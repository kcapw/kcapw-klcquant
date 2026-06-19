from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer

from .generation_runtime import GenerationConfig, _logit_summary
from .prompt_formatting import format_prompt
from .streamed_transformer_executor import ExecutorConfig, StreamedTransformerExecutor
from .utils import read_json


MODEL_PRESETS = {
    "20b": {"label": "GPT-OSS 20B (/mnt/ramdisk)", "model_dir": "/mnt/ramdisk", "support_dir": "/mnt/ramdisk"},
    "120b": {"label": "GPT-OSS 120B (workspace)", "model_dir": "model", "support_dir": "model_support"},
}


@dataclass
class ServerRuntimeSettings:
    model: str = "120b"
    prompt_format: str = "chat"
    system_prompt: str = "You are a helpful assistant. Answer directly and concisely."
    max_layers: int = 36
    max_new_tokens: int = 8
    max_context_tokens: int = 128
    expert_cache_mb: int = 8192
    hot_vram_budget_mb: int = 12288
    active_experts_per_token_cap: int = 0
    min_active_experts_per_token: int = 2
    max_active_experts_per_token: int = 4
    sticky_routing: bool = False
    predictive_prefetch: bool = False
    dynamic_expert_caps: bool = False
    cache_aware_routing: bool = False
    routing_semantic_guard: bool = True
    layer_aware_cache: bool = True
    adaptive_layer_quota: bool = False
    adaptive_layer_quota_strength: float = 1.5
    expert_matmul_precision: str = "bf16"
    expert_protect_tokens: int = 2
    sticky_routing_strength: float = 0.35
    sticky_candidate_margin: float = 0.50
    max_sticky_bonus: float = 0.25
    min_raw_route_overlap: int = 2
    cache_aware_routing_strength: float = 0.08
    expert_prefetch_limit: int = 3
    top_k: int = 8
    prefill_progress_interval: int = 4
    degeneration_guard: bool = True
    max_repeated_token_run: int = 8


def _coerce_settings(payload: dict[str, Any]) -> ServerRuntimeSettings:
    base = asdict(ServerRuntimeSettings())
    for key, value in payload.items():
        if key in base:
            base[key] = value
    for key in (
        "max_layers",
        "max_new_tokens",
        "max_context_tokens",
        "expert_cache_mb",
        "hot_vram_budget_mb",
        "active_experts_per_token_cap",
        "min_active_experts_per_token",
        "max_active_experts_per_token",
        "expert_prefetch_limit",
        "top_k",
        "prefill_progress_interval",
        "max_repeated_token_run",
        "expert_protect_tokens",
        "min_raw_route_overlap",
    ):
        base[key] = int(base[key])
    for key in (
        "sticky_routing_strength",
        "sticky_candidate_margin",
        "max_sticky_bonus",
        "cache_aware_routing_strength",
        "adaptive_layer_quota_strength",
    ):
        base[key] = float(base[key])
    for key in (
        "sticky_routing",
        "predictive_prefetch",
        "dynamic_expert_caps",
        "cache_aware_routing",
        "routing_semantic_guard",
        "layer_aware_cache",
        "adaptive_layer_quota",
        "degeneration_guard",
    ):
        base[key] = bool(base[key])
    if base["expert_matmul_precision"] not in {"fp32", "bf16"}:
        base["expert_matmul_precision"] = "fp32"
    return ServerRuntimeSettings(**base)


def _model_paths(settings: ServerRuntimeSettings) -> tuple[Path, Path]:
    preset = MODEL_PRESETS.get(settings.model, MODEL_PRESETS["20b"])
    return Path(preset["model_dir"]), Path(preset["support_dir"])


def _executor_config(settings: ServerRuntimeSettings, model_config: dict[str, Any]) -> ExecutorConfig:
    max_layers = min(max(int(settings.max_layers), 1), int(model_config["num_hidden_layers"]))
    locality_enabled = any(
        [
            settings.sticky_routing,
            settings.predictive_prefetch,
            settings.dynamic_expert_caps,
            settings.cache_aware_routing,
            settings.active_experts_per_token_cap > 0,
        ]
    )
    return ExecutorConfig(
        max_layers=max_layers,
        expert_cache_mb=max(int(settings.expert_cache_mb), 0),
        offload_kv_cache=True,
        use_quantized_overrides=False,
        execute_experts=True,
        kv_cache_precision="fp16",
        dtype=torch.bfloat16,
        hot_residency=True,
        hot_vram_budget_mb=max(int(settings.hot_vram_budget_mb), 0),
        pin_lm_head=True,
        pin_layer_tensors=True,
        routing_locality=locality_enabled,
        sticky_routing_strength=settings.sticky_routing_strength if settings.sticky_routing else 0.0,
        sticky_routing_decay=0.92,
        routing_semantic_guard=settings.routing_semantic_guard,
        sticky_candidate_margin=settings.sticky_candidate_margin,
        max_sticky_bonus=settings.max_sticky_bonus,
        min_raw_route_overlap=max(int(settings.min_raw_route_overlap), 0),
        max_hot_experts_per_layer=8 if settings.sticky_routing else 0,
        active_experts_per_token_cap=max(int(settings.active_experts_per_token_cap), 0),
        routing_exploration_margin=0.25,
        cache_aware_routing_strength=settings.cache_aware_routing_strength if settings.cache_aware_routing else 0.0,
        predictive_expert_prefetch=settings.predictive_prefetch,
        expert_prefetch_limit=max(int(settings.expert_prefetch_limit), 1),
        expert_async_prefetch=settings.predictive_prefetch,
        routing_prediction_window=16,
        routing_workload_window=64,
        dynamic_active_expert_cap=settings.dynamic_expert_caps,
        min_active_experts_per_token=max(int(settings.min_active_experts_per_token), 1),
        max_active_experts_per_token=max(int(settings.max_active_experts_per_token), 0),
        layer_aware_expert_cache=settings.layer_aware_cache,
        expert_matmul_precision=settings.expert_matmul_precision,
        expert_protect_tokens=max(int(settings.expert_protect_tokens), 1),
        adaptive_layer_quota=settings.adaptive_layer_quota,
        adaptive_layer_quota_strength=settings.adaptive_layer_quota_strength,
    )


def _cuda_current() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "allocated_gb": 0.0, "reserved_gb": 0.0}
    return {
        "available": True,
        "allocated_gb": round(torch.cuda.memory_allocated() / 2**30, 4),
        "reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 4),
        "max_allocated_gb": round(torch.cuda.max_memory_allocated() / 2**30, 4),
    }


def _active_experts_from_logs(layer_logs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active = []
    for layer in layer_logs:
        route = layer.get("route", {})
        active.append(
            {
                "layer": int(layer.get("layer", -1)),
                "experts": [int(x) for x in route.get("experts", [])],
                "raw_experts": [int(x) for x in route.get("raw_experts", [])],
                "active_expert_cap": route.get("active_expert_cap"),
                "sticky_adjusted": bool(route.get("sticky_adjusted", False)),
            }
        )
    return active


def _degeneration_signal(generated: list[int], generated_text: str, max_repeated_token_run: int) -> dict[str, Any]:
    if not generated:
        return {"triggered": False, "reason": None, "repeated_token_run": 0}
    run = 1
    for idx in range(len(generated) - 2, -1, -1):
        if generated[idx] != generated[-1]:
            break
        run += 1
    text_tail = generated_text[-240:].lower()
    repeated_ellipsis = text_tail.count("...") >= 8
    repeated_apology = text_tail.count("sorry") >= 5 or text_tail.count("apologies") >= 4
    repeated_problem = text_tail.count("problem") >= 4
    triggered = run >= max_repeated_token_run or repeated_ellipsis or repeated_apology or repeated_problem
    reason = None
    if run >= max_repeated_token_run:
        reason = "repeated_token"
    elif repeated_ellipsis:
        reason = "repeated_ellipsis"
    elif repeated_apology:
        reason = "repeated_apology"
    elif repeated_problem:
        reason = "repeated_problem"
    return {"triggered": triggered, "reason": reason, "repeated_token_run": run}


def _telemetry(
    executor: StreamedTransformerExecutor,
    layer_logs: list[dict[str, Any]],
    generated_count: int,
    elapsed_s: float,
    token_latency_s: float,
    decode_elapsed_s: float | None = None,
    recent_token_latencies: list[float] | None = None,
    decode_base: dict[str, Any] | None = None,
    previous_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stats = executor.runtime_stats()
    expert = stats.get("expert_cache", {})
    router = stats.get("router", {})
    locality = router.get("locality", {})
    active_rows = locality.get("active_expert_set_size_over_time") or []
    hot_items = expert.get("hot_items", [])[:16]
    end_to_end_tps = generated_count / max(elapsed_s, 1e-9)
    decode_tps = generated_count / max(decode_elapsed_s or 0.0, 1e-9)
    recent = recent_token_latencies or []
    recent_tps = len(recent) / max(sum(recent), 1e-9) if recent else 0.0
    decode_base = decode_base or {}
    previous_snapshot = previous_snapshot or decode_base
    decode_hits = int(expert.get("hits", 0)) - int(decode_base.get("hits", 0))
    decode_misses = int(expert.get("misses", 0)) - int(decode_base.get("misses", 0))
    token_hits = int(expert.get("hits", 0)) - int(previous_snapshot.get("hits", 0))
    token_misses = int(expert.get("misses", 0)) - int(previous_snapshot.get("misses", 0))
    token_evictions = int(expert.get("evictions", 0)) - int(previous_snapshot.get("evictions", 0))
    token_reread_mb = (
        float(expert.get("miss_loaded_mb", expert.get("loaded_mb", 0.0)))
        - float(previous_snapshot.get("miss_loaded_mb", previous_snapshot.get("loaded_mb", 0.0)))
    )
    decode_reread_mb = (
        float(expert.get("miss_loaded_mb", expert.get("loaded_mb", 0.0)))
        - float(decode_base.get("miss_loaded_mb", decode_base.get("loaded_mb", 0.0)))
    )
    token_reused_mb = float(expert.get("hit_mb", 0.0)) - float(previous_snapshot.get("hit_mb", 0.0))
    decode_reused_mb = float(expert.get("hit_mb", 0.0)) - float(decode_base.get("hit_mb", 0.0))
    return {
        "token_latency_s": round(token_latency_s, 6),
        "tokens_per_second": end_to_end_tps,
        "end_to_end_tokens_per_second": end_to_end_tps,
        "decode_tokens_per_second": decode_tps,
        "recent_tokens_per_second": recent_tps,
        "vram": _cuda_current(),
        "active_experts": _active_experts_from_logs(layer_logs),
        "active_set_size": active_rows[-1].get("active_expert_count") if active_rows else None,
        "cache_hit_rate": expert.get("hit_rate", 0.0),
        "decode_cache_hit_rate": decode_hits / max(decode_hits + decode_misses, 1),
        "token_cache_hit_rate": token_hits / max(token_hits + token_misses, 1),
        "expert_hits": expert.get("hits", 0),
        "expert_misses": expert.get("misses", 0),
        "evictions": expert.get("evictions", 0),
        "token_evictions": token_evictions,
        "rereads": expert.get("misses", 0),
        "reread_mb": expert.get("miss_loaded_mb", expert.get("loaded_mb", 0.0)),
        "reread_mb_per_token": expert.get("miss_loaded_mb", expert.get("loaded_mb", 0.0)) / max(generated_count, 1),
        "decode_reread_mb": decode_reread_mb,
        "decode_reread_mb_per_token": decode_reread_mb / max(generated_count, 1),
        "token_reread_mb": token_reread_mb,
        "reused_mb": expert.get("hit_mb", 0.0),
        "decode_reused_mb": decode_reused_mb,
        "token_reused_mb": token_reused_mb,
        "effective_useful_bandwidth": expert.get("effective_useful_bandwidth", 0.0),
        "transfer_amplification_ratio": expert.get("miss_loaded_mb", expert.get("loaded_mb", 0.0))
        / max(float(expert.get("hit_mb", 0.0)), 1e-9),
        "blocked_on_prefetch_s": expert.get("blocked_on_prefetch_s", 0.0),
        "materialization_ms": expert.get("avg_materialization_time_ms", 0.0),
        "dequant_ms": expert.get("avg_dequant_time_ms", 0.0),
        "prefetch_use_rate": expert.get("prefetch_use_rate", 0.0),
        "prefetch_waste_rate": 1.0 - float(expert.get("prefetch_use_rate", 0.0))
        if expert.get("prefetch_submitted", 0)
        else 0.0,
        "current_hot_experts": hot_items,
        "layer_residency": expert.get("layer_residency", []),
        "layer_activity": expert.get("layer_activity", []),
        "eviction_reasons": expert.get("eviction_reasons", {}),
        "routing_entropy": locality.get("routing_entropy_mean", 0.0),
        "reuse_distance": locality.get("reuse_distance", {}),
        "expert_thrash_mode": locality.get("expert_thrash_mode", False),
    }


async def stream_generation(websocket: Any, prompt: str, settings: ServerRuntimeSettings) -> None:
    model_dir, support_dir = _model_paths(settings)
    if not (support_dir / "config.json").exists():
        raise FileNotFoundError(f"missing support config: {support_dir / 'config.json'}")
    model_config = read_json(support_dir / "config.json")
    executor_config = _executor_config(settings, model_config)
    generation_config = GenerationConfig(
        max_new_tokens=settings.max_new_tokens,
        max_context_tokens=settings.max_context_tokens,
        top_k=settings.top_k,
        lm_head_chunk_rows=4096,
        stop_token_ids=[int(model_config.get("eos_token_id", 200002))],
    )
    tokenizer = AutoTokenizer.from_pretrained(str(support_dir), trust_remote_code=True)
    rendered = format_prompt(tokenizer, prompt, settings.prompt_format, settings.system_prompt)
    all_token_ids = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    token_ids = all_token_ids[-generation_config.max_context_tokens :]
    if not token_ids:
        token_ids = [tokenizer.eos_token_id or 0]

    executor = StreamedTransformerExecutor(model_dir, support_dir, "quantized_model", "cuda", executor_config)
    await websocket.send_json(
        {
            "type": "start",
            "settings": asdict(settings),
            "model": {
                "model_dir": str(model_dir),
                "support_dir": str(support_dir),
                "layers_total": int(model_config["num_hidden_layers"]),
                "layers_executed": executor_config.max_layers,
                "experts_per_layer": int(model_config["num_local_experts"]),
            },
            "context_tokens": len(token_ids),
            "prompt_tokens_total": len(all_token_ids),
            "context_truncated": len(all_token_ids) > len(token_ids),
            "context_kept": "tail",
        }
    )

    start = time.perf_counter()
    position = 0
    current = int(token_ids[-1])
    generated: list[int] = []
    generated_text = ""
    prefill_start = time.perf_counter()
    prefill_latency_s = 0.0
    decode_start: float | None = None
    recent_latencies: list[float] = []
    decode_base_snapshot: dict[str, Any] | None = None
    previous_expert_snapshot: dict[str, Any] | None = None

    with torch.no_grad():
        for idx, token in enumerate(token_ids[:-1]):
            step_start = time.perf_counter()
            _hidden, logs = executor.step_token(int(token), position)
            position += 1
            progress_interval = max(int(settings.prefill_progress_interval), 1)
            if idx == 0 or idx == len(token_ids[:-1]) - 1 or idx % progress_interval == progress_interval - 1:
                await websocket.send_json(
                    {
                        "type": "prefill",
                        "position": position,
                        "context_tokens": len(token_ids),
                        "elapsed_s": round(time.perf_counter() - start, 6),
                        "telemetry": _telemetry(
                            executor,
                            logs,
                            len(generated),
                            time.perf_counter() - start,
                            time.perf_counter() - step_start,
                            decode_elapsed_s=0.0,
                            recent_token_latencies=recent_latencies[-8:],
                        ),
                    }
                )
                await asyncio.sleep(0)

        prefill_latency_s = time.perf_counter() - prefill_start
        decode_start = time.perf_counter()
        decode_base_snapshot = executor.runtime_stats().get("expert_cache", {})
        previous_expert_snapshot = dict(decode_base_snapshot)
        for step in range(generation_config.max_new_tokens):
            token_start = time.perf_counter()
            hidden, logs = executor.step_token(current, position)
            raw_logits = executor.logits(hidden, generation_config.lm_head_chunk_rows).float()
            logits = torch.nan_to_num(raw_logits, nan=-1e9, posinf=1e9, neginf=-1e9)
            logprobs = torch.log_softmax(logits, dim=0)
            next_id = int(torch.argmax(logits).item())
            token_latency = time.perf_counter() - token_start
            recent_latencies.append(token_latency)
            generated.append(next_id)
            token_text = tokenizer.decode([next_id])
            generated_text += token_text
            degeneration = _degeneration_signal(generated, generated_text, max(int(settings.max_repeated_token_run), 1))
            summary = _logit_summary(logits, logprobs, generation_config.top_k)
            current = next_id
            position += 1
            await websocket.send_json(
                {
                    "type": "token",
                    "step": step,
                    "token_id": next_id,
                    "token_text": token_text,
                    "generated_text": generated_text,
                    "startup_latency_s": round(prefill_start - start, 6),
                    "prefill_latency_s": round(prefill_latency_s, 6),
                    "first_token_latency_s": round(token_latency, 6) if step == 0 else None,
                    "decode_elapsed_s": round(time.perf_counter() - decode_start, 6),
                    "degeneration": degeneration,
                    "top_token_ids": summary["top_token_ids"],
                    "top_logits": summary["top_logits"],
                    "entropy": summary["entropy"],
                    "telemetry": _telemetry(
                        executor,
                        logs,
                        len(generated),
                        time.perf_counter() - start,
                        token_latency,
                        decode_elapsed_s=time.perf_counter() - decode_start,
                        recent_token_latencies=recent_latencies[-8:],
                        decode_base=decode_base_snapshot,
                        previous_snapshot=previous_expert_snapshot,
                    ),
                }
            )
            previous_expert_snapshot = executor.runtime_stats().get("expert_cache", {})
            if generation_config.stop_token_ids and next_id in generation_config.stop_token_ids:
                break
            if settings.degeneration_guard and degeneration["triggered"]:
                await websocket.send_json(
                    {
                        "type": "guard_stop",
                        "reason": degeneration["reason"],
                        "generated_text": generated_text,
                        "generated_token_ids": generated,
                        "telemetry": _telemetry(
                            executor,
                            logs,
                            len(generated),
                            time.perf_counter() - start,
                            token_latency,
                            decode_elapsed_s=time.perf_counter() - decode_start,
                            recent_token_latencies=recent_latencies[-8:],
                            decode_base=decode_base_snapshot,
                            previous_snapshot=previous_expert_snapshot,
                        ),
                    }
                )
                break
            await asyncio.sleep(0)

    await websocket.send_json(
        {
            "type": "done",
            "generated_token_ids": generated,
            "generated_text": generated_text,
            "latency_s": round(time.perf_counter() - start, 6),
            "prefill_latency_s": round(prefill_latency_s, 6),
            "decode_latency_s": round(time.perf_counter() - decode_start, 6) if decode_start else 0.0,
            "runtime_stats": executor.runtime_stats(),
        }
    )


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>KLCQUANT Streamed Runtime</title>
  <style>
    :root { color-scheme: dark; --bg:#101418; --panel:#171d23; --line:#2a333d; --text:#e8edf2; --muted:#9aa7b3; --accent:#6fb7ff; --bad:#ff7a90; --good:#8ee6a8; }
    * { box-sizing: border-box; }
    body { margin: 0; font: 14px/1.4 system-ui, -apple-system, Segoe UI, sans-serif; background: var(--bg); color: var(--text); }
    header { display:flex; align-items:center; justify-content:space-between; padding: 12px 16px; border-bottom:1px solid var(--line); background:#0d1115; }
    h1 { font-size: 16px; margin: 0; font-weight: 650; }
    main { display:grid; grid-template-columns: 360px 1fr; min-height: calc(100vh - 49px); }
    aside { border-right:1px solid var(--line); padding: 14px; overflow:auto; }
    section { padding: 14px; overflow:auto; }
    label { display:block; color:var(--muted); font-size:12px; margin: 10px 0 4px; }
    input, textarea, select, button { width:100%; border:1px solid var(--line); background:#0d1115; color:var(--text); border-radius:6px; padding:8px; font:inherit; }
    textarea { min-height: 92px; resize: vertical; }
    button { cursor:pointer; background:#16324a; border-color:#24577f; font-weight:650; }
    button:disabled { opacity:.55; cursor:not-allowed; }
    .row { display:grid; grid-template-columns: 1fr 1fr; gap:8px; }
    .checks { display:grid; grid-template-columns: 1fr 1fr; gap: 6px 10px; margin-top:8px; }
    .checks label { display:flex; gap:7px; align-items:center; margin:0; font-size:13px; color:var(--text); }
    .checks input { width:auto; }
    .status { color: var(--muted); }
    .output { min-height: 108px; white-space: pre-wrap; font-size: 18px; padding: 12px; border:1px solid var(--line); background:#0d1115; border-radius:6px; }
    .grid { display:grid; grid-template-columns: repeat(4, minmax(140px,1fr)); gap:8px; margin-top:12px; }
    .metric { border:1px solid var(--line); background:var(--panel); border-radius:6px; padding:9px; min-height:62px; }
    .metric b { display:block; font-size:18px; margin-top:3px; overflow:hidden; text-overflow:ellipsis; }
    .metric span { color:var(--muted); font-size:12px; }
    .log { margin-top:12px; height:220px; overflow:auto; border:1px solid var(--line); background:#0d1115; border-radius:6px; padding:8px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size:12px; white-space:pre-wrap; }
    table { width:100%; border-collapse:collapse; margin-top:12px; font-size:12px; }
    th, td { border-bottom:1px solid var(--line); padding:6px; text-align:left; }
    th { color:var(--muted); font-weight:600; }
    .pill { display:inline-block; padding:2px 7px; border-radius:999px; background:#233140; color:var(--accent); }
    .bad { color: var(--bad); }
    .good { color: var(--good); }
  </style>
</head>
<body>
<header><h1>KLCQUANT Streamed Runtime</h1><div id="status" class="status">idle</div></header>
<main>
  <aside>
    <label>Model</label>
    <select id="model"><option value="120b">GPT-OSS 120B</option><option value="20b">GPT-OSS 20B (/mnt/ramdisk)</option></select>
    <label>Prompt</label>
    <textarea id="prompt">Hello.</textarea>
    <div class="row">
      <div><label>Max layers</label><input id="max_layers" type="number" value="36" min="1"></div>
      <div><label>New tokens</label><input id="max_new_tokens" type="number" value="8" min="1"></div>
    </div>
    <div class="row">
      <div><label>Context tokens</label><input id="max_context_tokens" type="number" value="128" min="1"></div>
      <div><label>Expert cache MB</label><input id="expert_cache_mb" type="number" value="8192" min="0"></div>
    </div>
    <div class="row">
      <div><label>Hot VRAM MB</label><input id="hot_vram_budget_mb" type="number" value="12288" min="0"></div>
      <div><label>Expert cap</label><input id="active_experts_per_token_cap" type="number" value="0" min="0"></div>
    </div>
    <div class="row">
      <div><label>Protect tokens</label><input id="expert_protect_tokens" type="number" value="2" min="1"></div>
      <div><label>Prefetch limit</label><input id="expert_prefetch_limit" type="number" value="3" min="1"></div>
    </div>
    <div class="checks">
      <label><input id="sticky_routing" type="checkbox"> sticky</label>
      <label><input id="predictive_prefetch" type="checkbox"> prefetch</label>
      <label><input id="dynamic_expert_caps" type="checkbox"> dynamic caps</label>
      <label><input id="cache_aware_routing" type="checkbox"> cache aware</label>
      <label><input id="routing_semantic_guard" type="checkbox" checked> route guard</label>
      <label><input id="layer_aware_cache" type="checkbox" checked> layer cache</label>
      <label><input id="adaptive_layer_quota" type="checkbox"> adaptive quota</label>
      <label><input id="degeneration_guard" type="checkbox" checked> guard</label>
    </div>
    <label>Prompt format</label>
    <select id="prompt_format"><option value="chat">chat</option><option value="raw">raw</option></select>
    <label>Expert matmul</label>
    <select id="expert_matmul_precision"><option value="bf16">bf16 fast</option><option value="fp32">fp32 stable</option></select>
    <div class="row">
      <div><label>Sticky margin</label><input id="sticky_candidate_margin" type="number" value="0.50" min="0" step="0.01"></div>
      <div><label>Max bonus</label><input id="max_sticky_bonus" type="number" value="0.25" min="0" step="0.01"></div>
    </div>
    <div class="row">
      <div><label>Raw overlap</label><input id="min_raw_route_overlap" type="number" value="2" min="0"></div>
      <div><label>Sticky strength</label><input id="sticky_routing_strength" type="number" value="0.35" min="0" step="0.01"></div>
    </div>
    <label>System prompt</label>
    <textarea id="system_prompt">You are a helpful assistant. Answer directly and concisely.</textarea>
    <button id="run" style="margin-top:12px">Run Stream</button>
  </aside>
  <section>
    <div class="output" id="output"></div>
    <div class="grid" id="metrics"></div>
    <table>
      <thead><tr><th>Layer</th><th>Experts</th><th>Raw</th><th>Cap</th><th>Adjusted</th></tr></thead>
      <tbody id="experts"></tbody>
    </table>
    <table>
      <thead><tr><th>Tok</th><th>Latency</th><th>Recent tok/s</th><th>Hit</th><th>Reread MB/tok</th><th>Evict</th><th>Entropy</th></tr></thead>
      <tbody id="history"></tbody>
    </table>
    <div class="log" id="log"></div>
  </section>
</main>
<script>
const $ = id => document.getElementById(id);
const fields = ["model","prompt_format","system_prompt","expert_matmul_precision","max_layers","max_new_tokens","max_context_tokens","expert_cache_mb","hot_vram_budget_mb","active_experts_per_token_cap","expert_protect_tokens","expert_prefetch_limit","sticky_candidate_margin","max_sticky_bonus","min_raw_route_overlap","sticky_routing_strength","sticky_routing","predictive_prefetch","dynamic_expert_caps","cache_aware_routing","routing_semantic_guard","layer_aware_cache","adaptive_layer_quota","degeneration_guard"];
let socket = null;
let tokenHistory = [];
function settings() {
  const s = {};
  for (const id of fields) {
    const el = $(id);
    if (el.type === "checkbox") s[id] = el.checked;
    else if (el.type === "number") s[id] = Number(el.value);
    else s[id] = el.value;
  }
  return s;
}
function metric(label, value, cls="") { return `<div class="metric"><span>${label}</span><b class="${cls}">${value ?? "-"}</b></div>`; }
function renderTelemetry(t) {
  if (!t) return;
  $("metrics").innerHTML = [
    metric("tok/s e2e", Number(t.end_to_end_tokens_per_second || t.tokens_per_second || 0).toFixed(4)),
    metric("tok/s decode", Number(t.decode_tokens_per_second || 0).toFixed(4)),
    metric("tok/s recent", Number(t.recent_tokens_per_second || 0).toFixed(4)),
    metric("token latency", Number(t.token_latency_s || 0).toFixed(3) + "s"),
    metric("VRAM allocated", (t.vram?.allocated_gb ?? 0) + " GB"),
    metric("hit decode", Number(t.decode_cache_hit_rate || 0).toFixed(3)),
    metric("evictions", t.evictions),
    metric("reread MB/tok", Number(t.decode_reread_mb_per_token || t.reread_mb_per_token || 0).toFixed(1)),
    metric("blocked prefetch", Number(t.blocked_on_prefetch_s || 0).toFixed(3) + "s", t.blocked_on_prefetch_s > 1 ? "bad" : ""),
    metric("active set", t.active_set_size),
    metric("routing entropy", Number(t.routing_entropy || 0).toFixed(3)),
    metric("mat ms", Number(t.materialization_ms || 0).toFixed(3)),
    metric("prefetch waste", Number(t.prefetch_waste_rate || 0).toFixed(3)),
    metric("useful bw", Number(t.effective_useful_bandwidth || 0).toFixed(3)),
    metric("thrash", t.expert_thrash_mode ? "yes" : "no", t.expert_thrash_mode ? "bad" : "good"),
  ].join("");
  $("experts").innerHTML = (t.active_experts || []).slice(-24).map(row =>
    `<tr><td>${row.layer}</td><td>${row.experts.join(", ")}</td><td>${row.raw_experts.join(", ")}</td><td>${row.active_expert_cap ?? ""}</td><td>${row.sticky_adjusted ? "yes" : ""}</td></tr>`
  ).join("");
}
function logLine(obj) {
  $("log").textContent += JSON.stringify(obj) + "\n";
  $("log").scrollTop = $("log").scrollHeight;
}
$("run").onclick = () => {
  if (socket) socket.close();
  $("output").textContent = "";
  $("log").textContent = "";
  $("history").innerHTML = "";
  tokenHistory = [];
  $("status").textContent = "connecting";
  $("run").disabled = true;
  socket = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket.onopen = () => socket.send(JSON.stringify({type:"start", prompt:$("prompt").value, settings:settings()}));
  socket.onmessage = ev => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "start") $("status").innerHTML = `<span class="pill">${msg.model.layers_executed}/${msg.model.layers_total} layers</span>`;
    if (msg.type === "prefill") { $("status").textContent = `prefill ${msg.position}/${msg.context_tokens}`; renderTelemetry(msg.telemetry); }
    if (msg.type === "token") { $("output").textContent = msg.generated_text; $("status").textContent = `token ${msg.step + 1}`; renderTelemetry(msg.telemetry); logLine({token:msg.token_text, id:msg.token_id, latency:msg.telemetry.token_latency_s}); }
    if (msg.type === "token") {
      tokenHistory.push({idx: msg.step + 1, telemetry: msg.telemetry});
      $("history").innerHTML = tokenHistory.slice(-64).map(row => {
        const t = row.telemetry || {};
        return `<tr><td>${row.idx}</td><td>${Number(t.token_latency_s || 0).toFixed(3)}</td><td>${Number(t.recent_tokens_per_second || 0).toFixed(4)}</td><td>${Number(t.token_cache_hit_rate || 0).toFixed(3)}</td><td>${Number(t.token_reread_mb || 0).toFixed(1)}</td><td>${t.token_evictions ?? 0}</td><td>${Number(t.routing_entropy || 0).toFixed(3)}</td></tr>`;
      }).join("");
    }
    if (msg.type === "guard_stop") { $("output").textContent = msg.generated_text; $("status").textContent = `guard stopped: ${msg.reason}`; renderTelemetry(msg.telemetry); logLine({guard_stop:msg.reason}); }
    if (msg.type === "done") { $("status").textContent = `done in ${msg.latency_s}s`; $("run").disabled = false; logLine({done:true, tokens:msg.generated_token_ids.length}); }
    if (msg.type === "error") { $("status").textContent = "error"; $("run").disabled = false; logLine(msg); }
  };
  socket.onclose = () => { $("run").disabled = false; if ($("status").textContent !== "done") $("status").textContent = "closed"; };
};
</script>
</body>
</html>
"""


def create_app():
    app = FastAPI(title="KLCQUANT Streamed Runtime")
    worker_lock = asyncio.Lock()

    @app.get("/")
    async def index() -> HTMLResponse:
        return HTMLResponse(HTML)

    @app.get("/api/models")
    async def models() -> JSONResponse:
        payload = {}
        for key, preset in MODEL_PRESETS.items():
            support = Path(preset["support_dir"])
            config = read_json(support / "config.json") if (support / "config.json").exists() else {}
            payload[key] = {**preset, "available": bool(config), "config": config}
        return JSONResponse(payload)

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            message = json.loads(await websocket.receive_text())
            if message.get("type") != "start":
                await websocket.send_json({"type": "error", "error": "first message must be type=start"})
                return
            prompt = str(message.get("prompt", "Hello."))
            settings = _coerce_settings(message.get("settings") or {})
            if worker_lock.locked():
                await websocket.send_json({"type": "error", "error": "inference worker is busy"})
                return
            async with worker_lock:
                await stream_generation(websocket, prompt, settings)
        except Exception as exc:
            await websocket.send_json({"type": "error", "error": f"{type(exc).__name__}: {exc}"})
        finally:
            await websocket.close()

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the KLCQUANT local streamed inference server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8008)
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import uvicorn

    uvicorn.run("klcquant.inference_server:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
