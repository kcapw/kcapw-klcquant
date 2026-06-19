# Experimental Status

This version of KLCquant is experimental and unfinished. It is currently functional for research, scanning, quantization experiments, streamed probes, and local runtime debugging, but it still needs more validation before it should be treated as a production inference runtime.
# KLCQUANT

KLCQUANT is an experimental Python research framework for adaptive tensor importance analysis and dynamic quantization of large sharded transformer models.

It is designed around a hard constraint: do not load the full model into VRAM. The default path reads the safetensors index, groups tensors by layer/global role, and can stream one group at a time through CUDA for measurement.

## Install

```bash
python -m pip install -e .
```

## Generate benchmark prompts

```bash
python -m klcquant.benchmark_runner generate-prompts --count 1000 --out prompts/generated_prompts.json
```

## Static shard analysis

This scans tensor metadata and small value samples from safetensors shards on CPU, then writes JSON reports with importance rankings and a quantization plan.

```bash
python -m klcquant.benchmark_runner --model-dir model --support-dir model_support --out-dir reports scan
```

## CUDA streamed scan

This loads one tensor group at a time into VRAM, records CUDA memory snapshots, computes value statistics, and unloads before moving to the next group.

```bash
python -m klcquant.benchmark_runner --model-dir model --support-dir model_support --out-dir reports scan --cuda-stream
```

## Optional inference profiling

If your installed `transformers` build supports the model architecture, you can run generation profiling with module hooks:

```bash
python -m klcquant.benchmark_runner \
  --support-dir model_support \
  --prompts prompts/generated_prompts.json \
  --prompt-count 100 \
  hf-profile \
  --hf-model-path . \
  --max-new-tokens 32
```

For a 120B-class model this may still require CPU/NVMe offload or a custom runtime. The shard streaming pieces are intentionally separated so they remain usable even when full Hugging Face inference is not feasible.

## Heatmaps

```bash
python -m klcquant.tensor_heatmap reports/klcquant-scan-YYYYMMDD-HHMMSS.json
```

## Partial Streamed Quantization

This prototype consumes an importance report, selects a bounded subset of tensors, loads one tensor at a time, quantizes on CUDA when available, writes packed artifacts to `quantized_model/`, and records reconstruction quality.

```bash
python -m klcquant.streamed_quant_runner \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --quantized-dir quantized_model \
  --max-tensors 12 \
  --max-original-mb 256
```

For a first low-risk pass, constrain to low-rank modes:

```bash
python -m klcquant.streamed_quant_runner \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --mode q3 --mode q4 \
  --max-tensors 12 \
  --max-original-mb 64
```

## Streamed Runtime Evaluation

The first inference-quality milestone is a low-VRAM streamed logit probe. It is not a full 120B autoregressive forward pass yet. It tokenizes prompts, loads only the required embedding rows, streams original and quantized runtime tensors group by group, unloads cold tensors, and computes vocabulary logits through chunked `lm_head` reads.

```bash
python -m klcquant.prompt_eval_runner \
  --model-dir model \
  --support-dir model_support \
  --quantized-dir quantized_model \
  --prompt-count 6 \
  --max-runtime-tensors 8 \
  --lm-head-chunk-rows 4096
```

Reports include token overlap, logit drift, probe perplexity drift, response-length deltas, hallucination-risk indicators, latency, VRAM peak, cache hit rate, runtime tensor hotness, and PNG charts for quality, VRAM residency, and tensor hotness.

## Minimal Streamed Autoregressive Generation

The next milestone is a real token-by-token streamed decoder prefix. It executes a small number of transformer blocks with real attention, real KV-cache updates, real router top-k selection, streamed MXFP4 selected-expert matmuls, quantized tensor overrides when present, and chunked `lm_head` logits.

```bash
python -m klcquant.autoregressive_runner \
  --model-dir model \
  --support-dir model_support \
  --quantized-dir quantized_model \
  --prompt-count 1 \
  --layer-depths 2,4 \
  --max-context-tokens 2 \
  --token-counts 1,2 \
  --expert-cache-mb 512
```

Progression runs can stop automatically when drift gets too high:

```bash
python -m klcquant.autoregressive_runner \
  --layer-depths 2,4,8,16 \
  --token-counts 1 \
  --stop-perplexity-drift 100 \
  --stop-sequence-overlap 0.25
```

## Local Streamed Inference Server

For interactive runtime debugging, start the single-user FastAPI websocket server:

```bash
klcquant-serve --host 127.0.0.1 --port 8008
```

Then open:

```text
http://127.0.0.1:8008
```

The debug UI supports the `/mnt/ramdisk` 20B model and the workspace 120B model, streams generated tokens as each decode step completes, and updates telemetry after every prefill/sample step.

Live runtime controls:

- model: `20b` or `120b`
- max layers, max new tokens, and max context tokens
- expert cache MB and hot residency MB
- active experts/token cap
- sticky routing
- predictive expert prefetch
- dynamic expert caps
- cache-aware routing

Telemetry includes rolling tokens/sec, per-token latency, active routed experts, cache hit rate, evictions, rereads, blocked-on-prefetch time, active set size, CUDA memory, hot experts, routing entropy, materialization/dequant time, reuse distance, and prefetch waste.

This server is intentionally lightweight and single-user. It uses greedy decoding and the same streamed executor path as the CLI probes, so it is meant for observing locality/reread behavior after architecture changes rather than serving production traffic.

## Modules

- `streamed_loader.py`: safetensors index parsing, layer/tensor grouping, streamed CUDA loading, async prefetch.
- `tensor_profiler.py`: static tensor statistics, CUDA stream scans, optional forward hooks.
- `importance_ranker.py`: scoring and rankings for critical, medium, and least-used tensors.
- `adaptive_quantizer.py`: adaptive fp16/q16/q8/q4/q3 policy and quantization utilities.
- `prompt_generator.py`: diverse prompt corpus generation across coding, reasoning, multilingual, math, long context, roleplay, edge cases, memory, logic, creative, and symbolic tasks.
- `benchmark_runner.py`: CLI report generation.
- `streamed_quant_runner.py`: partial resumable streamed quantization runner.
- `quantized_runtime_loader.py`: dynamic quantized artifact discovery and mixed-precision loading.
- `runtime_dequantizer.py`: q3/q4/q8/q16/fp16 runtime reconstruction.
- `runtime_prefetcher.py`: asynchronous quantized group prefetching.
- `streamed_inference.py`: prompt-conditioned streamed logit-probe engine.
- `prompt_eval_runner.py`: before/after prompt comparison reports and charts.
- `output_similarity.py`: token, logit, perplexity, and drift-risk metrics.
- `streamed_transformer_executor.py`: minimal streamed decoder block executor.
- `inference_server.py`: local FastAPI websocket server and HTML runtime telemetry UI.
- `streamed_attention.py`: chunk-conscious grouped-query attention and KV updates.
- `streamed_moe_router.py`: router top-k selection and expert activation tracking.
- `mxfp4_expert.py`: selected-expert MXFP4 unpacking, dequantization, and expert weight cache.
- `kv_cache_manager.py`: bounded/offloadable KV cache.
- `tensor_scheduler.py`: layer scheduling and cache statistics.
- `generation_runtime.py`: baseline vs quantized token generation helpers.
- `autoregressive_runner.py`: tiny streamed autoregressive comparison CLI.
- `q1_quantizer.py` / `q2_quantizer.py`: experimental packed ultra-low-bit tensor storage.
- `tensor_pruner.py`: magnitude pruning and sparse value/index reconstruction.
- `sparsity_tracker.py`: pruning density/sparsity metrics.
- `stability_guard.py`: drift/divergence guardrails for autoregressive experiments.

## Ultra-Low-Bit Experiments

Q1/Q2/pruning are intentionally restricted to ultra-low-importance tensors. Do not apply them to attention, router, embeddings, norms, or `lm_head`.

Example:

```bash
python -m klcquant.streamed_quant_runner \
  --quantized-dir quantized_experiments/mixed_q4_q2 \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --group layer_001 \
  --pattern 'model.layers.1.mlp.experts.down_proj_bias' \
  --force-mode q2 \
  --max-tensors 1
```

The first controlled ultra-low-bit run found that layer-0/layer-1 expert down-projection biases are more generation-sensitive than their static importance suggested: Q4/Q2/Q1/pruned variants diverged in a 2-layer, 4-token real-expert generation test. Treat that as a guardrail for the next policy pass.

## Runtime Sensitivity Calibration

Static importance is not enough for compression policy decisions. The calibration CLI perturbs one tensor at a time, rebuilds a temporary q8/q4/q2/q1/pruned artifact, runs tiny streamed autoregressive validation, and records the measured stability boundary in `reports/sensitivity_database.json`.

```bash
klcquant-calibrate \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --candidate model.layers.0.mlp.experts.down_proj_bias \
  --targets 2x4 \
  --modes q8,q4,q2,q1,pruned
```

The full research progression is available with `--targets 2x4,4x8,8x8`, but it is intentionally not the default. Calibration reports include tensor stability rankings, recommended precision tiers, instability heatmaps, role/layer clustering, and safety notes for tensors that should remain fp16/q8.

## Multi-Tensor Compression Orchestration

Once individual sensitivity exists, use the orchestrator to test whether tensors remain stable when compressed together. It builds mixed-precision temporary artifact sets, runs streamed autoregressive validation, stops progressive escalation on instability, and emits a compression safety graph plus architecture-wide maps.

```bash
klcquant-orchestrate \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --candidate model.layers.1.mlp.experts.down_proj_bias \
  --candidate model.layers.1.mlp.experts.gate_up_proj_bias \
  --profiles conservative \
  --targets 2x4
```

Broader bounded sweep with an early, middle, and later tensor plus an expert MXFP4 scale tensor:

```bash
klcquant-orchestrate \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --candidate model.layers.1.mlp.experts.down_proj_bias \
  --candidate model.layers.4.mlp.experts.gate_up_proj_scales \
  --candidate model.layers.7.mlp.experts.down_proj_bias \
  --profiles conservative,balanced,aggressive,ultra_low_vram \
  --targets 2x4,4x8,8x8 \
  --prompt-count 1 \
  --max-context-tokens 2
```

Expected output files:

- `reports/klcquant-orchestration-<run_id>.json`: complete run data, `summary_report`, safety graph, profile summaries, architecture maps, and per-step metrics.
- `reports/klcquant-orchestration-<run_id>.profiles.png`: profile/precision stability chart.
- `reports/klcquant-orchestration-<run_id>.layers.png`: layer criticality overlay.
- `calibration_runs/...`: temporary mixed-precision safetensor override artifacts.

Available runtime profiles are `conservative`, `balanced`, `aggressive`, and `ultra_low_vram`. They tune precision escalation and attach cache/prefetch residency guidance to the report; they do not override the streamed execution safety constraints.

Interpretation:

- `co_compressible` means the tested tensor pair stayed inside the configured stability guard for the specific target, prompt count, profile, and precision step. It is a local approval, not a global proof.
- `amplifies_instability` means combined compression produced more drift than singleton history predicted, or failed at a deeper/lower-precision step. Treat that edge as unsafe until a narrower policy is validated.
- Trust `aggressive` or `ultra_low_vram` only when the exact target depth/token count remains stable, sequence overlap stays high, perplexity drift is bounded, and no edge involving the compressed tensors is marked unsafe.
- Reject an aggressive/ultra-low result when it only passes shallow targets, stops early at deeper targets, has first-token divergence, or relies on tensors with sparse calibration history.

## Recursive Survivability And Recovery

The survivability runner measures token-by-token drift dynamics, KV-cache pressure, transfer bursts, and adaptive promotion recovery. It adds the profiles `recursive_safe`, `recovery_aggressive`, and `cache_preserving`.

```bash
klcquant-survive \
  --importance-report reports/klcquant-scan-YYYYMMDD-HHMMSS.json \
  --candidate model.layers.1.mlp.experts.down_proj_bias \
  --candidate model.layers.1.mlp.experts.gate_up_proj_bias \
  --profiles conservative,balanced,recursive_safe,recovery_aggressive \
  --token-depths 256,512,1024 \
  --max-layers 2 \
  --prompt-count 1
```

Expected output files:

- `reports/klcquant-survivability-<run_id>.json`: recursive drift, recovery attempts, KV survivability, pressure telemetry, and frontier data.
- `reports/klcquant-survivability-<run_id>.frontier.png`: stable / recoverable / unrecoverable frontier.
- `reports/klcquant-survivability-<run_id>.pressure.png`: transfer and KV pressure graph.

Interpretation:

- `drift_velocity` is the growth rate of cumulative token/routing/entropy instability across generated tokens.
- `overlap_decay_rate` is how quickly prefix token agreement falls as generation proceeds.
- `instability_acceleration` catches compounding failures where later tokens degrade faster than early tokens.
- A `degraded_but_recoverable` frontier point means the first precision setting failed, but promotion such as q2 -> q4 -> q8 extended survival or restored overlap.
- Treat `recovery_aggressive` as experimental unless recovery succeeds at the requested token depth and promotion-triggered pressure remains within VRAM limits.
- Prefer `cache_preserving` when q4 is viable but KV reset/rebuild costs would dominate, or when stale-cache divergence appears in q8/q4 KV experiments.
- `compression_policy.py`: rank-aware experimental compression policy.
- `compression_orchestrator.py`: progressive multi-tensor compression policy search.
- `adaptive_recovery_runner.py`: recursive survivability control and promotion recovery CLI.
- `recursive_drift_tracker.py`: token-step drift velocity, overlap decay, routing divergence, and instability acceleration.
- `runtime_pressure_telemetry.py`: residency, reload churn, transfer burst, and cache pressure summaries.
- `kv_survivability.py`: KV precision survivability and stale-cache divergence metrics.
- `multi_tensor_calibrator.py`: mixed override-set builder and streamed validation runner.
- `interaction_sensitivity.py`: compression safety graph and co-compression edge scoring.
- `instability_accumulator.py`: drift accumulation and early-stop logic.
- `drift_forecaster.py`: lightweight divergence and combination-risk forecasting.
- `runtime_sensitivity_probe.py`: q8/q4/q2/q1/pruned perturbation calibration CLI.
- `tensor_perturbation_runner.py`: single-tensor quantized override builder and streamed validation runner.
- `sensitivity_database.py`: persistent JSON database for runtime sensitivity records.
- `tensor_criticality_ranker.py`: dynamic criticality scoring and clustering helpers.
- `adaptive_precision_policy.py`: runtime-calibrated precision tier recommendations.
- `tensor_similarity.py`: tensor reconstruction quality metrics.
- `vram_monitor.py`: CUDA memory event logging.
- `tensor_heatmap.py`: visualization support.
- `runtime_cache.py`: simple byte-bounded LRU cache for streamed tensor groups.

Reports are JSON and include VRAM usage, estimated compression ratio, tensor rankings, quantization decisions, and generation metrics when inference profiling is available.
