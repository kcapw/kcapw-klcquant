#!/usr/bin/env bash
set -euo pipefail
python -m klcquant.benchmark_runner --model-dir model --support-dir model_support --out-dir reports scan --cuda-stream
