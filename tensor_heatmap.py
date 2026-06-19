from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .utils import read_json

LAYER_RE = re.compile(r"model\.layers\.(\d+)\.(.+)")


def _bucket(name: str) -> tuple[int, str] | None:
    match = LAYER_RE.match(name)
    if not match:
        return None
    rest = match.group(2)
    if "self_attn" in rest:
        kind = "attention"
    elif "router" in rest:
        kind = "router"
    elif "experts" in rest:
        kind = "experts"
    elif "norm" in rest:
        kind = "norm"
    else:
        kind = "other"
    return int(match.group(1)), kind


def render_heatmap(report_path: str | Path, out_path: str | Path | None = None) -> Path:
    report = read_json(report_path)
    rankings = report["tensor_importance_rankings"]
    kinds = ["attention", "router", "experts", "norm", "other"]
    max_layer = 0
    for item in rankings:
        bucket = _bucket(item["name"])
        if bucket:
            max_layer = max(max_layer, bucket[0])
    data = np.zeros((max_layer + 1, len(kinds)), dtype=np.float32)
    counts = np.zeros_like(data)
    for item in rankings:
        bucket = _bucket(item["name"])
        if not bucket:
            continue
        layer, kind = bucket
        col = kinds.index(kind)
        data[layer, col] += float(item.get("importance_score", 0.0))
        counts[layer, col] += 1
    data = data / np.maximum(counts, 1)

    fig, ax = plt.subplots(figsize=(10, max(5, (max_layer + 1) * 0.22)))
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_title("KLCQUANT Tensor Importance Heatmap")
    ax.set_xlabel("tensor family")
    ax.set_ylabel("layer")
    ax.set_xticks(range(len(kinds)), kinds)
    ax.set_yticks(range(max_layer + 1))
    fig.colorbar(im, ax=ax, label="mean importance")
    fig.tight_layout()
    target = Path(out_path) if out_path else Path(report_path).with_suffix(".heatmap.png")
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--out")
    args = parser.parse_args()
    print(render_heatmap(args.report, args.out))


if __name__ == "__main__":
    main()
