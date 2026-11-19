#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np

from fit_pointcloud_planes import read_ply


def estimate(points, bins=160):
    z = points[:, 2]
    z = z[np.isfinite(z)]
    lo, hi = np.percentile(z, [1, 99])
    hist, edges = np.histogram(z[(z >= lo) & (z <= hi)], bins=bins, range=(lo, hi))
    smooth = np.convolve(hist.astype(float), np.ones(5) / 5, mode="same")
    threshold = max(float(np.percentile(smooth, 70)), 1.0)
    peaks = [i for i in range(1, bins - 1) if smooth[i] >= smooth[i - 1] and smooth[i] >= smooth[i + 1] and smooth[i] >= threshold]
    peaks.sort(key=lambda i: smooth[i], reverse=True)
    selected = []
    for index in peaks:
        if all(abs(index - other) >= 5 for other in selected):
            selected.append(index)
    selected.sort()
    levels = [float((edges[i] + edges[i + 1]) / 2) for i in selected]
    gaps = [levels[i + 1] - levels[i] for i in range(len(levels) - 1)]
    return {"candidate_levels": levels, "candidate_gaps": gaps, "z_range": [float(lo), float(hi)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    files = sorted((Path(args.root) / "plys").glob("*.ply"))[: args.limit]
    rows = []
    for path in files:
        result = estimate(read_ply(path))
        result["file"] = path.name
        rows.append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps({"files": len(rows), "mean_candidate_levels": float(np.mean([len(r["candidate_levels"]) for r in rows])), "output": str(output)}))


if __name__ == "__main__":
    main()
