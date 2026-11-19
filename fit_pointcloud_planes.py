#!/usr/bin/env python3
import argparse
import json
import struct
from pathlib import Path

import numpy as np


def read_ply(path):
    raw = path.read_bytes()
    marker = b"end_header\n"
    start = raw.index(marker) + len(marker)
    count = int(next(line for line in raw[:start].splitlines() if line.startswith(b"element vertex")).split()[2])
    dtype = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    arr = np.frombuffer(raw, dtype=dtype, count=count, offset=start)
    return np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float64)


def read_pcd(path):
    raw = path.read_bytes()
    marker = b"DATA binary\n"
    start = raw.index(marker) + len(marker)
    count = int(next(line for line in raw[:start].splitlines() if line.startswith(b"POINTS")).split()[1])
    arr = np.frombuffer(raw, dtype="<f4", count=count * 4, offset=start).reshape(-1, 4)
    return arr[:, :3].astype(np.float64)


def fit(points, sample=20000):
    if len(points) > sample:
        points = points[np.linspace(0, len(points) - 1, sample).astype(int)]
    points = points[np.isfinite(points).all(axis=1)]
    center = points.mean(axis=0)
    _, values, vectors = np.linalg.svd(points - center, full_matrices=False)
    normal = vectors[-1]
    residual = np.abs((points - center) @ normal)
    return {"points": int(len(points)), "center": center.tolist(), "normal": normal.tolist(), "singular_values": values.tolist(), "median_plane_residual": float(np.median(residual)), "p95_plane_residual": float(np.percentile(residual, 95)), "bounds_min": points.min(axis=0).tolist(), "bounds_max": points.max(axis=0).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.root)
    files = sorted((root / "plys").glob("*.ply"))[: args.limit]
    rows = []
    for path in files:
        try:
            result = fit(read_ply(path))
            result.update({"file": path.name, "format": "ply", "status": "ok"})
        except Exception as exc:
            result = {"file": path.name, "format": "ply", "status": "error", "error": str(exc)}
        rows.append(result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    ok = [row for row in rows if row["status"] == "ok"]
    print(json.dumps({"files": len(rows), "ok": len(ok), "median_residual": float(np.median([row["median_plane_residual"] for row in ok])) if ok else None, "output": str(output)}))


if __name__ == "__main__":
    main()
