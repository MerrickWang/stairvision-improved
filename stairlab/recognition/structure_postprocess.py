#!/usr/bin/env python3
"""Clean YOLO stair-edge predictions with simple structural constraints."""

import argparse
import json
import math
from pathlib import Path


def midpoint(line):
    first, second = line["endpoints"][:2]
    return ((first[0] + second[0]) / 2, (first[1] + second[1]) / 2)


def line_length(line):
    first, second = line["endpoints"][:2]
    return math.hypot(first[0] - second[0], first[1] - second[1])


def clean(record, confidence=0.25, minimum_length_ratio=0.12, merge_y_ratio=0.035,
          *, short_edge_minimum_length_ratio=None):
    candidates = []
    for line in record.get("lines", []):
        if float(line.get("confidence", 0)) < confidence:
            continue
        if len(line.get("endpoints", [])) < 2:
            continue
        first, second = line["endpoints"][:2]
        diagonal = math.hypot(max(first[0], second[0]), max(first[1], second[1]), 1.0)
        ratio = minimum_length_ratio
        if short_edge_minimum_length_ratio is not None and line.get('short_edge_recovery'):
            ratio = short_edge_minimum_length_ratio
        if line_length(line) < ratio * diagonal:
            continue
        candidates.append(line)

    candidates.sort(key=lambda line: midpoint(line)[1])
    merged = []
    for line in candidates:
        match = None
        for index, previous in enumerate(merged):
            if line.get("class") != previous.get("class"):
                continue
            y_gap = abs(midpoint(line)[1] - midpoint(previous)[1])
            limit = merge_y_ratio * max(1.0, math.sqrt(max(line_length(line), line_length(previous))))
            if y_gap <= limit:
                match = index
                break
        if match is None:
            merged.append(line)
        elif line.get("confidence", 0) > merged[match].get("confidence", 0):
            merged[match] = line

    merged.sort(key=lambda line: (midpoint(line)[1], midpoint(line)[0]))
    convex_count = sum(int(line.get("class", -1)) == 0 for line in merged)
    result = dict(record)
    result["lines_structured"] = merged
    result["structured_convex_count"] = convex_count
    result["structured_total_count"] = len(merged)
    result["structure_status"] = "accepted" if convex_count >= 2 else "insufficient_lines"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    source = Path(args.input)
    files = sorted(source.glob("*.json")) if source.is_dir() else [source]
    rows = [clean(json.loads(path.read_text()), args.conf) for path in files]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    print(json.dumps({
        "images": len(rows),
        "raw_convex_total": sum(row.get("visible_convex_candidates", 0) for row in rows),
        "structured_convex_total": sum(row.get("structured_convex_count", 0) for row in rows),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
