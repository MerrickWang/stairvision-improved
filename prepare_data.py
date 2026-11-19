"""Convert the authors' stair-line annotations without changing official splits."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image
import yaml


def line_to_pose(row, width, height, coordinates):
    if len(row) != 5:
        raise ValueError(f'Expected class x1 y1 x2 y2, got {row}')
    cls, x1, y1, x2, y2 = map(float, row)
    if cls not in (0, 1):
        raise ValueError('Class must be convex=0 or concave=1')
    pts = np.array([[x1, y1], [x2, y2]], dtype=float)
    if coordinates == 'pixels':
        pts /= [width, height]
    if not np.isfinite(pts).all() or (pts < 0).any() or (pts > 1).any():
        raise ValueError(f'Out of bounds endpoints: {pts}')
    pts = pts[np.argsort(pts[:, 0])]
    if np.linalg.norm(pts[1] - pts[0]) < 1e-6:
        raise ValueError('Zero-length line')
    # Fixed 6-pixel padding around a line gives horizontal lines nonzero area.
    lo = np.maximum(pts.min(0) - [3 / width, 3 / height], 0)
    hi = np.minimum(pts.max(0) + [3 / width, 3 / height], 1)
    center, size = (lo + hi) / 2, hi - lo
    return [int(cls), *center, *size, *pts[0], 2, *pts[1], 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='data/raw')
    ap.add_argument('--output', default='data/prepared')
    ap.add_argument('--coordinates', choices=['pixels', 'normalized'], required=True)
    args = ap.parse_args()
    raw, out = Path(args.raw).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stats, hashes = {}, {}
    for split in ('train', 'val'):
        candidates = [p for p in raw.rglob('images') if p.parent.name == split]
        if len(candidates) != 1:
            raise ValueError(f'Expected one {split}/images directory, got {candidates}')
        source = candidates[0].parent
        images = sorted(p for p in (source / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        image_out, label_out = out / 'images' / split, out / 'labels' / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        records, objects = [], 0
        hashes[split] = set()
        for img in images:
            label = source / 'labels' / (img.stem + '.txt')
            depth = source / 'depthes' / (img.stem + '.png')
            if not label.exists() or not depth.exists():
                raise FileNotFoundError(f'Missing label or depth for {img}')
            with Image.open(img) as im:
                w, h = im.size
            rows = [line.split() for line in label.read_text().splitlines() if line.strip()]
            pose = [line_to_pose(row, w, h, args.coordinates) for row in rows]
            (label_out / label.name).write_text('\n'.join(' '.join(f'{v:.8g}' for v in row) for row in pose) + '\n')
            dest = image_out / img.name
            if not dest.exists():
                dest.symlink_to(img)
            hashes[split].add(hashlib.sha256(img.read_bytes()).hexdigest())
            records.append({'image': str(img), 'depth': str(depth), 'depth_kind': 'encoded_uint8', 'split': split})
            objects += len(pose)
        (out / f'depth_{split}.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        stats[split] = {'images': len(records), 'lines': objects}
    stats['identical_image_overlap'] = len(hashes['train'] & hashes['val'])
    stats['split_policy'] = 'authors official split; scene independence is not established'
    if stats['identical_image_overlap']:
        raise ValueError(f'Exact image leakage: {stats}')
    cfg = {'path': str(out), 'train': 'images/train', 'val': 'images/val', 'names': {0: 'convex', 1: 'concave'}, 'kpt_shape': [2, 3], 'flip_idx': [1, 0]}
    (out / 'stairs_pose.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / 'summary.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
