"""Filename-matched RGB/depth/segmentation data and geometric grid labels."""
import argparse
import json
from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import config

H, W = config.feature_size_h, config.feature_size_w
SY, SX = 512 / H, 512 / W


def grid_labels(lines, flip=False):
    """Clip each straight line against grid cells; preserve horizontal lines.

    Returns class-specific [Gaussian confidence, x1,y1,x2,y2, occupancy].
    This replaces the slow upstream converter and its divide-by-zero/boundary
    cases. If same-class segments share a cell, retain the longer segment.
    """
    targets = np.zeros((2, 6, H, W), dtype=np.float32)
    lengths = np.zeros((2, H, W), dtype=np.float32)
    collisions = 0
    for row in lines:
        if len(row) != 5:
            raise ValueError('Line labels must have class x1 y1 x2 y2')
        cls = int(row[0])
        if cls not in (0, 1) or float(cls) != row[0]:
            raise ValueError('Invalid line class')
        p, q = np.asarray(row[1:3], float), np.asarray(row[3:5], float)
        if not np.isfinite([p, q]).all() or (np.asarray([p, q]) < 0).any() or (np.asarray([p, q]) > 512).any():
            raise ValueError('Invalid pixel endpoints')
        if flip:
            p[0], q[0] = 512-p[0], 512-q[0]
        if p[0] > q[0] or (p[0] == q[0] and p[1] > q[1]):
            p, q = q, p
        delta = q-p
        length = np.linalg.norm(delta)
        if length < 1e-6:
            raise ValueError('Zero-length line')
        ts = [0., 1.]
        for axis, stride, cells in ((0, SX, W), (1, SY, H)):
            if abs(delta[axis]) > 1e-8:
                ts.extend(t for k in range(1, cells) if 0 < (t := (k*stride-p[axis])/delta[axis]) < 1)
        ts = np.unique(np.round(ts, 12))
        for t0, t1 in zip(ts[:-1], ts[1:]):
            if t1-t0 < 1e-9:
                continue
            a, b = p+t0*delta, p+t1*delta
            middle = (a+b)/2
            j, i = min(int(middle[0]/SX), W-1), min(int(middle[1]/SY), H-1)
            seglen = np.linalg.norm(b-a)
            if targets[cls, 5, i, j]:
                collisions += 1
                if seglen <= lengths[cls, i, j]:
                    continue
            loc = np.concatenate(((a-[j*SX, i*SY])/[SX, SY], (b-[j*SX, i*SY])/[SX, SY]))
            conf = np.exp(-.5*(min(t0, 1-t1)*length/max(length/2, 1e-6))**2)
            targets[cls, :, i, j] = [conf, *np.clip(loc, 0, 1), 1]
            lengths[cls, i, j] = seglen
    return targets, collisions


def prepare(raw, cache):
    raw, cache = Path(raw).resolve(), Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    report = {'grid': [H, W], 'converter': 'analytic grid clipping v1; longest same-class segment wins'}
    for split in ('train', 'val'):
        candidates = [p for p in raw.rglob('images') if p.parent.name == split]
        if len(candidates) != 1:
            raise ValueError(f'Ambiguous split: {split}')
        root = candidates[0].parent
        target = cache/split; target.mkdir(exist_ok=True)
        records, cells, conflicts = [], 0, 0
        for image in sorted((root/'images').glob('*.jpg')):
            depth, seg, label = [root/folder/(image.stem+ext) for folder, ext in [('depthes','.png'), ('segmentations','.png'), ('labels','.txt')]]
            if not all(p.exists() for p in (depth, seg, label)):
                raise FileNotFoundError(f'Missing paired annotation for {image}')
            rgb = np.array(Image.open(image)); d = np.array(Image.open(depth)); m = np.array(Image.open(seg))
            if rgb.shape != (512,512,3) or d.shape != (512,512) or m.shape != (512,512) or d.dtype != np.uint8:
                raise ValueError(f'Unexpected image geometry or depth type: {image}')
            if not set(np.unique(m)).issubset({0,128,255}):
                raise ValueError(f'Invalid semantic mask: {seg}')
            lines = [list(map(float,s.split())) for s in label.read_text().splitlines() if s.strip()]
            normal, c = grid_labels(lines); flipped, fc = grid_labels(lines, True)
            label_cache = target/(image.stem+'.npz')
            np.savez_compressed(label_cache, normal=normal, flipped=flipped)
            records.append({'image':str(image),'depth':str(depth),'segmentation':str(seg),'targets':str(label_cache)})
            cells += int(normal[:,5].sum()); conflicts += c+fc
        if not records:
            raise ValueError('Empty split')
        (cache/(split+'.jsonl')).write_text(''.join(json.dumps(x)+'\n' for x in records))
        report[split] = {'samples':len(records),'positive_cells':cells,'same_class_cell_conflicts_both_flips':conflicts}
    (cache/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


class StairPairs(Dataset):
    def __init__(self, manifest, train=False, limit=0):
        self.rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError('Empty dataset')
        self.augment = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = np.array(Image.open(row['image']).convert('RGB'))
        depth = np.array(Image.open(row['depth']))
        mask = np.array(Image.open(row['segmentation']))
        flip = self.augment and random.random() < .5
        with np.load(row['targets']) as z:
            targets = z['flipped' if flip else 'normal'].copy()
        if flip:
            image, depth, mask = [x[:,::-1].copy() for x in (image,depth,mask)]
        image = torch.from_numpy(image.copy()).permute(2,0,1).float()/127.5-1
        depth = torch.from_numpy(depth.copy()).float().unsqueeze(0)/255
        sem = np.stack([mask==value for value in (0,128,255)]).astype(np.float32)
        targets = torch.from_numpy(targets)
        return image, depth, (targets[0,:1],targets[0,1:5],targets[0,5:]), (targets[1,:1],targets[1,1:5],targets[1,5:]), torch.from_numpy(sem)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',default='/root/stairlab/data/raw');p.add_argument('--cache',default='data/prepared_depthout');a=p.parse_args();prepare(a.raw,a.cache)
