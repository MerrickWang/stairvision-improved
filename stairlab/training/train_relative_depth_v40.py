"""Fine-tune DA-V2 Small with scale/shift-invariant depth and optional tread ranking."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


SAMPLES = 48
MAX_PAIRS = 12
DECISION_MARGIN = 0.05


def tread_pairs(label_file, tread, raw, valid, near_is):
    """Sample comparable tread interiors in adjacent convex-edge bands."""
    h, w = raw.shape
    lines = []
    for line in label_file.read_text().splitlines():
        cls, x1, y1, x2, y2 = map(float, line.split())
        if x2 < x1:
            x1, x2, y1, y2 = x2, x1, y2, y1
        if cls == 0 and x2 - x1 >= w / 8:
            lines.append((x1, y1, x2, y2))
    lines.sort(key=lambda line: (line[1] + line[3]) / 2)
    indices = np.zeros((MAX_PAIRS, 2, SAMPLES), np.int64)
    signs = np.zeros(MAX_PAIRS, np.float32)
    num = 0
    for start in range(len(lines) - 2):
        three = lines[start:start + 3]
        left = max(0, int(np.ceil(max(line[0] for line in three))))
        right = min(w, int(np.floor(min(line[2] for line in three))) + 1)
        if right - left < w / 10:
            continue
        cols = np.arange(left, right)
        rows = np.arange(h)[:, None]
        bounds = [np.interp(cols, [x1, x2], [y1, y2]) for x1, y1, x2, y2 in three]
        candidates = []
        for top, bottom in zip(bounds, bounds[1:]):
            gap = bottom - top
            inset = np.maximum(1, gap * .14)
            band = (gap >= 5) & (rows >= top + inset) & (rows <= bottom - inset)
            yy, xx = np.nonzero(band & tread[:, left:right] & valid[:, left:right])
            if len(xx) < SAMPLES:
                break
            order = np.lexsort((yy, xx))
            chosen = order[np.linspace(0, len(order) - 1, SAMPLES, dtype=int)]
            candidates.append(yy[chosen] * w + xx[chosen] + left)
        if len(candidates) != 2:
            continue
        values = raw.ravel()[np.stack(candidates)]
        low, med, high = np.percentile(values, [25, 50, 75], axis=1)
        delta = med[0] - med[1]
        if abs(delta) < max(4, .25 * np.mean(high - low)):
            continue
        indices[num] = np.stack(candidates)
        signs[num] = np.sign(delta) * (1 if near_is == 'high' else -1)
        num += 1
        if num == MAX_PAIRS:
            break
    return indices, signs


class StairDepthPairs(Dataset):
    def __init__(self, root, split, processor, size, near_is):
        self.root = Path(root) / split
        self.files = sorted(p for p in (self.root / 'images').iterdir()
                            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if not self.files:
            raise ValueError(f'No images in {self.root}')
        self.processor, self.size, self.near_is = processor, size, near_is

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]
        with Image.open(path) as image, Image.open(self.root / 'depthes' / f'{path.stem}.png') as depth, \
             Image.open(self.root / 'segmentations' / f'{path.stem}.png') as mask:
            rgb = np.asarray(image.convert('RGB'))
            raw = np.asarray(depth)
            tread = np.asarray(mask) == 255
        if raw.dtype != np.uint8 or raw.shape != rgb.shape[:2] or tread.shape != raw.shape:
            raise ValueError(f'Invalid RGB, encoded depth or mask alignment: {path}')
        valid = (raw > 0) & np.any(rgb > 0, axis=-1)
        indices, signs = tread_pairs(self.root / 'labels' / f'{path.stem}.txt', tread, raw, valid, self.near_is)
        pixels = self.processor(images=rgb, return_tensors='pt', keep_aspect_ratio=False,
                                size={'height': self.size, 'width': self.size})['pixel_values'][0]
        target = raw.astype(np.float32) * ((1 if self.near_is == 'high' else -1) / 255)
        return pixels, target, valid, indices, signs


def standardize(values, valid):
    count = valid.sum((1, 2), keepdim=True).clamp_min(1)
    center = (values * valid).sum((1, 2), keepdim=True) / count
    deviation = values - center
    scale = (deviation.abs() * valid).sum((1, 2), keepdim=True) / count
    return deviation / scale.clamp_min(1e-6)


def compute_terms(model, batch, device):
    pixels, target, valid, indices, signs = [value.to(device, non_blocking=True) for value in batch]
    result = model(pixel_values=pixels).predicted_depth.float()
    pred = F.interpolate(result[:, None], target.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
    pred = standardize(pred, valid)
    error = (pred - standardize(target, valid)).abs()[valid]
    samples = pred.flatten(1).gather(1, indices.flatten(1)).reshape_as(indices)
    medians = samples.median(dim=-1).values
    margins = ((medians[:, :, 0] - medians[:, :, 1]) * signs)[signs != 0]
    return error, margins


@torch.inference_mode()
def evaluate(model, loader, device):
    model.eval()
    absolute = pixels = pairs = correct = decided = 0
    for batch in loader:
        error, margins = compute_terms(model, batch, device)
        absolute += error.sum().item()
        pixels += error.numel()
        pairs += margins.numel()
        correct += (margins >= DECISION_MARGIN).sum().item()
        decided += (margins.abs() >= DECISION_MARGIN).sum().item()
    if not pairs or not pixels:
        raise ValueError('Validation needs valid depth pixels and reference tread pairs')
    return {'images': len(loader.dataset), 'normalized_mae': absolute / pixels, 'valid_pixels': pixels,
            'pairs': pairs, 'decided': decided, 'correct': correct, 'incorrect': decided - correct,
            'correct_all': correct / pairs, 'coverage': decided / pairs,
            'accuracy_decided': correct / decided if decided else None}


def make_model(source, device):
    model = AutoModelForDepthEstimation.from_pretrained(source, local_files_only=True).to(device)
    model.backbone.requires_grad_(False)
    return model


def train_variant(args, val, device, variant, weight):
    torch.manual_seed(args.seed)
    model = make_model(args.model, device)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    dataset = StairDepthPairs(args.dataset, 'train', processor, args.size, args.near_is)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                        persistent_workers=args.workers > 0, pin_memory=device.type == 'cuda')
    optimizer = torch.optim.AdamW([param for param in model.parameters() if param.requires_grad], lr=args.lr)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.backbone.eval()
        total_pairs = 0
        for step, batch in enumerate(loader, 1):
            optimizer.zero_grad(set_to_none=True)
            error, margins = compute_terms(model, batch, device)
            loss = error.sum() / max(error.numel(), 1)
            if weight and margins.numel():
                loss = loss + weight * F.relu(.10 - margins).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss in {variant} epoch {epoch} batch {step}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_pairs += margins.numel()
            if step % 100 == 0 or step == len(loader):
                print(f'{variant} epoch {epoch}/{args.epochs} batch {step}/{len(loader)} loss {loss.item():.4f}', flush=True)
        metrics = {'epoch': epoch, 'train_images': len(dataset), 'train_pairs': total_pairs,
                   **evaluate(model, val, device)}
        history.append(metrics)
        print(json.dumps({'variant': variant, **metrics}), flush=True)
        (args.output / f'{variant}_history.json').write_text(json.dumps(history, indent=2) + '\n')
    decoder = {name: value.detach().cpu() for name, value in model.state_dict().items()
               if not name.startswith('backbone.')}
    torch.save({'version': '4.0', 'variant': variant, 'size': args.size, 'decoder': decoder},
               args.output / f'{variant}.pt')
    return history[-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--near-is', choices=['high', 'low'], required=True)
    parser.add_argument('--model', default='models/depth-anything-v2-small')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--size', type=int, default=392)
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--rank-weight', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if args.size < 14 or args.size % 14 or args.epochs < 1 or args.rank_weight <= 0:
        parser.error('Use a positive patch-aligned size, epoch count and ranking weight')
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.glob('*.pt')) or (args.output / 'results.json').exists():
        raise FileExistsError(f'Use a new experiment directory: {args.output}')
    cv2.setNumThreads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    dataset = StairDepthPairs(args.dataset, 'val', processor, args.size, args.near_is)
    val = DataLoader(dataset, batch_size=args.batch, num_workers=args.workers,
                     persistent_workers=args.workers > 0, pin_memory=device.type == 'cuda')
    source = Path(args.model) / 'source.json'
    report = {'version': '4.0', 'args': vars(args) | {'output': str(args.output)},
              'decision_margin': DECISION_MARGIN, 'model_source': json.loads(source.read_text()) if source.is_file() else None}
    baseline = make_model(args.model, device)
    report['original'] = evaluate(baseline, val, device)
    print(json.dumps({'variant': 'original', **report['original']}), flush=True)
    del baseline
    for variant, weight in [('decoder_only', 0), ('tread_rank', args.rank_weight)]:
        report[variant] = train_variant(args, val, device, variant, weight)
    (args.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
