"""Train the YOLO11 short-fragment head from cached original stair-edge labels."""

import argparse
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from yolo_fragments import GRID, SIZE, YoloFragments


class FragmentDataset(Dataset):
    def __init__(self, manifest, *, augment=False, limit=0, surface_aux=False):
        self.augment = augment
        entries = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        self.rows = []
        for row in entries[:limit or None]:
            image = cv2.imread(row['image'])
            if image is None or image.shape != (SIZE, SIZE, 3):
                raise ValueError(f"Expected a 512x512 RGB image: {row['image']}")
            with np.load(row['targets']) as cache:
                normal = cache['normal'][0].astype(np.float16)
                flipped = cache['flipped'][0].astype(np.float16) if augment else None
            if normal.shape != (6, *GRID):
                raise ValueError(f"Unexpected fragment grid: {row['targets']}")
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            surface = None
            if surface_aux:
                raw_mask = cv2.imread(row['segmentation'], cv2.IMREAD_GRAYSCALE)
                if raw_mask is None or raw_mask.shape != (SIZE, SIZE) or not np.isin(raw_mask, (0, 128, 255)).all():
                    raise ValueError(f"Invalid surface label: {row['segmentation']}")
                labels = np.where(raw_mask == 128, 1, np.where(raw_mask == 255, 2, 0)).astype('uint8')
                surface = torch.from_numpy(cv2.resize(labels, (64, 64), interpolation=cv2.INTER_NEAREST_EXACT))
            self.rows.append((torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
                              torch.from_numpy(normal), None if flipped is None else torch.from_numpy(flipped), surface))
        if not self.rows:
            raise ValueError(f'Empty data split: {manifest}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, normal, flipped, surface = self.rows[index]
        if self.augment and random.random() < .5:
            return (image.flip((-1,)), flipped) if surface is None else (image.flip((-1,)), flipped, surface.flip((-1,)))
        return (image, normal) if surface is None else (image, normal, surface)


def fragment_loss(logits, labels):
    positive = labels[:, 5] > 0
    target = positive.float()
    score = logits[:, 0]
    probability = score.sigmoid()
    focal = torch.where(positive, 1 - probability, probability).square()
    occupancy = (focal * F.binary_cross_entropy_with_logits(score, target, reduction='none')).sum()
    delta = F.smooth_l1_loss(logits[:, 1:].sigmoid(), labels[:, 1:5], beta=.15, reduction='none')
    location = (delta * target[:, None]).sum() / 4
    return (occupancy + 3 * location) / target.sum().clamp(min=1)


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    stats = torch.zeros(4, device=device)
    confusion = torch.zeros((3, 3), device=device)
    for image, labels, *surface in loader:
        image = image.to(device, non_blocking=True).float() / 255
        labels = labels.to(device, non_blocking=True).float()
        with torch.autocast(device.type, enabled=device.type == 'cuda'):
            output = model(image, with_surfaces=bool(surface))
        if surface:
            output, surface_logits = output
            expected_surface = surface[0].to(device, non_blocking=True).long()
            classes = surface_logits.argmax(1)
            confusion += torch.bincount((3 * expected_surface + classes).flatten(), minlength=9).reshape(3, 3)
        output = output.float()
        predicted, expected = output[:, 0] >= 0, labels[:, 5] > 0
        endpoints = (output[:, 1:].sigmoid() - labels[:, 1:5]).square()
        distance = (endpoints[:, :2].sum(1).sqrt() + endpoints[:, 2:].sum(1).sqrt()) / 2
        matched = predicted & expected & (distance <= .25)
        stats += torch.stack((matched.sum(), predicted.sum(), expected.sum(), fragment_loss(output, labels)))
    tp, predicted, expected, loss = stats.tolist()
    report = {'validation_loss': loss / len(loader), 'cell_precision': tp / max(1, predicted),
              'cell_recall': tp / max(1, expected), 'cell_f1': 2 * tp / max(1, predicted + expected)}
    if model.surface_head is not None:
        intersection = confusion.diag()
        union = confusion.sum(0) + confusion.sum(1) - intersection
        report['surface_miou'] = (intersection[1:] / union[1:].clamp_min(1)).mean().item()
    return report


def save_checkpoint(path, state):
    pending = path.with_suffix('.part')
    torch.save(state, pending)
    pending.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label-cache', required=True, type=Path)
    parser.add_argument('--backbone', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=24)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-val', type=int, default=0)
    parser.add_argument('--surface-aux', action='store_true', help='Supervise tread/riser segmentation during training')
    parser.add_argument('--freeze-features', action='store_true', help='Keep the original feature weights frozen')
    args = parser.parse_args()
    if min(args.epochs, args.batch) < 1:
        parser.error('Epochs and batch size must be positive')
    if (args.output / 'last.pt').exists():
        parser.error('Choose a new output directory to preserve existing checkpoints')
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    torch.manual_seed(42)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision('high')
    cv2.setNumThreads(1)
    model = YoloFragments(args.backbone, surface_aux=args.surface_aux).to(device)
    # Ultralytics export checkpoints can load with every parameter frozen.
    model.layers.requires_grad_(not args.freeze_features)
    train = FragmentDataset(args.label_cache / 'train.jsonl', augment=True, limit=args.limit_train, surface_aux=args.surface_aux)
    val = FragmentDataset(args.label_cache / 'val.jsonl', limit=args.limit_val, surface_aux=args.surface_aux)
    loaders = [DataLoader(split, batch_size=args.batch, shuffle=shuffle, pin_memory=device.type == 'cuda')
               for split, shuffle in ((train, True), (val, False))]
    heads = list(model.head.parameters())
    if model.surface_head is not None:
        heads.extend(model.surface_head.parameters())
    optimizer = torch.optim.AdamW([{'params': model.layers.parameters(), 'lr': 2e-4},
                                   {'params': heads, 'lr': 1e-3}], weight_decay=1e-3)
    surface_weights = torch.tensor([.4, 1., 1.], device=device)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=2e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    print(json.dumps({'train': len(train), 'val': len(val), 'grid': GRID, 'epochs': args.epochs,
                      'device': str(device), 'batch': args.batch, 'surface_aux': args.surface_aux,
                      'trainable_feature_parameters': sum(p.numel() for p in model.layers.parameters() if p.requires_grad)}), flush=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        tick = time.monotonic()
        model.train()
        running = torch.zeros((), device=device)
        for image, labels, *surface in loaders[0]:
            image = image.to(device, non_blocking=True).float() / 255
            labels = labels.to(device, non_blocking=True).float()
            gain = torch.rand((image.shape[0], 1, 1, 1), device=device) * .4 + .8
            image = (image * gain).clamp_(0, 1)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                output = model(image, with_surfaces=args.surface_aux)
            if args.surface_aux:
                output, surface_logits = output
            loss = fragment_loss(output.float(), labels)
            if args.surface_aux:
                loss = loss + .30 * F.cross_entropy(surface_logits.float(), surface[0].to(device, non_blocking=True).long(),
                                                    weight=surface_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach()
        scheduler.step()
        row = {'epoch': epoch, 'train_loss': running.item() / len(loaders[0]), **validate(model, loaders[1], device),
               'elapsed_s': round(time.monotonic() - tick, 1)}
        if not np.isfinite(row['train_loss']) or not np.isfinite(row['validation_loss']):
            raise RuntimeError(f'Non-finite loss at epoch {epoch}')
        weights = {'model': model.state_dict(), 'epoch': epoch, 'grid': GRID, 'metrics': row,
                   'train_images': len(train), 'val_images': len(val), 'backbone': args.backbone.name,
                   'surface_aux': args.surface_aux, 'freeze_features': args.freeze_features}
        save_checkpoint(args.output / 'last.pt', weights)
        if row['cell_f1'] > best:
            best = row['cell_f1']
            save_checkpoint(args.output / 'best.pt', weights)
        with (args.output / 'metrics.jsonl').open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
