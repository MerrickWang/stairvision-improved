"""Train a full-resolution stair-edge response head from existing line annotations."""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DenseStairHead(nn.Sequential):
    def __init__(self):
        layers = []
        for inch, dilation in ((3, 1), (12, 2), (12, 4), (12, 8), (12, 1)):
            layers.extend((nn.Conv2d(inch, 12, 3, padding=dilation, dilation=dilation), nn.ReLU()))
        super().__init__(*layers, nn.Conv2d(12, 1, 1))


class StairPatches(Dataset):
    def __init__(self, root, *, training):
        self.root, self.training = root, training
        self.paths = sorted(p for p in (root / 'images').iterdir() if p.suffix.lower() in {'.jpg', '.png'})

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        rng = random if self.training else random.Random(index)
        path = self.paths[index]
        image = cv2.imread(str(path))[:, :, ::-1]
        side = rng.randint(128, 256) if self.training else 192
        convex = np.zeros(image.shape[:2], dtype=np.uint8)
        concave = convex.copy()
        centers = []
        for line in (self.root / 'labels' / f'{path.stem}.txt').read_text().splitlines():
            if not line.strip():
                continue
            kind, x0, y0, x1, y1 = map(float, line.split())
            cv2.line(convex if kind == 0 else concave, (round(x0), round(y0)), (round(x1), round(y1)), 1, 2)
            if kind == 0:
                centers.append(((x0 + x1) / 2, (y0 + y1) / 2))
        x, y = rng.choice(centers) if centers else (image.shape[1] / 2, image.shape[0] / 2)
        left = int(np.clip(x - side / 2 + rng.uniform(-side/4, side/4), 0, image.shape[1] - side))
        top = int(np.clip(y - side / 2 + rng.uniform(-side/4, side/4), 0, image.shape[0] - side))
        crop = np.s_[top:top+side, left:left+side]
        rgb = cv2.resize(image[crop], (192, 192)).astype('float32') / 255
        target = cv2.resize(convex[crop], (192, 192), interpolation=cv2.INTER_NEAREST).astype('float32')
        hard = cv2.resize(concave[crop], (192, 192), interpolation=cv2.INTER_NEAREST) * (1 - target)
        if self.training and rng.random() < .5:
            rgb, target, hard = rgb[:, ::-1].copy(), target[:, ::-1].copy(), hard[:, ::-1].copy()
        return rgb.transpose(2, 0, 1), target[None], hard[None]


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    values = []
    with torch.set_grad_enabled(optimizer is not None):
        for rgb, target, hard in loader:
            rgb, target, hard = (v.to(device) for v in (rgb, target, hard))
            logits = model(rgb)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, target, weight=1 + 10 * target + 3 * hard)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            values.append(float(loss.detach()))
    return float(np.mean(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.manual_seed(38)
    random.seed(38)
    cv2.setNumThreads(0)
    train = StairPatches(args.dataset / 'train', training=True)
    val = StairPatches(args.dataset / 'val', training=False)
    if not len(train) or not len(val):
        parser.error('Expected nonempty train and val images with matching labels')
    training = DataLoader(train, batch_size=24, shuffle=True, num_workers=4)
    validation = DataLoader(val, batch_size=24, num_workers=4)
    model = DenseStairHead().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002)
    best = float('inf')
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        row = {'epoch': epoch, 'train_loss': run_epoch(model, training, args.device, optimizer),
               'val_loss': run_epoch(model, validation, args.device)}
        history.append(row)
        (args.output / 'training.json').write_text(json.dumps({'train': len(train), 'val': len(val),
                                                               'history': history}, indent=2) + '\n')
        print(row, flush=True)
        if row['val_loss'] < best:
            best = row['val_loss']
            torch.save(model.state_dict(), args.output / 'best.pt')


if __name__ == '__main__':
    main()
