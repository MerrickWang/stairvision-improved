#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class StairSegmentation(Dataset):
    def __init__(self, root, split, size):
        base = Path(root) / split
        self.images = sorted((base / "images").glob("*.jpg"))
        self.masks = base / "segmentations"
        self.size = size
        if not self.images:
            raise RuntimeError(f"No images found in {base / 'images'}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.masks / f"{image_path.stem}.png"
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise RuntimeError(f"Missing pair for {image_path.name}")
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        image = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255
        labels = np.zeros_like(mask, dtype=np.int64)
        labels[mask == 128] = 1
        labels[mask == 255] = 2
        return image, torch.from_numpy(labels)


class SmallSegNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/RGB-D stair dataset/RGB-D stair dataset")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--output", default="runs/segmentation/baseline")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train = StairSegmentation(args.root, "train", args.size)
    val = StairSegmentation(args.root, "val", args.size)
    train_loader = DataLoader(train, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=args.batch, shuffle=False, num_workers=4, pin_memory=True)
    model = SmallSegNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * images.size(0)
        model.eval()
        intersection = torch.zeros(3, device=device)
        union = torch.zeros(3, device=device)
        with torch.inference_mode():
            for images, labels in val_loader:
                pred = model(images.to(device)).argmax(1)
                labels = labels.to(device)
                for cls in range(3):
                    intersection[cls] += ((pred == cls) & (labels == cls)).sum()
                    union[cls] += ((pred == cls) | (labels == cls)).sum()
        iou = (intersection / union.clamp_min(1)).detach().cpu().tolist()
        row = {"epoch": epoch, "train_loss": total / len(train), "mIoU": sum(iou) / 3, "class_iou": iou}
        history.append(row)
        print(json.dumps(row), flush=True)
        torch.save({"model": model.state_dict(), "size": args.size, "classes": ["background", "tread", "riser"]}, output / "last.pt")
        if row["mIoU"] >= max(x["mIoU"] for x in history):
            torch.save({"model": model.state_dict(), "size": args.size, "classes": ["background", "tread", "riser"]}, output / "best.pt")
    (output / "metrics.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
