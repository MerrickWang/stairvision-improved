"""DA-V2 decoder fine-tuning on paired RGB / encoded uint8 depth maps."""
import argparse
import json
import random
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor
from depth_model import EncodedDepthModel, masked_loss


class Pairs(Dataset):
    def __init__(self, manifest, processor, size, limit=0, augment=False):
        self.rows = [json.loads(x) for x in Path(manifest).read_text().splitlines() if x.strip()]
        if limit:
            self.rows = self.rows[:limit]
        if not self.rows:
            raise ValueError('Empty dataset')
        self.processor, self.size, self.augment = processor, size, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        if row['depth_kind'] != 'encoded_uint8':
            raise ValueError('This trainer expects encoded_uint8, not metric or colored depth')
        rgb = Image.open(row['image']).convert('RGB')
        raw = np.array(Image.open(row['depth']))
        if raw.ndim != 2 or raw.dtype != np.uint8:
            raise ValueError(f'Expected single-channel uint8 depth, got {raw.shape} {raw.dtype}')
        if raw.shape != (rgb.height, rgb.width):
            raise ValueError('RGB/depth spatial alignment mismatch')
        # Ignore invalid zero depths and zero-padded letterbox borders.
        mask = (raw > 0) & np.any(np.array(rgb) > 0, axis=-1)
        depth = Image.fromarray(raw)
        mask = Image.fromarray(mask)
        if self.augment and random.random() < 0.5:
            rgb, depth, mask = [x.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for x in (rgb, depth, mask)]
        pixels = self.processor(images=rgb, return_tensors='pt',
            size={'height': self.size, 'width': self.size}, keep_aspect_ratio=False)['pixel_values'][0]
        target = torch.from_numpy(np.array(depth.resize((self.size, self.size), Image.Resampling.NEAREST), copy=True)).float() / 255
        mask = torch.from_numpy(np.array(mask.resize((self.size, self.size), Image.Resampling.NEAREST), copy=True)).bool()
        return pixels, target, mask


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='models/depth-anything-v2-small')
    p.add_argument('--train', default='data/prepared/depth_train.jsonl')
    p.add_argument('--val', default='data/prepared/depth_val.jsonl')
    p.add_argument('--output', default='runs/depth/baseline')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--size', type=int, default=294)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--limit-train', type=int, default=0)
    p.add_argument('--limit-val', type=int, default=0)
    p.add_argument('--unfreeze-backbone', action='store_true')
    p.add_argument('--resume')
    p.add_argument('--init')
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--min-delta', type=float, default=0.0001)
    a = p.parse_args()
    if a.size % 14:
        raise ValueError('--size must be divisible by the ViT patch size 14')
    out = Path(a.output)
    if (out / 'last.pt').exists() and not a.resume:
        raise FileExistsError('Output exists: use a new output directory or --resume')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoImageProcessor.from_pretrained(a.model, local_files_only=True)
    train = DataLoader(Pairs(a.train, processor, a.size, a.limit_train, True), batch_size=a.batch, shuffle=True, num_workers=a.workers)
    val = DataLoader(Pairs(a.val, processor, a.size, a.limit_val), batch_size=a.batch, num_workers=a.workers)
    model = EncodedDepthModel(a.model, not a.unfreeze_backbone).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    start, best = 0, float('inf')
    bad_epochs, stop_best = 0, float('inf')
    if a.patience < 1 or a.min_delta < 0 or (a.init and a.resume):
        raise ValueError('Invalid early stopping or initialization arguments')
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location='cpu', weights_only=False)['model'])
    if a.resume:
        ckpt = torch.load(a.resume, map_location='cpu', weights_only=False)
        for key in ('size', 'unfreeze_backbone', 'model'):
            if ckpt['args'][key] != vars(a)[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(ckpt['model']); optimizer.load_state_dict(ckpt['optimizer'])
        start, best = ckpt['epoch'] + 1, ckpt['best']
        bad_epochs = ckpt.get('bad_epochs', 0)
        stop_best = ckpt.get('stop_best', best)
    (out / 'config.json').write_text(json.dumps(vars(a), indent=2))
    print(json.dumps({'device': str(device), 'train_samples': len(train.dataset), 'val_samples': len(val.dataset), 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad), 'output_kind': model.output_kind}), flush=True)
    for epoch in range(start, a.epochs):
        model.train(); total, samples = 0., 0
        for step, (pixels, target, mask) in enumerate(train, 1):
            pixels, target, mask = [x.to(device) for x in (pixels, target, mask)]
            optimizer.zero_grad(set_to_none=True)
            pred = model(pixels)
            pred = torch.nn.functional.interpolate(pred[:, None], target.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
            loss, _ = masked_loss(pred, target, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            total += loss.item() * len(pixels); samples += len(pixels)
            if step % 100 == 0:
                print(json.dumps({'epoch': epoch + 1, 'step': step, 'steps': len(train), 'running_train_loss': total / samples}), flush=True)
        model.eval(); abs_sum, sq_sum, count = 0., 0., 0
        with torch.inference_mode():
            for pixels, target, mask in val:
                pixels, target, mask = [x.to(device) for x in (pixels, target, mask)]
                pred = model(pixels)
                pred = torch.nn.functional.interpolate(pred[:, None], target.shape[-2:], mode='bilinear', align_corners=False)[:, 0]
                error = (pred - target)[mask]
                abs_sum += error.abs().sum().item(); sq_sum += error.square().sum().item(); count += error.numel()
        if count == 0:
            raise ValueError('No valid validation pixels')
        mae = abs_sum / count
        if not np.isfinite(mae):
            raise FloatingPointError('Non-finite validation loss')
        improved = mae < best; best = min(best, mae)
        if mae < stop_best - a.min_delta:
            stop_best, bad_epochs = mae, 0
        else:
            bad_epochs += 1
        row = {'epoch': epoch + 1, 'train_loss': total / samples, 'val_encoded_mae': mae, 'val_encoded_rmse': (sq_sum / count) ** 0.5, 'valid_pixels': count}
        print(json.dumps(row), flush=True)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'best': best, 'args': vars(a), 'output_kind': model.output_kind, 'bad_epochs': bad_epochs, 'stop_best': stop_best}
        torch.save(ckpt, out / 'last.pt')
        if improved:
            torch.save(ckpt, out / 'best.pt')
        if bad_epochs >= a.patience:
            print(json.dumps({'early_stopped': True, 'epoch': epoch + 1, 'best_mae': best, 'patience': a.patience}), flush=True)
            break


if __name__ == '__main__':
    main()
