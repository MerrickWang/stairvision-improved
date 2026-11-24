"""Evaluate relative depth on the official RGB-D test frames with fixed preprocessing."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from transformers import AutoImageProcessor

from predict_relative_depth import colorize
from train_relative_depth_v40 import make_model, standardize


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'results.json').read_text())
    config = report['args']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(config['model'], device).eval()
    processor = AutoImageProcessor.from_pretrained(config['model'], local_files_only=True)
    files = sorted((args.test / 'images').glob('color_*.png'))
    if not files:
        raise ValueError('No official test RGB frames found')
    size = config['size']
    examples = {'color_0', 'color_74', 'color_110'}
    example_columns = {name: [] for name in examples}
    results = {}
    for variant in ('original', 'decoder_only', 'tread_rank'):
        if variant != 'original':
            checkpoint = torch.load(args.run / f'{variant}.pt', map_location='cpu', weights_only=True)
            model.load_state_dict(checkpoint['decoder'], strict=False)
        total_error, count = 0.0, 0
        rows = []
        for file in files:
            rgb = Image.open(file).convert('RGB')
            depth_file = args.test / 'depthes' / file.name.replace('color_', 'Depth_', 1)
            with Image.open(depth_file) as depth:
                raw = np.array(depth)
            if raw.dtype != np.uint8 or raw.shape != (rgb.height, rgb.width):
                raise ValueError(f'Invalid RGB/depth alignment for {file}')
            reference = -torch.from_numpy(raw).float().to(device)[None] / 255
            valid = reference < 0
            side = max(rgb.size)
            left, top = (side - rgb.width) // 2, (side - rgb.height) // 2
            padded = Image.new('RGB', (side, side))
            padded.paste(rgb, (left, top))
            pixels = processor(images=padded, return_tensors='pt', keep_aspect_ratio=False,
                               size={'height': size, 'width': size})['pixel_values'].to(device)
            prediction = model(pixel_values=pixels).predicted_depth.float()
            pred = F.interpolate(prediction[:, None], (side, side), mode='bilinear', align_corners=False)
            pred = pred[:, 0, top:top + rgb.height, left:left + rgb.width]
            error = (standardize(pred, valid) - standardize(reference, valid)).abs()[valid]
            if not error.numel() or not torch.isfinite(error).all():
                raise ValueError(f'Invalid depth evaluation pixels for {file}')
            total_error += error.sum().item()
            count += error.numel()
            rows.append({'image': file.name, 'normalized_mae': error.mean().item(), 'pixels': error.numel()})
            if file.stem in examples:
                panels = example_columns[file.stem]
                if not panels:
                    panels.append((rgb, 'RGB'))
                panels.append((colorize(pred[0].cpu().numpy())[0], variant))
        results[variant] = {'images': len(rows), 'valid_pixels': count,
                            'normalized_mae': total_error / count, 'per_image': rows}
        print(variant, len(rows), total_error / count, flush=True)

    (args.run / 'test_results.json').write_text(json.dumps(results, indent=2) + '\n')
    width, height = 212, 159
    grid = Image.new('RGB', (width * 4, (height + 20) * len(examples)), (28, 28, 28))
    draw = ImageDraw.Draw(grid)
    for row, name in enumerate(sorted(examples)):
        for col, (panel, title) in enumerate(example_columns[name]):
            x, y = col * width, row * (height + 20)
            grid.paste(panel.resize((width, height), Image.Resampling.LANCZOS), (x, y + 20))
            draw.text((x + 5, y + 4), f'{name} | {title}' if col == 0 else title, fill='white')
    grid.save(args.run / 'examples.jpg', quality=92)


if __name__ == '__main__':
    main()
