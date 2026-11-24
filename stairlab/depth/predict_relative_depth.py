"""Estimate relative inverse depth from RGB with official Depth Anything V2 weights."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from torch.nn.functional import interpolate
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}


def input_images(source):
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    files = sorted(path for path in source.iterdir()
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f'No input images in {source}')
    if len({path.stem.casefold() for path in files}) != len(files):
        raise ValueError('Input images must have unique stems to avoid overwriting outputs')
    return files


def colorize(depth):
    low, high = np.percentile(depth, [2, 98])
    scaled = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
    colors = cv2.applyColorMap(np.uint8(scaled * 255), cv2.COLORMAP_TURBO)
    return Image.fromarray(cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)), [float(low), float(high)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path, help='RGB image or image directory')
    parser.add_argument('--model', default='models/depth-anything-v2-small', help='Local pretrained model directory')
    parser.add_argument('--checkpoint', type=Path, help='Optional StairLab 4.0 decoder checkpoint')
    parser.add_argument('--output', type=Path, default=Path('runs/relative_depth'))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    files = input_images(args.source)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForDepthEstimation.from_pretrained(args.model, local_files_only=True)
    checkpoint = None
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        expected = {name for name in model.state_dict() if not name.startswith('backbone.')}
        if checkpoint.get('version') != '4.0' or set(checkpoint.get('decoder', {})) != expected:
            raise ValueError('Checkpoint does not match the StairLab 4.0 decoder format')
        model.load_state_dict(checkpoint['decoder'], strict=False)
    model.to(args.device).eval()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []

    with torch.inference_mode():
        for path in files:
            with Image.open(path) as image:
                rgb = image.convert('RGB')
            if checkpoint:
                size = checkpoint['size']
                side = max(rgb.size)
                left, top = (side - rgb.width) // 2, (side - rgb.height) // 2
                padded = Image.new('RGB', (side, side))
                padded.paste(rgb, (left, top))
                pixels = processor(images=padded, return_tensors='pt', keep_aspect_ratio=False,
                                   size={'height': size, 'width': size})['pixel_values'].to(args.device)
                output_shape = (side, side)
            else:
                pixels = processor(images=rgb, return_tensors='pt')['pixel_values'].to(args.device)
                output_shape = (rgb.height, rgb.width)
            prediction = model(pixel_values=pixels).predicted_depth
            depth = interpolate(prediction[:, None], output_shape,
                                mode='bilinear', align_corners=False)[0, 0].float().cpu().numpy()
            if checkpoint:
                depth = depth[top:top + rgb.height, left:left + rgb.width]
            if not np.isfinite(depth).all():
                raise ValueError(f'Non-finite depth values for {path}')

            values_path = args.output / f'{path.stem}_depth.npy'
            preview_path = args.output / f'{path.stem}_depth.png'
            preview, display_range = colorize(depth)
            np.save(values_path, depth)
            preview.save(preview_path)
            records.append({'image': str(path), 'shape_hw': list(depth.shape),
                            'values': values_path.name, 'preview': preview_path.name,
                            'preview_p02_p98': display_range})
            print(f'{len(records)}/{len(files)} {path.name}', flush=True)

    model_source = Path(args.model) / 'source.json'
    report = {'model': args.model, 'device': args.device,
              'depth_kind': 'relative_inverse_not_meters', 'larger_values': 'nearer',
              'dtype': 'float32', 'images': records}
    if model_source.is_file():
        report['model_source'] = json.loads(model_source.read_text(encoding='utf-8'))
    if checkpoint:
        report['checkpoint'] = {'path': str(args.checkpoint), 'variant': checkpoint['variant'],
                                'input_size': checkpoint['size']}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
