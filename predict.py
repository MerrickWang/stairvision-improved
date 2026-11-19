"""Batch CLI: stair-line keypoints plus original or fine-tuned relative depth."""
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from ultralytics import YOLO
from depth_model import EncodedDepthModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--yolo', required=True)
    p.add_argument('--depth-model', default='models/depth-anything-v2-small')
    p.add_argument('--depth-checkpoint')
    p.add_argument('--output', default='runs/predict')
    p.add_argument('--conf', type=float, default=0.25)
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    yolo = YOLO(a.yolo)
    processor = AutoImageProcessor.from_pretrained(a.depth_model, local_files_only=True)
    if a.depth_checkpoint:
        ckpt = torch.load(a.depth_checkpoint, map_location='cpu', weights_only=False)
        depth = EncodedDepthModel(a.depth_model)
        depth.load_state_dict(ckpt['model']); kind = ckpt['output_kind']
    else:
        depth = AutoModelForDepthEstimation.from_pretrained(a.depth_model, local_files_only=True)
        kind = 'relative_inverse_not_meters'
    depth.to(device).eval()
    source = Path(a.source)
    files = [source] if source.is_file() else sorted(p for p in source.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not files:
        raise ValueError('No input images')
    for path in files:
        rgb = Image.open(path).convert('RGB')
        result = yolo.predict(source=str(path), conf=a.conf, verbose=False, device=device)[0]
        pixels = processor(images=rgb, return_tensors='pt')['pixel_values'].to(device)
        with torch.inference_mode():
            pred = depth(pixels) if a.depth_checkpoint else depth(pixel_values=pixels).predicted_depth
            arr = torch.nn.functional.interpolate(pred[:, None], (rgb.height, rgb.width), mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        np.save(out / f'{path.stem}_depth.npy', arr)
        low, high = np.percentile(arr, [2, 98])
        heat = np.uint8(np.clip((arr-low) / max(high-low, 1e-6), 0, 1) * 255)
        cv2.imwrite(str(out / f'{path.stem}_depth.png'), cv2.applyColorMap(heat, cv2.COLORMAP_TURBO))
        cv2.imwrite(str(out / f'{path.stem}_lines.jpg'), result.plot(kpt_line=False))
        rows = []
        if result.keypoints is not None:
            for cls, conf, points in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), result.keypoints.data.tolist()):
                rows.append({'class': int(cls), 'confidence': conf, 'endpoints': points})
        record = {'image': str(path), 'lines': rows, 'visible_convex_candidates': sum(x['class'] == 0 for x in rows), 'depth_kind': kind, 'height_m': None, 'note': 'Convex candidates are not a validated stair count. Relative/encoded depth cannot establish metric height.'}
        (out / f'{path.stem}.json').write_text(json.dumps(record, indent=2))
        print(path.name, len(rows), 'lines', kind)


if __name__ == '__main__':
    main()
