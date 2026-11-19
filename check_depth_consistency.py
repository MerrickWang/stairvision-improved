#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

from depth_model import EncodedDepthModel
from train_segmentation import SmallSegNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--seg-model", required=True)
    parser.add_argument("--depth-model", required=True)
    parser.add_argument("--model-dir", default="models/depth-anything-v2-small")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seg_state = torch.load(args.seg_model, map_location="cpu", weights_only=False)
    seg = SmallSegNet().to(device)
    seg.load_state_dict(seg_state["model"])
    seg.eval()
    depth_state = torch.load(args.depth_model, map_location="cpu", weights_only=False)
    depth = EncodedDepthModel(args.model_dir).to(device)
    depth.load_state_dict(depth_state["model"])
    depth.eval()
    processor = AutoImageProcessor.from_pretrained(args.model_dir, local_files_only=True)

    rows = []
    files = sorted(Path(args.images).glob("*.jpg"))[: args.limit]
    for path in files:
        rgb = Image.open(path).convert("RGB")
        width, height = rgb.size
        pixels = processor(images=rgb, return_tensors="pt")["pixel_values"].to(device)
        with torch.inference_mode():
            predicted = depth(pixels)
            depth_map = torch.nn.functional.interpolate(predicted[:, None], (height, width), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        small = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(small[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255).unsqueeze(0).to(device)
        with torch.inference_mode():
            labels = seg(tensor).argmax(1)[0].cpu().numpy()
        labels = cv2.resize(labels.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        means = []
        for cls in (1, 2):
            values = depth_map[labels == cls]
            means.append(float(np.median(values)) if values.size else None)
        valid = [value for value in means if value is not None]
        rows.append({"image": str(path), "tread_depth": means[0], "riser_depth": means[1], "depth_order_consistent": len(valid) == 2 and means[0] != means[1]})
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    comparable = [row for row in rows if row["depth_order_consistent"]]
    print(json.dumps({"images": len(rows), "images_with_two_regions": len(comparable), "output": str(output)}))


if __name__ == "__main__":
    main()
