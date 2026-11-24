"""Trace reference convex edges through the StairLab short-fragment pipeline."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from stairlab.recognition.depthout_refine import DepthOutFragments, coverage
from stairlab.recognition.depthout_stitch import components, predict_steps
from stairlab.experiments.predict_depthout_only import matched_edges
from stairlab.recognition.yolo_fragments import SIZE, YoloFragments, decode_batch


def edge_support(fragments, scores, edge, diagonal):
    delta = edge[1] - edge[0]
    length = np.linalg.norm(delta)
    axis = delta / length
    normal = np.array([-axis[1], axis[0]])
    projected = (fragments - edge[0]) @ axis
    offsets = (fragments - edge[0]) @ normal
    intervals = np.clip(np.sort(projected, axis=1), 0, length)
    overlap = np.ptp(intervals, axis=1) >= 2
    tolerance = max(3, .006 * diagonal)
    aligned = overlap & (np.abs(offsets).max(axis=1) <= tolerance)
    nearby = overlap & (np.abs(offsets).max(axis=1) <= 3 * tolerance)
    chains = []
    for ids in components(np.flatnonzero(aligned), projected, .045 * diagonal):
        chain = projected[ids]
        span = float(np.ptp(chain))
        weights = np.minimum(np.linalg.norm(fragments[ids, 1] - fragments[ids, 0], axis=1), .035 * diagonal)
        chains.append({'n': len(ids), 'span_px': span,
                       'coverage': coverage(np.sort(chain, axis=1) - chain.min(), span),
                       'weight_px': float(weights @ scores[ids])})
    return {'nearby': int(nearby.sum()), 'aligned': int(aligned.sum()),
            'reference_coverage': coverage(intervals[aligned], length), 'chains': chains}


def trace(decoded, mask, reference, confidence):
    fragments, scores = decoded
    selected = scores >= confidence
    result = predict_steps((fragments[selected], scores[selected]), mask)
    stages = {'stitched': result['stitched_edges'], **result['edge_stages']}
    diagonal = np.hypot(*mask.shape)
    matches = {stage: [j + 1 for j in matched_edges(lines, reference, diagonal)]
               for stage, lines in stages.items()}
    return {'fragments': fragments[selected].tolist(), 'scores': scores[selected].tolist(),
            'stages': stages, 'matched_reference_ids': matches, 'visible_count': result['visible_step_count'],
            'steps': result['steps'], 'merge_events': result['merge_events'],
            'blocked_merges': result['blocked_merges'],
            'reference_support': [edge_support(fragments[selected], scores[selected], edge, diagonal)
                                  for edge in reference]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'backbone', 'weights', 'depthout-checkout', 'depthout-weights', 'output'):
        parser.add_argument(f'--{field}', type=Path, required=True)
    parser.add_argument('--image', action='append', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sources = [Path(json.loads(line)['image']) for line in args.manifest.read_text().splitlines() if line.strip()]
    selected = [path for path in sources if path.name in args.image]
    if len(selected) != len(set(args.image)) or {path.name for path in selected} != set(args.image):
        parser.error('Each requested image must appear exactly once in the manifest')
    images = [cv2.imread(str(path)) for path in selected]
    if any(image is None or image.shape != (SIZE, SIZE, 3) for image in images):
        parser.error(f'Expected readable {SIZE}x{SIZE} validation images')
    checkpoint = torch.load(args.weights, weights_only=True, map_location='cpu')
    model = YoloFragments(args.backbone, surface_aux=checkpoint.get('surface_aux', False)).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    semantic = DepthOutFragments(args.depthout_checkout, args.depthout_weights, args.device)
    tensor = torch.from_numpy(np.stack([image[..., ::-1] for image in images]))
    tensor = tensor.to(args.device).permute(0, 3, 1, 2).float() / 255
    with torch.inference_mode():
        yolo = decode_batch(model(tensor), .1)
    depthout = semantic.predict(images, .1, with_masks=True)
    cases = {}
    for path, decoded_yolo, (decoded_depthout, mask) in zip(selected, yolo, depthout, strict=True):
        label = path.parent.parent / 'labels' / (path.stem + '.txt')
        rows = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
        reference = np.array([row[1:5] for row in rows if int(row[0]) == 0]).reshape(-1, 2, 2)
        cases[path.name] = {'reference_edges': reference.tolist(), 'models': {}}
        for name, decoded in (('yolo_surface_aux', decoded_yolo), ('depthout', decoded_depthout)):
            cases[path.name]['models'][name] = {f'{confidence:.2f}': trace(decoded, mask, reference, confidence)
                                                for confidence in (.30, .10)}
    report = {'grouping_version': '2.4', 'baseline_confidence': .30, 'diagnostic_confidence': .10,
              'yolo_epoch': checkpoint['epoch'],
              'weight_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for name, path in (('yolo_surface_aux', args.weights), ('depthout', args.depthout_weights))},
              'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
