"""Run StairLab 1.0, 3.0 and 3.9 on a common split and score available line labels."""

from stairlab.recognition.geometry import square


import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from ultralytics import YOLO

from stairlab.recognition import depthout_stitch
from stairlab.recognition.depthout_refine import DepthOutFragments
from stairlab.recognition.pipeline import run_case
from stairlab.experiments.run_zoom_audit import sha
from stairlab.recognition.structure_postprocess import clean
from stairlab.recognition.dense_head import DenseStairHead
from stairlab.recognition.yolo_fragments import SIZE, YoloFragments


def load_v1(path):
    spec = importlib.util.spec_from_file_location('archived_stairlab_v1', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.resolve().parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def read_labels(path, offset, scale):
    if not path or not path.is_file():
        return None
    data = [list(map(float, line.split())) for line in path.read_text().splitlines() if line.strip()]
    return (np.array([row[1:5] for row in data if int(row[0]) == 0], dtype=float).reshape(-1, 2, 2)+offset)*scale


def match(groups, truth):
    if not groups or not len(truth):
        return 0
    errors = []
    for group in groups:
        by_member = []
        for segment in group:
            a = np.asarray(segment)
            by_member.append(np.minimum(np.linalg.norm(a-truth, axis=2).mean(axis=1),
                                        np.linalg.norm(a[::-1]-truth, axis=2).mean(axis=1)))
        errors.append(np.min(by_member, axis=0))
    errors = np.array(errors)
    # Original StairLab endpoint rule, in the common square coordinate system.
    threshold = .03 * np.hypot(SIZE, SIZE)
    ii, jj = linear_sum_assignment(np.where(errors <= threshold, errors, 1e6))
    return int(np.count_nonzero(errors[ii, jj] <= threshold))


def preview(image, counts, edges, points, name):
    panels = []
    for version in ('1.0', '3.0', '3.9'):
        out = image.copy()
        for group in edges[version]:
            for line in group:
                a, b = np.round(line).astype(int)
                cv2.line(out, tuple(a), tuple(b), (50, 220, 250), 2, cv2.LINE_AA)
        if version == '1.0':
            for point in points:
                cv2.circle(out, tuple(np.rint(point).astype(int)), 4, (70, 90, 255), -1)
        canvas = cv2.copyMakeBorder(out, 34, 0, 0, 0, cv2.BORDER_CONSTANT, value=(24, 24, 24))
        cv2.putText(canvas, f'{name} | V{version} | count {counts[version]}', (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .51, (245, 245, 245), 1, cv2.LINE_AA)
        panels.append(canvas)
    return np.hstack(panels)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'output', 'legacy-code', 'pose', 'seg', 'yolo-weights', 'depthout-checkout',
                 'depthout-weights', 'dense-weights', 'protocol'):
        p.add_argument(f'--{name}', type=Path, required=True)
    p.add_argument('--labels', type=Path)
    p.add_argument('--locked-rois', type=Path)
    p.add_argument('--save-overlays', action='store_true')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batch', type=int, default=8)
    a = p.parse_args()
    paths = sorted(x for x in a.images.iterdir() if x.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not paths or a.batch < 1:
        p.error('Need nonempty images and a positive batch size')
    protocol = json.loads(a.protocol.read_text(encoding='utf-8'))
    hashes = {'yolo_surface_aux': sha(a.yolo_weights), 'depthout': sha(a.depthout_weights)}
    if hashes != protocol['checkpoints_sha256'] or sha(Path(depthout_stitch.__file__)) != protocol['counting_source_sha256']:
        p.error('Frozen model or inference source mismatch')
    legacy = load_v1(a.legacy_code)
    pose, seg = YOLO(str(a.pose)), YOLO(str(a.seg))
    ckpt = torch.load(a.yolo_weights, map_location='cpu', weights_only=True)
    yolo = YoloFragments('yolo11n-pose.yaml', surface_aux=ckpt.get('surface_aux', False)).to(a.device).eval()
    yolo.load_state_dict(ckpt['model'])
    depth = DepthOutFragments(a.depthout_checkout, a.depthout_weights, a.device)
    dense = DenseStairHead().to(a.device).eval()
    dense.load_state_dict(torch.load(a.dense_weights, map_location='cpu', weights_only=True))
    locked = json.loads(a.locked_rois.read_text(encoding='utf-8')) if a.locked_rois else {}
    confidence = protocol['configuration']['fragment_confidence']
    a.output.mkdir(parents=True, exist_ok=True)
    if a.save_overlays:
        (a.output/'overlays').mkdir(exist_ok=True)
    rows = []
    opts = dict(conf=.25, imgsz=640, device=a.device, verbose=False)
    with torch.inference_mode():
        for start in range(0, len(paths), a.batch):
            batch = paths[start:start+a.batch]
            raw = [cv2.imread(str(path)) for path in batch]
            if any(x is None for x in raw):
                p.error('An input image failed to decode')
            pose_out = pose.predict([str(path) for path in batch], **opts)
            seg_out = seg.predict([str(path) for path in batch], retina_masks=True, **opts)
            for path, image, pr, sr in zip(batch, raw, pose_out, seg_out, strict=True):
                lines = legacy.extract_lines(pr)
                surfaces = legacy.extract_surfaces(sr, pr.orig_shape)
                one = legacy.count_visible_steps(surfaces, lines, pr.orig_shape)
                prepared, offset, scale = square(image)
                latest = run_case(prepared, locked.get(path.name, {}).get('roi_xyxy'), yolo, depth, dense,
                                  a.device, confidence, include_geometry=True)
                # 1.0's fused steps contain points, not line endpoints. Its
                # independent detector metric uses its structured convex lines.
                one_lines = clean({'lines': lines})['lines_structured']
                g1 = [[((np.asarray(row['endpoints'])[:, :2]+offset)*scale).tolist()]
                      for row in one_lines if row['class'] == 0]
                edges = {'1.0': g1, '3.0': latest['edges_3_0'], '3.9': latest['edges_3_9']}
                counts = {'1.0': one['visible_step_count'], '3.0': len(edges['3.0']), '3.9': latest['count_3_9_refine']}
                points = [[(step['x']+offset[0])*scale, (step['y']+offset[1])*scale] for step in one['steps']]
                labels = read_labels(a.labels/f'{path.stem}.txt' if a.labels else None, offset, scale)
                if a.labels and labels is None:
                    p.error(f'Missing label for {path.name}')
                row = {'image': path.name, 'counts': counts, 'roi': latest['roi'], 'base_3_9': latest['base'],
                       'refinements_3_9': latest['refinements'], 'line_predictions': {k: len(v) for k,v in edges.items()}}
                if labels is not None:
                    row.update(reference_convex_lines=len(labels), matches={k: match(v, labels) for k,v in edges.items()})
                if a.save_overlays:
                    cv2.imwrite(str(a.output/'overlays'/f'{path.stem}.jpg'), preview(prepared, counts, edges, points, path.name))
                rows.append(row)
            if len(rows) % 80 == 0 or len(rows) == len(paths):
                print(f'Processed {len(rows)}/{len(paths)}', flush=True)
    report = {'images': len(rows), 'checkpoints': {**hashes, 'pose_1_0': sha(a.pose), 'segment_1_0': sha(a.seg),
                                                 'dense': sha(a.dense_weights)},
              'legacy_code_sha256': sha(a.legacy_code), 'per_image': rows}
    (a.output/'results.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
