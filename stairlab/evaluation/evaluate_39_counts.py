"""Paired physical-count audit for 3.8 and the two 3.9 modes."""

from stairlab.recognition.pipeline import propose_roi, groups, prep, run_case


import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from stairlab.recognition import depthout_stitch
from stairlab.recognition.depthout_refine import DepthOutFragments
from stairlab.recognition.depthout_stitch import predict_steps
from stairlab.recognition.full_flight_tracks import refine_flight
from stairlab.recognition.perspective_step_tracks import recover_perspective_tracks
from stairlab.recognition.geometry import rgb_tensor
from stairlab.experiments.run_zoom_audit import sha
from stairlab.recognition.geometry import zoom
from stairlab.recognition.dense_head import DenseStairHead
from stairlab.recognition.yolo_fragments import SIZE, YoloFragments, decode_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('cases', 'protocol', 'yolo-weights', 'depthout-checkout', 'depthout-weights', 'dense-weights', 'output'):
        p.add_argument(f'--{name}', required=True, type=Path)
    p.add_argument('--device', default='cuda:0')
    a = p.parse_args()
    protocol = json.loads(a.protocol.read_text(encoding='utf-8'))
    hashes = {'yolo_surface_aux': sha(a.yolo_weights), 'depthout': sha(a.depthout_weights)}
    if hashes != protocol['checkpoints_sha256'] or sha(Path(depthout_stitch.__file__)) != protocol['counting_source_sha256']:
        p.error('Frozen baseline mismatch')
    ckpt = torch.load(a.yolo_weights, map_location='cpu', weights_only=True)
    yolo = YoloFragments('yolo11n-pose.yaml', surface_aux=ckpt.get('surface_aux', False)).to(a.device).eval()
    yolo.load_state_dict(ckpt['model'])
    depth = DepthOutFragments(a.depthout_checkout, a.depthout_weights, a.device)
    dense = DenseStairHead().to(a.device).eval()
    dense.load_state_dict(torch.load(a.dense_weights, map_location='cpu', weights_only=True))
    results = []
    with torch.inference_mode():
        for case in json.loads(a.cases.read_text(encoding='utf-8')):
            path = Path(case['path'])
            if case.get('source_sha256') and sha(path) != case['source_sha256']:
                p.error(f'Locked image drift: {path.name}')
            image = cv2.imread(str(path))
            if image is None or image.shape[:2] != (SIZE, SIZE):
                p.error(f'Expected readable {SIZE} × {SIZE} image: {path.name}')
            counts = run_case(image, case.get('roi'), yolo, depth, dense, a.device,
                              protocol['configuration']['fragment_confidence'])
            results.append({**case, **counts})
    a.output.write_text(json.dumps({'cases_sha256': sha(a.cases), 'dense_weights_sha256': sha(a.dense_weights),
                                    'results': results}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
