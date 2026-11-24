"""Paired physical-count audit for 3.8 and the two 3.9 modes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import depthout_stitch
from depthout_refine import DepthOutFragments
from depthout_stitch import predict_steps
from full_flight_tracks import refine_flight
from perspective_step_tracks import recover_perspective_tracks
from run_far_edge_audit import rgb_tensor
from run_zoom_audit import sha, zoom
from train_dense_stair_edges import DenseStairHead
from yolo_fragments import SIZE, YoloFragments, decode_batch


def propose_roi(mask):
    """Use the spatial extent of substantial stair-surface components."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(np.uint8(mask > 0))
    regions = stats[1:count]
    regions = regions[regions[:, 4] >= max(24, .0015 * mask.size)]
    if not len(regions):
        return None
    left, top = regions[:, :2].min(axis=0)
    right, bottom = (regions[:, :2] + regions[:, 2:4]).max(axis=0)
    pad = max(12, round(.08 * max(right-left, bottom-top)))
    box = [max(0, left-pad), max(0, top-pad), min(SIZE, right+pad), min(SIZE, bottom+pad)]
    if max(box[2]-box[0], box[3]-box[1]) >= .95 * SIZE:
        return None
    return list(map(int, box))


def groups(result):
    return [step['edge_segments'] for step in result['steps']]


def prep(data, mask, image, confidence):
    return groups(predict_steps(data, mask, recover_short_edges=True, fragment_confidence=confidence,
                                texture_image=image, split_dense_tracks=True))


def run_case(image, locked_roi, yolo, depth, dense, device, confidence, *, include_geometry=False):
    roi = locked_roi
    if roi is None:
        _, mask = depth.predict([image], confidence, with_masks=True)[0]
        roi = propose_roi(mask)
    if roi is None:
        crop, scale, offset = image, 1, [0, 0]
    else:
        crop, scale, offset = zoom(image, roi)
    frames = [image, crop]
    fragments = decode_batch(yolo(rgb_tensor(frames, device)), confidence)
    masks = [result[1] for result in depth.predict(frames, confidence, with_masks=True)]
    whole = prep(fragments[0], masks[0], image, confidence)
    output = {'roi': roi, 'base': 'whole', 'count_3_4': len(whole), 'count_3_8': len(whole),
              'count_3_9_refine': len(whole), 'count_3_9_add': len(whole), 'new_tracks': [], 'refinements': 0}
    if include_geometry:
        original = predict_steps(fragments[0], masks[0], fragment_confidence=confidence)
        output.update(edges_3_0=groups(original), edges_3_9=whole)
    if roi is None:
        return output
    local = [[(np.asarray(line)*scale + offset).tolist() for line in step]
             for step in prep(fragments[1], masks[1], crop, confidence)]
    mapped = np.concatenate([fragments[1][0]*scale + offset, fragments[0][0]])
    scores = np.concatenate([fragments[1][1], fragments[0][1]])
    heat = dense(rgb_tensor([crop], device)).sigmoid()[0, 0].cpu().numpy()
    heat = cv2.warpAffine(heat, np.array([[scale, 0, offset[0]], [0, scale, offset[1]]]), (SIZE, SIZE))
    added, diagnosis = recover_perspective_tracks(image, mapped, scores, local, roi, dense_response=heat, joint=True)
    if diagnosis.get('fit_error', float('inf')) > .40:
        return output
    exterior = [step for step in whole if all(max(y for _, y in line) < roi[1] or min(y for _, y in line) >= roi[3]
                                               for line in step)]
    anchored = local + exterior + added
    extended, new, refinements = refine_flight(image, mapped, scores, anchored, roi)
    output.update(base='crop', count_3_8=len(anchored), count_3_9_refine=len(extended)-len(new),
                  count_3_9_add=len(extended), new_tracks=new, refinements=len(refinements))
    if include_geometry:
        output['edges_3_9'] = extended[:len(anchored)]
    return output


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
