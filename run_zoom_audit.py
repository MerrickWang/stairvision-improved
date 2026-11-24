"""Compare reference stair edges using the same 3.4 model before and after cropping."""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

import depthout_stitch
from depthout_refine import DepthOutFragments
from depthout_stitch import predict_steps
from yolo_fragments import SIZE, YoloFragments, decode_batch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def zoom(image, box):
    x0, y0, x1, y1 = box
    cut = image[y0:y1, x0:x1]
    h, w = cut.shape[:2]
    side = max(h, w)
    px, py = (side - w) // 2, (side - h) // 2
    canvas = np.zeros((side, side, 3), dtype=cut.dtype)
    canvas[py:py + h, px:px + w] = cut
    return cv2.resize(canvas, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR), side / SIZE, [x0 - px, y0 - py]


def overlap_error(segment, reference, minimum_overlap, tolerance):
    left = max(segment[0, 0], reference[0, 0])
    right = min(segment[1, 0], reference[1, 0])
    if right - left < minimum_overlap:
        return np.inf
    xs = np.linspace(left, right, 5)
    offsets = np.abs(np.interp(xs, segment[:, 0], segment[:, 1]) -
                     np.interp(xs, reference[:, 0], reference[:, 1]))
    return float(offsets.mean()) if offsets.max() <= tolerance else np.inf


def order(edge):
    points = np.asarray(edge)
    return points[np.argsort(points[:, 0], kind='stable')]


def stage_matches(groups, references, tolerance):
    if not groups:
        return {}
    errors = np.array([[min(overlap_error(order(segment), ref, .38 * np.ptp(ref[:, 0]), tolerance)
                               for segment in group) for group in groups] for ref in references])
    ii, jj = linear_sum_assignment(np.where(np.isfinite(errors), errors, 1e6))
    return {int(i + 1): {'prediction_index': int(j + 1), 'mean_vertical_error_px': float(errors[i, j])}
            for i, j in zip(ii, jj) if np.isfinite(errors[i, j])}


def trace(data, mask, image, to_original, references, tolerance, confidence):
    decoded, scores = data
    result = predict_steps(data, mask, recover_short_edges=True, fragment_confidence=confidence,
                           texture_image=image, split_dense_tracks=True)
    fragment_support = []
    candidates = [order(segment) for segment in to_original(decoded)]
    for ref in references:
        aligned = [segment.tolist() for segment in candidates
                   if np.isfinite(overlap_error(segment, ref, 3, tolerance))]
        fragment_support.append({'segments': len(aligned), 'aligned_segments': aligned})
    mapped = {key: [[to_original(line['endpoints']).tolist()] for line in stage]
              for key, stage in [('stitched', result['stitched_edges']),
                                 ('structured', result['edge_stages']['structured_convex'])]}
    mapped['grouped'] = [[to_original(edge).tolist() for edge in step['edge_segments']] for step in result['steps']]
    matches = {stage: stage_matches(groups, references, tolerance) for stage, groups in mapped.items()}
    return {'raw_fragment_count': len(decoded), 'reference_fragment_support': fragment_support,
            'visible_step_count': result['visible_step_count'], 'mapped_steps_by_stage': mapped,
            'matched_references': matches}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'probes', 'protocol', 'yolo-weights', 'depthout-checkout', 'depthout-weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    conf = json.loads(args.protocol.read_text(encoding='utf-8'))
    hashes = {'yolo_surface_aux': sha(args.yolo_weights), 'depthout': sha(args.depthout_weights)}
    if hashes != conf['checkpoints_sha256'] or sha(Path(depthout_stitch.__file__)) != conf['counting_source_sha256']:
        parser.error('Frozen checkpoint or counting source mismatch')
    confidence = conf['configuration']['fragment_confidence']
    cases = json.loads(args.probes.read_text(encoding='utf-8'))
    state = torch.load(args.yolo_weights, map_location='cpu', weights_only=True)
    yolo = YoloFragments('yolo11n-pose.yaml', surface_aux=state.get('surface_aux', False)).to(args.device).eval()
    yolo.load_state_dict(state['model'])
    depthout = DepthOutFragments(args.depthout_checkout, args.depthout_weights, args.device)
    output = {}
    for name, case in cases.items():
        source = args.images / name
        if sha(source) != case['source_sha256']:
            parser.error(f'Prepared image drift for {name}')
        image = cv2.imread(str(source))
        cropped, inverse, offset = zoom(image, case['roi_xyxy'])
        views = [image, cropped]
        tensor = torch.from_numpy(np.stack([x[..., ::-1] for x in views]))
        tensor = tensor.to(args.device).permute(0, 3, 1, 2).float() / 255
        with torch.inference_mode():
            yolo_data = decode_batch(yolo(tensor), confidence)
        depth_data = depthout.predict(views, confidence, with_masks=True)
        refs = np.array(case['references'], dtype=float)
        methods = {'yolo': {}, 'depthout': {}}
        for index, scene in enumerate(('full', 'zoom')):
            scale, origin = (1, [0, 0]) if index == 0 else (inverse, offset)

            def to_original(points):
                return np.asarray(points) * scale + origin

            df, mask = depth_data[index]
            for method, data in (('yolo', yolo_data[index]), ('depthout', df)):
                methods[method][scene] = trace(data, mask, views[index], to_original, refs,
                                               case['tolerance_px'], confidence)
        output[name] = {'zoom_scale': 1 / inverse, 'case': case, 'models': methods}
    args.output.write_text(json.dumps({'locked_probes_sha256': sha(args.probes), 'cases': output}, indent=2) + '\n')


if __name__ == '__main__':
    main()
