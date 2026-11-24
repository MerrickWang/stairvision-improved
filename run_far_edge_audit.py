"""Ablate native cropping, a dense edge head and ordered fitting on locked probes."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

import depthout_stitch
from depthout_refine import DepthOutFragments, coverage
from depthout_stitch import predict_steps
from perspective_step_tracks import recover_perspective_tracks
from run_zoom_audit import order, overlap_error, sha, stage_matches, zoom
from train_dense_stair_edges import DenseStairHead
from yolo_fragments import SIZE, YoloFragments, decode_batch


def native_zoom(image, roi, inverse, offset):
    native_h, native_w = image.shape[:2]
    side = max(native_h, native_w)
    px, py = (side - native_w) // 2, (side - native_h) // 2
    axis = (np.arange(SIZE) + .5) * inverse
    xx, yy = np.meshgrid(axis + offset[0], axis + offset[1])
    view = cv2.remap(image, (xx * side / SIZE - .5 - px).astype('float32'),
                     (yy * side / SIZE - .5 - py).astype('float32'), cv2.INTER_LINEAR)
    x0, y0, x1, y1 = roi
    view[(xx < x0) | (xx >= x1) | (yy < y0) | (yy >= y1)] = 0
    return view


def oracle_support(fragments, references, tolerance):
    if not references or not len(fragments):
        return []
    fragments = [order(segment) for segment in fragments]
    refs = [order(edge) for edge in references]
    error = np.array([[overlap_error(segment, ref, 3, tolerance) for ref in refs] for segment in fragments])
    owners = error.argmin(axis=1)
    report = []
    for i, ref in enumerate(refs):
        left, right = ref[:, 0]
        span = right - left
        ids = (owners == i) & np.isfinite(error[:, i])
        intervals = np.array([np.clip(f[:, 0], left, right) - left for f, keep in zip(fragments, ids) if keep])
        ratio = coverage(intervals, span)
        report.append({'edge': i + 1, 'nearby': int(np.isfinite(error[:, i]).sum()),
                       'exclusive': int(ids.sum()), 'covered_fraction': ratio, 'adequate': bool(ids.sum() >= 2 and ratio >= .40)})
    return report


def rgb_tensor(images, device):
    data = np.stack([im[..., ::-1] for im in images])
    return torch.from_numpy(data).permute(0, 3, 1, 2).to(device).float() / 255


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('images', 'native', 'probes', 'protocol', 'yolo-weights', 'depthout-checkout',
                 'depthout-weights', 'dense-weights', 'output'):
        parser.add_argument(f'--{name}', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding='utf-8'))
    hashes = {'yolo_surface_aux': sha(args.yolo_weights), 'depthout': sha(args.depthout_weights)}
    if protocol['checkpoints_sha256'] != hashes or sha(Path(depthout_stitch.__file__)) != protocol['counting_source_sha256']:
        parser.error('Frozen baseline mismatch')
    confidence = protocol['configuration']['fragment_confidence']
    state = torch.load(args.yolo_weights, map_location='cpu', weights_only=True)
    yolo = YoloFragments('yolo11n-pose.yaml', surface_aux=state.get('surface_aux', False)).to(args.device).eval()
    yolo.load_state_dict(state['model'])
    dense = DenseStairHead().to(args.device).eval()
    dense.load_state_dict(torch.load(args.dense_weights, map_location='cpu', weights_only=True))
    depth = DepthOutFragments(args.depthout_checkout, args.depthout_weights, args.device)
    cases = json.loads(args.probes.read_text(encoding='utf-8'))
    reports = {}
    for name, case in cases.items():
        image_path = args.images / name
        if sha(image_path) != case['source_sha256']:
            parser.error(f'Prepared image drift for {name}')
        image = cv2.imread(str(image_path))
        native = cv2.imread(str(args.native / f'native_{name}'))
        if native is None or max(native.shape[:2]) <= SIZE:
            parser.error(f'No native resolution source for {name}')
        direct = native_zoom(native, [0, 0, SIZE, SIZE], 1, [0, 0])
        small = lambda view: cv2.resize(view.astype('float32'), (32, 32), interpolation=cv2.INTER_AREA)
        if np.mean(abs(small(direct) - small(image))) > 12:
            parser.error(f'Higher resolution input does not match the locked image for {name}')
        roi = case['roi_xyxy']
        crop, scale, offset = zoom(image, roi)
        native_crop = native_zoom(native, roi, scale, offset)
        views = [image, crop, native_crop]
        with torch.inference_mode():
            data = decode_batch(yolo(rgb_tensor(views[:2], args.device)), confidence)
            data.append(decode_batch(yolo(rgb_tensor([image, native_crop], args.device)), confidence)[1])
            heatmaps = dense(rgb_tensor(views[1:], args.device)).sigmoid()[:, 0].cpu().numpy()
        masks = [v[1] for v in depth.predict(views[:2], confidence, with_masks=True)]
        masks.append(depth.predict([image, native_crop], confidence, with_masks=True)[1][1])
        reports[name] = result = {'reference_count': case.get('known_whole_image_count'),
                                  'minimum_count': case.get('known_whole_image_minimum'),
                                  'native_size': native.shape[:2], 'source_kind': case.get('source_kind', 'original'),
                                  'native_sha256': sha(args.native / f'native_{name}'),
                                  'oracle': {}, 'variants': {}}
        reference = case.get('references', [])

        def evaluate(groups, added=None):
            return {'count': len(groups), 'added': [] if added is None else added,
                    'matches': stage_matches(groups, np.asarray(reference), case['tolerance_px']) if reference else {}}

        full = predict_steps(data[0], masks[0], recover_short_edges=True, texture_image=image, split_dense_tracks=True)
        outside = [s['edge_segments'] for s in full['steps']
                   if all(max(y for _, y in edge) < roi[1] or min(y for _, y in edge) >= roi[3]
                          for edge in s['edge_segments'])]
        result['variants']['whole_3_4'] = evaluate([s['edge_segments'] for s in full['steps']])
        for index, label in enumerate(('resize_then_crop', 'native_crop'), 1):
            step = predict_steps(data[index], masks[index], recover_short_edges=True,
                                 texture_image=views[index], split_dense_tracks=True)
            local = [[(np.asarray(edge) * scale + offset).tolist() for edge in s['edge_segments']] for s in step['steps']]
            mapped = data[index][0] * scale + offset
            result['oracle'][label] = oracle_support(mapped, reference, case['tolerance_px']) if reference else []
            result['variants'][label] = evaluate(local)
            heat = cv2.warpAffine(heatmaps[index - 1], np.array([[scale, 0, offset[0]], [0, scale, offset[1]]]),
                                  (SIZE, SIZE), flags=cv2.INTER_LINEAR)
            fragments = np.concatenate([mapped, data[0][0]])
            scores = np.concatenate([data[index][1], data[0][1]])
            for method, response, joint in (('perspective', None, False), ('response_head', heat, False),
                                            ('ordered_fit', None, True), ('combined', heat, True)):
                added, detail = recover_perspective_tracks(image, fragments, scores, local, roi,
                                                           dense_response=response, joint=joint)
                use_crop = detail.get('fit_error', float('inf')) <= .40
                groups = local + outside + added if use_crop else [s['edge_segments'] for s in full['steps']]
                result['variants'][f'{label}_{method}'] = {
                    **evaluate(groups, added), 'selected_base': label if use_crop else 'whole', 'diagnostics': detail}
    args.output.write_text(json.dumps({'probes_sha256': sha(args.probes), 'head_sha256': sha(args.dense_weights),
                                       'cases': reports}, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
