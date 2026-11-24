"""Compare a YOLO fragment checkpoint using the shared DepthOut stitcher."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from depthout_refine import DepthOutFragments
from depthout_stitch import predict_steps
from predict_depthout_only import matched_edges, render
from yolo_fragments import SIZE, YoloFragments, decode_batch


def summarize(rows):
    truth = np.array([row['reference_convex_lines'] for row in rows])
    report = {}
    for name, key in (('stitched', 'stitched_count'), ('grouped', 'visible_step_count')):
        count = np.array([row[key] for row in rows])
        matches = sum(row[f'matched_{name}'] for row in rows)
        report[name] = {'mae': float(np.abs(count - truth).mean()), 'exact_accuracy': float((count == truth).mean()),
                        'line_recall': matches / max(1, int(truth.sum())),
                        'line_precision': matches / max(1, int(count.sum())), 'matched': matches,
                        'predictions': int(count.sum())}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('manifest', 'backbone', 'weights', 'depthout-checkout', 'depthout-weights', 'output'):
        parser.add_argument(f'--{field}', required=True, type=Path)
    parser.add_argument('--batch', default=8, type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--confidence', default=.5, type=float)
    parser.add_argument('--preview', action='append', default=[])
    parser.add_argument('--recover-short-edges', action='store_true', help='Enable experimental 3.2 short-edge recovery')
    parser.add_argument('--recover-dense-edges', action='store_true', help='Recover short edges in dense level sequences')
    parser.add_argument('--filter-tread-texture', action='store_true', help='Reject trailing texture boundaries and weak seams')
    parser.add_argument('--split-dense-tracks', action='store_true', help='Split a fitted edge with two supported parallel tracks')
    args = parser.parse_args()
    if args.batch < 1 or not 0 < args.confidence <= 1:
        parser.error('Supply a positive batch size and a confidence in (0, 1]')
    sources = [Path(json.loads(line)['image']) for line in args.manifest.read_text().splitlines() if line.strip()]
    if not sources:
        parser.error('Validation manifest is empty')
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.weights, weights_only=True, map_location='cpu')
    model = YoloFragments(args.backbone, surface_aux=checkpoint.get('surface_aux', False)).to(args.device).eval()
    model.load_state_dict(checkpoint['model'])
    semantic = DepthOutFragments(args.depthout_checkout, args.depthout_weights, args.device)
    rows = []
    for i in range(0, len(sources), args.batch):
        paths = sources[i:i + args.batch]
        images = [cv2.imread(str(p)) for p in paths]
        if any(image is None or image.shape != (SIZE, SIZE, 3) for image in images):
            raise ValueError(f'Expected original {SIZE}x{SIZE} stair images')
        rgb = np.stack([cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images])
        tensor = torch.from_numpy(rgb).to(args.device).permute(0, 3, 1, 2).float() / 255
        floor = min(args.confidence, .10) if args.recover_dense_edges else args.confidence
        with torch.inference_mode():
            fragments = decode_batch(model(tensor), floor)
        surfaces = semantic.predict(images, with_masks=True)
        for path, image, data, (_, mask) in zip(paths, images, fragments, surfaces, strict=True):
            result = predict_steps(data, mask, recover_short_edges=args.recover_short_edges,
                                   recover_dense_edges=args.recover_dense_edges, fragment_confidence=args.confidence,
                                   texture_image=image if args.filter_tread_texture else None,
                                   split_dense_tracks=args.split_dense_tracks)
            stitched, grouped = result['stitched_edges'], result['edge_stages']['fused']
            label = path.parent.parent / 'labels' / f'{path.stem}.txt'
            coords = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
            truth = np.array([line[1:5] for line in coords if int(line[0]) == 0]).reshape(-1, 2, 2)
            diagonal = np.hypot(*mask.shape)
            assigned = matched_edges(grouped, truth, diagonal)
            rows.append({'image': path.name, 'reference_convex_lines': len(truth), 'grid_fragments': result['grid_fragments'],
                         'stitched_count': len(stitched), 'visible_step_count': len(grouped),
                         'matched_stitched': len(matched_edges(stitched, truth, diagonal)), 'matched_grouped': len(assigned),
                         'matched_grouped_reference_ids': assigned})
            if args.recover_short_edges:
                rows[-1].update(short_edge_candidates=result['short_edge_candidate_count'],
                                short_edges_recovered=result['short_edges_recovered'])
            rows[-1].update({key: result[key] for key in ('dense_edge_candidates', 'dense_edges_recovered',
                                                         'texture_edges_rejected', 'split_track_events') if key in result})
            if path.name in args.preview:
                render(args.output / f'{path.stem}_comparison.jpg', image, data[0],
                       [line['endpoints'] for line in stitched], [line['endpoints'] for line in grouped], truth)
                if args.recover_short_edges or args.recover_dense_edges or args.filter_tread_texture or args.split_dense_tracks:
                    (args.output / f'{path.stem}_prediction.json').write_text(json.dumps(result, indent=2) + '\n')
        if len(rows) % 80 == 0:
            print(f'Processed {len(rows)}/{len(sources)}', flush=True)
    report = {'model': 'YOLO11 short-fragment head', 'grouping_surfaces': 'StairNet-DepthOut',
              'epoch': checkpoint['epoch'], 'images': len(rows), 'confidence': args.confidence, **summarize(rows)}
    if args.split_dense_tracks:
        report.update(workflow_version='3.4', split_dense_tracks=True, recover_dense_edges=args.recover_dense_edges,
                      filter_tread_texture=args.filter_tread_texture)
    elif args.recover_dense_edges or args.filter_tread_texture:
        report.update(workflow_version='3.3', recover_dense_edges=args.recover_dense_edges,
                      filter_tread_texture=args.filter_tread_texture)
    elif args.recover_short_edges:
        report['workflow_version'] = '3.2'
    (args.output / 'results.json').write_text(json.dumps({**report, 'per_image': rows}, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
