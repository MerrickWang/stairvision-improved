"""Predict and evaluate visible stair edges without a YOLO model."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from depthout_refine import DepthOutFragments
from depthout_stitch import predict_steps


def matched_edges(lines, truth, diagonal):
    if not lines or not len(truth):
        return []
    points = np.array([line['endpoints'] for line in lines])
    direct = np.linalg.norm(points[:, None] - truth[None], axis=-1).mean(axis=-1)
    flipped = np.linalg.norm(points[:, None] - truth[None, :, ::-1], axis=-1).mean(axis=-1)
    distance = np.minimum(direct, flipped) / diagonal
    indices = linear_sum_assignment(np.where(distance <= .03, distance, 1e6))
    return sorted(indices[1][distance[indices] <= .03].tolist())


def render(path, image, fragments, stitched, groups, truth):
    panels = []
    for title, lines in (('GRID FRAGMENTS', fragments), ('STITCHED EDGES', stitched), ('GROUPED LEVELS', groups)):
        panel = image.copy()
        for a, b in truth:
            cv2.line(panel, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (245, 235, 30), 2, cv2.LINE_AA)
        for index, (a, b) in enumerate(lines, 1):
            a, b = tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))
            cv2.line(panel, a, b, (40, 180, 255), 2, cv2.LINE_AA)
            if title == 'GROUPED LEVELS':
                center = tuple(np.rint((np.array(a) + b) / 2).astype(int))
                cv2.putText(panel, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, .58, (18, 18, 18), 4, cv2.LINE_AA)
                cv2.putText(panel, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, .58, (255, 255, 255), 1, cv2.LINE_AA)
        panel = cv2.copyMakeBorder(panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(26, 26, 26))
        cv2.putText(panel, f'{title}  {len(lines)}', (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .62, (245, 245, 245), 1)
        panels.append(panel)
    if not cv2.imwrite(str(path), np.hstack(panels)):
        raise OSError(f'Could not save {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('source', 'depthout-checkout', 'depthout-weights', 'output'):
        parser.add_argument(f'--{key}', required=True, type=Path)
    parser.add_argument('--labels', type=Path, help='Directory of original pixel-space convex/concave labels')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--confidence', type=float, default=0.5)
    parser.add_argument('--preview', action='append', default=[])
    parser.add_argument('--recover-short-edges', action='store_true', help='Enable experimental 3.2 short-edge recovery')
    parser.add_argument('--recover-dense-edges', action='store_true', help='Recover short edges in dense level sequences')
    parser.add_argument('--filter-tread-texture', action='store_true', help='Reject trailing texture boundaries and weak seams')
    parser.add_argument('--split-dense-tracks', action='store_true', help='Split a fitted edge with two supported parallel tracks')
    args = parser.parse_args()
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg'}) if args.source.is_dir() else [args.source]
    if args.batch < 1 or not paths or not 0 < args.confidence <= 1:
        parser.error('A nonempty source, positive batch size, and confidence in (0, 1] are required')
    model = DepthOutFragments(args.depthout_checkout, args.depthout_weights, args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for start in range(0, len(paths), args.batch):
        batch = paths[start:start + args.batch]
        images = [cv2.imread(str(path)) for path in batch]
        if any(image is None for image in images):
            raise ValueError(f'Unreadable image: {next(p for p, image in zip(batch, images) if image is None)}')
        floor = min(args.confidence, .10) if args.recover_dense_edges else args.confidence
        for path, image, (fragments, mask) in zip(batch, images, model.predict(images, floor, with_masks=True), strict=True):
            result = predict_steps(fragments, mask, recover_short_edges=args.recover_short_edges,
                                   recover_dense_edges=args.recover_dense_edges, fragment_confidence=args.confidence,
                                   texture_image=image if args.filter_tread_texture else None,
                                   split_dense_tracks=args.split_dense_tracks)
            groups = result['edge_stages']['fused']
            row = {'image': path.name, 'grid_fragments': result['grid_fragments'],
                   'stitched_count': len(result['stitched_edges']), 'visible_step_count': result['visible_step_count']}
            if args.recover_short_edges:
                row.update(short_edge_candidates=result['short_edge_candidate_count'],
                           short_edges_recovered=result['short_edges_recovered'])
            row.update({key: result[key] for key in ('dense_edge_candidates', 'dense_edges_recovered',
                                                     'texture_edges_rejected', 'split_track_events') if key in result})
            truth = np.empty((0, 2, 2))
            if args.labels:
                labels = [list(map(float, line.split())) for line in (args.labels / (path.stem + '.txt')).read_text().splitlines() if line.strip()]
                truth = np.array([label[1:5] for label in labels if int(label[0]) == 0]).reshape(-1, 2, 2)
                diagonal = np.hypot(*mask.shape)
                stitched_matches = matched_edges(result['stitched_edges'], truth, diagonal)
                grouped_matches = matched_edges(groups, truth, diagonal)
                row.update(reference_convex_lines=len(truth), matched_stitched=len(stitched_matches),
                           matched_grouped=len(grouped_matches), matched_grouped_reference_ids=grouped_matches)
            if len(paths) == 1 or path.name in args.preview:
                (args.output / f'{path.stem}_prediction.json').write_text(json.dumps(result, indent=2) + '\n')
                render(args.output / f'{path.stem}_comparison.jpg', image, fragments[0],
                       [line['endpoints'] for line in result['stitched_edges']],
                       [line['endpoints'] for line in groups], truth)
            rows.append(row)
        if len(rows) % 80 == 0:
            print(f'Processed {len(rows)}/{len(paths)}', flush=True)
    report = {'model': 'StairNet-DepthOut', 'images': len(rows), 'fragment_confidence': args.confidence}
    if args.split_dense_tracks:
        report.update(workflow_version='3.4', split_dense_tracks=True, recover_dense_edges=args.recover_dense_edges,
                      filter_tread_texture=args.filter_tread_texture)
    elif args.recover_dense_edges or args.filter_tread_texture:
        report.update(workflow_version='3.3', recover_dense_edges=args.recover_dense_edges,
                      filter_tread_texture=args.filter_tread_texture)
    elif args.recover_short_edges:
        report['workflow_version'] = '3.2'
    if args.labels:
        report['reference'] = 'annotated convex edges, proxy for visible physical levels'
        reference = np.array([row['reference_convex_lines'] for row in rows])
        for name, key in (('stitched', 'stitched_count'), ('grouped', 'visible_step_count')):
            count = np.array([row[key] for row in rows])
            matched = sum(row[f'matched_{name}'] for row in rows)
            report[name] = {'mae': float(np.abs(count - reference).mean()), 'exact_accuracy': float((count == reference).mean()),
                            'matched': matched, 'line_recall': matched / max(1, int(reference.sum())),
                            'line_precision': matched / max(1, int(count.sum())), 'predictions': int(count.sum())}
    (args.output / 'results.json').write_text(json.dumps({**report, 'per_image': rows}, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
