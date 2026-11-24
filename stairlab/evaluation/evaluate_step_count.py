"""Compare step counts against annotated visible convex stair edges."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

from stairlab.recognition.step_count import PRIMARY_CONFIDENCE, SECONDARY_CONFIDENCE, extract_lines, extract_surfaces
from stairlab.recognition.step_grouping import STEP_COUNT_VERSION, count_grouped_steps


STAGES = ('raw_convex', 'structured_convex', 'deduplicated', 'grouped_v22', 'grouped_v23', 'fused')


def match_count(edges, targets, diagonal):
    if not edges or not len(targets):
        return 0
    pred = np.array([[point[:2] for point in edge['endpoints'][:2]] for edge in edges])
    direct = np.linalg.norm(pred[:, None] - targets[None], axis=-1).mean(-1)
    reverse = np.linalg.norm(pred[:, None] - targets[None, :, ::-1], axis=-1).mean(-1)
    distance = np.minimum(direct, reverse) / diagonal
    cost = np.where(distance <= 0.03, distance, 1e6)
    i, j = linear_sum_assignment(cost)
    return int(np.sum(distance[i, j] <= 0.03))


def summarize(rows, *, line_metrics=True):
    reference = np.array([row['reference'] for row in rows])
    report = {}
    for method in (*STAGES, 'riser_instances'):
        errors = np.array([row[method] for row in rows]) - reference
        report[method] = {'mae': float(np.abs(errors).mean()),
                          'exact_accuracy': float((errors == 0).mean()),
                          'mean_signed_error': float(errors.mean()),
                          'underestimated_images': int((errors < 0).sum()),
                          'overestimated_images': int((errors > 0).sum())}
        if line_metrics and method in STAGES:
            matched = sum(row['matched'][method] for row in rows)
            predictions = sum(row[method] for row in rows)
            report[method].update(matched=matched, line_recall=matched / max(int(reference.sum()), 1),
                                  line_precision=matched / max(predictions, 1))
    before = np.abs(np.array([row['structured_convex'] for row in rows]) - reference)
    after = np.abs(np.array([row['fused'] for row in rows]) - reference)
    report['paired_images_vs_structured'] = {'improved': int((after < before).sum()),
                                             'unchanged': int((after == before).sum()),
                                             'worsened': int((after > before).sum())}
    return report


def manual_summary(rows, annotations):
    by_name = {row['image']: row for row in rows}
    selected = []
    for annotation in annotations:
        if annotation['count'] is not None:
            selected.append({**by_name[annotation['image']], 'reference': annotation['count'],
                             'scene': annotation['scene']})
    if not selected:
        raise ValueError('Manual annotations contain no scorable images')
    scenes = {}
    for scene in sorted({row['scene'] for row in selected}):
        group = [row for row in selected if row['scene'] == scene]
        scenes[scene] = {'images': len(group), **summarize(group, line_metrics=False)}
    return {'images': len(selected), 'excluded': len(annotations) - len(selected),
            'reference': 'manual visible physical levels, one count per level',
            **summarize(selected, line_metrics=False),
            'scenes': scenes,
            'per_image': [{key: row[key] for key in ('image', 'scene', 'reference', 'deduplicated', 'grouped_v22', 'grouped_v23', 'fused')}
                          for row in selected]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Split directory with images/ and labels/')
    parser.add_argument('--pose', required=True)
    parser.add_argument('--seg', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--device', default='0')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--manual-labels', type=Path, help='JSON file with separate visible-step annotations')
    parser.add_argument('--audit-output', type=Path, help='Save merge diagnostics for images where 2.2 worsened the error')
    args = parser.parse_args()
    images = sorted(p for p in (args.data / 'images').iterdir()
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if not images or args.batch < 1:
        parser.error('Supply a split with images and a positive batch size')
    pose, seg = YOLO(args.pose), YOLO(args.seg)
    options = dict(imgsz=args.imgsz, device=args.device, verbose=False)
    rows, audit = [], []
    for offset in range(0, len(images), args.batch):
        paths = images[offset:offset + args.batch]
        sources = [str(path) for path in paths]
        poses = pose.predict(sources, conf=SECONDARY_CONFIDENCE, **options)
        segments = seg.predict(sources, conf=PRIMARY_CONFIDENCE, retina_masks=True, **options)
        for path, pose_result, seg_result in zip(paths, poses, segments, strict=True):
            lines = extract_lines(pose_result)
            surfaces = extract_surfaces(seg_result, pose_result.orig_shape)
            counts = count_grouped_steps(surfaces, lines, pose_result.orig_shape)
            previous = count_grouped_steps(surfaces, lines, pose_result.orig_shape, version='2.2')
            prior = count_grouped_steps(surfaces, lines, pose_result.orig_shape, version='2.3')
            label = args.data / 'labels' / f'{path.stem}.txt'
            targets = [list(map(float, line.split()[1:5])) for line in label.read_text().splitlines()
                       if line.strip() and int(float(line.split()[0])) == 0]
            targets = np.array(targets).reshape(-1, 2, 2)
            stages = {stage: counts['edge_stages'][stage] for stage in ('structured_convex', 'deduplicated', 'fused')}
            stages['grouped_v22'] = previous['edge_stages']['fused']
            stages['grouped_v23'] = prior['edge_stages']['fused']
            stages['raw_convex'] = [line for line in lines
                                    if line['class'] == 0 and line['confidence'] >= PRIMARY_CONFIDENCE]
            diagonal = float(np.hypot(*pose_result.orig_shape))
            rows.append({'image': path.name, 'reference': len(targets),
                         **{stage: len(edges) for stage, edges in stages.items()},
                         'riser_instances': sum(cls == 1 for cls, _, _ in surfaces),
                         'suspected_gaps': len(counts['suspected_gaps']),
                         'endpoint_merges': [event for event in counts['merge_events']
                                             if event['reason'] == 'shared_endpoint_local_spacing'],
                         'matched': {stage: match_count(edges, targets, diagonal) for stage, edges in stages.items()}})
            row = rows[-1]
            if args.audit_output and abs(row['grouped_v22'] - row['reference']) > abs(row['deduplicated'] - row['reference']):
                audit.append({**row, 'annotations': targets.tolist(),
                              'groups_v22': previous['steps'], 'groups_v23': counts['steps'],
                              'merges_v22': previous['merge_events'], 'merges_v23': counts['merge_events'],
                              'blocked_v23': counts['blocked_merges']})
        if len(rows) % (args.batch * 10) == 0:
            print(f'Processed {len(rows)}/{len(images)}', flush=True)
    night_rows = [row for row in rows if 'night' in row['image'].lower()]
    report = {'images': len(rows), 'step_count_version': STEP_COUNT_VERSION,
              'reference': 'annotated visible convex edges (step-count proxy)',
              'pose': args.pose, 'seg': args.seg, 'imgsz': args.imgsz, 'confidence': PRIMARY_CONFIDENCE,
              'secondary_confidence': SECONDARY_CONFIDENCE, 'endpoint_threshold_diagonal': 0.03,
              **summarize(rows)}
    if night_rows:
        report['night_named_subset'] = {'images': len(night_rows), **summarize(night_rows)}
    if args.manual_labels:
        annotations = json.loads(args.manual_labels.read_text(encoding='utf-8'))['images']
        report['manual_sample'] = manual_summary(rows, annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**report, 'per_image': rows}, indent=2) + '\n')
    if args.audit_output:
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
