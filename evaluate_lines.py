"""Class-aware endpoint matching. Thin-box mAP alone is not a line metric."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data', default='data/prepared')
    p.add_argument('--split', choices=['train', 'val'], default='val')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--threshold', type=float, default=0.03, help='Mean endpoint distance / image diagonal')
    p.add_argument('--output', default='runs/line_metrics.json')
    a = p.parse_args()
    root = Path(a.data); model = YOLO(a.weights)
    tp = fp = fn = 0; errors = []; count_errors = []
    for result in model.predict(source=str(root/'images'/a.split), stream=True, conf=a.conf, verbose=False, device=0):
        h, w = result.orig_shape
        label = root/'labels'/a.split/(Path(result.path).stem+'.txt')
        rows = [list(map(float, x.split())) for x in label.read_text().splitlines() if x.strip()]
        gt_classes = np.array([x[0] for x in rows], dtype=int)
        gt = np.array([[[x[5]*w, x[6]*h], [x[8]*w, x[9]*h]] for x in rows]).reshape(-1, 2, 2)
        classes = result.boxes.cls.cpu().numpy().astype(int)
        points = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else np.zeros((0, 2, 2))
        count_errors.append(abs(int((classes == 0).sum()) - int((gt_classes == 0).sum())))
        for cls in (0, 1):
            g, q = gt[gt_classes == cls], points[classes == cls]
            if not len(g) or not len(q):
                fn += len(g); fp += len(q); continue
            direct = np.linalg.norm(g[:, None] - q[None, :], axis=-1).mean(-1)
            swapped = np.linalg.norm(g[:, None] - q[None, :, ::-1], axis=-1).mean(-1)
            distance = np.minimum(direct, swapped) / np.hypot(w, h)
            # Prefer the maximum number of valid matches before minimizing error.
            cost = np.where(distance <= a.threshold, distance, 1e6)
            gi, qi = linear_sum_assignment(cost)
            good = distance[gi, qi] <= a.threshold
            hits = int(good.sum()); tp += hits; fn += len(g)-hits; fp += len(q)-hits
            errors.extend(distance[gi[good], qi[good]].tolist())
    precision, recall = tp / max(tp+fp, 1), tp / max(tp+fn, 1)
    report = {'images': len(count_errors), 'tp': tp, 'fp': fp, 'fn': fn, 'precision': precision, 'recall': recall,
        'f1': 2*precision*recall/max(precision+recall, 1e-12),
        'matched_endpoint_error_diagonal': float(np.mean(errors)) if errors else None,
        'convex_count_mae': float(np.mean(count_errors)), 'convex_count_exact_accuracy': float(np.mean(np.array(count_errors)==0)),
        'confidence_threshold': a.conf, 'endpoint_threshold_diagonal': a.threshold,
        'note': 'Count metric uses annotated visible convex edges, not total physical staircase steps.'}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(report, indent=2)); print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
