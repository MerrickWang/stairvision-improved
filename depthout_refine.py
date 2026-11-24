"""Refine the positions of existing StairLab groups with DepthOut convex fragments."""

from pathlib import Path
import sys

import numpy as np


class DepthOutFragments:
    def __init__(self, checkout, weights, device):
        import torch

        checkout = Path(checkout).resolve()
        if not (checkout / 'nets/NetV3.py').is_file():
            raise ValueError('Supply a StairNet-DepthOut checkout containing nets/NetV3.py')
        sys.path.insert(0, str(checkout))
        try:
            from nets.NetV3 import StairNet_DepthOut
        finally:
            sys.path.pop(0)
        checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
        self.model = StairNet_DepthOut(width=checkpoint['args']['width']).to(device).eval()
        self.model.load_state_dict(checkpoint['model'])
        self.device = device

    def predict(self, images, confidence=0.5, *, with_masks=False):
        import cv2
        import torch

        inputs, transforms = [], []
        for image in images:
            height, width = image.shape[:2]
            side = max(height, width)
            y, x = (side - height) // 2, (side - width) // 2
            rgb = np.zeros((side, side, 3), dtype=np.uint8)
            rgb[y:y + height, x:x + width] = image[..., ::-1]
            inputs.append(cv2.resize(rgb, (512, 512)))
            transforms.append((height, width, side, x, y))
        tensor = torch.from_numpy(np.stack(inputs)).permute(0, 3, 1, 2).to(self.device).float() / 127.5 - 1
        with torch.inference_mode():
            outputs = self.model(tensor)
        scores, positions = outputs[:2]
        fragments = [decode_fragments(s[0], p, transform, confidence) for s, p, transform
                     in zip(scores.cpu().numpy(), positions.cpu().numpy(), transforms)]
        if not with_masks:
            return fragments
        labels = outputs[4].argmax(dim=1).byte().cpu().numpy()
        return [(data, cv2.resize(label, (side, side), interpolation=cv2.INTER_NEAREST)[y:y + height, x:x + width])
                for data, label, (height, width, side, x, y) in zip(fragments, labels, transforms)]


def decode_fragments(scores, positions, transform, confidence):
    height, width, side, x, y = transform
    rows, cols = np.nonzero(scores >= confidence)
    strides = np.array([side / scores.shape[1], side / scores.shape[0]])
    ends = (positions[:, rows, cols].T.reshape(-1, 2, 2) + np.column_stack((cols, rows))[:, None]) * strides
    ends -= [x, y]
    valid = (np.isfinite(ends).all(axis=(1, 2)) & (ends >= 0).all(axis=(1, 2)) &
             (ends < [width, height]).all(axis=(1, 2)) &
             (np.linalg.norm(ends[:, 0] - ends[:, 1], axis=1) >= 3))
    return ends[valid], scores[rows, cols][valid]


def coverage(intervals, span):
    covered, right = 0.0, 0.0
    for left, end in sorted(intervals, key=lambda interval: interval[0]):
        covered += max(0, end - max(left, right))
        right = max(right, end)
    return covered / span


def refine_edge(endpoints, fragments, confidence, shape):
    """Fit a consensus of fragments in a corridor, preserving tangential extent."""
    start, end = np.asarray(endpoints, dtype=float)[:, :2]
    length = np.linalg.norm(end - start)
    diagonal = np.hypot(*shape)
    if length < 0.08 * diagonal or len(fragments) < 5:
        return None
    axis = (end - start) / length
    normal = np.array([-axis[1], axis[0]])
    projected = (fragments - start) @ axis
    offsets = (fragments - start) @ normal
    span = np.ptp(projected, axis=1)
    limit = diagonal * 0.052
    useful = ((projected.min(axis=1) >= -0.10 * length) &
              (projected.max(axis=1) <= 1.10 * length) &
              (np.max(np.abs(offsets), axis=1) <= limit) & (span >= 3) &
              (np.ptp(offsets, axis=1) <= 0.48 * span))
    x, y, scores = projected[useful], offsets[useful], confidence[useful]
    if len(x) < 5:
        return None
    intervals = np.clip(np.sort(x, axis=1), 0, length)
    residual = max(2.5, diagonal * 0.006)
    # Test slopes between separated fragment centers. Individual grid fragments
    # can be too short to provide a stable estimate of a line's direction.
    xc, yc = x.mean(axis=1), y.mean(axis=1)
    first, second = np.nonzero(np.triu(np.abs(xc[:, None] - xc[None, :]) >= 0.35 * length, 1))
    slopes = (yc[second] - yc[first]) / (xc[second] - xc[first])
    intercepts = yc[first] - slopes * xc[first]
    hypotheses = (np.abs(slopes) <= 0.31) & (np.abs(intercepts) <= limit) & (np.abs(intercepts + slopes * length) <= limit)
    if not hypotheses.any():
        return None
    slopes, intercepts = slopes[hypotheses], intercepts[hypotheses]
    if len(slopes) > 2048:
        sample = np.linspace(0, len(slopes) - 1, 2048, dtype=int)
        slopes, intercepts = slopes[sample], intercepts[sample]
    agreement = np.max(np.abs(y[None] - (slopes[:, None, None] * x[None] + intercepts[:, None, None])), axis=2) <= residual
    support = agreement @ (scores * span[useful])
    best = int(np.argmax(support))
    inliers = agreement[best]
    if inliers.sum() < 5:
        return None
    extent = intervals[inliers]
    covered = coverage(extent, length)
    if covered < 0.48 or extent[:, 0].min() > 0.23 * length or extent[:, 1].max() < 0.77 * length:
        return None
    # Two nearly equal but separated lines are ambiguous in a crowded stair flight.
    middle = intercepts + slopes * (length / 2)
    if np.any((support >= 0.86 * support[best]) & (np.abs(middle - middle[best]) > 2.4 * residual)):
        return None
    slope, intercept = np.polyfit(x[inliers].ravel(), y[inliers].ravel(), 1, w=np.repeat(np.sqrt(scores[inliers]), 2))
    correction = np.array([intercept, intercept + slope * length])
    if np.max(np.abs(correction)) > limit or np.max(np.abs(correction)) < 1:
        return None
    refined = np.array([start, end]) + correction[:, None] * normal
    if np.any(refined < 0) or np.any(refined >= [shape[1], shape[0]]):
        return None
    return refined.tolist(), {'fragment_count': int(inliers.sum()), 'covered_fraction': covered}


def refine_group_edges(edges, fragment_data, shape):
    fragments, scores = fragment_data
    result = []
    original = [np.asarray(line['endpoints'], dtype=float)[:, :2] for line in edges]
    for i, edge in enumerate(edges):
        proposal = refine_edge(edge['endpoints'], fragments, scores, shape)
        if proposal is None:
            result.append(edge)
            continue
        points, evidence = proposal
        before, after = original[i], np.array(points)
        # Keep ownership with the original group when neighboring rails compete.
        def endpoint_distance(a, b):
            return min(np.linalg.norm(a - b, axis=1).mean(), np.linalg.norm(a - b[::-1], axis=1).mean())

        move = endpoint_distance(before, after)
        if any(j != i and endpoint_distance(other, after) + 2 < move for j, other in enumerate(original)):
            result.append(edge)
            continue
        endpoints = [[*point, *old[2:]] for point, old in zip(points, edge['endpoints'])]
        result.append({**edge, 'endpoints': endpoints, 'depthout_evidence': evidence})
    return result
