"""Perspective-guided fitting of short stair tracks while retaining existing steps."""

import cv2
import numpy as np
from scipy.signal import find_peaks


def select_ordered_tracks(evidence, levels):
    """Optimize compatible tracks between each pair of retained anchor levels."""
    selected = []
    for left, right in zip(levels[:-1], levels[1:]):
        candidates = sorted(e for e in evidence if left < e[0] < right)
        # Anchors are fixed. Skipping a candidate is permitted when it lacks evidence.
        best, paths = [0.0], [[]]
        nodes = [(left, 0., 0., None), *candidates]
        for i, (y, pitch, quality, edge) in enumerate(candidates, 1):
            options = [(best[j] - .20 * abs((y - previous[0]) / pitch - round((y - previous[0]) / pitch)), j)
                       for j, previous in enumerate(nodes[:i]) if y - previous[0] >= max(3, .46 * pitch)]
            if not options:
                best.append(-np.inf)
                paths.append([])
                continue
            merit, j = max(options)
            best.append(merit + min(2., quality / 70))
            paths.append(paths[j] + [(y, edge)])
        best_exit = [(0., 0)] + [(best[i] - .20 * abs((right - y) / pitch - round((right - y) / pitch)), i)
                                  for i, (y, pitch, _, _) in enumerate(candidates, 1)
                                  if right - y >= max(3, .46 * pitch)]
        selected.extend(paths[max(best_exit)[1]])
    return selected


def recover_perspective_tracks(image, fragments, scores, anchors, roi, *, dense_response=None, joint=False):
    x0, y0, x1, y1 = roi
    width = x1 - x0
    center = (x0 + x1) / 2
    lines = np.asarray([sorted(edge) for group in anchors for edge in group], dtype=float).reshape(-1, 2, 2)
    spans = np.ptp(lines[:, :, 0], axis=1)
    lines = lines[(lines[:, 0, 0] < center) & (lines[:, 1, 0] > center) & (spans >= .24 * width)]
    report = {'anchors': len(lines)}
    if len(lines) < 6:
        return [], report
    slopes = (lines[:, 1, 1] - lines[:, 0, 1]) / np.ptp(lines[:, :, 0], axis=1)
    slope = float(np.median(slopes))
    if abs(slope) > .22:
        return [], report
    levels = np.array(sorted(np.interp(center, edge[:, 0], edge[:, 1]) for edge in lines))
    gaps = np.diff(levels)
    regular = (gaps >= .014 * width) & (gaps <= .080 * width)
    if regular.sum() < 4:
        return [], report

    # Under pinhole projection, the derivative of image position along
    # equally spaced levels is proportional to squared distance from the vanishing position.
    mids = (levels[1:] + levels[:-1])[regular] / 2
    observed = gaps[regular]
    horizons = np.linspace(y0 - .5 * (y1 - y0), levels[0] - .12 * (y1 - y0), 48)
    scale = (mids[None, :] - horizons[:, None]) ** 2
    factors = np.median(observed / scale, axis=1)
    errors = np.median(np.abs(np.log((scale * factors[:, None]) / observed)), axis=1)
    idx = int(np.argmin(errors))
    horizon, factor = float(horizons[idx]), float(factors[idx])
    report.update(horizon=horizon, fit_error=float(errors[idx]))
    if errors[idx] > .40:
        return [], report

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    xs = np.arange(round(center - .14 * width), round(center + .14 * width))
    if not len(xs):
        return [], report
    row = np.arange(y0 + 3, y1 - 3)
    ys = np.rint(row[:, None] + slope * (xs - center)).astype(int)
    above = np.clip(ys - 1, 0, gray.shape[0] - 1)
    below = np.clip(ys + 1, 0, gray.shape[0] - 1)
    response = gray[above, xs] - gray[below, xs]
    contrast = np.median(response, axis=1)
    peaks = find_peaks(contrast, prominence=7, distance=3)[0]
    dense_vote = None
    if dense_response is not None:
        dense_vote = np.median(dense_response[np.clip(ys, 0, gray.shape[0] - 1), xs], axis=1)
        dense_peaks = find_peaks(dense_vote, prominence=.035, distance=3)[0]
        peaks = np.union1d(peaks, dense_peaks)
    offsets = (fragments[:, :, 1] - slope * (fragments[:, :, 0] - center)).mean(axis=1)
    eligible = []
    for peak in peaks:
        y = row[peak]
        p = float(np.clip(factor * (y - horizon) ** 2, 4, .09 * width))
        j = np.searchsorted(levels, y)
        if j == 0 or j == len(levels):
            continue
        left, right = y - levels[j - 1], levels[j] - y
        if min(left, right) < max(2.7, .36 * p) or max(left, right) > 2.25 * p:
            continue
        visible_rgb = (response[peak] >= 12).mean() >= .58 and contrast[peak] >= 15
        visible_dense = dense_vote is not None and dense_vote[peak] >= .28
        if not visible_rgb and not visible_dense:
            continue
        strength = max(float(contrast[peak]), 35 * float(dense_vote[peak])) if visible_dense else float(contrast[peak])
        eligible.append((float(y), p, strength))
    if not eligible:
        return [], report

    proposals = np.array([item[0] for item in eligible])
    hypotheses = np.concatenate([levels, proposals])
    distances = np.abs(offsets[:, None] - hypotheses)
    owner = distances.argmin(axis=1)
    fragment_slopes = np.abs(np.diff(fragments[:, :, 1], axis=1).ravel() - slope *
                             np.diff(fragments[:, :, 0], axis=1).ravel())
    xcenters = fragments[:, :, 0].mean(axis=1)
    evidence = []
    for i, (y, p, strength) in enumerate(eligible):
        k = len(levels) + i
        keep = ((owner == k) & (distances[:, k] <= min(2.5, .36 * p)) & (fragment_slopes < 2.5)
                & (abs(xcenters - center) < .29 * width))
        bins = np.unique(np.round(xcenters[keep] / 6))
        if len(bins) < 3 or np.ptp(xcenters[keep]) < .11 * width:
            continue
        target = [[float(x), float(y + slope * (x - center))] for x in (xs[0], xs[-1])]
        # Guard against an existing angled edge crossing the proposed track.
        if any(np.min(abs(np.array(target)[:, 1] - np.interp(xs[[0, -1]], line[:, 0], line[:, 1]))) < 2.2
               for line in lines if line[0, 0] <= xs[0] and line[1, 0] >= xs[-1]):
            continue
        evidence.append((y, p, strength * float(scores[keep].sum()), target))
    if joint:
        accepted = select_ordered_tracks(evidence, levels)
        report['pending_positions_y'] = [y for y, _, _ in eligible if all(abs(y - v) >= 3 for v, _ in accepted)]
    else:
        accepted = []
        for y, pitch, _, target in sorted(evidence, key=lambda e: -e[2]):
            if all(abs(y - earlier) >= max(3, .46 * pitch) for earlier, _ in accepted):
                accepted.append((y, target))
    report['candidates'] = len(eligible)
    return [[edge] for _, edge in sorted(accepted)], report
