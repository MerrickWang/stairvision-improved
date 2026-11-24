"""Add independent, fragment-supported tracks without replacing established groups."""

import numpy as np

from stairlab.recognition.depthout_refine import coverage


def recover_tracks(fragments, scores, anchors, roi, *, diagnostics=None):
    stats = diagnostics if diagnostics is not None else {}
    x0, y0, x1, y1 = roi
    width = x1 - x0
    middle = (x0 + x1) / 2
    lines = np.asarray([sorted(edge, key=lambda point: point[0]) for group in anchors for edge in group],
                       dtype=float).reshape(-1, 2, 2)
    usable = (lines[:, 0, 0] <= middle) & (lines[:, 1, 0] >= middle) & (np.ptp(lines[:, :, 0], axis=1) >= .24 * width)
    lines = lines[usable]
    stats['anchors'] = len(lines)
    if len(lines) < 6 or len(fragments) < 6:
        return []
    slopes = np.diff(lines[:, :, 1], axis=1).ravel() / np.diff(lines[:, :, 0], axis=1).ravel()
    slope = float(np.median(slopes))
    if abs(slope) > .25:
        return []
    levels = np.array(sorted(np.interp(middle, line[:, 0], line[:, 1]) for line in lines))
    gaps = np.diff(levels)
    regular = gaps[(gaps >= .015 * width) & (gaps <= .062 * width)]
    if len(regular) < 4:
        return []
    pitch = float(np.median(regular))
    stats.update(pitch=pitch, gap_windows=0, trial_offsets=0, support_rejected=0, coverage_rejected=0,
                 center_rejected=0, separation_rejected=0)
    left = fragments[:, :, 0].min(axis=1)
    right = fragments[:, :, 0].max(axis=1)
    inside = ((left >= x0) & (right <= x1) & (fragments[:, :, 1].min(axis=1) > y0 + 4)
              & (fragments[:, :, 1].max(axis=1) < y1 - 4) & (right - left >= 3))
    fragments, scores = fragments[inside], scores[inside]
    offsets = fragments[:, :, 1] - slope * (fragments[:, :, 0] - middle)
    separation = max(2.8, .43 * pitch)
    additions = []
    for a, b in zip(levels[:-1], levels[1:]):
        if not 1.42 * pitch <= b - a <= 2.8 * pitch:
            continue
        stats['gap_windows'] += 1
        best = None
        for y in np.unique(np.round(offsets.mean(axis=1), 1)):
            if min(y - a, b - y) < separation or max(y - a, b - y) > 1.45 * pitch:
                continue
            stats['trial_offsets'] += 1
            keep = np.max(np.abs(offsets - y), axis=1) <= min(2.2, .30 * pitch)
            group = fragments[keep]
            if len(group) < 5 or len(np.unique(np.round(group[:, :, 0].mean(axis=1) / 6))) < 5:
                stats['support_rejected'] += 1
                continue
            intervals = np.sort(group[:, :, 0], axis=1)
            lo, hi = intervals.min(), intervals.max()
            if hi - lo < .18 * width or coverage(intervals - lo, hi - lo) < .70:
                stats['coverage_rejected'] += 1
                continue
            if not lo <= middle <= hi:
                stats['center_rejected'] += 1
                continue
            edge = [[float(lo), float(y + slope * (lo - middle))], [float(hi), float(y + slope * (hi - middle))]]
            # Independently supported tracks must remain separated across the common footprint.
            conflict = False
            for line in lines:
                start, stop = max(lo, line[0, 0]), min(hi, line[1, 0])
                if stop - start >= 12:
                    xs = np.array([start, stop])
                    delta = y + slope * (xs - middle) - np.interp(xs, line[:, 0], line[:, 1])
                    if np.min(abs(delta)) < separation or delta[0] * delta[1] <= 0:
                        conflict = True
                        break
            if conflict:
                stats['separation_rejected'] += 1
                continue
            value = (hi - lo) * float(np.mean(scores[keep]))
            if best is None or value > best[0]:
                best = value, edge
        if best is not None:
            additions.append([best[1]])
    return additions
