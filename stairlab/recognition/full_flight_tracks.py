"""Search a dense flight for supported tracks, including outside anchor intervals."""

import cv2
import numpy as np
from scipy.signal import find_peaks


def refine_flight(image, fragments, scores, anchors, roi):
    x0, y0, x1, y1 = roi
    width = x1 - x0
    center = (x0 + x1) / 2
    groups = [[list(map(list, edge)) for edge in group] for group in anchors]
    central = [(i, np.asarray(sorted(edge), dtype=float)) for i, group in enumerate(groups) for edge in group
               if min(p[0] for p in edge) <= center <= max(p[0] for p in edge)
               and abs(edge[1][0] - edge[0][0]) >= .18 * width]
    if len(central) < 6 or len(fragments) < 5:
        return groups, [], []
    lines = [line for _, line in central]
    slope = float(np.median([(b[1] - a[1]) / (b[0] - a[0]) for a, b in lines]))
    levels = np.array([np.interp(center, line[:, 0], line[:, 1]) for line in lines])
    gaps = np.diff(np.sort(levels))
    regular = gaps[(gaps >= .014 * width) & (gaps <= .08 * width)]
    if len(regular) < 4 or abs(slope) > .22:
        return groups, [], []
    pitch = float(np.median(regular))
    xs = np.arange(round(center - .13 * width), round(center + .13 * width))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rows = np.arange(y0 + 3, y1 - 3)
    yy = np.rint(rows[:, None] + slope * (xs - center)).astype(int)
    response = gray[np.clip(yy - 1, 0, gray.shape[0] - 1), xs] - gray[np.clip(yy + 1, 0, gray.shape[0] - 1), xs]
    brightness = np.median(response, axis=1)
    peaks = [idx for idx in find_peaks(brightness, prominence=8, distance=3)[0]
             if brightness[idx] >= 17 and (response[idx] >= 12).mean() >= .60]
    if not peaks:
        return groups, [], []

    # Competing candidates own distinct fragments. A steep or isolated fragment
    # cannot supply evidence for several tightly spaced levels.
    y = rows[peaks]
    offsets = fragments[:, :, 1] - slope * (fragments[:, :, 0] - center)
    distances = abs(offsets.mean(axis=1)[:, None] - y)
    owner = distances.argmin(axis=1)
    mids = fragments[:, :, 0].mean(axis=1)
    evidence = []
    for i, p in enumerate(y):
        keep = ((owner == i) & (distances[:, i] <= 2.6) & (np.ptp(offsets, axis=1) <= 2.4)
                & (mids > xs[0] - .09 * width) & (mids < xs[-1] + .09 * width))
        if np.unique(np.round(mids[keep] / 6)).size < 3 or np.ptp(mids[keep]) < .059 * width:
            continue
        lo = max(xs[0], fragments[keep, :, 0].min() - 3)
        hi = min(xs[-1], fragments[keep, :, 0].max() + 3)
        if hi - lo < .10 * width:
            continue
        edge = [[float(t), float(p + slope * (t - center))] for t in (lo, hi)]
        evidence.append((float(brightness[peaks[i]] * scores[keep].sum()), float(p), edge))

    additions, refinements = [], []
    used_groups = set()
    new_levels = []
    for _, p, edge in sorted(evidence, reverse=True):
        a, b = np.asarray(edge)
        near = []
        for index, line in central:
            lo, hi = max(a[0], line[0, 0]), min(b[0], line[1, 0])
            if hi - lo < 12:
                continue
            samples = np.linspace(lo, hi, 5)
            offsets = p + slope * (samples - center) - np.interp(samples, line[:, 0], line[:, 1])
            if np.min(abs(offsets)) <= max(2.9, .36 * pitch) or offsets[0] * offsets[-1] < 0:
                near.append((float(np.mean(abs(offsets))), index))
        if near:
            if len({index for _, index in near}) > 1:
                continue
            error, index = min(near)
            if index not in used_groups and error <= max(3.8, .48 * pitch):
                groups[index].append(edge)
                used_groups.add(index)
                refinements.append({'group': index + 1, 'edge': edge})
                # A second, stronger transition on the same bright tread can
                # refine this group's position without adding another count.
                position = int(p - rows[0])
                radius = min(4, int(.50 * pitch))
                options = [j for j in peaks if 0 < abs(j - position) <= radius
                           and brightness[j] >= max(35, 2.5 * brightness[position])
                           and np.min(brightness[min(j, position):max(j, position)]) >= -35
                           and all(other == index or abs(level - rows[j]) > 3
                                   for (other, _), level in zip(central, levels, strict=True))]
                if options:
                    delta = max(options, key=lambda j: brightness[j]) - position
                    shifted = [[x, y + delta] for x, y in edge]
                    groups[index].append(shifted)
                    refinements.append({'group': index + 1, 'edge': shifted, 'same_tread_snap': True})
            continue
        # Favor a supported continuation of the flight over isolated texture.
        all_levels = np.concatenate([levels, new_levels])
        gap = float(np.min(abs(all_levels - p)))
        if not max(3.2, .43 * pitch) <= gap <= 1.9 * pitch:
            continue
        additions.append([edge])
        new_levels.append(p)
    return groups + additions, additions, refinements
