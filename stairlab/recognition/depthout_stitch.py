"""Reconstruct full convex edges from StairNet-DepthOut grid fragments alone."""

import cv2
import numpy as np

from stairlab.recognition.depthout_refine import coverage
from stairlab.recognition.step_count import convex_boundary, surface_boundary
from stairlab.recognition.step_grouping import count_grouped_steps


def components(indices, projected, gap):
    intervals = np.sort(projected[indices], axis=1)
    order = np.argsort(intervals[:, 0], kind='stable')
    part = []
    right = -np.inf
    for index in order:
        left, end = intervals[index]
        if part and left > right + gap:
            yield np.array(part)
            part = []
            right = -np.inf
        part.append(indices[index])
        right = max(right, end)
    if part:
        yield np.array(part)


def stitch_fragments(fragments, scores, shape):
    """Extract a series of collinear, connected fragment chains."""
    lines, _, _ = _stitch(fragments, scores, shape)
    return lines


def _stitch(fragments, scores, shape, *, profile='primary'):
    profiles = {'primary': (.08, .12, np.inf, .52, .065, .045, 4),
                'short': (.05, .065, .12, .72, .065, .045, 4),
                'dense': (.025, .047, .18, .70, .032, .024, 3),
                'split': (.025, .085, .26, .72, .044, .024, 4)}
    seed, min_span, max_span, min_cover, min_weight, gap_ratio, min_fragments = profiles[profile]
    active = np.ones(len(fragments), dtype=bool)
    if len(fragments) < min_fragments:
        return [], active, []
    diagonal = float(np.hypot(*shape))
    tolerance = max(1.2, .0025 * diagonal) if profile == 'split' else max(3.0, .006 * diagonal)
    gap = gap_ratio * diagonal
    lengths = np.linalg.norm(fragments[:, 1] - fragments[:, 0], axis=1)
    centers = fragments.mean(axis=1)
    left, right = np.triu_indices(len(fragments), 1)
    vectors = centers[right] - centers[left]
    norms = np.linalg.norm(vectors, axis=1)
    valid = (norms >= seed * diagonal) & (norms <= .63 * diagonal)
    directions = vectors[valid] / norms[valid, None]
    anchors = centers[left[valid]]
    if not len(anchors):
        return [], active, []
    # Limit consensus work in frames with many occupied grid cells.
    if len(anchors) > 4096:
        selected = np.linspace(0, len(anchors) - 1, 4096, dtype=int)
        anchors, directions = anchors[selected], directions[selected]
    normals = directions[:, ::-1] * [-1, 1]
    intercept = (anchors * normals).sum(axis=1)
    support = np.logical_and.reduce([
        np.abs(normals @ fragments[:, end].T - intercept[:, None]) <= tolerance for end in (0, 1)])
    weights = np.minimum(lengths, .035 * diagonal) * scores
    output, chains = [], []
    while active.sum() >= min_fragments:
        rank = support @ (active * weights)
        choices = np.flatnonzero(rank >= min_weight * diagonal)
        choices = choices[np.argsort(-rank[choices], kind='stable')[:48]]
        best = None
        for h in choices:
            indices = np.flatnonzero(support[h] & active)
            projected = fragments @ directions[h]
            for chain in components(indices, projected, gap):
                extent = projected[chain]
                first, last = extent.min(), extent.max()
                if len(chain) < min_fragments or not min_span * diagonal <= last - first <= max_span * diagonal:
                    continue
                covered = coverage(np.sort(extent, axis=1) - first, last - first)
                if covered < min_cover:
                    continue
                score = float(weights[chain].sum() * covered)
                if best is None or score > best[0]:
                    best = score, chain
        if best is None:
            break
        _, chain = best
        points = fragments[chain].reshape(-1, 2)
        point_weights = np.repeat(scores[chain], 2)
        origin = np.average(points, axis=0, weights=point_weights)
        centered = points - origin
        _, _, bases = np.linalg.svd(centered * np.sqrt(point_weights[:, None]), full_matrices=False)
        axis = bases[0]
        t = centered @ axis
        ends = origin + np.array([t.min(), t.max()])[:, None] * axis
        ends = ends[np.argsort(ends[:, 0], kind='stable')]
        ends = np.clip(ends, [0, 0], [shape[1] - 1, shape[0] - 1])
        output.append({'class': 0, 'confidence': float(np.mean(scores[chain])),
                       'endpoints': ends.tolist(), 'fragment_count': len(chain)})
        chains.append(chain)
        active[chain] = False
    return output, active, chains


def semantic_surfaces(mask):
    """Split DepthOut semantic labels into surface regions for existing grouping."""
    regions = []
    for kind in (1, 2):
        total, labels, stats, _ = cv2.connectedComponentsWithStats((mask == kind).astype('uint8'))
        for index in range(1, total):
            if stats[index, cv2.CC_STAT_AREA] >= max(24, mask.size * .001):
                regions.append((kind - 1, 1.0, labels == index))
    return regions


def short_edge_evidence(line, primary, risers, shape):
    edge = convex_boundary(line, shape)
    if edge is None or len(edge.x) < 14:
        return None
    diagonal = np.hypot(*shape)
    x, y = np.mean(line['endpoints'], axis=0)
    slope = np.diff(edge.y[[0, -1]])[0] / (edge.x[-1] - edge.x[0])
    offsets = []
    for known in primary:
        (left, top), (right, bottom) = known['endpoints']
        if right <= left or not left - .13 * diagonal <= x <= right + .13 * diagonal:
            continue
        neighbor_slope = (bottom - top) / (right - left)
        if abs(np.arctan(slope) - np.arctan(neighbor_slope)) > np.deg2rad(19):
            continue
        offsets.append(top + (x - left) * neighbor_slope - y)
    if any(abs(offset) <= .018 * diagonal for offset in offsets):
        return None

    for riser in risers:
        _, a, b = np.intersect1d(edge.x, riser.x, assume_unique=True, return_indices=True)
        if np.count_nonzero(np.abs(edge.y[a] - riser.y[b]) <= .014 * diagonal) >= .60 * len(edge.x):
            return 'riser_upper_boundary'

    # With weak face evidence, require a strong chain and two other levels
    # whose extrapolated spacing supports a missing internal or terminal step.
    if line['confidence'] < .78 or len(offsets) < 2:
        return None
    distances = sorted(offsets, key=abs)
    first, second = distances[:2]
    adjacent = min(abs(first), abs(second))
    expected = abs(second - first) if first * second > 0 else max(abs(first), abs(second))
    if .035 * diagonal <= adjacent and .60 * expected <= adjacent <= 1.60 * expected:
        return 'local_level_spacing'
    return None


def dense_spacing(line, known, shape):
    """Check for an internal gap or a short continuation of a resolved dense sequence."""
    diagonal = np.hypot(*shape)
    (x0, y0), (x1, y1) = line['endpoints']
    x, y = (x0 + x1) / 2, (y0 + y1) / 2
    angle = np.arctan2(y1 - y0, x1 - x0)
    offsets = []
    for other in known:
        (left, top), (right, bottom) = other['endpoints']
        overlap = min(x1, right) - max(x0, left)
        if right <= left or overlap < .50 * (x1 - x0):
            continue
        if abs(angle - np.arctan2(bottom - top, right - left)) > np.deg2rad(12):
            continue
        offsets.append(top + (x - left) * (bottom - top) / (right - left) - y)
    if len(offsets) < 3 or min(map(abs, offsets)) < .0075 * diagonal:
        return False
    above = sorted(-d for d in offsets if d < 0)
    below = sorted(d for d in offsets if d > 0)
    limit = .036 * diagonal
    if above and below:
        near, far = sorted((above[0], below[0]))
        if far > min(limit, 1.6 * near):
            return False
        return any(.60 * near <= side[1] - side[0] <= 1.65 * far
                   for side in (above, below) if len(side) >= 2)
    first, second, third = (above or below)[:3]
    gap, next_gap = second - first, third - second
    return (.60 * gap <= first <= min(limit, 1.42 * gap)
            and .46 * next_gap <= gap <= 1.45 * next_gap)


def split_compound_edges(lines, chains, fragments, scores, shape):
    """Refit a chain that consumed fragments from two nearby parallel tracks."""
    diagonal = np.hypot(*shape)
    ends = np.array([line['endpoints'] for line in lines], dtype=float).reshape(-1, 2, 2)
    widths = ends[:, 1, 0] - ends[:, 0, 0]
    slopes = (ends[:, 1, 1] - ends[:, 0, 1]) / np.maximum(widths, 1)
    replaced, diagnostics = [], []
    used = np.zeros(len(fragments), dtype=bool)
    for index, (line, chain) in enumerate(zip(lines, chains, strict=True)):
        replaced.append(line)
        if len(chain) < 7 or not .10 * diagonal <= widths[index] <= .23 * diagonal or abs(slopes[index]) > .23:
            continue
        left, right = ends[index, :, 0]
        x = (left + right) / 2
        heights = ends[:, 0, 1] + (x - ends[:, 0, 0]) * slopes
        local = ((ends[:, 0, 0] <= left + .20 * widths[index]) &
                 (ends[:, 1, 0] >= right - .20 * widths[index]) &
                 (np.abs(slopes - slopes[index]) < .14))
        offsets = (heights - heights[index])[local]
        above, below = -offsets[offsets < 0], offsets[offsets > 0]
        if len(offsets) < 5 or not len(above) or not len(below):
            continue
        upper, lower = above.min(), below.min()
        if not .025 * diagonal <= upper + lower <= .045 * diagonal:
            continue
        x_coords = fragments[:, :, 0]
        y_fit = heights[index] + (x_coords - x) * slopes[index]
        potential = np.flatnonzero(~used & (np.abs(fragments[:, :, 1] - y_fit).max(axis=1) <= .012 * diagonal)
                                   & (x_coords.min(axis=1) >= left - .018 * diagonal)
                                   & (x_coords.max(axis=1) <= right + .018 * diagonal))
        tracks, _, track_chains = _stitch(fragments[potential], scores[potential], shape, profile='split')
        tracks = [(track, potential[support]) for track, support in zip(tracks, track_chains, strict=True)
                  if track['confidence'] >= .48 and len(np.intersect1d(potential[support], chain)) >= 3]
        if len(tracks) != 2:
            continue
        tracks.sort(key=lambda item: np.mean(item[0]['endpoints'], axis=0)[1])
        first, second = [np.array(t['endpoints']) for t, _ in tracks]
        spans = [edge[1, 0] - edge[0, 0] for edge in (first, second)]
        lo, hi = max(first[0, 0], second[0, 0]), min(first[1, 0], second[1, 0])
        if hi - lo < .74 * min(spans) or min(spans) < .67 * widths[index]:
            continue
        samples = np.linspace(lo, hi, 5)
        top, bottom = [np.interp(samples, edge[:, 0], edge[:, 1]) for edge in (first, second)]
        gap = bottom - top
        pitch = (upper + lower) / 3
        if (gap.min() < max(.005 * diagonal, .43 * pitch) or gap.max() > 1.38 * pitch
                or np.ptp(gap) > .58 * pitch):
            continue
        main_y = heights[index] + (samples[2] - x) * slopes[index]
        distances = np.array([top[2] - (heights[index] - upper),
                              heights[index] + lower - bottom[2]])
        if not top[2] < main_y < bottom[2] or np.min(distances) < .50 * pitch or np.max(distances) > 1.70 * pitch:
            continue
        pair = [{**track, 'split_track_pair': index} for track, _ in tracks]
        for _, support in tracks:
            used[support] = True
        replaced[-1:] = pair
        diagnostics.append({'original': line, 'tracks': pair})
    return replaced, diagnostics, used


def tread_texture_lines(lines, image, mask):
    """Find the trailing edge of fine texture bands and weak seams next to them."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    texture = cv2.GaussianBlur(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 1.15)), (0, 0), 3.2)
    h, w = gray.shape
    diagonal = np.hypot(h, w)
    profiles = {}
    for idx, line in enumerate(lines):
        (x0, y0), (x1, y1) = line['endpoints']
        width = x1 - x0
        if width < .11 * diagonal or abs(y1 - y0) > .30 * width:
            continue
        x = np.rint(np.linspace(x0 + .12 * width, x1 - .12 * width, 72)).astype(int)
        y = np.rint(np.interp(x, [x0, x1], [y0, y1])).astype(int)
        valid = (y >= 10) & (y < h - 10) & (x >= 0) & (x < w)
        x, y = x[valid], y[valid]
        if len(x) < 36:
            continue
        above, below = np.median(texture[y - 9, x]), np.median(texture[y + 9, x])
        contrast = np.abs(gray[y - 3, x] - gray[y + 3, x])
        same_surface = mask[y - 6, x] == mask[y + 6, x]
        profiles[idx] = (x0, x1, y0, (y1 - y0) / width, above, below,
                         np.median(contrast) < 9 and (contrast >= 12).mean() < .36 and same_surface.mean() >= .80)

    def separation(a, b, minimum_overlap):
        left, right = max(a[0], b[0]), min(a[1], b[1])
        if right - left < minimum_overlap or abs(a[3] - b[3]) > .12:
            return None
        x = (left + right) / 2
        return b[2] + b[3] * (x - b[0]) - a[2] - a[3] * (x - a[0])

    rejected = {}
    for top in profiles.values():
        if top[5] < max(8, 2.7 * top[4]):
            continue
        for trailing_id, trailing in profiles.items():
            if trailing[4] < max(8, 2.7 * trailing[5]):
                continue
            width = min(top[1] - top[0], trailing[1] - trailing[0])
            band = separation(top, trailing, .65 * width)
            if width < .22 * diagonal or band is None or not .026 * diagonal <= band <= .105 * diagonal:
                continue
            for next_edge in profiles.values():
                gap = separation(trailing, next_edge, .55 * width)
                if gap is None or not 1.25 * band <= gap <= .27 * diagonal or next_edge[6]:
                    continue
                rejected[trailing_id] = 'trailing_texture_band'
                for i, seam in profiles.items():
                    if not seam[6]:
                        continue
                    before = separation(trailing, seam, .07 * diagonal)
                    after = separation(seam, next_edge, .07 * diagonal)
                    if before is not None and after is not None and min(before, after) >= .023 * diagonal:
                        rejected[i] = 'weak_same_surface_seam'
    return rejected


def predict_steps(fragment_data, mask, *, recover_short_edges=False, recover_dense_edges=False,
                  fragment_confidence=.30, texture_image=None, split_dense_tracks=False):
    fragments, scores = fragment_data
    strong = np.flatnonzero(scores >= fragment_confidence) if recover_dense_edges else np.arange(len(scores))
    lines, remaining, primary_chains = _stitch(fragments[strong], scores[strong], mask.shape)
    surfaces = semantic_surfaces(mask)
    extra = {}
    options = {}
    if split_dense_tracks:
        lines, splits, used = split_compound_edges(lines, primary_chains, fragments[strong], scores[strong], mask.shape)
        remaining &= ~used
        extra['split_track_events'] = splits
    if recover_short_edges:
        unused = strong[remaining]
        candidates, _, chains = _stitch(fragments[unused], scores[unused], mask.shape, profile='short')
        risers = [surface_boundary(region, conf, upper=True) for kind, conf, region in surfaces if kind == 1]
        recovered = []
        for line, chain in zip(candidates, chains):
            reason = short_edge_evidence(line, lines + recovered, risers, mask.shape)
            if reason is not None:
                recovered.append({**line, 'short_edge_recovery': reason})
                # Preserve rejected chains for the dense pass.
                remaining[np.searchsorted(strong, unused[chain])] = False
        lines.extend(recovered)
        extra.update(short_edge_candidate_count=len(candidates), short_edges_recovered=recovered)
        options['short_edge_minimum_length_ratio'] = .09
    if recover_dense_edges:
        residual = scores < fragment_confidence
        residual[strong[remaining]] = True
        candidates, _, chains = _stitch(fragments[residual], scores[residual], mask.shape, profile='dense')
        residual_scores = scores[residual]
        eligible = [line for line, chain in zip(candidates, chains)
                    if np.count_nonzero(residual_scores[chain] >= fragment_confidence) >= 2
                    and line['confidence'] >= max(.25, fragment_confidence)]
        recovered, pending = [], []
        for line in eligible:
            if dense_spacing(line, lines + recovered, mask.shape):
                recovered.append({**line, 'short_edge_recovery': 'dense_level_spacing'})
            else:
                pending.append(line)
        # A newly confirmed terminal level may support one farther candidate.
        if recovered:
            known = lines + recovered
            recovered.extend({**line, 'short_edge_recovery': 'dense_level_spacing'} for line in pending
                             if dense_spacing(line, known, mask.shape))
        lines.extend(recovered)
        extra.update(dense_edge_candidates=len(candidates), dense_edges_recovered=recovered)
        options['short_edge_minimum_length_ratio'] = .09
    count_lines = lines
    if texture_image is not None:
        rejected = tread_texture_lines(lines, texture_image, mask)
        extra['texture_edges_rejected'] = [{**lines[i], 'texture_rejection': reason}
                                           for i, reason in rejected.items()]
        count_lines = [line for i, line in enumerate(lines) if i not in rejected]
    counts = count_grouped_steps(surfaces, count_lines, mask.shape, version='2.4', **options)
    return {'source_model': 'StairNet-DepthOut', 'grid_fragments': len(strong),
            'stitched_edges': lines, **counts, **extra}
