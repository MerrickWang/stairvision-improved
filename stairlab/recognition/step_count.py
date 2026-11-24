"""Estimate visible steps from stair surface masks and convex edges."""

from dataclasses import dataclass

import numpy as np

from stairlab.recognition.structure_postprocess import clean, midpoint


STEP_COUNT_VERSION = '2.1'
PRIMARY_CONFIDENCE = 0.25
SECONDARY_CONFIDENCE = 0.10


@dataclass
class Boundary:
    x: np.ndarray
    y: np.ndarray
    confidence: float


def surface_boundary(mask, confidence, *, upper):
    """Take the upper riser edge or lower tread edge at each visible column."""
    rows, cols = np.nonzero(mask)
    if not len(rows):
        return None
    x, indices = np.unique(cols, return_inverse=True)
    y = np.full(len(x), mask.shape[0] if upper else -1)
    (np.minimum.at if upper else np.maximum.at)(y, indices, rows)
    # The image crop can look like a horizontal surface edge.
    visible = (y > 1) & (y < mask.shape[0] - 2)
    return Boundary(x[visible], y[visible], confidence)


def convex_boundary(line, shape):
    first, second = sorted(line['endpoints'][:2], key=lambda point: point[0])
    left = max(0, int(np.ceil(first[0])))
    right = min(shape[1] - 1, int(np.floor(second[0])))
    if right <= left:
        return None
    x = np.arange(left, right + 1)
    y = np.interp(x, [first[0], second[0]], [first[1], second[1]])
    visible = (y > 1) & (y < shape[0] - 2)
    return Boundary(x[visible], y[visible], float(line['confidence']))


def boundary_agreement(first, second, tolerance, minimum_width):
    common, a, b = np.intersect1d(first.x, second.x, assume_unique=True, return_indices=True)
    required = max(minimum_width, 0.35 * min(len(first.x), len(second.x)))
    if len(common) < required:
        return 0.0
    near = np.abs(first.y[a] - second.y[b]) <= tolerance
    if np.count_nonzero(near) < required or near.mean() < 0.6:
        return 0.0
    return float(near.mean() * len(common) / min(len(first.x), len(second.x)))


def edge_geometry(first, second, minimum_width):
    """Compare two edges over their shared horizontal footprint."""
    if first is None or second is None:
        return None
    common, a, b = np.intersect1d(first.x, second.x, assume_unique=True, return_indices=True)
    if len(common) < max(minimum_width, 0.55 * min(len(first.x), len(second.x))):
        return None
    p, q = first.y[a], second.y[b]
    span = common[-1] - common[0]
    slope_p, slope_q = (p[-1] - p[0]) / span, (q[-1] - q[0]) / span
    angle = abs(np.arctan(slope_p) - np.arctan(slope_q))
    normal_distance = np.max(np.abs(p - q)) / np.hypot(1, (slope_p + slope_q) / 2)
    return float(angle), float(normal_distance)


def snap_to_surface(line, edge, surface, tolerance, minimum_width):
    """Fit the agreed part of a surface and adjust its edge without changing x."""
    shared, a, b = np.intersect1d(edge.x, surface.x, assume_unique=True, return_indices=True)
    near = np.abs(edge.y[a] - surface.y[b]) <= tolerance
    if near.sum() < max(minimum_width, 0.6 * len(edge.x)):
        return line
    x, y = shared[near], surface.y[b][near]
    slope, offset = np.polyfit(x, y, 1)
    if np.median(np.abs(y - (slope * x + offset))) > tolerance / 3:
        return line
    endpoints = [[point[0], float(slope * point[0] + offset), *point[2:]]
                 for point in line['endpoints']]
    if any(abs(new[1] - old[1]) > tolerance
           for new, old in zip(endpoints, line['endpoints'])):
        return line
    return {**line, 'endpoints': endpoints}


def local_gaps(edges, minimum_width, height):
    """Locate internal gaps consistent with one missing edge in a local sequence."""
    ordered = sorted((edge for edge in edges if edge is not None and len(edge.x) >= minimum_width),
                     key=lambda edge: np.median(edge.y))
    for index, (upper, lower) in enumerate(zip(ordered, ordered[1:])):
        shared = np.intersect1d(upper.x, lower.x, assume_unique=True)
        if len(shared) < max(minimum_width, 0.55 * min(len(upper.x), len(lower.x))):
            continue
        x = float(np.median(shared))
        upper_y, lower_y = (np.interp(x, edge.x, edge.y) for edge in (upper, lower))
        gap = lower_y - upper_y
        if gap < height * 0.05:
            continue
        neighbors = []
        for neighbor, anchor, sign in ((index - 1, upper_y, -1), (index + 2, lower_y, 1)):
            if 0 <= neighbor < len(ordered):
                edge = ordered[neighbor]
                if edge.x[0] <= x <= edge.x[-1]:
                    spacing = sign * (np.interp(x, edge.x, edge.y) - anchor)
                    if spacing >= height * 0.015:
                        neighbors.append(spacing)
        if neighbors and max(neighbors) <= 1.5 * min(neighbors):
            spacing = np.median(neighbors)
            if 1.65 <= gap / spacing <= 2.45:
                y = (np.interp(shared, upper.x, upper.y) + np.interp(shared, lower.x, lower.y)) / 2
                yield Boundary(shared, y, 0.0), max(3.0, 0.22 * gap)


def separate_split_tracks(first, second):
    pair = first.get('split_track_pair')
    return pair is not None and pair == second.get('split_track_pair')


def prepare_edge_groups(surfaces, lines, shape, *, short_edge_minimum_length_ratio=None):
    """Prepare the structured baseline and the 2.1 geometry-filtered candidates."""
    height, width = shape
    minimum_width = max(4, round(width * 0.04))
    tolerance = max(3, round(height * 0.02))
    strict_gap = max(2, height * 0.009)
    boundaries = []
    for kind, confidence, mask in surfaces:
        boundary = surface_boundary(mask, confidence, upper=kind == 1)
        if boundary is not None and len(boundary.x) >= minimum_width:
            boundaries.append((kind, boundary))

    def support(edge):
        if edge is None or len(edge.x) < minimum_width:
            return {}
        matches = {}
        for index, (kind, surface) in enumerate(boundaries):
            score = boundary_agreement(edge, surface, tolerance, minimum_width)
            if score > matches.get(kind, (0.0, 0))[0]:
                matches[kind] = (score, index)
        return matches

    high_lines = [line for line in lines if line['confidence'] >= PRIMARY_CONFIDENCE]
    structured = clean({'lines': high_lines}, short_edge_minimum_length_ratio=short_edge_minimum_length_ratio)
    baseline = [line for line in structured['lines_structured'] if line['class'] == 0]
    convex = sorted(baseline, key=lambda line: line['confidence'], reverse=True)
    groups = []
    for line in convex:
        edge = convex_boundary(line, shape)
        evidence = support(edge)
        for group in groups:
            if separate_split_tracks(line, group['line']):
                continue
            geometry = edge_geometry(edge, group['edge'], minimum_width)
            if geometry is None:
                continue
            angle, distance = geometry
            shared_surface = any(kind in group['support'] and group['support'][kind][1] == match[1]
                                 for kind, match in evidence.items())
            if angle <= np.deg2rad(7) and (distance <= strict_gap or
                                            (distance <= tolerance and shared_surface)):
                group['merged_edges'] += 1
                break
        else:
            groups.append({'line': line, 'edge': edge, 'support': evidence, 'merged_edges': 0})

    return baseline, groups, boundaries, support, minimum_width, tolerance


def count_visible_steps(surfaces, lines, shape):
    """Return 2.1 candidates and the intermediate edge sets for comparison."""
    baseline, groups, boundaries, support, minimum_width, tolerance = prepare_edge_groups(
        surfaces, lines, shape
    )
    height, _ = shape
    refined = []
    for group in groups:
        line = group['line']
        if group['support']:
            _, index = max(group['support'].values())
            line = snap_to_surface(line, group['edge'], boundaries[index][1], tolerance, minimum_width)
        refined.append(line)

    low_lines = []
    for line in lines:
        if (line['class'] == 0 and len(line['endpoints']) >= 2
                and SECONDARY_CONFIDENCE <= line['confidence'] < PRIMARY_CONFIDENCE):
            edge = convex_boundary(line, shape)
            evidence = support(edge)
            if evidence:
                low_lines.append((line, edge, evidence))

    occupied = [group['edge'] for group in groups]
    recovered = []
    unresolved = []
    for target, gap_tolerance in local_gaps(occupied.copy(), minimum_width, height):
        matches = []
        for index, (line, edge, evidence) in enumerate(low_lines):
            score = boundary_agreement(edge, target, gap_tolerance, minimum_width)
            if score < 0.6:
                continue
            if any(geometry is not None and geometry[0] <= np.deg2rad(7) and geometry[1] <= tolerance
                   for geometry in (edge_geometry(edge, known, minimum_width) for known in occupied)):
                continue
            matches.append((score, line['confidence'], index))
        if not matches:
            unresolved.append({'x': float(np.median(target.x)), 'y': float(np.median(target.y))})
            continue
        _, _, index = max(matches)
        line, edge, evidence = low_lines.pop(index)
        occupied.append(edge)
        recovered.append({'line': line, 'support': evidence, 'merged_edges': 0})

    steps = []
    for group, line, recovered_step in (
        [(group, line, False) for group, line in zip(groups, refined)] +
        [(group, group['line'], True) for group in recovered]
    ):
        x, y = midpoint(line)
        support_names = ['tread' if kind == 0 else 'riser' for kind in sorted(group['support'])]
        steps.append({'x': float(x), 'y': float(y), 'evidence': ['convex_edge', *support_names],
                      'confidence': 'low' if recovered_step or not support_names else 'high',
                      'merged_edges': group['merged_edges'], 'recovered': recovered_step})

    steps.sort(key=lambda step: (step['y'], step['x']))
    return {'visible_step_count': len(steps), 'steps': steps, 'step_count_version': STEP_COUNT_VERSION,
            'structured_convex_count': len(baseline), 'suspected_gaps': unresolved,
            'edge_stages': {'structured_convex': baseline, 'deduplicated': [group['line'] for group in groups],
                            'refined': refined, 'fused': refined + [group['line'] for group in recovered]}}


def extract_surfaces(result, shape):
    """Read retina masks in original image coordinates."""
    if result.masks is None:
        return []
    masks = result.masks.data.cpu().numpy()
    if masks.shape[1:] != tuple(shape):
        raise ValueError('Segmentation masks must be predicted with retina_masks=True')
    return [(int(cls), float(conf), mask > 0.5) for cls, conf, mask in
            zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(), masks) if int(cls) in (0, 1)]


def extract_lines(result):
    if result.keypoints is None:
        return []
    return [{'class': int(cls), 'confidence': float(conf), 'endpoints': points}
            for cls, conf, points in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist(),
                                         result.keypoints.data.tolist())]
