"""Group partial and shifted convex edges into visible steps."""

from dataclasses import dataclass
from itertools import product

import numpy as np

from step_count import boundary_agreement, midpoint, prepare_edge_groups, separate_split_tracks


STEP_COUNT_VERSION = '2.4'


@dataclass
class Candidate:
    group: dict
    endpoints: np.ndarray
    angle: float
    supports: dict
    nearby: dict

    @property
    def profile(self):
        return self.group['edge']


def angle_distance(first, second):
    difference = abs(first - second)
    return min(difference, np.pi - difference)


def shared_surface(first, second):
    return any(index in second.nearby.get(kind, ())
               for kind, indices in first.nearby.items() for index in indices)


def conflicting_surfaces(first, second):
    return any(index != second.supports[kind][1] for kind, (_, index) in first.supports.items()
               if kind in second.supports)


def separated_level_distances(first, second, candidates, x, distances, height):
    low, high = sorted(np.interp(x, edge.profile.x, edge.profile.y) for edge in (first, second))
    scale = max(height * 0.018, 1.6 * distances)
    above, below = [], []
    for other in candidates:
        edge = other.profile
        if edge is None or not len(edge.x) or not edge.x[0] <= x <= edge.x[-1]:
            continue
        if angle_distance(other.angle, first.angle) > np.deg2rad(15):
            continue
        y = np.interp(x, edge.x, edge.y)
        if low - y >= scale:
            above.append(low - y)
        if y - high >= scale:
            below.append(y - high)
    return [min(side) for side in (above, below) if side]


def same_step(first, second, candidates, shape, minimum_width):
    if separate_split_tracks(first.group['line'], second.group['line']) or conflicting_surfaces(first, second):
        return False
    height, width = shape
    angle = angle_distance(first.angle, second.angle)
    same_surface = shared_surface(first, second)
    p, q = first.profile, second.profile
    overlap = np.intersect1d(p.x, q.x, assume_unique=True) if p is not None and q is not None else []
    min_span = min(len(p.x), len(q.x)) if p is not None and q is not None else 0

    if len(overlap) >= max(minimum_width, 0.35 * min_span):
        if angle > np.deg2rad(15):
            return False
        xs = np.quantile(overlap, [0.15, 0.5, 0.85])
        separation = np.abs(np.interp(xs, p.x, p.y) - np.interp(xs, q.x, q.y))
        gap = float(np.median(separation))
        if separation.max() > height * 0.042:
            return False
        if gap <= height * 0.012 and angle <= np.deg2rad(8):
            return True
        neighbors = separated_level_distances(first, second, candidates, xs[1], gap, height)
        if neighbors and gap > 0.38 * min(neighbors):
            return False
        return (same_surface and gap <= height * 0.033) or (
            len(neighbors) == 2 and gap <= height * 0.023 and angle <= np.deg2rad(10)
        )

    # A change in direction at a joint can be a corner of the same step.
    # Require the surfaces to agree and the segments to meet near their ends.
    if not same_surface or angle > np.deg2rad(75) or len(overlap) > max(minimum_width, 0.25 * min_span):
        return False
    joints = [(np.linalg.norm(a - b), i, j)
              for (i, a), (j, b) in product(enumerate(first.endpoints), enumerate(second.endpoints))]
    distance, i, j = min(joints)
    if distance > 0.038 * np.hypot(height, width):
        return False
    away_a = first.endpoints[1 - i] - first.endpoints[i]
    away_b = second.endpoints[1 - j] - second.endpoints[j]
    return float(np.dot(away_a, away_b)) <= 0


def overlap_columns(first, second):
    p, q = first.profile, second.profile
    if p is None or q is None:
        return np.empty(0)
    return np.intersect1d(p.x, q.x, assume_unique=True)


def regular_pair(first, second, columns, minimum_width):
    if first.profile is None or second.profile is None:
        return False
    return len(columns) >= max(minimum_width, 0.35 * min(len(first.profile.x), len(second.profile.x)))


def plausible_adjacent_levels(first, second, candidates, columns, height):
    """Protect a pair when the same interval repeats on both sides of it."""
    x = float(np.median(columns))
    low, high = sorted(np.interp(x, edge.profile.x, edge.profile.y) for edge in (first, second))
    gap = high - low
    if gap < max(2, height * 0.005):
        return False
    # Distant levels can be only a few pixels apart. Require the ordering and
    # separation to persist across the shared span before protecting such a pair.
    xs = np.quantile(columns, [0.2, 0.8])
    p, q = first.profile, second.profile
    separation = np.interp(xs, p.x, p.y) - np.interp(xs, q.x, q.y)
    if separation[0] * separation[1] <= 0 or np.min(np.abs(separation)) < 0.55 * gap:
        return False
    distances = ([], [])
    for other in candidates:
        profile = other.profile
        if profile is None or not len(profile.x) or not profile.x[0] <= x <= profile.x[-1]:
            continue
        if angle_distance(other.angle, first.angle) > np.deg2rad(12):
            continue
        y = np.interp(x, profile.x, profile.y)
        for side, distance in enumerate((low - y, y - high)):
            if distance >= 0.65 * gap:
                distances[side].append(distance)
    return all(side and min(side) <= 1.40 * gap for side in distances)


def corner_proposal(first, second, shape):
    height, width = shape
    angle = angle_distance(first.angle, second.angle)
    if not np.deg2rad(14) <= angle <= np.deg2rad(78):
        return None
    distance, i, j = min((np.linalg.norm(a - b), i, j)
                         for (i, a), (j, b) in product(enumerate(first.endpoints), enumerate(second.endpoints)))
    a, b = first.endpoints[i], second.endpoints[j]
    away_a = first.endpoints[1 - i] - a
    away_b = second.endpoints[1 - j] - b
    if away_a[0] * away_b[0] >= 0 or abs(a[1] - b[1]) > height * 0.037:
        return None
    shared = shared_surface(first, second)
    limit = (0.064 if shared else 0.035) * np.hypot(height, width)
    if distance > limit:
        return None
    if away_a[0] < 0:
        return first, second, a, b, float(distance), shared
    return second, first, b, a, float(distance), shared


def compatible_corners(first, second, shape):
    a, b, x, y, _, _ = first
    c, d, u, v, _, _ = second
    if any(one is two for one in (a, b) for two in (c, d)):
        return False
    if angle_distance(a.angle, c.angle) > np.deg2rad(16) or angle_distance(b.angle, d.angle) > np.deg2rad(16):
        return False
    distance_left, distance_right = x[1] - u[1], y[1] - v[1]
    if distance_left * distance_right <= 0:
        return False
    height, width = shape
    gap = (abs(distance_left) + abs(distance_right)) / 2
    drift = abs((x[0] + y[0] - u[0] - v[0]) / 2)
    return height * 0.018 <= gap <= height * 0.25 and drift <= max(width * 0.055, 0.65 * gap)


def diagnostic(first, second, reason):
    return {'reason': reason,
            'edges': [first.endpoints.tolist(), second.endpoints.tolist()],
            'surfaces': [{kind: instance for kind, (_, instance) in edge.supports.items()}
                         for edge in (first, second)],
            'nearby_surfaces': [{kind: sorted(instances) for kind, instances in edge.nearby.items()}
                                for edge in (first, second)],
            'shared_surface': shared_surface(first, second)}


def pair_corner_groups(groups, shape, minimum_width, events):
    proposals = []
    for i, first in enumerate(groups):
        for j in range(i + 1, len(groups)):
            for a, b in product(first, groups[j]):
                if regular_pair(a, b, overlap_columns(a, b), minimum_width):
                    continue
                proposal = corner_proposal(a, b, shape)
                if proposal:
                    proposals.append((i, j, proposal))

    eligible = []
    for i, j, proposal in proposals:
        if any(separate_split_tracks(a.group['line'], b.group['line']) for a, b in product(groups[i], groups[j])):
            continue
        if proposal[-1] or any(compatible_corners(proposal, other, shape) for _, _, other in proposals):
            eligible.append((i, j, proposal))
    eligible.sort(key=lambda item: (not item[2][-1], item[2][-2]))
    used, selected, paired = set(), [], []
    for i, j, proposal in eligible:
        if i in used or j in used:
            continue
        left, right, a, b, _, shared = proposal
        if any((a[1] - previous[2][1]) * (b[1] - previous[3][1]) < 0
               and angle_distance(left.angle, previous[0].angle) <= np.deg2rad(16)
               and angle_distance(right.angle, previous[1].angle) <= np.deg2rad(16)
               for previous in selected):
            continue
        used.update((i, j))
        selected.append(proposal)
        paired.append(groups[i] + groups[j])
        events.append(diagnostic(left, right, 'corner_shared_surface' if shared else 'corner_repeated_order'))
    return paired + [group for i, group in enumerate(groups) if i not in used]


def same_endpoint_level(first, second, candidates, shape, minimum_width):
    """Check a shared visible endpoint against the level spacing of the flight."""
    height, width = shape
    if separate_split_tracks(first.group['line'], second.group['line']):
        return False
    if conflicting_surfaces(first, second) or angle_distance(first.angle, second.angle) > np.deg2rad(13):
        return False
    columns = overlap_columns(first, second)
    if not regular_pair(first, second, columns, minimum_width):
        return False
    p, q = first.profile, second.profile
    if len(columns) < 0.65 * min(len(p.x), len(q.x)):
        return False
    if plausible_adjacent_levels(first, second, candidates, columns, height):
        return False
    distances = np.linalg.norm(first.endpoints - second.endpoints, axis=1)
    for side in np.argsort(distances):
        a, b = first.endpoints[side], second.endpoints[side]
        if distances[side] > 0.031 * np.hypot(height, width):
            break
        y_gap = abs(a[1] - b[1])
        x = (a[0] + b[0]) / 2
        if not width * 0.025 <= x <= width * 0.975 or y_gap > height * 0.023:
            continue
        low, high = sorted((a[1], b[1]))
        # Ignore other copies at this endpoint and require two distinct levels
        # from the same direction and horizontal footprint.
        clear = max(height * 0.023, 2.25 * y_gap)
        signed_distances = []
        for other in candidates:
            if angle_distance(first.angle, other.angle) > np.deg2rad(15):
                continue
            common = overlap_columns(first, other)
            if not regular_pair(first, other, common, minimum_width):
                continue
            y = other.endpoints[side, 1]
            distance = y - low if y < low else y - high
            if abs(distance) >= clear:
                signed_distances.append(distance)
        levels = []
        for distance in sorted(signed_distances, key=abs):
            if all(abs(distance - known) >= clear for known in levels):
                levels.append(distance)
        if len(levels) < 2 or y_gap > 0.35 * min(map(abs, levels)):
            continue
        xs = np.quantile(columns, [0.15, 0.5, 0.85])
        gap = float(np.max(np.abs(np.interp(xs, p.x, p.y) - np.interp(xs, q.x, q.y))))
        if gap <= min(height * 0.052, 0.45 * min(map(abs, levels))):
            return True
    return False


def merge_endpoint_groups(groups, candidates, shape, minimum_width, events):
    final = []
    for incoming in groups:
        for members in final:
            # Every member must agree. A turn already matched by the corner
            # pass must not absorb another group through only one of its sides.
            pairs = list(product(incoming, members))
            if all(same_endpoint_level(a, b, candidates, shape, minimum_width) for a, b in pairs):
                events.append(diagnostic(incoming[0], members[0], 'shared_endpoint_local_spacing'))
                members.extend(incoming)
                break
        else:
            final.append(incoming.copy())
    return final


def count_grouped_steps(surfaces, lines, shape, *, version=STEP_COUNT_VERSION,
                        short_edge_minimum_length_ratio=None):
    if version not in ('2.2', '2.3', STEP_COUNT_VERSION):
        raise ValueError(f'Unsupported grouping version: {version}')
    baseline, deduped, boundaries, _, minimum_width, tolerance = prepare_edge_groups(
        surfaces, lines, shape, short_edge_minimum_length_ratio=short_edge_minimum_length_ratio)
    candidates = []
    for group in deduped:
        endpoints = np.array(sorted((point[:2] for point in group['line']['endpoints'])))
        delta = endpoints[1] - endpoints[0]
        angle = float(np.arctan2(delta[1], delta[0]))
        nearby = {}
        edge = group['edge']
        if edge is not None and len(edge.x) >= minimum_width:
            for index, (kind, surface) in enumerate(boundaries):
                if boundary_agreement(edge, surface, 1.45 * tolerance, minimum_width) >= 0.60:
                    nearby.setdefault(kind, set()).add(index)
        candidates.append(Candidate(group, endpoints, angle, group['support'], nearby))

    # Build groups around a strong edge and check all members before adding one.
    # This keeps a succession of nearby edges from chaining two levels together.
    groups, events, blocked = [], [], []
    for candidate in candidates:
        for members in groups:
            reasons = []
            for member in members:
                columns = overlap_columns(candidate, member)
                regular = regular_pair(candidate, member, columns, minimum_width)
                if version != '2.2' and not regular:
                    break
                if not same_step(candidate, member, candidates, shape, minimum_width):
                    if version != '2.2' and regular and conflicting_surfaces(candidate, member):
                        geometry = angle_distance(candidate.angle, member.angle)
                        x = float(np.median(columns))
                        gap = abs(np.interp(x, candidate.profile.x, candidate.profile.y)
                                  - np.interp(x, member.profile.x, member.profile.y))
                        if geometry <= np.deg2rad(15) and gap <= shape[0] * .033:
                            blocked.append(diagnostic(candidate, member, 'different_surface_boundaries'))
                    break
                if version != '2.2' and plausible_adjacent_levels(candidate, member, candidates, columns, shape[0]):
                    blocked.append(diagnostic(candidate, member, 'repeated_adjacent_spacing'))
                    break
                reason = 'corner_shared_surface' if not regular else (
                    'overlap_shared_surface' if shared_surface(candidate, member) else 'overlap_geometry')
                reasons.append((member, reason))
            else:
                members.append(candidate)
                events.extend(diagnostic(candidate, member, reason) for member, reason in reasons)
                break
        else:
            groups.append([candidate])

    if version != '2.2':
        groups = pair_corner_groups(groups, shape, minimum_width, events)
    if version == '2.4':
        groups = merge_endpoint_groups(groups, candidates, shape, minimum_width, events)

    angles = np.array([candidate.angle for candidate in candidates])
    direction = 0.5 * np.arctan2(np.sin(2 * angles).sum(), np.cos(2 * angles).sum())
    normal = np.array([-np.sin(direction), np.cos(direction)])
    groups.sort(key=lambda members: float(np.dot(midpoint(members[0].group['line']), normal)))
    steps, edges = [], []
    for members in groups:
        line = members[0].group['line']
        x, y = midpoint(line)
        kinds = {kind for member in members for kind in member.supports}
        steps.append({'x': float(x), 'y': float(y),
                      'evidence': ['convex_edge', *('tread' if kind == 0 else 'riser' for kind in sorted(kinds))],
                      'confidence': 'high' if kinds else 'low', 'recovered': False,
                      'merged_edges': len(members) - 1 + sum(member.group['merged_edges'] for member in members),
                      'edge_segments': [member.group['line']['endpoints'] for member in members]})
        edges.append(line)

    return {'visible_step_count': len(steps), 'steps': steps, 'step_count_version': version,
            'structured_convex_count': len(baseline), 'suspected_gaps': [],
            'merge_events': events, 'blocked_merges': blocked,
            'edge_stages': {'structured_convex': baseline, 'deduplicated': [group['line'] for group in deduped],
                            'grouped': edges, 'fused': edges}}
