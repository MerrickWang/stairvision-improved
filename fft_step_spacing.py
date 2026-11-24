"""Experimental 3.5: recover supported short edges in locally periodic flights."""

import numpy as np
from scipy.signal import find_peaks

from depthout_refine import coverage
from step_grouping import count_grouped_steps
from depthout_stitch import semantic_surfaces


def fft_spacing(signal, minimum, maximum):
    """Estimate the first repeat interval, using zero padding for linear correlation."""
    signal = np.asarray(signal, dtype=float)
    spectrum = np.fft.rfft(signal, n=2 * len(signal))
    correlation = np.fft.irfft(np.abs(spectrum) ** 2, n=2 * len(signal))[:len(signal)]
    if correlation[0] <= 0:
        return None
    autocorrelation = correlation / correlation[0]
    peaks = [p for p in find_peaks(autocorrelation)[0]
             if minimum <= p <= maximum and autocorrelation[p] >= .26]
    if not peaks:
        return None
    best = max(autocorrelation[p] for p in peaks)
    lag = min(p for p in peaks if autocorrelation[p] >= .76 * best)
    return int(lag), float(autocorrelation[lag])


def recover_fft_edges(data, mask, baseline, *, allow_jitter=False):
    """Keep original lines and fit only fragment-supported candidates near repeated levels."""
    fragments, scores = data
    height, width = mask.shape
    rejected = {tuple(map(tuple, line['endpoints'])) for line in baseline['texture_edges_rejected']}
    existing = [line for line in baseline['stitched_edges'] if tuple(map(tuple, line['endpoints'])) not in rejected]
    anchors = np.asarray([line['endpoints'] for line in baseline['edge_stages']['grouped']], dtype=float).reshape(-1, 2, 2)
    dx = np.diff(anchors[:, :, 0], axis=1).ravel()
    if np.count_nonzero(dx >= .14 * width) < 4 or len(fragments) < 8:
        return baseline, [], []
    broad = anchors[dx >= .14 * width]
    slope = float(np.median(np.diff(broad[:, :, 1], axis=1).ravel() / np.diff(broad[:, :, 0], axis=1).ravel()))
    if abs(slope) > .35:
        return baseline, [], []
    x = float(np.median(broad[:, :, 0].mean(axis=1)))
    near = np.abs(fragments[:, :, 0].mean(axis=1) - x) <= .20 * width
    frags, weights = fragments[near], scores[near]
    offsets = frags[:, :, 1] - slope * (frags[:, :, 0] - x)
    y = offsets.mean(axis=1)
    known = broad[:, :, 1].mean(axis=1) - slope * (broad[:, :, 0].mean(axis=1) - x)
    votes = np.bincount(np.clip(np.rint(y).astype(int), 0, height - 1), weights=weights, minlength=height)
    prominence = max(.65, (.09 if allow_jitter else .13) * votes.max())
    peaks = find_peaks(votes, prominence=prominence, distance=3)[0]
    added, windows = [], []
    for start in range(0, height - 79, 20 if allow_jitter else 40):
        local = peaks[(peaks >= start) & (peaks < start + 80)]
        if len(local) < 5:
            continue
        signal = np.zeros(80)
        signal[local - start] = 1
        if allow_jitter:
            signal = np.convolve(signal, [.22, .56, .22], mode='same')
        period = fft_spacing(signal, 5, 24)
        if period is None:
            continue
        lag, strength = period
        windows.append({'start': start, 'lag_px': lag, 'correlation': strength})
        for p in local:
            if np.min(np.abs(known - p)) <= max(2.8, .40 * lag):
                continue
            if min(np.min(abs(known - (p - lag))), np.min(abs(known - (p + lag)))) > .28 * lag:
                continue
            if any(np.min(abs(local - (p + sign * lag))) > .28 * lag for sign in (-1, 1)):
                continue
            ids = np.flatnonzero(np.max(abs(offsets - p), axis=1) <= min(2.8, .31 * lag))
            if len(ids) < 4:
                continue
            intervals = np.sort(frags[ids, :, 0], axis=1)
            left, right = float(intervals.min()), float(intervals.max())
            if right - left < .105 * width or coverage(intervals - left, right - left) < .65:
                continue
            line = {'class': 0, 'confidence': float(weights[ids].mean()), 'short_edge_recovery': 'fft_spacing',
                    'endpoints': [[left, float(p + slope * (left - x))], [right, float(p + slope * (right - x))]],
                    'fragment_count': len(ids)}
            if all(0 <= pt[1] < height for pt in line['endpoints']):
                added.append(line)
                known = np.append(known, p)
    if not added:
        return baseline, added, windows
    counts = count_grouped_steps(semantic_surfaces(mask), existing + added, mask.shape,
                                 version='2.4', short_edge_minimum_length_ratio=.09)
    return {**baseline, **counts, 'stitched_edges': baseline['stitched_edges'] + added}, added, windows
