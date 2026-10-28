#!/usr/bin/env python3
"""
Behavioural measures computed from recorded runs: time near the caregiver, stuck periods, need
correlation and frequency, vocal bouts (first defined for the npj paper's robot experiment).
"""
from itertools import combinations

import numpy as np
from scipy import stats
from scipy.signal import hilbert

BINS = 50            # heat-map grid of the stuck-period rule, as in the npj paper


def stuck_mask(child_xy, child_state, arena_centre, arena_size, bins=50):
    """Flag samples where the exploring child stays in one place for too long.

    Paper section 6.3: passages when the robot was in the same arena position for a
    duration more than two standard deviations above the mean duration. Position is
    the cell of the heat-map grid. Only exploration is flagged: a child that stops
    next to the caregiver to ask for care is behaving as intended.
    """
    low = np.asarray(arena_centre) - arena_size / 2.0
    cells = np.clip(((child_xy - low) / arena_size * bins).astype(int), 0, bins - 1)
    cell_id = cells[:, 0] * bins + cells[:, 1]

    # Runs of consecutive samples in the same cell
    change = np.flatnonzero(np.diff(cell_id) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(cell_id)]))
    lengths = ends - starts
    threshold = lengths.mean() + 2 * lengths.std()

    mask = np.zeros(len(cell_id), dtype=bool)
    for start, end, length in zip(starts, ends, lengths):
        if length > threshold and child_state[start:end].mean() > 0.5:
            mask[start:end] = True
    return mask


def near_caregiver(child_xy, parent_xy, half_side):
    """True where the child is inside the square of the given half side around the caregiver."""
    offset = np.abs(child_xy - parent_xy)
    return (offset[:, 0] <= half_side) & (offset[:, 1] <= half_side)


def arena_limits(run):
    low = np.asarray(run.arena_centre) - run.arena_size / 2.0
    return low, low + run.arena_size


def half_square(run):
    """Half side of the 'close to the caregiver' square: the central quarter of the arena."""
    return run.arena_size / 4.0


def analyse(run, source):
    """Per-run quantities shared by the figures."""
    child, parent = run.positions(source)
    stuck = stuck_mask(child, run.child_state, run.arena_centre, run.arena_size, BINS)
    near = near_caregiver(child, parent, half_square(run))
    return dict(child=child, parent=parent, distance=run.distance(source), stuck=stuck, near=near)


def spans(mask, t):
    """(start, end) times of the contiguous True stretches of a mask."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(t[s], t[e]) for s, e in zip(starts, ends)]


def block_bootstrap_ci(values, block, n_boot=2000, seed=0):
    """95% interval of the mean of an autocorrelated series, resampling whole blocks."""
    values = np.asarray(values, dtype=float)
    n_blocks = max(1, len(values) // block)
    block_means = values[:n_blocks * block].reshape(n_blocks, -1).mean(axis=1)
    rng = np.random.default_rng(seed)
    resampled = rng.choice(block_means, size=(n_boot, n_blocks)).mean(axis=1)
    return np.percentile(resampled, [2.5, 97.5])


def proximity_tests(counts):
    """Chi-squared test and Bonferroni-corrected pairwise z-tests on time near the caregiver.

    counts: {regime name: (samples near, samples total)}
    Returns (chi2, dof, n, p, [(name_a, name_b, z, p_corrected), ...]).
    As in the paper, every recorded sample counts as one observation.
    """
    names = list(counts)
    table = np.array([[counts[k][0], counts[k][1] - counts[k][0]] for k in names])
    chi2, p, dof, _ = stats.chi2_contingency(table)

    pairs = list(combinations(names, 2))
    pairwise = []
    for a, b in pairs:
        near_a, n_a = counts[a]
        near_b, n_b = counts[b]
        pooled = (near_a + near_b) / (n_a + n_b)
        se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
        z = (near_a / n_a - near_b / n_b) / se if se > 0 else 0.0
        p_pair = min(1.0, 2 * stats.norm.sf(abs(z)) * len(pairs))
        pairwise.append((a, b, z, p_pair))
    return chi2, dof, int(table.sum()), p, pairwise


def need_correlation(run):
    """Correlation coefficient between the child's and the caregiver's needs (paper eq. 6)."""
    if run.diverged or run.child_need.std() == 0 or run.parent_need.std() == 0:
        return np.nan
    return float(np.corrcoef(run.child_need, run.parent_need)[0, 1])


def oscillation_frequency(need, duration):
    """Oscillation frequency (Hz) estimated from the number of zero crossings."""
    if not np.isfinite(need).all():
        return np.nan
    crossings = np.count_nonzero(np.diff(np.sign(need)) != 0)
    return crossings / (2.0 * duration) if duration > 0 else np.nan


def relative_phase(run):
    """Mean phase of the caregiver's need relative to the child's (degrees, -180 to 180)."""
    if run.diverged:
        return np.nan
    phase_c = np.angle(hilbert(run.child_need - run.child_need.mean()))
    phase_p = np.angle(hilbert(run.parent_need - run.parent_need.mean()))
    return float(np.degrees(np.angle(np.mean(np.exp(1j * (phase_p - phase_c))))))


def vocal_bouts(vocal):
    """Number of separate vocalisation bouts in a 0/1 series."""
    vocal = np.asarray(vocal) > 0
    return int(np.count_nonzero(vocal[1:] & ~vocal[:-1]) + (1 if vocal[0] else 0))
