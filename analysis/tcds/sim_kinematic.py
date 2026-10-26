#!/usr/bin/env python3
"""
Kinematic stand-in for the Gazebo strange-situation experiment, driven by the decoupled
controller (model.simulate_decoupled). Used to generate many runs for the inverse problem.

It mirrors ros/Attachment-Model: arena 2.61 m centred on the caregiver box (0.325, 0.325),
box 0.3 m, caregiver on the box, child spawned at (1.0, 0.325) facing the box. Behaviours
follow behaviors.py at 10 Hz: exploration at 0.2 m/s with a random turn rate re-drawn
during even seconds and in-place turning until the next multiple of 5 s when the sonar
reads < 0.105 m; approach turns towards the tags (0.1 m/s, +/-0.2 rad/s), searches by
rotating at 0.3 rad/s when no tag is visible and stops at 0.2 m. The child calls while it
approaches and sees a tag; the caregiver calls while its need state is in approach.
The camera model (field of view, range) is a simplification of MiRo's two cameras.
"""
import numpy as np

import model

ARENA_CENTRE = np.array([0.325, 0.325])
ARENA_SIZE = 2.61
BOX_HALF = 0.15
ROBOT_R = 0.08
SPAWN = (1.0, 0.325, 3.14)
CAREGIVER = np.array([0.375, 0.325])
DT = 0.1
EXPLORE_SPEED = 0.12             # effective exploration speed (m/s) matched to Gazebo (commanded 0.2 m/s)
FOV = np.radians((-15, 60))      # left eye covers bearings -15..60 deg, right eye mirrored
TAG_RANGE = 1.9
TAG_MIN = 0.5                    # closer than this to the box centre the tags leave the cameras' view


def needs_10hz(eAv, eAm, seconds, initial=model.INITIAL):
    """Decoupled-controller needs sampled at 10 Hz (initial: one start for all runs or one per run).

    As in the ROS controller, a run that overflows keeps publishing +/-inf until the first NaN,
    after which the controller thread stops and the last published value persists.
    """
    ticks = int(seconds * model.TICK_HZ)
    with np.errstate(all="ignore"):
        d = model.simulate_decoupled(eAv, eAm, ticks, initial=initial, guard=False)
    view = model.own_view(d)[2::3]                      # 3 controller ticks per 10 Hz sample
    for t in range(1, len(view)):
        bad = np.isnan(view[t])
        view[t][bad] = view[t - 1][bad]
    return view                                         # [T, N, 4]


def _ray_to_walls(pos, heading):
    """Distance from pos along heading to the arena wall or the box (sonar)."""
    lo, hi = ARENA_CENTRE - ARENA_SIZE / 2, ARENA_CENTRE + ARENA_SIZE / 2
    d = np.array([np.cos(heading), np.sin(heading)])
    best = np.inf
    for axis in (0, 1):
        if abs(d[axis]) > 1e-9:
            for wall in (lo[axis], hi[axis]):
                t = (wall - pos[axis]) / d[axis]
                if t > 0:
                    best = min(best, t)
    # box as an axis-aligned square (slab method)
    blo, bhi = ARENA_CENTRE - BOX_HALF, ARENA_CENTRE + BOX_HALF
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (blo - pos) / d, (bhi - pos) / d
    tmin, tmax = np.nanmax(np.minimum(t1, t2)), np.nanmin(np.maximum(t1, t2))
    if tmax >= max(tmin, 0):
        best = min(best, max(tmin, 0))
    return best - ROBOT_R


def _tags_seen(pos, heading):
    vec = ARENA_CENTRE - pos
    dist = np.hypot(*vec)
    if dist > TAG_RANGE or dist < TAG_MIN:
        return False, False
    bearing = (np.arctan2(vec[1], vec[0]) - heading + np.pi) % (2 * np.pi) - np.pi
    left = FOV[0] <= bearing <= FOV[1]
    right = -FOV[1] <= bearing <= -FOV[0]
    return left, right


def _keep_inside(pos):
    lo = ARENA_CENTRE - ARENA_SIZE / 2 + ROBOT_R
    hi = ARENA_CENTRE + ARENA_SIZE / 2 - ROBOT_R
    pos = np.clip(pos, lo, hi)
    rel = pos - ARENA_CENTRE
    if np.all(np.abs(rel) < BOX_HALF + ROBOT_R):        # push out of the box along the shallow axis
        axis = np.argmax(np.abs(rel))
        rel[axis] = np.sign(rel[axis] or 1.0) * (BOX_HALF + ROBOT_R)
        pos = ARENA_CENTRE + rel
    return pos


def run_behaviour(needs, seed=0):
    """Child path and calls for one run. needs: [T, 4] at 10 Hz. Returns dict of [T] arrays."""
    rng = np.random.default_rng(seed)
    T = len(needs)
    pos = np.array(SPAWN[:2], float)
    heading = SPAWN[2]
    turn_rate, spin = 0.0, rng.choice([1, -1]) * np.pi / 3
    exploring = True
    out = {k: np.zeros(T) for k in ("x", "y", "call_c", "call_p", "approach_c")}
    for t in range(T):
        sec = int(t * DT)
        approach = needs[t, 0] > 0
        call_c = False
        if approach:
            left, right = _tags_seen(pos, heading)
            sonar = _ray_to_walls(pos, heading)
            if left and not right:
                v, w = 0.1, 0.2
            elif right and not left:
                v, w = 0.1, -0.2
            elif left and right:
                v, w = (0.0 if sonar < 0.2 else 0.1), 0.0
            else:
                v, w = 0.0, 0.3
            call_c = left or right
        else:
            sonar = _ray_to_walls(pos, heading)
            if not exploring and sec % 5 == 0:
                exploring = True
            turning = (not exploring) or (sonar < 0.105)
            if exploring and sonar < 0.105:
                exploring = False
            if not turning:
                if sec % 2 == 0:
                    turn_rate = rng.uniform(-np.pi / 2, np.pi / 2)
                v, w = EXPLORE_SPEED, turn_rate
            else:
                if sec % 5 == 0:
                    spin = rng.choice([1, -1]) * np.pi / 3
                v, w = 0.0, 2 * spin
        heading = heading + w * DT
        step = v * DT * np.array([np.cos(heading), np.sin(heading)])
        if _ray_to_walls(pos, heading) < np.hypot(*step):
            step = 0.0 * step                                 # blocked: wheels slip in place
        pos = _keep_inside(pos + step)
        out["x"][t], out["y"][t] = pos
        out["call_c"][t] = float(call_c)
        out["call_p"][t] = float(needs[t, 2] > 0)
        out["approach_c"][t] = float(approach)
    return out


def features(child_xy, parent_xy, call_c, call_p, dt=DT, arena=ARENA_SIZE):
    """Behavioural features available to an outside observer."""
    dist = np.hypot(*(child_xy - parent_xy).T)
    half = arena / 4.0
    near = np.mean(np.all(np.abs(child_xy - parent_xy) <= half, axis=1))
    minutes = len(dist) * dt / 60.0

    def bouts(v):
        v = np.asarray(v) > 0
        return (np.count_nonzero(v[1:] & ~v[:-1]) + (1 if v[0] else 0)) / minutes

    smooth = np.convolve(dist - dist.mean(), np.ones(20) / 20, mode="same")
    crossings = np.count_nonzero(np.diff(np.sign(smooth)) != 0) / (2 * minutes)
    return np.array([near, dist.mean(), dist.std(), np.mean(call_c), bouts(call_c),
                     np.mean(call_p), bouts(call_p), crossings])


FEATURE_NAMES = ["time near caregiver", "mean distance", "distance s.d.", "child calling",
                 "child call bouts/min", "caregiver calling", "caregiver call bouts/min",
                 "distance cycles/min"]
