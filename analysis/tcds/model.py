#!/usr/bin/env python3
"""
The attachment controllers re-implemented outside ROS, vectorised over parameter sets.

Three variants (see the repository README):
  coupled   (M1) -- CentralAttachmentController: one integrator holds both agents; the
                    physical and emotional distances are functions of the shared state.
  decoupled (M2) -- IndividualAttachmentController: every agent integrates its own oscillator
                    together with an internal model of the partner, and obtains the two
                    distances from the learned coupling networks, fed with its own state.
  sensed    (M0) -- the Living Machines controller (ros_mommy_daddy/central_control.py):
                    one integrator, distances supplied from outside (odometry, vocal contact).

Integration follows the controllers: RK4, h = 0.01 per control tick, ticks at 30 Hz, with
the distances held constant during a tick.
"""
import numpy as np

from nnweights import load_networks

PARAMS = dict(a=1.0, c=1.0, b=0.7, k0=0.8, delta=0.1, h=0.01)
INITIAL = (-1.0, 1.0, -1.0, 1.0)     # x_c, y_c, x_p, y_p as in the controllers
TICK_HZ = 30.0                       # controller loop rate (rospy.Rate(30))
BLOWUP = 10.0                        # |state| above this counts as divergence

_NETS = None


def networks():
    global _NETS
    if _NETS is None:
        _NETS = load_networks()
    return _NETS


def vector_field(s, eAv, eAm, dp, de, p=PARAMS):
    """Right-hand side used by all controllers. s: [..., 4] = (x_c, y_c, x_p, y_p)."""
    x1, y1, x2, y2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    dx1 = -p["a"] * (4.0 * p["c"] * x1 ** 3 - 2.0 * x1) - y1
    dy1 = p["b"] * x1 - eAm * (y2 + y1 - dp) - eAv * (y2 - y1 - de)
    dx2 = -p["a"] * (4.0 * p["c"] * x2 ** 3 - 2.0 * x2) - y2
    dy2 = p["b"] * x2 - eAm * (y2 + y1 + dp) - eAv * (y2 - y1 - de)
    return np.stack([dx1, dy1, dx2, dy2], axis=-1)


def vector_field_halves(s, eAv, eAm, dp1, de1, dp2, de2, p=PARAMS):
    """As vector_field, with separate couplings for the child half (1) and the caregiver half (2)."""
    x1, y1, x2, y2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    dx1 = -p["a"] * (4.0 * p["c"] * x1 ** 3 - 2.0 * x1) - y1
    dy1 = p["b"] * x1 - eAm * (y2 + y1 - dp1) - eAv * (y2 - y1 - de1)
    dx2 = -p["a"] * (4.0 * p["c"] * x2 ** 3 - 2.0 * x2) - y2
    dy2 = p["b"] * x2 - eAm * (y2 + y1 + dp2) - eAv * (y2 - y1 - de2)
    return np.stack([dx1, dy1, dx2, dy2], axis=-1)


def rk4_halves(s, eAv, eAm, c1, c2, p=PARAMS):
    h = p["h"]
    f = lambda z: vector_field_halves(z, eAv, eAm, *c1, *c2, p)
    k1 = f(s); k2 = f(s + h * k1 / 2); k3 = f(s + h * k2 / 2); k4 = f(s + h * k3)
    return s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def rk4(s, eAv, eAm, dp, de, p=PARAMS):
    h = p["h"]
    k1 = vector_field(s, eAv, eAm, dp, de, p)
    k2 = vector_field(s + h * k1 / 2, eAv, eAm, dp, de, p)
    k3 = vector_field(s + h * k2 / 2, eAv, eAm, dp, de, p)
    k4 = vector_field(s + h * k3, eAv, eAm, dp, de, p)
    return s + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0


def analytic_distances(s, p=PARAMS):
    """Distances of the central controller: functions of the accumulated needs."""
    yc, yp = s[..., 1], s[..., 3]
    dp = (yc + yp) / 2.0 + yc
    de = p["k0"] * ((yc - yp) / 2.0 + yc) + p["delta"] * p["k0"]
    return dp, de


def learned_distances(s, owner, eAv, eAm, net_agent="parent", nets=None):
    """Distances an agent obtains from the coupling networks (IndividualAttachmentController).

    owner: 'child' or 'parent', whose own need is fed to the networks. The controller loads
    the networks trained for the caregiver for both robots (net_agent='parent').
    Physical input: [|y_c - y_p| of the internal model, own x, own y, ambivalence].
    Emotional input: [evidence trace, partner call, own call, own x, own y, avoidance], with
    the vocal channel left at zero as in the controller.
    """
    nets = nets or networks()
    x_own = s[..., 0] if owner == "child" else s[..., 2]
    y_own = s[..., 1] if owner == "child" else s[..., 3]
    gap = np.abs(s[..., 1] - s[..., 3])
    eAm = np.broadcast_to(eAm, gap.shape)
    eAv = np.broadcast_to(eAv, gap.shape)
    zeros = np.zeros_like(gap)
    dp = nets[("physical", net_agent)](np.stack([gap, x_own, y_own, eAm], axis=-1))
    de = nets[("emotional", net_agent)](np.stack([zeros, zeros, zeros, x_own, y_own, eAv], axis=-1))
    return dp, de


def _prepare(eAv, eAm):
    eAv, eAm = np.broadcast_arrays(np.atleast_1d(np.asarray(eAv, float)),
                                   np.atleast_1d(np.asarray(eAm, float)))
    return eAv.copy(), eAm.copy()


def _initial(n, initial):
    """Starting state for N runs: one (x_c, y_c, x_p, y_p) for all, or one row per run."""
    initial = np.asarray(initial, float)
    return np.tile(initial, (n, 1)) if initial.ndim == 1 else initial.copy()


def simulate_coupled(eAv, eAm, ticks, initial=INITIAL, p=PARAMS, distances="analytic", stride=1):
    """M1: a single integrator for the dyad. Returns states [ticks/stride, N, 4] and distances.

    stride > 1 records every stride-th tick (the integration is unchanged)."""
    eAv, eAm = _prepare(eAv, eAm)
    s = _initial(len(eAv), initial)
    states = np.empty((ticks // stride, len(eAv), 4))
    dists = np.empty((ticks // stride, len(eAv), 2))
    for t in range(ticks):
        dp, de = analytic_distances(s, p)
        s = _guard(rk4(s, eAv, eAm, dp, de, p))
        if (t + 1) % stride == 0:
            k = (t + 1) // stride - 1
            states[k], dists[k, :, 0], dists[k, :, 1] = s, dp, de
    return states, dists


def simulate_decoupled(eAv, eAm, ticks, initial=INITIAL, p=PARAMS, net_agent="parent",
                       coupling="learned", guard=True, stride=1, partner="learned"):
    """M2: every agent integrates its own copy of the dyad (self + internal partner model).

    coupling='learned' uses the trained networks (as in Gazebo); 'analytic' gives each copy
    the central controller's distance functions (an ablation that isolates the networks).
    guard=False lets a run overflow to inf/NaN as the ROS controller does (its loop stops at
    the first NaN), instead of freezing it at NaN once |state| > 1e6.
    net_agent='own' gives each agent the networks trained for it (an ablation; the controller
    loads the caregiver's networks for both robots).
    partner='analytic' integrates each agent's partner model with the analytic coupling of the
    standard model (the controller's 'baseline' mode); only the agent's own half is learned.
    Returns dict with 'child' and 'parent' copies, each [ticks/stride, N, 4], and their distances.
    """
    eAv, eAm = _prepare(eAv, eAm)
    out = {}
    for owner in ("child", "parent"):
        nets_for = owner if net_agent == "own" else net_agent
        s = _initial(len(eAv), initial)
        states = np.empty((ticks // stride, len(eAv), 4))
        dists = np.empty((ticks // stride, len(eAv), 2))
        for t in range(ticks):
            if coupling == "learned":
                dp, de = learned_distances(s, owner, eAv, eAm, nets_for)
            else:
                dp, de = analytic_distances(s, p)
            if partner == "analytic":
                base = analytic_distances(s, p)
                c_own, c_partner = (dp, de), base
                s = rk4_halves(s, eAv, eAm, *((c_own, c_partner) if owner == "child" else (c_partner, c_own)), p)
            else:
                s = rk4(s, eAv, eAm, dp, de, p)
            s = _guard(s) if guard else s
            if (t + 1) % stride == 0:
                k = (t + 1) // stride - 1
                states[k], dists[k, :, 0], dists[k, :, 1] = s, dp, de
        out[owner] = states
        out[owner + "_distances"] = dists
    return out


def simulate_sensed(eAv, eAm, dp_series, de_series, initial=INITIAL, p=PARAMS):
    """M0: the Living Machines controller driven by externally supplied distances."""
    eAv, eAm = _prepare(eAv, eAm)
    ticks = len(dp_series)
    s = _initial(len(eAv), initial)
    states = np.empty((ticks, len(eAv), 4))
    for t in range(ticks):
        s = _guard(rk4(s, eAv, eAm, dp_series[t], de_series[t], p))
        states[t] = s
    return states


def _guard(s):
    """Freeze runs that blew up so that NaN/overflow does not spread through the batch."""
    bad = ~np.isfinite(s).all(axis=-1) | (np.abs(s).max(axis=-1) > 1e6)
    if bad.any():
        s = s.copy()
        s[bad] = np.nan
    return s


def own_view(decoupled):
    """What each robot acts on: the child copy's (x_c, y_c) and the parent copy's (x_p, y_p)."""
    child, parent = decoupled["child"], decoupled["parent"]
    return np.stack([child[..., 0], child[..., 1], parent[..., 2], parent[..., 3]], axis=-1)


def diverged(states):
    """True per run where any state left the bounded region."""
    s = np.nan_to_num(states, nan=np.inf)
    return (np.abs(s) > BLOWUP).any(axis=(0, 2))


def actions(states):
    """Action read out from the needs: 0 approach (x > 0), 1 explore, as in the controllers."""
    return np.heaviside(-states[..., [0, 2]], 0).astype(int)
