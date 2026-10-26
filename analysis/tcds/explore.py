#!/usr/bin/env python3
"""
How decoupling the dyad changes its dynamics: coupled (M1) vs decoupled (M2) controllers.

Writes results/explore_strips.npz and results/explore_grid.npz, used by the paper figures.
"""
import os
import time
import argparse

import numpy as np
from scipy.signal import hilbert

import model

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
H = model.PARAMS["h"]
GAZEBO_RATE = H * model.TICK_HZ        # model time per simulated second (0.3)


def first_divergence(states):
    """Model time at which each run first leaves |state| <= BLOWUP (inf if never)."""
    s = np.nan_to_num(states, nan=np.inf)
    out = np.abs(s).max(axis=2) > model.BLOWUP
    hit = out.any(axis=0)
    t = np.where(hit, out.argmax(axis=0) * H, np.inf)
    return t


def corr(u, v):
    ok = np.isfinite(u).all() and np.isfinite(v).all()
    if not ok or u.std() < 1e-6 or v.std() < 1e-6:
        return np.nan
    return float(np.corrcoef(u, v)[0, 1])


def freq(x, duration):
    """Oscillation frequency in cycles per model time unit (zero crossings / 2)."""
    if not np.isfinite(x).all():
        return np.nan
    crossings = np.count_nonzero(np.diff(np.sign(x)) != 0)
    return crossings / (2.0 * duration)


def rel_phase(u, v):
    """Mean phase of v relative to u in degrees, and phase-locking value (0-1)."""
    if not (np.isfinite(u).all() and np.isfinite(v).all()) or u.std() < 1e-6 or v.std() < 1e-6:
        return np.nan, np.nan
    dphi = np.angle(hilbert(v - v.mean())) - np.angle(hilbert(u - u.mean()))
    z = np.mean(np.exp(1j * dphi))
    return float(np.degrees(np.angle(z))), float(np.abs(z))


def poincare(x, y):
    """y at upward zero crossings of x (Poincare section x = 0, dx > 0)."""
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        return np.array([])
    idx = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0))
    frac = -x[idx] / (x[idx + 1] - x[idx])
    return y[idx] + frac * (y[idx + 1] - y[idx])


def summarise(child_view, parent_view, burn, within=None):
    """Per-run measures from the child's (x_c, y_c) and parent's (x_p, y_p) series [T, N]."""
    xc, yc, xp, yp = child_view
    n = xc.shape[1]
    keep = slice(burn, None)
    dur = (xc.shape[0] - burn) * H
    out = {k: np.full(n, np.nan) for k in
           ("r", "freq_c", "freq_p", "phase", "plv", "approach_c", "approach_p", "amp_c", "amp_p")}
    sections = []
    for i in range(n):
        u, v = xc[keep, i], xp[keep, i]
        out["r"][i] = corr(u, v)
        out["freq_c"][i] = freq(u, dur)
        out["freq_p"][i] = freq(v, dur)
        out["phase"][i], out["plv"][i] = rel_phase(u, v)
        if np.isfinite(u).all():
            out["approach_c"][i] = np.mean(u > 0)
            out["amp_c"][i] = np.ptp(yc[keep, i])
        if np.isfinite(v).all():
            out["approach_p"][i] = np.mean(v > 0)
            out["amp_p"][i] = np.ptp(yp[keep, i])
        sections.append(poincare(u, yc[keep, i]))
    out["sections"] = sections
    return out


def run_variant(kind, eAv, eAm, ticks, burn):
    t0 = time.time()
    if kind == "coupled":
        s, _ = model.simulate_coupled(eAv, eAm, ticks)
        view = (s[..., 0], s[..., 1], s[..., 2], s[..., 3])
        res = summarise(view, None, burn)
        res["t_div"] = first_divergence(s)
    else:
        d = model.simulate_decoupled(eAv, eAm, ticks, coupling=kind.split("_")[1])
        ch, pa = d["child"], d["parent"]
        view = (ch[..., 0], ch[..., 1], pa[..., 2], pa[..., 3])
        res = summarise(view, None, burn)
        res["t_div"] = np.minimum(first_divergence(ch), first_divergence(pa))
        res["t_div_child_copy"] = first_divergence(ch)
        res["t_div_parent_copy"] = first_divergence(pa)
        # what each agent believes: its own oscillator against its internal partner model
        res["r_within_child"] = np.array([corr(ch[burn:, i, 0], ch[burn:, i, 2]) for i in range(len(eAv))])
        res["r_within_parent"] = np.array([corr(pa[burn:, i, 0], pa[burn:, i, 2]) for i in range(len(eAv))])
        # how well each internal model tracks the real partner
        res["r_model_parent"] = np.array([corr(ch[burn:, i, 2], pa[burn:, i, 2]) for i in range(len(eAv))])
        res["r_model_child"] = np.array([corr(pa[burn:, i, 0], ch[burn:, i, 0]) for i in range(len(eAv))])
        err_p = ch[burn:, :, 3] - pa[burn:, :, 3]
        err_c = pa[burn:, :, 1] - ch[burn:, :, 1]
        res["mismatch_parent"] = np.sqrt(np.nanmean(err_p ** 2, axis=0))
        res["mismatch_child"] = np.sqrt(np.nanmean(err_c ** 2, axis=0))
        # belief mismatch over time for later plotting (downsampled)
        res["mismatch_trace"] = np.abs(ch[::50, :, 3] - pa[::50, :, 3])
    print(f"  {kind}: {len(eAv)} runs, {ticks} ticks, {time.time() - t0:.0f}s")
    return res


def strips(ticks, burn, step):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    out = {"values": values}
    for axis in ("avoidant", "ambivalent"):
        eAv = values if axis == "avoidant" else np.zeros_like(values)
        eAm = values if axis == "ambivalent" else np.zeros_like(values)
        for kind in ("coupled", "decoupled_learned", "decoupled_analytic"):
            res = run_variant(kind, eAv, eAm, ticks, burn)
            for k, v in res.items():
                out[f"{axis}/{kind}/{k}"] = np.array(v, dtype=object) if k == "sections" else v
    return out


def grid(ticks, burn, step):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    AV, AM = np.meshgrid(values, values, indexing="ij")
    out = {"values": values}
    for kind in ("coupled", "decoupled_learned"):
        res = run_variant(kind, AV.ravel(), AM.ravel(), ticks, burn)
        for k in ("r", "freq_c", "freq_p", "phase", "plv", "approach_c", "approach_p", "t_div"):
            out[f"{kind}/{k}"] = res[k].reshape(AV.shape)
        if kind != "coupled":
            for k in ("r_within_child", "r_within_parent", "mismatch_parent"):
                out[f"{kind}/{k}"] = res[k].reshape(AV.shape)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--what", choices=("strips", "grid", "gazebo", "all"), default="all")
    args = parser.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    np.seterr(all="ignore")
    if args.what in ("strips", "all"):
        print("strips (long horizon, 300 model time units)")
        np.savez(os.path.join(RESULTS, "explore_strips.npz"), **strips(30000, 10000, 0.02), allow_pickle=True)
    if args.what in ("gazebo", "all"):
        # the horizon of a 180 s Gazebo run: 54 model time units, no burn-in
        print("strips (Gazebo horizon, 180 s)")
        np.savez(os.path.join(RESULTS, "explore_strips_gazebo.npz"),
                 **strips(int(180 * model.TICK_HZ), 0, 0.02), allow_pickle=True)
    if args.what in ("grid", "all"):
        print("grid")
        np.savez(os.path.join(RESULTS, "explore_grid.npz"), **grid(20000, 6000, 0.05), allow_pickle=True)
