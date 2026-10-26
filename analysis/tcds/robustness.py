#!/usr/bin/env python3
"""
Robustness of the results to the starting needs, and the observation-driven agent simulator.

  strips     (O1)  coupled and decoupled controllers along both strips, 40 random starts per value
  strips180        the same over the 180 s horizon of a Gazebo run
  grid       (O1)  regime fractions over the (eAv, eAm) plane, 20 random starts per cell
  stability  (O1b) how quickly a small difference between the starting needs grows from the
                   identical start under avoidance (decoupled controller)
  nets       (O2)  decoupled controller with each agent using its own networks
  notebook   (O3)  the agent simulator the networks were trained in, from random starts, with
                   learned and analytic coupling: how well the networks estimate the coupling

Random starts are U(-1, 1) for (x_c, y_c, x_p, y_p) from a fixed generator, the same set for
every parameter value. Writes results/robust_<part>.npz.

    python3 analysis/tcds/robustness.py [strips grid stability nets notebook]
"""
import os
import sys
import time

import numpy as np

import model
import sim_agents as sa
from explore import corr, freq

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
H = model.PARAMS["h"]
STRIDE = 10                      # record every 10th tick (0.1 model time)


def random_starts(n, seed=0):
    return np.random.default_rng(seed).uniform(-1.0, 1.0, (n, 4))


def escaped(states):
    s = np.nan_to_num(states, nan=np.inf)
    return (np.abs(s) > model.BLOWUP).any(axis=(0, 2))


def measures(child_view, parent_view, burn):
    """Per-run measures; child_view / parent_view are the child's and the caregiver's copies [T, N, 4]
    (the same array twice for the coupled controller)."""
    n = child_view.shape[1]
    dur = (child_view.shape[0] - burn) * STRIDE * H
    keep = slice(burn, None)
    out = {k: np.full(n, np.nan) for k in ("r", "belief_c", "belief_p", "track_r", "track_rmse",
                                           "freq_c", "freq_p", "approach_c", "approach_p")}
    esc = escaped(child_view) | escaped(parent_view)
    for i in range(n):
        if esc[i]:
            continue
        xc, xp = child_view[keep, i, 0], parent_view[keep, i, 2]
        out["r"][i] = corr(xc, xp)
        out["belief_c"][i] = corr(child_view[keep, i, 0], child_view[keep, i, 2])
        out["belief_p"][i] = corr(parent_view[keep, i, 0], parent_view[keep, i, 2])
        out["track_r"][i] = corr(child_view[keep, i, 2], xp)                  # child's model of the caregiver
        out["track_rmse"][i] = np.sqrt(np.mean((child_view[keep, i, 3] - parent_view[keep, i, 3]) ** 2))
        out["freq_c"][i], out["freq_p"][i] = freq(xc, dur), freq(xp, dur)
        out["approach_c"][i], out["approach_p"][i] = np.mean(xc > 0), np.mean(xp > 0)
    out["escaped"] = esc
    return out


def run_variant(kind, eAv, eAm, init, ticks, burn, **kw):
    t0 = time.time()
    with np.errstate(all="ignore"):
        if kind == "coupled":
            s, _ = model.simulate_coupled(eAv, eAm, ticks, initial=init, stride=STRIDE)
            res = measures(s, s, burn)
        else:
            d = model.simulate_decoupled(eAv, eAm, ticks, initial=init, stride=STRIDE, **kw)
            res = measures(d["child"], d["parent"], burn)
    print(f"  {kind}: {len(eAv)} runs, {time.time() - t0:.0f}s", flush=True)
    return res


def strips(n_starts=40, step=0.02, ticks=30000, burn_ticks=10000, net_agent="parent", tag="strips"):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    starts = random_starts(n_starts)
    out = {"values": values, "starts": starts}
    for axis in ("avoidant", "ambivalent"):
        v = np.repeat(values, n_starts)
        eAv = v if axis == "avoidant" else np.zeros_like(v)
        eAm = v if axis == "ambivalent" else np.zeros_like(v)
        init = np.tile(starts, (len(values), 1))
        kinds = ("coupled", "decoupled") if net_agent == "parent" else ("decoupled",)
        for kind in kinds:
            res = run_variant(kind, eAv, eAm, init, ticks, burn_ticks // STRIDE, net_agent=net_agent) \
                if kind == "decoupled" else run_variant(kind, eAv, eAm, init, ticks, burn_ticks // STRIDE)
            for key, val in res.items():
                out[f"{axis}/{kind}/{key}"] = val.reshape(len(values), n_starts)
    np.savez(os.path.join(RESULTS, f"robust_{tag}.npz"), **out)


def grid(n_starts=20, step=0.05, ticks=20000, burn_ticks=6000):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 3)
    starts = random_starts(n_starts, seed=1)
    AV, AM = np.meshgrid(values, values, indexing="ij")
    eAv = np.repeat(AV.ravel(), n_starts)
    eAm = np.repeat(AM.ravel(), n_starts)
    init = np.tile(starts, (AV.size, 1))
    out = {"values": values, "starts": starts}
    for kind in ("coupled", "decoupled"):
        res = run_variant(kind, eAv, eAm, init, ticks, burn_ticks // STRIDE)
        for key, val in res.items():
            out[f"{kind}/{key}"] = val.reshape(len(values), len(values), n_starts)
    np.savez(os.path.join(RESULTS, "robust_grid.npz"), **out)


def stability(ticks=60000):
    """Time until the two agents' needs differ by more than 0.1, starting from the identical start
    with the caregiver's accumulated need raised by `size`."""
    sizes = np.array([1e-9, 1e-7, 1e-5, 1e-3, 1e-1])
    eavs = np.array([0.1, 0.2, 0.3])
    S, E = np.meshgrid(sizes, eavs, indexing="ij")
    init = np.tile(np.array(model.INITIAL, float), (S.size, 1))
    init[:, 3] += S.ravel()
    with np.errstate(all="ignore"):
        d = model.simulate_decoupled(E.ravel(), np.zeros(S.size), ticks, initial=init, stride=STRIDE)
        c, _ = model.simulate_coupled(E.ravel(), np.zeros(S.size), ticks, initial=init, stride=STRIDE)
    out = {"sizes": sizes, "eavs": eavs}
    last = slice(-int(100 / (STRIDE * H)), None)          # last 100 model time units
    for name, xc, xp in (("decoupled", d["child"][..., 0], d["parent"][..., 2]), ("coupled", c[..., 0], c[..., 2])):
        # departure from the synchronous state: first time the needs differ by more than 0.1
        # (sizes up to 1e-3); r_end shows whether they return to it
        gap = np.abs(np.nan_to_num(xc - xp, nan=np.inf)) > 0.1
        t_break = np.where(gap.any(axis=0), (gap.argmax(axis=0) + 1) * STRIDE * H, np.inf)
        out[f"{name}/t_break"] = t_break.reshape(S.shape)                     # model time
        out[f"{name}/t_break_s"] = (t_break / (H * model.TICK_HZ)).reshape(S.shape)   # robot seconds
        out[f"{name}/r_end"] = np.array([corr(xc[last, i], xp[last, i]) for i in range(S.size)]).reshape(S.shape)
    print("  decoupled: robot seconds until the needs first differ by 0.1 (rows = size, cols = eps_Av):\n",
          np.round(out["decoupled/t_break_s"]), "\n  final r:\n", np.round(out["decoupled/r_end"], 2))
    print("  coupled:", np.round(out["coupled/t_break_s"]).tolist(), "final r", np.round(out["coupled/r_end"], 2).tolist())
    np.savez(os.path.join(RESULTS, "robust_stability.npz"), **out)


def notebook(n_starts=40, steps=20000, burn=4000):
    """The agent simulator with learned (networks fed sensed distance and calls) and analytic coupling."""
    values = np.round(np.arange(0.0, 1.0 + 1e-9, 0.1), 2)
    starts = random_starts(n_starts, seed=2)
    nets = model.networks()
    out = {"values": values}
    for axis, learned in (("ambivalent", ("physical",)), ("avoidant", ("emotional",))):
        v = np.repeat(values, n_starts)
        amb = v if axis == "ambivalent" else np.zeros_like(v)
        av = v if axis == "avoidant" else np.zeros_like(v)
        init = np.tile(starts, (len(values), 1))
        for mode in ("analytic", "learned"):
            t0 = time.time()
            with np.errstate(all="ignore"):
                rec = sa.simulate(amb, av, steps, seed=0, learned=learned if mode == "learned" else (), nets=nets,
                                  initial=init)
            n = len(v)
            keep = slice(burn, None)
            dist = np.linalg.norm(rec["child_pos"] - rec["parent_pos"], axis=-1)
            r = np.full(n, np.nan); near = np.full(n, np.nan); fc = np.full(n, np.nan)
            est_r = {k: np.full(n, np.nan) for k in ("c", "p")}
            est_rmse = {k: np.full(n, np.nan) for k in ("c", "p")}
            kind = "phys" if axis == "ambivalent" else "emo"
            target = rec["target_phys"] if axis == "ambivalent" else rec["target_emo"]
            dur = (steps - burn) * sa.NB["dt"]
            for i in range(n):
                xc, xp = rec["xc"][keep, i], rec["xp"][keep, i]
                if not (np.isfinite(xc).all() and np.isfinite(xp).all()) or max(np.abs(xc).max(), np.abs(xp).max()) > 10:
                    continue
                r[i] = corr(xc, xp)
                fc[i] = freq(xc, dur)
                near[i] = np.mean(dist[keep, i] < 0.5)
                for a in ("c", "p"):
                    est = rec[f"net_{kind}_{a}"][keep, i]
                    est_r[a][i] = corr(est, target[keep, i])
                    est_rmse[a][i] = np.sqrt(np.mean((est - target[keep, i]) ** 2))
            for key, val in (("r", r), ("near", near), ("freq_c", fc), ("est_r_c", est_r["c"]),
                             ("est_r_p", est_r["p"]), ("est_rmse_c", est_rmse["c"]), ("est_rmse_p", est_rmse["p"])):
                out[f"{axis}/{mode}/{key}"] = val.reshape(len(values), n_starts)
            print(f"  notebook {axis} {mode}: {time.time() - t0:.0f}s", flush=True)
    np.savez(os.path.join(RESULTS, "robust_notebook.npz"), **out)


if __name__ == "__main__":
    np.seterr(all="ignore")
    parts = sys.argv[1:] or ["stability", "strips", "nets", "notebook", "grid"]
    for part in parts:
        print(part, flush=True)
        if part == "nets":
            strips(n_starts=40, step=0.05, net_agent="own", tag="nets")
        elif part == "strips180":
            # the horizon of a 180 s Gazebo run (54 model time units), measured from the start
            strips(n_starts=40, step=0.02, ticks=int(180 * model.TICK_HZ), burn_ticks=0, tag="strips180")
        else:
            globals()[part]()
