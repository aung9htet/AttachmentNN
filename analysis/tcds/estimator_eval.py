#!/usr/bin/env python3
"""
How well do the learned coupling functions reproduce the analytic coupling? (agent simulator)

1. Open-loop accuracy: networks evaluated on the inputs of analytically coupled runs, at the
   ten training parameter values and at held-out values.
2. Input ablations: which signals the estimate relies on (distance, vocal exchange, own state).
3. Closed-loop fidelity: dyad measures with learned vs analytic coupling.
Writes results/estimator_eval.npz.
"""
import os
import numpy as np

import sim_agents as sa
from nnweights import load_networks
from explore import corr, freq

HERE = os.path.dirname(os.path.abspath(__file__))
T = 10000
SEEDS = range(5)
TRAIN_VALUES = np.linspace(0, 1, 10)
HELDOUT = np.array([0.05, 0.35, 0.65, 0.95])


def r2(y, yhat):
    ss = np.sum((y - y.mean()) ** 2)
    return 1 - np.sum((y - yhat) ** 2) / ss if ss > 1e-9 else np.nan


def inputs(rec, kind, agent, ablate=None):
    """Network inputs as in the notebook's training datasets, optionally ablated."""
    x = rec["xc"] if agent == "child" else rec["xp"]
    y = rec["yc"] if agent == "child" else rec["yp"]
    if kind == "physical":
        dist = sa.filtered_distance(rec["child_pos"], rec["parent_pos"])
        par = np.broadcast_to(rec["amb"], x.shape)
        if ablate == "distance":
            dist = np.full_like(dist, dist.mean())
        if ablate == "own_state":
            x, y = np.full_like(x, x.mean()), np.full_like(y, y.mean())
        return np.stack([dist, x, y, par], -1)
    ev = rec["ev_c"] if agent == "child" else rec["ev_p"]
    vp, vc = rec["voc_p"], rec["voc_c"]
    par = np.broadcast_to(rec["av"], x.shape)
    if ablate == "vocal":            # the embodied controller's silent channel
        ev, vp, vc = np.zeros_like(ev), np.zeros_like(vp), np.zeros_like(vc)
    if ablate == "own_state":
        x, y = np.full_like(x, x.mean()), np.full_like(y, y.mean())
    return np.stack([ev, vp, vc, x, y, par], -1)


def open_loop(nets):
    out = {}
    values = np.concatenate([TRAIN_VALUES, HELDOUT])
    for kind, target_key in (("physical", "target_phys"), ("emotional", "target_emo")):
        amb = values if kind == "physical" else np.zeros_like(values)
        av = values if kind == "emotional" else np.zeros_like(values)
        for ablate in (None, "distance" if kind == "physical" else "vocal", "own_state"):
            scores = {a: np.full((len(values), len(SEEDS)), np.nan) for a in ("child", "parent")}
            rmse = {a: np.full((len(values), len(SEEDS)), np.nan) for a in ("child", "parent")}
            for s in SEEDS:
                rec = sa.simulate(amb, av, T, seed=s)
                for agent in ("child", "parent"):
                    pred = nets[(kind, agent)](inputs(rec, kind, agent, ablate))
                    for i in range(len(values)):
                        yt = rec[target_key][200:, i]; yp = pred[200:, i]
                        if np.isfinite(yt).all():
                            scores[agent][i, s] = r2(yt, yp)
                            rmse[agent][i, s] = np.sqrt(np.mean((yt - yp) ** 2))
            tag = ablate or "full"
            for agent in ("child", "parent"):
                out[f"open/{kind}/{tag}/{agent}/r2"] = scores[agent]
                out[f"open/{kind}/{tag}/{agent}/rmse"] = rmse[agent]
            print(kind, tag, "median R2 child/parent:",
                  np.round(np.nanmedian(scores["child"], 1), 2), np.round(np.nanmedian(scores["parent"], 1), 2))
    out["open/values"] = values
    out["open/n_train"] = len(TRAIN_VALUES)
    return out


def dyad_measures(rec, burn=2000):
    n = rec["xc"].shape[1]
    keep = slice(burn, None)
    dur = (rec["xc"].shape[0] - burn) * sa.NB["dt"]
    dist = np.linalg.norm(rec["child_pos"] - rec["parent_pos"], axis=-1)
    m = {k: np.full(n, np.nan) for k in ("r", "freq_c", "near", "call_c", "call_p", "mean_dist")}
    for i in range(n):
        m["r"][i] = corr(rec["xc"][keep, i], rec["xp"][keep, i])
        m["freq_c"][i] = freq(rec["xc"][keep, i], dur)
        if np.isfinite(dist[keep, i]).all():
            m["near"][i] = np.mean(dist[keep, i] < 0.5)
            m["mean_dist"][i] = dist[keep, i].mean()
        m["call_c"][i] = rec["voc_c"][keep, i].mean()
        m["call_p"][i] = rec["voc_p"][keep, i].mean()
    return m


def closed_loop(nets, seeds=range(10)):
    out = {}
    values = np.round(np.arange(0.0, 1.0001, 0.1), 2)
    for axis, learned in (("ambivalent", ("physical",)), ("avoidant", ("emotional",))):
        amb = values if axis == "ambivalent" else np.zeros_like(values)
        av = values if axis == "avoidant" else np.zeros_like(values)
        for mode in ("analytic", "learned"):
            acc = {}
            for s in seeds:
                rec = sa.simulate(amb, av, T, seed=s, learned=learned if mode == "learned" else (), nets=nets)
                for k, v in dyad_measures(rec).items():
                    acc.setdefault(k, []).append(v)
            for k, v in acc.items():
                out[f"closed/{axis}/{mode}/{k}"] = np.array(v)     # [seeds, values]
            print(axis, mode, "r:", np.round(np.nanmean(acc["r"], 0), 2), "near:", np.round(np.nanmean(acc["near"], 0), 2))
    out["closed/values"] = values
    return out


if __name__ == "__main__":
    np.seterr(all="ignore")
    nets = load_networks()
    res = open_loop(nets)
    res.update(closed_loop(nets))
    np.savez(os.path.join(HERE, "results", "estimator_eval.npz"), **res)
