#!/usr/bin/env python3
"""
Per-run measures of the Gazebo runs, for the TCDS figures and tables.

For every run with recorded robot paths (experiment_data_34 on) it computes, over the first
`horizon` seconds:
  behaviour   time near the caregiver (stuck periods excluded), mean distance, child and caregiver
              calling, call reciprocity (fraction of the child's calling time with a caregiver
              call within +/-3 s)
  needs       realised need correlation r(x_c, x_p), oscillation frequencies, regime of each
              agent (oscillating, quenched in approach or exploration, escaped)
  internal    (runs with controller logs) each agent's belief r(own need, partner model),
              partner-model tracking (child's model of the caregiver vs the caregiver's own need),
              need frequencies and the detuning between the child's model of the caregiver and
              the caregiver's own need (and between the two agents' own needs),
              and coupling estimation: each agent's coupling signal against the coupling computed
              from both robots' true needs (Eq. 4 of the draft)
Writes results/gazebo_runs.csv.

    python3 analysis/tcds/gazebo.py [--data DIR]
"""
import os
import sys
import csv
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from load import load_all, default_data_dir   # noqa: E402
import metrics                                # noqa: E402
import model                                  # noqa: E402

HORIZON = 180.0
RECIPROCITY_WINDOW = 30          # samples at 10 Hz (+/-3 s)
QUENCH_WINDOW = 90.0             # an agent whose need does not change sign for this long is quenched
K0, DELTA = model.PARAMS["k0"], model.PARAMS["delta"]


def start_type(run):
    return "random" if run.random_start else "identical"


def group_key(run):
    return (run.controller, run.distance_source or "", run.networks, start_type(run), run.avoidant, run.ambivalent)


def corr(u, v):
    u, v = np.asarray(u, float), np.asarray(v, float)
    ok = np.isfinite(u) & np.isfinite(v)
    if ok.sum() < 10 or u[ok].std() < 1e-6 or v[ok].std() < 1e-6:
        return np.nan
    return float(np.corrcoef(u[ok], v[ok])[0, 1])


def reciprocity(child_calls, carer_calls, window=RECIPROCITY_WINDOW):
    child_calls = np.asarray(child_calls) > 0
    carer = np.asarray(carer_calls) > 0
    if not child_calls.any():
        return np.nan
    near = np.convolve(carer.astype(float), np.ones(2 * window + 1), mode="same") > 0
    return float(np.mean(near[child_calls]))


def agent_regime(x, t, y=None):
    """'escaped', 'quenched-approach', 'quenched-explore' or 'oscillating' for one need series
    (x: need, y: accumulated need)."""
    x = np.asarray(x, float)
    both = x if y is None else np.concatenate((x, np.asarray(y, float)))
    if not np.isfinite(both).all() or np.abs(both).max() > model.BLOWUP:
        return "escaped"
    late = t >= t[-1] - QUENCH_WINDOW
    if np.all(x[late] > 0):
        return "quenched-approach"
    if np.all(x[late] < 0):
        return "quenched-explore"
    return "oscillating"


def dyad_regime(rc, rp):
    if "escaped" in (rc, rp):
        return "escaped"
    q = [r.startswith("quenched") for r in (rc, rp)]
    return "both quenched" if all(q) else "one quenched" if any(q) else "oscillating"


def own_series(log, name, mode):
    """Own (x, y) of an agent from its controller log."""
    s = log["state"]
    if mode == "notebook":
        return s[:, 0], s[:, 1]
    return (s[:, 0], s[:, 1]) if name == "child" else (s[:, 2], s[:, 3])


def internal_measures(run, horizon):
    out = {}
    if not run.logs:
        return out
    if run.controller == "coupled":
        log = run.logs["central"]
        keep = log["t"] <= log["t"][0] + horizon
        out["belief_c"] = out["belief_p"] = corr(log["state"][keep, 0], log["state"][keep, 2])
        duration = float(log["t"][keep][-1] - log["t"][0])
        out["f_own_c"] = metrics.oscillation_frequency(log["state"][keep, 0], duration)
        out["f_partner"] = metrics.oscillation_frequency(log["state"][keep, 2], duration)
        out["detune_own"] = abs(out["f_own_c"] - out["f_partner"])
        return out
    child, carer = run.logs["child"], run.logs["parent"]
    t0 = child["t"][0]
    keep_c = child["t"] <= t0 + horizon
    keep_p = carer["t"] <= t0 + horizon
    t = child["t"][keep_c]
    xc, yc = (v[keep_c] for v in own_series(child, "child", run.controller))
    xp_own, yp_own = (np.interp(t, carer["t"][keep_p], v[keep_p]) for v in own_series(carer, "parent", run.controller))
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    out["f_own_c"] = metrics.oscillation_frequency(xc, duration)
    out["f_partner"] = metrics.oscillation_frequency(xp_own, duration)
    out["detune_own"] = abs(out["f_own_c"] - out["f_partner"])
    if run.controller in ("decoupled", "observed", "baseline", "baseline-observed"):
        sc, sp = child["state"][keep_c], carer["state"][keep_p]
        out["belief_c"] = corr(sc[:, 0], sc[:, 2])
        out["belief_p"] = corr(sp[:, 2], sp[:, 0])
        out["track_r"] = corr(sc[:, 2], xp_own)
        out["track_rmse"] = float(np.sqrt(np.nanmean((sc[:, 3] - yp_own) ** 2)))
        out["track_rmse_x"] = float(np.sqrt(np.nanmean((sc[:, 2] - xp_own) ** 2)))
        # frequency of the child's model of the caregiver against the caregiver's own need
        out["f_model"] = metrics.oscillation_frequency(sc[:, 2], duration)
        out["detune_model"] = abs(out["f_model"] - out["f_partner"])
    # coupling estimation: each agent's coupling signal vs the coupling of the true needs
    target_phys = (yc + yp_own) / 2.0 + yc
    target_emo = K0 * ((yc - yp_own) / 2.0 + yc) + DELTA * K0
    signal, target = ("physical", target_phys) if run.ambivalent > 0 else ("emotional", target_emo)
    if run.ambivalent > 0 or run.avoidant > 0:
        est_c = child[signal][keep_c]
        est_p = np.interp(t, carer["t"][keep_p], carer[signal][keep_p])
        out["est_r_c"] = corr(est_c, target)
        out["est_r_p"] = corr(est_p, target)
        out["est_rmse_c"] = float(np.sqrt(np.nanmean((est_c - target) ** 2)))
        out["est_rmse_p"] = float(np.sqrt(np.nanmean((est_p - target) ** 2)))
    return out


def run_measures(run, horizon=HORIZON):
    n = int(np.searchsorted(run.t, horizon, side="right"))
    a = metrics.analyse(run, "gt")
    keep = ~a["stuck"][:n]
    t = run.t[:n]
    rc = agent_regime(run.child_need[:n], t, run.child_acc[:n])
    rp = agent_regime(run.parent_need[:n], t, run.parent_acc[:n])
    m = dict(index=run.index, batch=run.batch or ("D0" if run.index >= 34 else "legacy"),
             controller=run.controller, distance=run.distance_source or "", networks=run.networks,
             start=start_type(run),
             avoidant=run.avoidant, ambivalent=run.ambivalent, seed=run.seed,
             duration=round(float(t[-1]), 1),
             near=float(np.mean(a["near"][:n][keep])), mean_distance=float(np.mean(a["distance"][:n][keep])),
             stuck=float(np.mean(a["stuck"][:n])),
             child_calling=float(np.mean(run.vocal_child[:n] > 0)),
             carer_calling=float(np.mean(run.vocal_parent[:n] > 0)),
             child_bouts=metrics.vocal_bouts(run.vocal_child[:n]),
             reciprocity=reciprocity(run.vocal_child[:n], run.vocal_parent[:n]),
             r=corr(run.child_need[:n], run.parent_need[:n]) if rc != "escaped" and rp != "escaped" else np.nan,
             freq_c=metrics.oscillation_frequency(run.child_need[:n], t[-1] - t[0]),
             freq_p=metrics.oscillation_frequency(run.parent_need[:n], t[-1] - t[0]),
             regime_c=rc, regime_p=rp, regime=dyad_regime(rc, rp), diverged=run.diverged)
    m.update(internal_measures(run, horizon))
    return m


def windowed(run, window=120.0, step=30.0):
    """Measures in sliding windows over a long run: realised r and call reciprocity."""
    rows = []
    for start in np.arange(0.0, run.t[-1] - window + 1e-9, step):
        sel = (run.t >= start) & (run.t < start + window)
        rows.append(dict(t=start + window / 2, r=corr(run.child_need[sel], run.parent_need[sel]),
                         reciprocity=reciprocity(run.vocal_child[sel], run.vocal_parent[sel]),
                         child_calling=float(np.mean(run.vocal_child[sel] > 0)),
                         carer_calling=float(np.mean(run.vocal_parent[sel] > 0))))
    return rows


FIELDS = ["index", "batch", "controller", "distance", "networks", "start", "avoidant", "ambivalent", "seed", "duration",
          "near", "mean_distance", "stuck", "child_calling", "carer_calling", "child_bouts", "reciprocity",
          "r", "freq_c", "freq_p", "regime_c", "regime_p", "regime", "diverged",
          "belief_c", "belief_p", "track_r", "track_rmse", "track_rmse_x", "est_r_c", "est_r_p", "est_rmse_c", "est_rmse_p",
          "f_own_c", "f_model", "f_partner", "detune_own", "detune_model"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    np.seterr(all="ignore")
    runs = [r for r in load_all(args.data or default_data_dir()) if r.has_positions]
    rows = [run_measures(r) for r in runs]
    out = os.path.join(HERE, "results", "gazebo_runs.csv") if args.data is None else \
        os.path.join(args.data, "gazebo_runs.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k in FIELDS})
    print(f"{len(rows)} runs -> {out}")


if __name__ == "__main__":
    main()
