#!/usr/bin/env python3
"""
Check recorded runs against their controllers (runs with controller logs, experiment_data_54 on).

  decoupled, coupled  replay the controller offline from the run's logged starting needs and
                      compare with the needs the robots recorded (the needs do not depend on
                      behaviour in these modes, so they must match)
  observed, notebook  behaviour feeds back, so instead check that the network inputs the
                      controller logged match what the robots recorded: the inter-robot
                      distance (ground truth or odometry) and both robots' calls

    python3 analysis/tcds/check_runs.py [--data DIR] [--from-index 54]
Writes results/run_checks.csv (or <data>/run_checks.csv for another data directory).
"""
import os
import sys
import csv
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from load import load_all, default_data_dir   # noqa: E402
import model                                  # noqa: E402
from validate_replay import best_alignment    # noqa: E402


def replay_check(run):
    """Correlation between recorded and re-simulated needs (child, caregiver)."""
    ticks = int((run.duration + 1) * model.TICK_HZ)
    tsim = np.arange(1, ticks + 1) * model.PARAMS["h"]
    with np.errstate(all="ignore"):
        if run.controller == "coupled":
            s, _ = model.simulate_coupled([run.avoidant], [run.ambivalent], ticks, initial=run.initial)
            sim_c, sim_p = s[:, 0, 0], s[:, 0, 2]
            sim_all = s[:, 0, :]
        else:
            d = model.simulate_decoupled([run.avoidant], [run.ambivalent], ticks, initial=run.initial,
                                         partner="analytic" if run.controller == "baseline" else "learned",
                                         net_agent="own" if run.networks == "own" else "parent")
            sim_c, sim_p = d["child"][:, 0, 0], d["parent"][:, 0, 2]
            sim_all = model.own_view(d)[:, 0, :]
    within = tsim <= (run.duration + 0.5) * model.PARAMS["h"] * model.TICK_HZ
    sim_c, sim_p, sim_all, tsim = sim_c[within], sim_p[within], sim_all[within], tsim[within]
    # needs and accumulated needs, as Run.diverged
    sim_div = not np.isfinite(sim_all).all() or np.abs(sim_all).max() > model.BLOWUP
    if run.diverged or sim_div:
        ok = run.diverged == sim_div
        return ok, f"diverged: recorded {run.diverged}, replay {sim_div}", np.nan, np.nan
    rc = best_alignment(run.child_need, tsim, sim_c, run.t)[0]
    rp = best_alignment(run.parent_need, tsim, sim_p, run.t)[0]
    ok = min(rc, rp) >= (0.9999 if run.controller == "coupled" else 0.999)
    return ok, f"replay r child {rc:.4f}, caregiver {rp:.4f}", rc, rp


def trimmed(log):
    """Controller logs are saved while the control thread may still append: cut to a common length."""
    n = min(len(v) for k, v in log.items() if k != "meta")
    return {k: (v if k == "meta" else v[:n]) for k, v in log.items()}


def log_check(run):
    """Do the logged network inputs match the recorded distance and calls?"""
    messages, ok = [], True
    key = "gt_distance" if run.distance_source == "gt" else "odom_distance"
    for name in ("child", "parent"):
        log = trimmed(run.logs[name])
        recorded = np.load(os.path.join(run.path, f"{name}_data.npy"), allow_pickle=True).item()
        rec_time = np.asarray(recorded["time"], float)
        rec_dist = np.asarray(recorded[key], float)[:len(rec_time)]
        calls = np.asarray(recorded[f"vocal_{name}"], float)[:len(rec_time)] > 0.5
        # the controller samples the same topics at 30 Hz; compare at the robot's 10 Hz sample times
        good = np.isfinite(rec_dist) & (rec_time >= log["t"][0]) & (rec_time <= log["t"][-1])
        err = np.nanmedian(np.abs(np.interp(rec_time[good], log["t"], log["distance_m"]) - rec_dist[good]))
        own = np.interp(rec_time[good], log["t"], log["own_call"]) > 0.5
        agree = np.mean(own == calls[good])
        ok &= bool(err < 0.02 and agree > 0.95)
        messages.append(f"{name}: distance error {100 * err:.1f} cm, own-call agreement {100 * agree:.0f}%")
    return ok, "; ".join(messages), np.nan, np.nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    parser.add_argument("--from-index", type=int, default=54)
    args = parser.parse_args()
    data_dir = args.data or default_data_dir()
    runs = [r for r in load_all(data_dir) if r.logs and r.index >= (args.from_index if args.data is None else 0)]
    rows = []
    for r in runs:
        if r.controller in ("decoupled", "coupled", "baseline"):
            ok, msg, rc, rp = replay_check(r)
        else:
            ok, msg, rc, rp = log_check(r)
        rows.append(dict(index=r.index, batch=r.batch, controller=r.controller, distance=r.distance_source or "",
                         networks=r.networks,
                         avoidant=r.avoidant, ambivalent=r.ambivalent, seed=r.seed,
                         initial=" ".join(f"{v:.3f}" for v in r.initial), duration=round(r.duration, 1),
                         rtf=round(r.meta["recorded_duration_s"] / r.meta["wall_duration_s"], 3)
                         if r.meta.get("wall_duration_s") else "",
                         diverged=r.diverged, ok=ok, check=msg))
        print(f"{'ok  ' if ok else 'FAIL'} {r.index:4d} {r.controller:9s} {r.distance_source or '':4s} "
              f"av {r.avoidant:.1f} am {r.ambivalent:.1f} seed {r.seed:3d}  {msg}")
    out = os.path.join(HERE, "results", "run_checks.csv") if args.data is None else os.path.join(data_dir, "run_checks.csv")
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{sum(r['ok'] for r in rows)}/{len(rows)} runs pass; wrote {out}")


if __name__ == "__main__":
    main()
