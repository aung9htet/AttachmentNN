#!/usr/bin/env python3
"""
Calibrate the kinematic stand-in's exploration speed on Gazebo runs that the inverse test does
not use, then repeat the transfer test of inverse.py with the calibrated speed.

The calibration runs are the coupled-reference and observed runs on the two strips (same arena,
spawn and action policy as the test runs, different controllers); the test runs of inverse.py
are the decoupled runs. For each calibration run the stand-in is driven by the run's own logged
needs, and the statistic matched is the child's mean ground-truth speed while it explores.
Writes results/calibration.json.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from load import load_all   # noqa: E402
import sim_kinematic as sk  # noqa: E402
import inverse              # noqa: E402

SPEEDS = np.round(np.arange(0.08, 0.2001, 0.01), 2)
SECONDS = 180


def explore_speed(xy, need, dt=sk.DT):
    """Mean speed (m/s) over the samples in which the child explores (need < 0)."""
    v = np.hypot(*np.diff(xy, axis=0).T) / dt
    explore = np.asarray(need[1:]) < 0
    return float(v[explore].mean()) if explore.any() else np.nan


def calibration_runs():
    return [r for r in load_all() if r.has_positions and not r.diverged
            and r.controller in ("coupled", "observed") and min(r.avoidant, r.ambivalent) == 0]


def stand_in_speed(run, speed):
    n = min(int(SECONDS * 10), len(run.t))
    needs = np.stack([run.child_need, run.child_acc, run.parent_need, run.parent_acc], 1)[:n]
    sk.EXPLORE_SPEED = speed
    b = sk.run_behaviour(needs, seed=1000 + max(run.seed, 0))
    return explore_speed(np.stack([b["x"], b["y"]], 1), needs[:, 0])


def transfer(speed):
    """inverse.py's transfer test with the stand-in at the given exploration speed."""
    sk.EXPLORE_SPEED = speed
    Xtr, Ytr, groups = inverse.training_set()
    Xte, Yte, _, _, _ = inverse.gazebo_set()
    out = {}
    for name, cols in (("child", inverse.CHILD_ONLY), ("dyad", inverse.DYAD)):
        res = inverse.fit_eval(Xtr, Ytr, groups, Xte, Yte, cols)
        out[name] = dict(acc=float(np.mean(res["test_pred"] == Yte[:, 2])),
                         mae=[float(v) for v in np.abs(res["test_eps"] - Yte[:, :2]).mean(0)])
    return out


if __name__ == "__main__":
    np.seterr(all="ignore")
    runs = calibration_runs()
    gazebo = np.array([explore_speed(r.child_gt[:SECONDS * 10], r.child_need[:SECONDS * 10]) for r in runs])
    test = [r for r in load_all() if r.has_positions and r.controller == "decoupled" and r.networks == "default"
            and min(r.avoidant, r.ambivalent) == 0 and not r.diverged]
    gazebo_test = np.array([explore_speed(r.child_gt[:SECONDS * 10], r.child_need[:SECONDS * 10]) for r in test])
    sim = np.array([[stand_in_speed(r, s) for r in runs] for s in SPEEDS])
    error = np.nanmean((sim - gazebo[None, :]) ** 2, axis=1)
    best = float(SPEEDS[np.argmin(error)])
    out = dict(n_calibration_runs=len(runs), gazebo_explore_speed=float(np.nanmean(gazebo)),
               gazebo_explore_speed_test_runs=float(np.nanmean(gazebo_test)),
               speeds=SPEEDS.tolist(), stand_in_explore_speed=np.nanmean(sim, 1).tolist(),
               squared_error=error.tolist(), calibrated_speed=best, current_speed=0.12)
    print(f"{len(runs)} calibration runs: Gazebo explore speed {out['gazebo_explore_speed']:.3f} m/s "
          f"(test runs {out['gazebo_explore_speed_test_runs']:.3f}); best stand-in speed {best:.2f} m/s")
    out["transfer"] = transfer(best)
    print("transfer at the calibrated speed:", out["transfer"])
    with open(os.path.join(HERE, "results", "calibration.json"), "w") as f:
        json.dump(out, f, indent=1)
