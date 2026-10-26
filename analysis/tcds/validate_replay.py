#!/usr/bin/env python3
"""Check that the offline decoupled model reproduces the needs recorded in Gazebo."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load import load_all
import model

def best_alignment(rec, sim_t, sim, rec_t):
    """Fit the model-time offset and rate that best match a recorded series."""
    best = (-np.inf, None, None)
    for rate in (0.28, 0.29, 0.3, 0.31):
        for off in np.arange(-3.0, 3.0, 0.05):
            tm = off + rate * rec_t
            ok = (tm >= sim_t[0]) & (tm <= sim_t[-1])
            if ok.sum() < 0.8 * len(rec):
                continue
            pred = np.interp(tm[ok], sim_t, sim)
            r = np.corrcoef(pred, rec[ok])[0, 1]
            if r > best[0]:
                best = (r, off, rate)
    return best

if __name__ == "__main__":
    runs = [r for r in load_all() if r.index >= 34]
    eAv = np.array([r.avoidant for r in runs]); eAm = np.array([r.ambivalent for r in runs])
    ticks = int(320 * model.TICK_HZ)
    d = model.simulate_decoupled(eAv, eAm, ticks)
    tsim = np.arange(1, ticks + 1) * model.PARAMS["h"]
    for i, r in enumerate(runs):
        out = []
        for name, rec, copy, col in (("x_c", r.child_need, "child", 0), ("x_p", r.parent_need, "parent", 2)):
            sim = d[copy][:, i, col]
            if r.diverged or not np.isfinite(sim).all():
                out.append(f"{name}: rec_div={r.diverged} sim_div={not np.isfinite(sim).all() or np.abs(sim).max()>10}")
                continue
            rr, off, rate = best_alignment(rec, tsim, sim, r.t)
            out.append(f"{name}: r={rr:.3f} off={off:.2f} rate={rate}")
        print(f"run {r.index:2d} eAv={r.avoidant:.1f} eAm={r.ambivalent:.1f} | " + " | ".join(out))
