#!/usr/bin/env python3
"""
The simulated strange situation of the npj draft (Section 6.2), regenerated.

Point agents in the agent-based simulator (sim_agents.simulate) with the analytic coupling of the
canonical model: the caregiver is held near a central base, the child explores the corners or returns
to the base, physical distance is the Euclidean distance and emotional contact is the exchange of
binary calls. The emotional coupling uses the sign of the canonical model (+delta k0 Av in the slow
equation, as in attachment_model_inference and the npj analysis), so the strongly avoidant child
settles in exploration. Five regimes, 10,000 steps each, as in the npj draft.

    python3 analysis/tcds/ssp_sim.py      -> results/sim_ssp.npz
"""
import os
from itertools import combinations

import numpy as np
from scipy import stats

import sim_agents as sa

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

REGIMES = [("secure", 0.0, 0.1), ("moderately avoidant", 0.4, 0.0), ("strongly avoidant", 0.9, 0.0),
           ("moderately ambivalent", 0.0, 0.4), ("strongly ambivalent", 0.0, 0.9)]
STEPS = 10000
CANONICAL_SIGN = dict(sa.NB, delta=-sa.NB["delta"])   # +delta k0 Av in the slow equation


def main(seed=0):
    names = [r[0] for r in REGIMES]
    av = np.array([r[1] for r in REGIMES]); am = np.array([r[2] for r in REGIMES])
    rec = sa.simulate(am, av, STEPS, seed=seed, p=CANONICAL_SIGN)
    child, parent = rec["child_pos"], rec["parent_pos"]
    near = (np.abs(child - parent) <= 0.5).all(-1)            # square of half the arena's width
    counts = np.stack([near.sum(0), (~near).sum(0)], 1)
    chi2, p, dof, _ = stats.chi2_contingency(counts)
    pairs = []
    for i, j in combinations(range(len(names)), 2):
        n1, n2 = counts[i].sum(), counts[j].sum()
        p1, p2 = counts[i, 0] / n1, counts[j, 0] / n2
        pool = (counts[i, 0] + counts[j, 0]) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2)) if 0 < pool < 1 else np.nan
        pb = min(1.0, 2 * stats.norm.sf(abs(z)) * 10) if np.isfinite(z) else np.nan   # Bonferroni, 10 pairs
        pairs.append((i, j, z, pb))
    xc, xp = rec["xc"], rec["xp"]
    r = np.array([np.corrcoef(xc[:, k], xp[:, k])[0, 1] if xc[:, k].std() > 1e-3 and xp[:, k].std() > 1e-3
                  else np.nan for k in range(len(names))])
    np.savez(os.path.join(RES, "sim_ssp.npz"), names=np.array(names), av=av, am=am, near=near.mean(0),
             counts=counts, chi2=chi2, p=p, dof=dof, pairs=np.array(pairs, float), r=r,
             child_pos=child[:, :, :], parent_pos=parent, dist=rec["dist"], voc_c=rec["voc_c"], voc_p=rec["voc_p"],
             xc=xc, xp=xp, steps=STEPS)
    for k, n in enumerate(names):
        print(f"  {n:22s} near {100 * near.mean(0)[k]:5.1f}%  r {r[k]:+.2f}  child calls {100 * rec['voc_c'][:, k].mean():.0f}%")
    print(f"  chi2 = {chi2:.1f}, df = {dof}, N = {counts.sum()}, p = {p:.2g}")
    for i, j, z, pb in pairs:
        print(f"    {names[i]} vs {names[j]}: z = {z:.1f}, p_Bonferroni = {pb:.2g}")


if __name__ == "__main__":
    main()
