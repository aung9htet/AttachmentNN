#!/usr/bin/env python3
"""
Figures made for the TCDS draft that the paper does not use (archived). They reuse the helpers of
analysis/tcds/figures.py and write to archive/analysis/figures/ (not in git).

    python3 archive/analysis/extra_figures.py
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "analysis", "tcds"))
import figures                                                                        # noqa: E402
from figures import metrics, exemplar_runs, tidy, panel, save                         # noqa: E402
from figures import DCOL, INK, MUTED, CHILD, CAREGIVER                                # noqa: E402


def fig_gazebo_timelines():
    """2 x 2 exemplar timelines at eps = 0.2: shared needs vs private models (columns), avoidance vs
    ambivalence (rows), same random start."""
    runs = exemplar_runs()[:4]                       # shared/private x avoidance/ambivalence
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]         # (row, col) of each exemplar in EXEMPLARS order
    fig, axes = plt.subplots(2, 2, figsize=(DCOL, 2.55), sharex=True, sharey=True)
    for (r, row, title), (i, j) in zip(runs, order):
        ax = axes[i, j]
        a = metrics.analyse(r, "gt")
        n = min(len(r.t), 1800)
        t = r.t[:n]
        dist = a["distance"][:n]
        for s0, s1 in metrics.spans(r.child_state[:n] == 0, t):
            ax.axvspan(s0, s1 + 0.1, color="#e8f0fb", lw=0, zorder=0)
        ax.plot(t, dist, color=INK, lw=0.8)
        top = 1.5
        for y, series, col in ((top + 0.12, r.vocal_child[:n], CHILD), (top + 0.4, r.vocal_parent[:n], CAREGIVER)):
            for s0, s1 in metrics.spans(series > 0, t):
                ax.plot([s0, s1 + 0.1], [y, y], color=col, lw=2.6, solid_capstyle="butt")
        ax.set_xlim(0, 180); ax.set_ylim(0, top + 0.55)
        ax.set_yticks([0, 0.5, 1.0, 1.5])
        if j == 0:
            ax.set_ylabel("distance (m)", fontsize=6.5)
        if i == 1:
            ax.set_xlabel("time (s)")
        tidy(ax)
        recip = row["reciprocity"]
        design, coupling = title.split(", ")
        panel(ax, "abcd"[2 * i + j], coupling.capitalize())
        ax.title.set_fontsize(7)
        ax.text(1.0, 1.035, f"coordination {row['r']:+.2f}, reciprocity "
                            f"{'n/a' if not np.isfinite(recip) else f'{recip:.2f}'}",
                transform=ax.transAxes, fontsize=6, color=MUTED, ha="right", va="bottom")
    handles = [Line2D([], [], color=INK, lw=0.8, label="child–caregiver distance"),
               Patch(facecolor="#e8f0fb", label="child in approach"),
               Line2D([], [], color=CHILD, lw=2.6, label="child calls"),
               Line2D([], [], color=CAREGIVER, lw=2.6, label="caregiver calls")]
    fig.tight_layout(h_pad=0.8, w_pad=1.0)
    fig.subplots_adjust(top=0.86)
    for j, design in enumerate(("Coupled reference", "Decoupled")):
        pos = axes[0, j].get_position()
        fig.text((pos.x0 + pos.x1) / 2, 0.975, design, ha="center", va="top", fontsize=7.5, color=INK, fontweight="bold")
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07), fontsize=6.2)
    save(fig, "fig_gazebo_timelines")


if __name__ == "__main__":
    np.seterr(all="ignore")
    figures.OUT = os.path.join(HERE, "figures")
    fig_gazebo_timelines()
