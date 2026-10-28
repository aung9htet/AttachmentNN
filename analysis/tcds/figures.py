#!/usr/bin/env python3
"""
Figures of the TCDS paper, written to analysis/tcds/figures (PDF, with PNG previews), sized for an
IEEE two-column page.

Run after canonical.py, ssp_sim.py, explore.py, robustness.py, estimator_eval.py, gazebo.py and inverse.py:
    python3 analysis/tcds/figures.py [names]       # no names: every figure
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, ANALYSIS)

from load import load_all          # noqa: E402
import metrics                     # noqa: E402
import model                       # noqa: E402
import sim_kinematic as sk         # noqa: E402
import gazebo as gz                # noqa: E402
import csv                         # noqa: E402
from scipy import stats            # noqa: E402

OUT = os.path.join(HERE, "figures")
RES = os.path.join(HERE, "results")

# IEEE geometry (inches) and the validated categorical palette (print, white surface)
COL, DCOL = 3.5, 7.16
CHILD, CAREGIVER, DECOUPLED = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, FAINT, GRID = "#0b0b0b", "#52514e", "#b9b8b2", "#e4e3df"
QUENCH, ONESIDED = "#d9d8d4", "#efeeea"
DIVERGING = LinearSegmentedColormap.from_list(
    "sync", ["#e34948", "#f0a3a0", "#f0efec", "#86b6ef", "#2a78d6"])   # anti-sync .. sync
SEQ = LinearSegmentedColormap.from_list("seq", ["#ffffff", "#cde2fb", "#86b6ef", "#2a78d6", "#104281"])

RATE = model.PARAMS["h"] * model.TICK_HZ   # model time per second of simulated robot time

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.3, "lines.markersize": 3.5, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "axes.titlelocation": "left",
})


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf")


def tidy(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)


def panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".strip(), loc="left", fontsize=7.5, color=INK)


NOUN = {"avoidant": r"avoidance $\varepsilon_{Av}$", "ambivalent": r"ambivalence $\varepsilon_{Am}$"}


# ---------------------------------------------------------------- Gazebo run table

# controller modes: (controller, distance) -> (label, colour, line style, marker)
MODES = {
    ("coupled", ""): ("Coupled reference", MUTED, "-", "s"),
    ("decoupled", ""): ("Decoupled", DECOUPLED, "-", "o"),
    ("observed", "gt"): ("Observed", CHILD, "-", "^"),
    ("observed", "odom"): ("Observed (odometry)", CHILD, "--", "v"),
    ("baseline", ""): ("Baseline", "#6250d6", "-", "D"),
    ("baseline-observed", "gt"): ("Baseline-observed", CAREGIVER, "-", "P"),
    ("notebook", "gt"): ("Notebook", "#eda100", "-", "X"),
}


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def gazebo_table():
    with open(os.path.join(RES, "gazebo_runs.csv")) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("avoidant", "ambivalent", "near", "mean_distance", "child_calling", "carer_calling",
                  "reciprocity", "r", "belief_c", "track_r", "track_rmse", "track_rmse_x", "est_r_c", "est_r_p", "duration",
                  "f_own_c", "f_model", "f_partner", "detune_own", "detune_model"):
            r[k] = _num(r.get(k))
        r["seed"] = int(r["seed"])
        r["index"] = int(r["index"])
        if r.get("networks") == "own":       # sensing ablation: open-loop robots with their own role's networks
            r["controller"] += "-own"
    return rows


def select(rows, controller, start, axis=None, distance="", batches=None):
    """Runs of one controller and start type on one strip (axis) or anywhere (axis=None)."""
    out = []
    for r in rows:
        if r["controller"] != controller or r["start"] != start or r["distance"] != distance:
            continue
        if batches and r["batch"] not in batches:
            continue
        if r["batch"] in ("E3", "E4") and not (batches and r["batch"] in batches):
            continue
        if axis == "avoidant" and r["ambivalent"] != 0:
            continue
        if axis == "ambivalent" and r["avoidant"] != 0:
            continue
        out.append(r)
    return out


def by_value(runs, axis, key, include_escaped=False):
    """{eps: [values]} for one strip."""
    groups = {}
    for r in runs:
        if not include_escaped and r["regime"] == "escaped":
            continue
        groups.setdefault(r[axis], []).append(r[key])
    return {k: np.asarray(v, float) for k, v in sorted(groups.items())}


def mean_ci(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return np.nan, np.nan
    if len(v) == 1:
        return v[0], 0.0
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def plot_points(ax, groups, color, marker, offset=0.0, hollow=False, ms=3.2, label=None, err="ci"):
    """Mean with 95% t-interval (err='ci') or standard deviation (err='sd') over runs."""
    xs, ms_, cis = [], [], []
    for x, vals in groups.items():
        m, ci = mean_ci(vals)
        if err == "sd":
            v = np.asarray(vals, float); v = v[np.isfinite(v)]
            ci = v.std(ddof=1) if len(v) > 1 else 0.0
        if np.isfinite(m):
            xs.append(x + offset); ms_.append(m); cis.append(ci)
    if not xs:
        return
    ax.errorbar(xs, ms_, yerr=cis, fmt=marker, ms=ms, color=color, mfc="white" if hollow else color,
                mec=color, mew=0.8, elinewidth=0.7, capsize=1.5, label=label, zorder=5)


def median_band(ax, values, x, color, ls="-", label=None, min_valid=0.5):
    """Median and IQR over random starts (columns), drawn where at least min_valid of them are finite."""
    finite = np.isfinite(values)
    ok = finite.mean(axis=1) >= min_valid
    q = np.full((3, len(x)), np.nan)
    for i in np.flatnonzero(ok):
        q[:, i] = np.percentile(values[i][finite[i]], [25, 50, 75])
    ax.fill_between(x, q[0], q[2], color=color, alpha=0.16, lw=0)
    ax.plot(x, q[1], color=color, ls=ls, label=label)


# ---------------------------------------------------------------- Fig 3: coupled vs decoupled along the strips

def fig_strips():
    rand = np.load(os.path.join(RES, "robust_strips180.npz"))
    ident = np.load(os.path.join(RES, "explore_strips_gazebo.npz"), allow_pickle=True)
    rows = gazebo_table()
    v = rand["values"]
    fig, axes = plt.subplots(2, 2, figsize=(DCOL, 2.95), sharex=True, gridspec_kw=dict(height_ratios=[2.1, 1]))
    for col, axis in enumerate(("avoidant", "ambivalent")):
        ax = axes[0, col]
        median_band(ax, rand[f"{axis}/coupled/r"], v, MUTED, label="Coupled reference, random starts")
        median_band(ax, rand[f"{axis}/decoupled/r"], v, DECOUPLED, label="Decoupled, random starts")
        ax.plot(ident["values"], ident[f"{axis}/decoupled_learned/r"], color=DECOUPLED, ls="--", lw=1.0,
                label="Decoupled, identical start")
        belief = rand[f"{axis}/decoupled/belief_c"]
        ok = np.isfinite(belief).mean(1) >= 0.5
        ax.plot(v[ok], np.nanmedian(belief[ok], axis=1), color=DECOUPLED, ls=":", lw=1.1,
                label=r"Decoupled: internal $r(x_c,\hat{x}_p)$")
        plot_points(ax, by_value(select(rows, "coupled", "random", axis), axis, "r"), MUTED, "s", -0.012,
                    label="Gazebo, Coupled reference")
        plot_points(ax, by_value(select(rows, "decoupled", "random", axis), axis, "r"), DECOUPLED, "o", 0.0,
                    label="Gazebo, Decoupled (random)")
        plot_points(ax, by_value(select(rows, "decoupled", "identical", axis), axis, "r"), DECOUPLED, "o", 0.012,
                    hollow=True, label="Gazebo, Decoupled (identical)")
        ax.axhline(0, color=FAINT, lw=0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("need correlation $r(x_c, x_p)$")
        tidy(ax)
        panel(ax, "ab"[col], f"{NOUN[axis].split(' ')[0].capitalize()} strip")
        ax = axes[1, col]
        for kind, color in (("coupled", MUTED), ("decoupled", DECOUPLED)):
            ax.plot(v, rand[f"{axis}/{kind}/escaped"].mean(axis=1), color=color)
        esc_id = np.isfinite(ident[f"{axis}/decoupled_learned/t_div"]).astype(float)
        ax.plot(ident["values"], esc_id, color=DECOUPLED, ls="--", lw=1.0)
        for controller, start, color, marker, hollow, off in (("coupled", "random", MUTED, "s", False, -0.012),
                                                               ("decoupled", "random", DECOUPLED, "o", False, 0.0),
                                                               ("decoupled", "identical", DECOUPLED, "o", True, 0.012)):
            g = {}
            for r in select(rows, controller, start, axis):
                g.setdefault(r[axis], []).append(r["regime"] == "escaped")
            xs = np.array(sorted(g)) + off
            ax.plot(xs, [np.mean(g[k]) for k in sorted(g)], marker, ms=3.2, color=color,
                    mfc="white" if hollow else color, mec=color, mew=0.8, ls="")
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("fraction escaped")
        ax.set_xlabel(NOUN[axis])
        ax.set_xlim(-0.03, 1.0)
        tidy(ax)
        panel(ax, "cd"[col])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12), columnspacing=1.1, handlelength=2.0)
    fig.tight_layout(h_pad=0.4, w_pad=1.2)
    save(fig, "fig_strips")


# ---------------------------------------------------------------- Fig 4: belief and reality

def fig_belief():
    rows = gazebo_table()
    long = np.load(os.path.join(RES, "robust_strips.npz"))
    run_row = next(r for r in rows if r["batch"] == "E4" and r["controller"] == "decoupled"
                   and r["ambivalent"] == 0.2 and r["seed"] == 1)
    run = next(r for r in load_all() if r.index == run_row["index"])
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.2), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    ax = axes[0]
    child, carer = run.logs["child"], run.logs["parent"]
    t0 = child["t"][0]
    tc, tp = child["t"] - t0, carer["t"] - t0
    keep_c, keep_p = tc <= 240, tp <= 240
    ax.plot(tc[keep_c], child["state"][keep_c, 0], color=CHILD, label=r"child's own need $x_c$")
    ax.plot(tc[keep_c], child["state"][keep_c, 2], color=CHILD, ls=":", label=r"child's model of the caregiver $\hat{x}_p$")
    ax.plot(tp[keep_p], carer["state"][keep_p, 2], color=CAREGIVER, label=r"caregiver's own need $x_p$")
    ax.axhline(0, color=FAINT, lw=0.5)
    ax.set_xlim(0, 240); ax.set_xlabel("time (s)"); ax.set_ylabel("need")
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.3), fontsize=5.8, handlelength=1.8, columnspacing=0.8)
    tidy(ax)
    panel(ax, "a", rf"Gazebo run {run.index}: $\varepsilon_{{Am}}=0.2$, random start")
    v = long["values"]
    for ax, axis, letter in ((axes[1], "avoidant", "b"), (axes[2], "ambivalent", "c")):
        for key, color, ls, label in (("belief_c", DECOUPLED, ":", r"internal $r(x_c,\hat{x}_p)$"),
                                      ("r", DECOUPLED, "-", r"coordination $r(x_c,x_p)$"),
                                      ("track_r", CAREGIVER, "-.", r"partner tracking $r(\hat{x}_p,x_p)$")):
            vals = long[f"{axis}/decoupled/{key}"]
            ok = np.isfinite(vals).mean(1) >= 0.5
            ax.plot(v[ok], np.nanmedian(vals[ok], axis=1), color=color, ls=ls, label=label)
        cvals = long[f"{axis}/coupled/r"]
        ok = np.isfinite(cvals).mean(1) >= 0.5
        ax.plot(v[ok], np.nanmedian(cvals[ok], axis=1), color=MUTED, lw=0.9, label="Coupled reference: realised")
        ax.axhline(0, color=FAINT, lw=0.5)
        ax.set_ylim(-1.1, 1.1); ax.set_xlim(-0.02, 0.8); ax.set_xlabel(NOUN[axis])
        if axis == "avoidant":
            ax.set_ylabel("correlation (median of 40)")
            ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(1.12, -0.3), fontsize=5.8, handlelength=1.8,
                      columnspacing=0.8)
        tidy(ax)
        panel(ax, letter, NOUN[axis].split(" ")[0].capitalize() + ", random starts")
    fig.tight_layout(w_pad=0.6)
    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig_belief")


def fig_belief_log():
    """Column-width version of panel (a) of fig_belief: one robot's controller log (main text)."""
    rows = gazebo_table()
    run_row = next(r for r in rows if r["batch"] == "E4" and r["controller"] == "decoupled"
                   and r["ambivalent"] == 0.2 and r["seed"] == 1)
    run = next(r for r in load_all() if r.index == run_row["index"])
    fig, ax = plt.subplots(figsize=(COL, 1.75))
    child, carer = run.logs["child"], run.logs["parent"]
    t0 = child["t"][0]
    tc, tp = child["t"] - t0, carer["t"] - t0
    keep_c, keep_p = tc <= 180, tp <= 180
    ax.plot(tc[keep_c], child["state"][keep_c, 0], color=CHILD, label=r"child's need $x_c$")
    ax.plot(tc[keep_c], child["state"][keep_c, 2], color=CHILD, ls=":", label=r"child's model $\hat{x}_p$")
    ax.plot(tp[keep_p], carer["state"][keep_p, 2], color=CAREGIVER, label=r"caregiver's need $x_p$")
    ax.axhline(0, color=FAINT, lw=0.5)
    ax.set_xlim(0, 180); ax.set_ylim(-1.05, 1.4); ax.set_xlabel("time (s)"); ax.set_ylabel("need")
    ax.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.04), fontsize=5.6, handlelength=1.6,
              columnspacing=0.8, handletextpad=0.4)
    tidy(ax)
    fig.tight_layout()
    save(fig, "fig_belief_log")


# ---------------------------------------------------------------- Fig 5: regime map from random starts

def fig_maps():
    g = np.load(os.path.join(RES, "robust_grid.npz"))
    rows = gazebo_table()
    v = g["values"]
    step = v[1] - v[0]
    edges = np.concatenate([v - step / 2, [v[-1] + step / 2]])
    fig, axes = plt.subplots(1, 2, figsize=(DCOL * 0.8, 2.6), sharey=True)
    for ax, kind, letter, title in zip(axes, ("coupled", "decoupled"), "ab",
                                       ("Coupled reference", "Decoupled")):
        esc = g[f"{kind}/escaped"].astype(bool)
        quench = ((g[f"{kind}/freq_c"] == 0) | (g[f"{kind}/freq_p"] == 0)) & ~esc
        osc = ~esc & ~quench
        r = np.where(osc, g[f"{kind}/r"], np.nan)
        med = np.full(esc.shape[:2], np.nan)
        for (i, j), _ in np.ndenumerate(med):
            if osc[i, j].mean() >= 0.5:
                med[i, j] = np.nanmedian(r[i, j])
        mesh = ax.pcolormesh(edges, edges, med, cmap=DIVERGING, vmin=-1, vmax=1, shading="flat")
        for (i, j), _ in np.ndenumerate(med):
            if np.isfinite(med[i, j]):
                continue
            if esc[i, j].mean() >= 0.5:
                kw = dict(facecolor="white", hatch="////", edgecolor=FAINT)
            elif quench[i, j].mean() >= 0.5:
                kw = dict(facecolor=QUENCH, edgecolor="none")
            else:
                kw = dict(facecolor="white", hatch="....", edgecolor=FAINT)
            ax.add_patch(Rectangle((v[j] - step / 2, v[i] - step / 2), step, step, lw=0, **kw))
        # Gazebo: majority regime per parameter pair (random starts, strips and off-axis points)
        pts = {}
        for row in rows:
            if row["controller"] != kind or row["start"] != "random" or row["batch"] == "E4":
                continue
            pts.setdefault((row["avoidant"], row["ambivalent"]), []).append(row["regime"])
        for (av, am), regimes in pts.items():
            major = max(set(regimes), key=regimes.count)
            marker = {"escaped": "x", "oscillating": "o"}.get(major, "s")
            ax.plot(am, av, marker, ms=3.5, mfc="none" if marker != "x" else INK, mec=INK, mew=0.8, color=INK)
        ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(edges[0], edges[-1])
        ax.set_aspect("equal")
        ax.set_xlabel(NOUN["ambivalent"])
        tidy(ax, grid=None)
        panel(ax, letter, title)
    axes[0].set_ylabel(NOUN["avoidant"])
    cax = fig.add_axes([0.92, 0.32, 0.015, 0.45])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("median need correlation\n(oscillating starts)", fontsize=6.5)
    cb.ax.tick_params(labelsize=6)
    handles = [Patch(facecolor=QUENCH, label="mostly quenched"),
               Patch(facecolor="white", hatch="////", edgecolor=FAINT, label="mostly escape"),
               Patch(facecolor="white", hatch="....", edgecolor=FAINT, label="mixed"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK, label="Gazebo: oscillating"),
               Line2D([], [], marker="s", ls="", mfc="none", mec=INK, label="Gazebo: quenched"),
               Line2D([], [], marker="x", ls="", color=INK, label="Gazebo: escaped")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.47, -0.04), fontsize=5.8,
               columnspacing=0.9)
    fig.subplots_adjust(left=0.08, right=0.9, bottom=0.2, top=0.92, wspace=0.08)
    save(fig, "fig_maps")


# ---------------------------------------------------------------- Fig: estimator and partner model by mode

ESTIMATION_MODES = [("decoupled", ""), ("observed", "gt"), ("observed", "odom"), ("baseline", ""),
                    ("baseline-observed", "gt"), ("notebook", "gt")]


def fig_estimation():
    rows = gazebo_table()
    fig, axes = plt.subplots(2, 2, figsize=(DCOL, 3.4), sharex="col")
    modes = [m for m in ESTIMATION_MODES if any(r["controller"] == m[0] and r["distance"] == m[1] for r in rows)]
    offsets = np.linspace(-0.025, 0.025, len(modes))
    for col, axis in enumerate(("avoidant", "ambivalent")):
        for row, (key, ylabel) in enumerate((("est_r_c", "estimate vs true coupling ($r$)"),
                                             ("track_r", r"$r(\hat{x}_p, x_p)$"))):
            ax = axes[row, col]
            for (controller, distance), off in zip(modes, offsets):
                if key == "track_r" and controller == "notebook":
                    continue
                label, color, ls, marker = MODES[(controller, distance)]
                groups = by_value(select(rows, controller, "random", axis, distance), axis, key)
                groups = {k: g for k, g in groups.items() if k > 0}
                plot_points(ax, groups, color, marker, off, hollow=(distance == "odom"), label=label)
                xs = sorted(groups)
                ax.plot(np.array(xs) + off, [mean_ci(groups[x])[0] for x in xs], color=color, ls=ls, lw=0.7, alpha=0.7)
            ax.axhline(0, color=FAINT, lw=0.5)
            ax.set_ylim(-1.1, 1.1)
            ax.set_xlim(0.05, 0.95)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 1:
                ax.set_xlabel(NOUN[axis])
            tidy(ax)
            panel(ax, "abcd"[row * 2 + col], ("Learned estimate" if row == 0 else "Partner tracking") + ", "
                  + NOUN[axis].split(" ")[0])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), columnspacing=1.2)
    fig.tight_layout(h_pad=0.8, w_pad=1.2)
    save(fig, "fig_estimation")


# ---------------------------------------------------------------- Fig: anchoring the partner model (R4)

ANCHOR_DESIGNS = [("coupled", ""), ("decoupled", ""), ("observed", "gt"), ("baseline", ""), ("baseline-observed", "gt")]
# the variants of the three-readouts figure, with the network-assignment ablation next to its variant
READOUT_VARIANTS = [("coupled", ""), ("decoupled", ""), ("decoupled-own", ""), ("observed", "gt"), ("baseline", ""),
                    ("baseline-own", ""), ("baseline-observed", "gt")]
READOUT_SHORT = {("coupled", ""): "Coupled", ("decoupled", ""): "Decoupled", ("decoupled-own", ""): "own nets",
                 ("observed", "gt"): "Observed", ("baseline", ""): "Baseline", ("baseline-own", ""): "own nets",
                 ("baseline-observed", "gt"): "Baseline-obs."}


def readout_runs(rows, controller, distance):
    """Random-start runs under ambivalence, 0.1 <= eps_Am <= 0.4, that did not escape."""
    return [r for r in select(rows, controller, "random", "ambivalent", distance)
            if 0 < r["ambivalent"] <= 0.4 and r["regime"] != "escaped"]


def fig_anchoring():
    """The three readouts by variant, per-run values pooled over 0.1 <= eps_Am <= 0.4 (runs that did not escape)."""
    rows = gazebo_table()
    panels = [("est_r_c", r"$r(d,\ \Phi(\mathrm{true}))$", "Target", False),
              ("track_r", r"$r(\hat{x}_p, x_p)$", "Tracking", False),
              ("track_rmse_x", r"RMSE of $\hat{x}_p$", "Tracking error", False),
              ("detune_model", r"$|f(\hat{x}_p) - f(x_p)|$ (mHz)", "Detuning", False),
              ("r", r"$r(x_c, x_p)$", "Realised", True),
              ("reciprocity", "fraction of child calls", "Reciprocity", True)]
    variants = [v for v in READOUT_VARIANTS if readout_runs(rows, *v)]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 6, figsize=(DCOL, 2.05))
    handles = {}
    for ax, (key, ylabel, title, with_coupled), letter in zip(axes, panels, "abcdef"):
        for k, (controller, distance) in enumerate(variants):
            if controller == "coupled" and not with_coupled:
                continue
            own = controller.endswith("-own")
            label, color, ls, marker = MODES[(controller[:-4] if own else controller, distance)]
            runs = readout_runs(rows, controller, distance)
            v = np.array([r[key] for r in runs], float)
            v = v[np.isfinite(v)] * (1000 if key == "detune_model" else 1)
            style = dict(marker=marker, ms=2.4, ls="", alpha=0.6, mfc="none" if own else color, mec=color,
                         mew=0.6 if own else 0)
            ax.plot(k + rng.uniform(-0.2, 0.2, len(v)), v, **style)
            m, ci = mean_ci(v)
            ax.errorbar([k], [m], yerr=[ci], fmt="_", ms=8, mew=1.4, color=INK, elinewidth=0.9, capsize=2, zorder=6)
            name = label + (", own networks" if own else "")
            handles[name] = Line2D([], [], **dict(style, ms=4, alpha=1), label=f"{name} ($n={len(runs)}$)")
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(variants) - 0.4)
        if key in ("r", "track_r", "est_r_c"):
            ax.axhline(0, color=FAINT, lw=0.5)
            ax.set_ylim(-1.1, 1.1)
        elif key == "reciprocity":
            ax.set_ylim(-0.03, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.set_ylabel(ylabel, labelpad=1)
        tidy(ax)
        panel(ax, letter, title)
        ax.title.set_fontsize(7)
    fig.legend(handles=list(handles.values()), loc="lower center", ncol=4, fontsize=6, frameon=False,
               handletextpad=0.2, columnspacing=1.0, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(w_pad=0.5, rect=(0, 0.17, 1, 1))
    save(fig, "fig_anchoring")


# ---------------------------------------------------------------- Fig: long runs

def fig_long():
    runs = [r for r in load_all() if r.batch == "E4"]
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.0), sharex=True)
    styles = {("coupled", 0.2, 0.0): (MUTED, "-", "Coupled reference, avoidance 0.2"),
              ("coupled", 0.0, 0.2): (MUTED, "--", "Coupled reference, ambivalence 0.2"),
              ("decoupled", 0.2, 0.0): (DECOUPLED, "-", "Decoupled, avoidance 0.2"),
              ("decoupled", 0.0, 0.2): (DECOUPLED, "--", "Decoupled, ambivalence 0.2")}
    seen = set()
    for r in sorted(runs, key=lambda x: (x.controller, x.avoidant, x.seed)):
        color, ls, label = styles[(r.controller, r.avoidant, r.ambivalent)]
        w = gz.windowed(r, window=120.0, step=15.0)
        t = [x["t"] for x in w]
        lab = None if label in seen else label
        seen.add(label)
        axes[0].plot(t, [x["reciprocity"] for x in w], color=color, ls=ls, lw=1.0, label=lab)
        axes[1].plot(t, [x["r"] for x in w], color=color, ls=ls, lw=1.0)
    axes[0].set_ylabel("call reciprocity"); axes[0].set_ylim(-0.03, 1.05)
    axes[1].set_ylabel("need correlation $r(x_c,x_p)$"); axes[1].set_ylim(-1.1, 1.1)
    axes[1].axhline(0, color=FAINT, lw=0.5)
    for ax, letter, title in ((axes[0], "a", "Call reciprocity"),
                              (axes[1], "b", "Coordination of the needs")):
        ax.set_xlabel("time (s), 120 s windows")
        ax.set_xlim(0, 900)
        tidy(ax)
        panel(ax, letter, title)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_long")


# ---------------------------------------------------------------- Fig: Gazebo exemplars

# (batch, controller, avoidance, ambivalence, seed, title)
EXEMPLARS = [
    ("E2", "coupled", 0.2, 0.0, 1, "Coupled reference, avoidance 0.2"),
    ("E1r", "decoupled", 0.2, 0.0, 1, "Decoupled, avoidance 0.2"),
    ("E2", "coupled", 0.0, 0.2, 1, "Coupled reference, ambivalence 0.2"),
    ("E1r", "decoupled", 0.0, 0.2, 1, "Decoupled, ambivalence 0.2"),
    ("E1r", "decoupled", 0.0, 0.7, 1, "Decoupled, ambivalence 0.7"),
]


def exemplar_runs():
    rows = gazebo_table()
    by_index = {r.index: r for r in load_all() if r.index >= 54}
    out = []
    for batch, controller, av, am, seed, title in EXEMPLARS:
        row = next(r for r in rows if r["batch"] == batch and r["controller"] == controller and r["avoidant"] == av
                   and r["ambivalent"] == am and r["seed"] == seed and r["start"] == "random")
        out.append((by_index[row["index"]], row, title))
    return out


def fig_gazebo_regimes():
    fig = plt.figure(figsize=(DCOL, 7.0))
    gs = fig.add_gridspec(len(EXEMPLARS), 2, width_ratios=[1, 3.6], hspace=0.75, wspace=0.18)
    for k, (r, row, title) in enumerate(exemplar_runs()):
        a = metrics.analyse(r, "gt")
        n = min(len(r.t), 1800)
        t = r.t[:n]
        ax = fig.add_subplot(gs[k, 0])
        child = a["child"][:n]
        ax.plot(child[:, 0], child[:, 1], color=CHILD, lw=0.7)
        ax.plot(*child[0], "o", color=CHILD, ms=3)
        low, high = metrics.arena_limits(r)
        half = metrics.half_square(r)
        centre = a["parent"][:n].mean(axis=0)
        ax.add_patch(Rectangle(low, r.arena_size, r.arena_size, fill=False, lw=0.6, edgecolor=FAINT))
        ax.add_patch(Rectangle(centre - half, 2 * half, 2 * half, fill=False, ls=":", lw=0.7, edgecolor=MUTED))
        box = np.asarray(r.arena_centre) - r.box_size / 2
        ax.add_patch(Rectangle(box, r.box_size, r.box_size, facecolor=CAREGIVER, alpha=0.35, edgecolor=CAREGIVER, lw=0.6))
        ax.set_xlim(low[0] - 0.03, high[0] + 0.03); ax.set_ylim(low[1] - 0.03, high[1] + 0.03)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=7, loc="left", color=INK)
        ax.text(0.0, -0.08, f"run {r.index}, seed {r.seed}", transform=ax.transAxes, fontsize=5.8, color=MUTED, va="top")
        ax = fig.add_subplot(gs[k, 1])
        dist = a["distance"][:n]
        for s0, s1 in metrics.spans(r.child_state[:n] == 0, t):
            ax.axvspan(s0, s1 + 0.1, color="#e8f0fb", lw=0, zorder=0)
        for s0, s1 in metrics.spans(a["stuck"][:n], t):
            ax.axvspan(s0, s1 + 0.1, facecolor="none", hatch="////", edgecolor=FAINT, lw=0, zorder=0)
        ax.plot(t, dist, color=INK, lw=0.8)
        top = max(1.45, np.nanmax(dist) + 0.1)
        for y, series, col, lab in ((top + 0.14, r.vocal_child[:n], CHILD, "child calls"),
                                    (top + 0.46, r.vocal_parent[:n], CAREGIVER, "caregiver calls")):
            for s0, s1 in metrics.spans(series > 0, t):
                ax.plot([s0, s1 + 0.1], [y, y], color=col, lw=3.2, solid_capstyle="butt")
            ax.text(181, y, lab, fontsize=5.6, va="center", color=MUTED)
        recip = row["reciprocity"]
        ax.text(0.0, 1.02, f"need correlation {row['r']:+.2f}  ·  call reciprocity "
                           f"{'n/a' if not np.isfinite(recip) else f'{recip:.2f}'}  ·  "
                           f"{100 * row['near']:.0f}% of time near the caregiver  ·  {row['regime']}",
                transform=ax.transAxes, fontsize=6, color=MUTED)
        ax.set_xlim(0, 180); ax.set_ylim(0, top + 0.6)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_ylabel("distance (m)", fontsize=6.5)
        tidy(ax)
        if k == len(EXEMPLARS) - 1:
            ax.set_xlabel("time (s)")
        else:
            ax.set_xticklabels([])
    handles = [Line2D([], [], color=CHILD, lw=0.8, label="child path"),
               Patch(facecolor=CAREGIVER, alpha=0.35, label="caregiver box"),
               Line2D([], [], color=MUTED, ls=":", label="near-caregiver square"),
               Line2D([], [], color=INK, lw=0.8, label="child-caregiver distance"),
               Patch(facecolor="#e8f0fb", label="child in approach"),
               Patch(facecolor="white", hatch="////", edgecolor=FAINT, label="stuck (excluded)")]
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, 0.035), fontsize=6)
    save(fig, "fig_gazebo_regimes")


# ---------------------------------------------------------------- Fig: dynamics of the coupled-oscillator model (npj Section 4)

STRIP_COLOURS = {"avoidant": CHILD, "ambivalent": CAREGIVER}


def _correlation_panel(ax, c, letter, legend_anchor=(0.62, 0.62), legend_cols=1):
    """Correlation of the two needs along the strips: random starts (median, IQR) and the anti-synchronised start."""
    v = c["strips/values"]
    for axis, col in STRIP_COLOURS.items():
        rr = c[f"strips/{axis}/random/r"]
        ok = np.isfinite(rr).mean(1) >= 0.5
        q = np.nanpercentile(rr[ok], [25, 50, 75], axis=1)
        ax.fill_between(v[ok], q[0], q[2], color=col, alpha=0.18, lw=0)
        ax.plot(v[ok], q[1], color=col, lw=1.1, label=NOUN[axis].split(" ")[0])
        anti = c[f"strips/{axis}/anti/r"]
        ax.plot(v, anti, color=col, ls="--", lw=0.8)
    ax.axhline(0, color=FAINT, lw=0.5); ax.set_ylim(-1.1, 1.1); ax.set_xlim(0, 1)
    ax.set_xlabel(r"coupling $\varepsilon$"); ax.set_ylabel(r"need correlation $r(x_c,x_p)$")
    ax.legend(loc="upper center", bbox_to_anchor=legend_anchor, ncol=legend_cols, fontsize=5.8, handlelength=1.4)
    tidy(ax); panel(ax, letter, "Correlation"); ax.title.set_fontsize(7)


def _phase_panel(fig, ax, c, letter, legend_anchor=(0.45, -0.17)):
    """Mean relative phase over the plane from the anti-synchronised start, with steady states, escapes and
    irregular oscillation marked."""
    pl = c["plane/values"]; step = pl[1] - pl[0]
    phase = np.where(c["plane/regime"] == 0, c["plane/phase"], np.nan)
    im = ax.imshow(phase.T, origin="lower", extent=(-step / 2, 1 + step / 2, -step / 2, 1 + step / 2),
                   cmap="coolwarm_r", vmin=0, vmax=180, aspect="equal")
    reg = c["plane/regime"]; nsec = c["plane/n_section_values"]
    for i, av in enumerate(pl):
        for j, am in enumerate(pl):
            box = dict(xy=(av - step / 2, am - step / 2), width=step, height=step, lw=0)
            if reg[i, j] == 1:
                ax.add_patch(Rectangle(**box, facecolor=QUENCH))
            elif reg[i, j] == 2:
                ax.add_patch(Rectangle(**box, facecolor="white", hatch="////", edgecolor=FAINT))
            elif nsec[i, j] >= 3:
                ax.plot(av, am, ".", ms=1.6, color=INK)
    ax.set_xlim(-step / 2, 1 + step / 2); ax.set_ylim(-step / 2, 1 + step / 2)
    ax.set_xlabel(NOUN["avoidant"]); ax.set_ylabel(NOUN["ambivalent"])
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03, ticks=[0, 90, 180])
    cb.set_label("relative phase (deg)", fontsize=6.2); cb.ax.tick_params(labelsize=6)
    panel(ax, letter, "Relative phase"); ax.title.set_fontsize(7)
    handles = [Patch(facecolor=QUENCH, label="steady state"),
               Patch(facecolor="white", hatch="////", edgecolor=FAINT, label="escape"),
               Line2D([], [], marker=".", ls="", color=INK, label="irregular")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=legend_anchor, ncol=3, fontsize=5.6,
              handlelength=1.2, columnspacing=0.6)


def _dynamics_figure(form, name):
    c = np.load(os.path.join(RES, "canonical.npz" if form == "canonical" else "canonical_distance.npz"))
    fig = plt.figure(figsize=(DCOL, 3.15))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], hspace=0.62, wspace=0.42)
    v = c["strips/values"]
    # (a) correlation, random starts (median, IQR) and the anti-synchronised start
    _correlation_panel(fig.add_subplot(gs[0, 0]), c, "a")
    # (b) frequency of the child's need, random starts
    ax = fig.add_subplot(gs[0, 1])
    for axis, col in STRIP_COLOURS.items():
        ff = c[f"strips/{axis}/random/freq"] * 100
        ok = np.isfinite(ff).mean(1) >= 0.5
        ax.plot(v[ok], np.nanmedian(ff[ok], 1), color=col, lw=1.1)
    ax.set_xlim(0, 1); ax.set_ylim(bottom=0)
    ax.set_xlabel(r"coupling $\varepsilon$"); ax.set_ylabel("cycles per 100 time units")
    tidy(ax); panel(ax, "b", "Frequency"); ax.title.set_fontsize(7)
    # (c,d) bifurcation diagrams: x_p at upward crossings of y_c = 0, and steady states
    pv = c["poincare/values"]
    for k, axis in enumerate(("ambivalent", "avoidant")):
        ax = fig.add_subplot(gs[1, k])
        counts = c[f"poincare/{axis}/counts"]
        pts = c[f"poincare/{axis}/points"]
        xs = np.repeat(pv, counts)
        ax.plot(xs, pts, ",", color=STRIP_COLOURS[axis], alpha=0.6, rasterized=True)
        ax.plot(xs, pts, "o", ms=0.4, color=STRIP_COLOURS[axis], alpha=0.5, mew=0, rasterized=True)
        final = c[f"poincare/{axis}/final"][:, 2]
        steady = counts == 0
        esc = c[f"poincare/{axis}/escaped"]
        ax.plot(pv[steady & ~esc], final[steady & ~esc], ".", ms=1.2, color=INK)
        if esc.any():
            ax.plot(pv[esc], np.full(esc.sum(), 1.05), "x", ms=2, color=MUTED, mew=0.5)
        ax.set_xlim(0, 1); ax.set_ylim(-1.15, 1.15)
        ax.set_xlabel(NOUN[axis]); ax.set_ylabel(r"$x_p$ at $y_c=0$")
        tidy(ax); panel(ax, "cd"[k], "Bifurcation, " + NOUN[axis].split(" ")[0]); ax.title.set_fontsize(7)
    # (e) relative phase over the plane, anti-synchronised start
    _phase_panel(fig, fig.add_subplot(gs[:, 2]), c, "e")
    save(fig, name)


def fig_canonical():
    _dynamics_figure("canonical", "fig_canonical")


def fig_canonical_distance():
    _dynamics_figure("distance", "fig_canonical_distance")


def fig_regimes():
    """Column-width version of panels (a) and (e) of fig_canonical (main text; the full figure is in the supplement)."""
    c = np.load(os.path.join(RES, "canonical.npz"))
    fig = plt.figure(figsize=(COL, 1.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    _correlation_panel(ax, c, "a", legend_anchor=(0.4, -0.3))
    phase_ax = fig.add_subplot(gs[0, 1])
    _phase_panel(fig, phase_ax, c, "b", legend_anchor=(0.4, -0.3))
    # give (a) the height of the square map, so the titles and legends line up
    fig.canvas.draw()
    pa, pb = ax.get_position(), phase_ax.get_position()
    ax.set_position([pa.x0, pb.y0, pa.width, pb.height])
    save(fig, "fig_regimes")


# ---------------------------------------------------------------- Fig: simulated strange situation (npj Section 6.2)

SSP_TITLES = {"secure": r"Secure, $\varepsilon_{Am}$ 0.1", "moderately avoidant": r"$\varepsilon_{Av}$ 0.4",
              "strongly avoidant": r"$\varepsilon_{Av}$ 0.9", "moderately ambivalent": r"$\varepsilon_{Am}$ 0.4",
              "strongly ambivalent": r"$\varepsilon_{Am}$ 0.9"}
SSP_TRACE_HANDLES = [Line2D([], [], color=INK, lw=0.8, label="distance"),
                     Line2D([], [], color=CHILD, lw=3, label="child calls"),
                     Line2D([], [], color=CAREGIVER, lw=3, label="caregiver calls")]


def _ssp_traces(ax, d, k):
    """Distance between the agents and their calls over the last 1500 steps of regime k."""
    t = np.arange(int(d["steps"]))
    w = slice(-1500, None)
    ax.plot(t[w] - t[w][0], d["dist"][w, k], color=INK, lw=0.6)
    for y, series, col in ((1.62, d["voc_c"][w, k], CHILD), (1.82, d["voc_p"][w, k], CAREGIVER)):
        on = series > 0
        ax.fill_between(t[w] - t[w][0], y - 0.07, y + 0.07, where=on, color=col, lw=0, step="mid")
    ax.set_ylim(0, 1.95); ax.set_xlim(0, 1500)
    ax.set_yticks([0, 0.5, 1.0, 1.5]) if k == 0 else ax.set_yticks([])
    if k == 0:
        ax.set_ylabel("distance", fontsize=6.5)
    ax.set_xlabel("step", fontsize=6.5)
    ax.tick_params(labelsize=5.8)
    tidy(ax)


def fig_sim_ssp():
    d = np.load(os.path.join(RES, "sim_ssp.npz"))
    names = [str(n) for n in d["names"]]
    n = len(names)
    fig = plt.figure(figsize=(DCOL, 2.75))
    gs = fig.add_gridspec(2, n, height_ratios=[1.25, 1], hspace=0.45, wspace=0.3)
    last = slice(-3000, None)
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[0, k])
        cp = d["child_pos"][last, k]
        ax.plot(cp[:, 0], cp[:, 1], color=CHILD, lw=0.35, alpha=0.8)
        pp = d["parent_pos"][last, k]
        ax.plot(pp[:, 0], pp[:, 1], color=CAREGIVER, lw=0.6)
        ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False, ls=":", lw=0.7, edgecolor=MUTED))
        ax.set_xlim(-1.05, 1.05); ax.set_ylim(-1.05, 1.05); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(FAINT)
        ax.set_title(f"({'abcde'[k]}) {SSP_TITLES[name]}", fontsize=6.6, loc="left", color=INK)
        ax.text(0.5, -0.1, f"near caregiver {100 * d['near'][k]:.0f}%", transform=ax.transAxes, ha="center",
                va="top", fontsize=6, color=MUTED)
        _ssp_traces(fig.add_subplot(gs[1, k]), d, k)
    handles = [Line2D([], [], color=CHILD, lw=0.8, label="child path"), Line2D([], [], color=CAREGIVER, lw=0.8, label="caregiver"),
               Line2D([], [], color=MUTED, ls=":", label="near-caregiver square")] + SSP_TRACE_HANDLES
    fig.legend(handles=handles, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.13), fontsize=6)
    save(fig, "fig_sim_ssp")


def fig_sim_traces():
    """The distance-and-calls row of fig_sim_ssp, with the time near the caregiver in each title (main text; the
    full figure, with the paths, is in the supplement)."""
    d = np.load(os.path.join(RES, "sim_ssp.npz"))
    names = [str(n) for n in d["names"]]
    fig, axes = plt.subplots(1, len(names), figsize=(DCOL, 1.4))
    for k, (ax, name) in enumerate(zip(axes, names)):
        _ssp_traces(ax, d, k)
        ax.set_title(f"({'abcde'[k]}) {SSP_TITLES[name]}\nnear caregiver {100 * d['near'][k]:.0f}%",
                     fontsize=6.6, loc="left", color=INK, linespacing=1.3)
    fig.legend(handles=SSP_TRACE_HANDLES, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=6)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.1, 1, 1))
    save(fig, "fig_sim_traces")


# ---------------------------------------------------------------- Fig: Gazebo sweeps

def fig_gazebo_sweeps():
    rows = gazebo_table()
    inv = np.load(os.path.join(RES, "inverse.npz"))
    Xtr, Ytr = inv["Xtr"], inv["Ytr"]
    seeds = len(Ytr) // len(np.unique(Ytr[:, :2], axis=0))
    rand_tr = np.tile(np.arange(seeds) >= 10, len(Ytr) // seeds)      # stand-in runs from random starts
    cols = [("near", "time near caregiver (%)", 0), ("child_calling", "child calling (%)", 3),
            ("carer_calling", "caregiver calling (%)", 5), ("reciprocity", "call reciprocity (%)", None)]
    fig, axes = plt.subplots(2, 4, figsize=(DCOL, 3.2), sharex="row")
    for row, axis in enumerate(("avoidant", "ambivalent")):
        k = 0 if axis == "avoidant" else 1
        other = 1 - k
        for c, (key, label, fi) in enumerate(cols):
            ax = axes[row, c]
            if fi is not None:
                sel = (Ytr[:, other] == 0) & rand_tr
                sv = np.unique(Ytr[sel, k])
                m = np.array([Xtr[sel & (Ytr[:, k] == u), fi].mean() for u in sv]) * 100
                sd = np.array([Xtr[sel & (Ytr[:, k] == u), fi].std() for u in sv]) * 100
                ax.fill_between(sv, m - sd, m + sd, color=DECOUPLED, alpha=0.15, lw=0)
                ax.plot(sv, m, color=DECOUPLED, lw=0.8, alpha=0.8, label="stand-in, Decoupled (mean $\\pm$ s.d.)")
            for controller, start, color, marker, hollow, off, lab in (
                    ("coupled", "random", MUTED, "s", False, -0.012, "Gazebo, Coupled reference"),
                    ("decoupled", "random", DECOUPLED, "o", False, 0.0, "Gazebo, Decoupled (random)"),
                    ("decoupled", "identical", DECOUPLED, "o", True, 0.012, "Gazebo, Decoupled (identical)")):
                g = by_value(select(rows, controller, start, axis), axis, key, include_escaped=True)
                g = {x: 100 * vals for x, vals in g.items()}
                plot_points(ax, g, color, marker, off, hollow=hollow, ms=2.8, label=lab + " (mean $\\pm$ s.d.)", err="sd")
            ax.set_ylim(-3, 103)
            ax.set_xlabel(NOUN[axis])
            ax.set_ylabel(label)
            ax.set_xlim(-0.04, 0.94)
            tidy(ax)
            panel(ax, "abcdefgh"[row * 4 + c])
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "fig_gazebo_sweeps")


# ---------------------------------------------------------------- Fig: inverse problem

def fig_inverse():
    inv = np.load(os.path.join(RES, "inverse.npz"))
    classes = [c for c in inv["classes"]]
    Yte = inv["Yte"]
    fig, axes = plt.subplots(1, 3, figsize=(DCOL * 0.85, 2.0), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
    for ax, name, letter, title in ((axes[0], "child", "a", "Child only"), (axes[1], "dyad", "b", "Both agents")):
        m = inv[f"{name}/confusion"]
        ax.imshow(m / np.maximum(1, m.sum(1, keepdims=True)), cmap=SEQ, vmin=0, vmax=1)
        for (i, j), c in np.ndenumerate(m):
            frac = c / max(1, m[i].sum())
            ax.text(j, i, str(c), ha="center", va="center", fontsize=7, color="white" if frac > 0.6 else INK)
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(classes, rotation=30, ha="right"); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("generating label" if name == "child" else "")
        panel(ax, letter, f"{title}: {inv[f'{name}/test_acc'] * 100:.0f}%")
        ax.title.set_fontsize(6.8)
    ax = axes[2]
    axis_av = Yte[:, 0] > 0
    true = Yte[:, 0] + Yte[:, 1]
    for name, mk, off in (("child", "o", -0.01), ("dyad", "s", 0.01)):
        pred = inv[f"{name}/test_eps"]
        est = np.where(axis_av, pred[:, 0], pred[:, 1])
        for sel, col in ((axis_av, CHILD), (~axis_av & (true > 0), CAREGIVER)):
            g = {}
            for x, e in zip(true[sel], est[sel]):
                g.setdefault(round(x, 2), []).append(e)
            xs = sorted(g)
            ax.errorbar(np.array(xs) + off, [np.mean(g[x]) for x in xs], yerr=[np.std(g[x]) for x in xs], fmt=mk,
                        ms=2.8, color=col, mfc=col if name == "dyad" else "white", mec=col, mew=0.8, elinewidth=0.6)
    ax.plot([0, 0.9], [0, 0.9], color=FAINT, lw=0.6)
    ax.set_xlabel(r"generating $\varepsilon$ (Gazebo runs)"); ax.set_ylabel(r"estimated $\varepsilon$ (mean $\pm$ s.d.)")
    ax.set_xlim(-0.05, 0.95); ax.set_ylim(-0.05, 0.95); ax.set_aspect("equal")
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, label="child only"),
                       Line2D([], [], marker="s", ls="", color=MUTED, label="both agents"),
                       Line2D([], [], marker="s", ls="", color=CHILD, label="avoidance runs"),
                       Line2D([], [], marker="s", ls="", color=CAREGIVER, label="ambivalence runs")],
              loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=5.6, handletextpad=0.3, columnspacing=0.8)
    tidy(ax, grid="both")
    panel(ax, "c", "Parameter estimate")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_inverse")


# ---------------------------------------------------------------- Fig: stability of the identical start (appendix)

def fig_stability():
    s = np.load(os.path.join(RES, "robust_stability.npz"))
    fig, ax = plt.subplots(figsize=(COL, 1.9))
    for j, (eav, color) in enumerate(zip(s["eavs"], (DECOUPLED, CHILD, "#6250d6"))):
        ax.plot(s["sizes"], s["decoupled/t_break_s"][:, j], "o-", color=color, ms=3, label=rf"$\varepsilon_{{Av}}={eav:.1f}$")
    ax.set_xscale("log")
    ax.set_xlabel(r"initial difference in accumulated need $|y_c-y_p|$")
    ax.set_ylabel("time to leave synchrony (s)")
    ax.axhline(180, color=FAINT, lw=0.6, ls=":")
    ax.text(1.3e-9, 190, "length of a 180 s run", fontsize=5.8, color=MUTED)
    ax.legend(loc="upper right")
    tidy(ax, grid="both")
    fig.tight_layout()
    save(fig, "fig_stability")


# ---------------------------------------------------------------- Fig 6: learned coupling offline

def fig_estimator():
    e = np.load(os.path.join(RES, "estimator_eval.npz"))
    vals = e["open/values"]
    ntr = int(e["open/n_train"])
    order = np.argsort(vals)
    fig, axes = plt.subplots(1, 4, figsize=(DCOL, 1.9), gridspec_kw=dict(width_ratios=[1.25, 1.1, 1, 1]))
    ax = axes[0]
    for kind, ls in (("physical", "-"), ("emotional", "--")):
        for agent, col in (("child", CHILD), ("parent", CAREGIVER)):
            rmse = np.nanmedian(e[f"open/{kind}/full/{agent}/rmse"], axis=1)
            ax.plot(vals[order], rmse[order], color=col, ls=ls, lw=1.0)
            ax.plot(vals[:ntr], rmse[:ntr], "o", color=col, ms=2.5)
            ax.plot(vals[ntr:], rmse[ntr:], "o", mfc="white", mec=col, ms=3, mew=0.8)
    ax.set_yscale("log")
    ax.set_ylim(2e-3, 40)
    ax.set_xlabel(r"coupling parameter $\varepsilon$"); ax.set_ylabel("RMSE of coupling estimate")
    ax.legend(handles=[Line2D([], [], color=INK, ls="-", label="physical"), Line2D([], [], color=INK, ls="--", label="emotional"),
                       Line2D([], [], color=CHILD, label="child net"), Line2D([], [], color=CAREGIVER, label="caregiver net"),
                       Line2D([], [], marker="o", ls="", color=MUTED, ms=2.5, label="trained value"),
                       Line2D([], [], marker="o", ls="", mfc="white", mec=MUTED, ms=3, label="held out")],
              loc="upper right", fontsize=5.5, handlelength=1.6, ncol=2, columnspacing=0.8)
    tidy(ax); panel(ax, "a", "Open-loop accuracy")
    # ablations: median R2 over the parameter values where the target varies (training range up to 0.67)
    ax = axes[1]
    rng = (vals > 0.05) & (vals < 0.7)
    groups = [("physical", ("full", "distance", "own_state"), ("all inputs", "no distance", "no own state")),
              ("emotional", ("full", "vocal", "own_state"), ("all inputs", "silent vocal", "no own state"))]
    x0 = 0
    ticks, labels = [], []
    for kind, tags, names in groups:
        for k, (tag, name) in enumerate(zip(tags, names)):
            for a, (agent, col) in enumerate((("child", CHILD), ("parent", CAREGIVER))):
                med = np.nanmedian(e[f"open/{kind}/{tag}/{agent}/r2"][rng])
                ax.bar(x0 + a * 0.38, np.clip(med, -1, 1), width=0.34, color=col, edgecolor="white", lw=0.5)
                if med < -1:                       # clipped bar: mark it and print the value
                    ax.text(x0 + a * 0.38, -1.0, f"{med:.1f}", rotation=90, ha="center", va="bottom", fontsize=5,
                            color="white")
            ticks.append(x0 + 0.19); labels.append(name)
            x0 += 1
        x0 += 0.5
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=5.8)
    ax.axhline(0, color=FAINT, lw=0.5)
    ax.text(1.2, 1.08, "physical", ha="center", fontsize=6, color=MUTED)
    ax.text(4.7, 1.08, "emotional", ha="center", fontsize=6, color=MUTED)
    ax.set_ylim(-1.05, 1.15); ax.set_ylabel(r"median $R^2$, $0.1\leq\varepsilon\leq0.67$")
    tidy(ax); panel(ax, "b", "Input ablation")
    ax.title.set_position((0, 1.06))
    cv = e["closed/values"]
    for ax, axis, letter in ((axes[2], "ambivalent", "c"), (axes[3], "avoidant", "d")):
        for mode, col, ls in (("analytic", MUTED, "--"), ("learned", DECOUPLED, "-")):
            r = e[f"closed/{axis}/{mode}/r"]
            m, sd = np.nanmean(r, 0), np.nanstd(r, 0)
            ax.fill_between(cv, m - sd, m + sd, color=col, alpha=0.18, lw=0)
            ax.plot(cv, m, color=col, ls=ls, label="analytic coupling" if mode == "analytic" else "learned coupling")
        ax.set_ylim(-1.1, 1.1); ax.set_xlabel(NOUN[axis])
        if axis == "ambivalent":
            ax.set_ylabel("need correlation (closed loop)")
        else:
            ax.legend(loc="lower right", fontsize=5.8, handlelength=1.8)
        tidy(ax); panel(ax, letter, "Closed loop")
    fig.tight_layout(w_pad=0.9)
    save(fig, "fig_estimator")


# ---------------------------------------------------------------- Fig: arena frame

def fig_arena():
    img = plt.imread(os.path.join(RES, "arena_frame_34.png"))
    fig, ax = plt.subplots(figsize=(COL * 0.8, COL * 0.8 * 480 / 640))
    ax.imshow(img)
    ax.set_axis_off()
    ax.annotate("caregiver on box\n(AprilTags on each face)", xy=(325, 240), xytext=(140, 110), fontsize=6.5,
                color="white", arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    ax.annotate("child", xy=(435, 245), xytext=(470, 140), fontsize=6.5, color="white",
                arrowprops=dict(arrowstyle="-", color="white", lw=0.6))
    ax.text(330, 470, "2.61 m", ha="center", va="bottom", fontsize=6.5, color="white")
    save(fig, "fig_arena")


# ---------------------------------------------------------------- Fig: odometry (appendix)

def fig_odometry():
    runs = [r for r in load_all() if r.batch == "E1r"]
    fig, axes = plt.subplots(1, 2, figsize=(DCOL * 0.8, 2.1), gridspec_kw=dict(width_ratios=[1, 1.7]))
    r = runs[0]
    ax = axes[0]
    ax.plot(r.child_gt[:, 0], r.child_gt[:, 1], color=CHILD, lw=0.7, label="ground truth")
    ax.plot(r.child_odom[:, 0], r.child_odom[:, 1], color=CAREGIVER, lw=0.7, label="wheel odometry")
    low, high = metrics.arena_limits(r)
    ax.add_patch(Rectangle(low, r.arena_size, r.arena_size, fill=False, lw=0.6, edgecolor=FAINT))
    ax.set_aspect("equal"); ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper center", fontsize=5.8, bbox_to_anchor=(0.5, -0.28), ncol=2, columnspacing=0.8)
    tidy(ax, grid=None); panel(ax, "a", f"Child path, run {r.index}")
    ax = axes[1]
    grid_t = np.arange(0, 180, 1.0)
    errs = []
    for x in runs:
        e = np.hypot(*(x.child_odom - x.child_gt).T)
        ax.plot(x.t, e, color=FAINT, lw=0.4)
        errs.append(np.interp(grid_t, x.t, e))
    ax.plot(grid_t, np.median(errs, 0), color=INK, lw=1.2, label=f"median of {len(runs)} runs")
    ax.axhline(r.arena_size, color=MUTED, ls=":", lw=0.6)
    ax.text(5, r.arena_size + 0.05, "arena width", fontsize=6, color=MUTED)
    ax.set_xlim(0, 180); ax.set_xlabel("time (s)"); ax.set_ylabel("odometry error (m)")
    ax.legend(loc="upper left")
    tidy(ax); panel(ax, "b", "Odometry error (private-model runs, random starts)")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig_odometry")


if __name__ == "__main__":
    np.seterr(all="ignore")
    which = sys.argv[1:] or ["regimes", "canonical", "canonical_distance", "sim_traces", "sim_ssp", "strips", "belief",
                             "belief_log", "maps", "estimator", "estimation", "anchoring", "long", "gazebo_regimes",
                             "gazebo_sweeps", "inverse", "stability", "arena", "odometry"]
    for name in which:
        globals()[f"fig_{name}"]()
