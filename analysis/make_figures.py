#!/usr/bin/env python3
"""
Figures and statistics for the robot experiments, following the npj paper
("Modelling Human Attachment using Coupled Oscillators in a Simulated and
Embodied Robot Dyad"), plus improved versions of the same figures.

Usage (inside the container):
    /workspace/.venv/bin/python /root/analysis/make_figures.py
    /workspace/.venv/bin/python /root/analysis/make_figures.py --source odom

Output:
    analysis/figures/paper_style/   same layout as the paper's figures
    analysis/figures/improved/      redesigned versions
    analysis/results/               statistics and per-run summary tables
"""
import os
import csv
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from load import load_all, find_run, default_data_dir
import metrics

HERE = os.path.dirname(os.path.abspath(__file__))

# The three regimes of the paper's robot experiment: (name, avoidant, ambivalent)
REGIMES = [
    ("Secure", 0.0, 0.1),
    ("Strongly avoidant", 0.9, 0.0),
    ("Moderately ambivalent", 0.0, 0.4),
]

# Colours: child and caregiver keep one colour each across the improved figures
CHILD = "#2a78d6"
PARENT = "#eb6834"
INK = "#0b0b0b"
MUTED = "#52514e"
FAINT = "#b9b8b2"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
OCCUPANCY = LinearSegmentedColormap.from_list(
    "occupancy", [SURFACE, "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

# Parameter names as nouns for labels
NOUN = {"avoidant": "avoidance", "ambivalent": "ambivalence"}

BINS = 50            # heat-map grid, as in the paper
BLOCK_SECONDS = 10   # block length for bootstrap intervals


# ----------------------------------------------------------------------------- helpers

def save(fig, folder, name):
    out_dir = os.path.join(HERE, "figures", folder)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{name}.{ext}")
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
        fix_owner(path)
    fix_owner(out_dir)
    fix_owner(os.path.dirname(out_dir))
    plt.close(fig)
    print(f"  wrote figures/{folder}/{name}.png")


def fix_owner(path):
    """Files written from the (root) container get the owner of the analysis folder."""
    try:
        st = os.stat(HERE)
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass


def spans(mask, t):
    """(start, end) times of the contiguous True stretches of a mask."""
    mask = np.asarray(mask, dtype=bool)
    edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
    return [(t[s], t[e]) for s, e in zip(starts, ends)]


def clean_axes(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, color=FAINT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)


def arena_limits(run):
    low = np.asarray(run.arena_centre) - run.arena_size / 2.0
    return low, low + run.arena_size


def half_square(run):
    """Half side of the 'close to the caregiver' square: the central quarter of the arena."""
    return run.arena_size / 4.0


def draw_arena(ax, run, parent_xy, improved=True):
    """Arena outline, caregiver box and the dotted proximity square."""
    low, high = arena_limits(run)
    half = half_square(run)
    centre = parent_xy.mean(axis=0)
    ax.add_patch(Rectangle(centre - half, 2 * half, 2 * half, fill=False, linestyle=":",
                           linewidth=1.2, edgecolor=MUTED if improved else "k", zorder=3))
    if improved:
        ax.add_patch(Rectangle(low, run.arena_size, run.arena_size, fill=False, linewidth=1, edgecolor=FAINT))
        box = np.asarray(run.arena_centre) - run.box_size / 2.0
        ax.add_patch(Rectangle(box, run.box_size, run.box_size, facecolor=PARENT, alpha=0.25,
                               edgecolor=PARENT, linewidth=1, zorder=4))
    pad = 0.05
    ax.set_xlim(low[0] - pad, high[0] + pad)
    ax.set_ylim(low[1] - pad, high[1] + pad)
    ax.set_aspect("equal")


def analyse(run, source):
    """Per-run quantities shared by the figures."""
    child, parent = run.positions(source)
    stuck = metrics.stuck_mask(child, run.child_state, run.arena_centre, run.arena_size, BINS)
    near = metrics.near_caregiver(child, parent, half_square(run))
    return dict(child=child, parent=parent, distance=run.distance(source), stuck=stuck, near=near)


def occupancy(run, child_xy, keep):
    low, high = arena_limits(run)
    hist, _, _ = np.histogram2d(child_xy[keep, 0], child_xy[keep, 1], bins=BINS,
                                range=[[low[0], high[0]], [low[1], high[1]]])
    return hist.T / max(1, keep.sum())


def regime_runs(runs):
    found = [(name, find_run(runs, av, am)) for name, av, am in REGIMES]
    return [(name, run) for name, run in found if run is not None]


def sweep_runs(runs, parameter):
    """Longest run with positions per value of one parameter, the other held at zero."""
    other = "ambivalent" if parameter == "avoidant" else "avoidant"
    values = sorted({getattr(r, parameter) for r in runs
                     if r.has_positions and getattr(r, other) == 0})
    picked = []
    for value in values:
        run = find_run(runs, **{parameter: value, other: 0.0})
        if run is not None:
            picked.append(run)
    return picked


# ----------------------------------------------------------------------------- paper-style figures

def paper_fig11(regimes, source):
    """Left: child trajectory relative to the caregiver. Right: distance and vocalisation."""
    fig, axes = plt.subplots(len(regimes), 2, figsize=(11, 3.6 * len(regimes)),
                             gridspec_kw={"width_ratios": [1, 2.2]}, squeeze=False)
    for (name, run), (ax_traj, ax_dist) in zip(regimes, axes):
        a = analyse(run, source)
        ax_traj.plot(a["child"][:, 0], a["child"][:, 1], color="tab:blue", linewidth=0.8)
        ax_traj.plot(a["parent"][:, 0], a["parent"][:, 1], "ks", markersize=4)
        draw_arena(ax_traj, run, a["parent"], improved=False)
        ax_traj.set_xlabel("x (m)")
        ax_traj.set_ylabel("y (m)")
        ax_traj.set_title(f"{name} ($\\epsilon_{{Av}}$={run.avoidant:g}, $\\epsilon_{{Am}}$={run.ambivalent:g})")

        iteration = np.arange(len(run.t))
        centre_distance = np.hypot(*(a["child"] - np.asarray(run.arena_centre)).T)
        normalised = np.minimum(1.0, centre_distance / (run.arena_size / np.sqrt(2)))
        ax_dist.fill_between(iteration, 0, run.vocal_child * 0.2, step="mid", color="tab:blue", alpha=0.8,
                             label="child vocalisation")
        ax_dist.plot(iteration, normalised, color="tab:orange", linewidth=1.2, label="distance")
        for start, end in spans(a["stuck"], iteration):
            ax_dist.axvspan(start, end, color="0.85", label="stuck (excluded)")
        ax_dist.set_ylim(0, 1)
        ax_dist.set_xlabel("iteration")
        ax_dist.set_ylabel("normalised distance from centre")
        handles, labels = ax_dist.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax_dist.legend(unique.values(), unique.keys(), fontsize=8, loc="upper right")
    fig.tight_layout()
    save(fig, "paper_style", "fig11_trajectory_distance_vocalisation")


def paper_fig12(regimes, source):
    """Heat maps of the child's position on a 50 x 50 grid, stuck periods removed."""
    fig, axes = plt.subplots(1, len(regimes), figsize=(4.2 * len(regimes), 4), squeeze=False)
    for (name, run), ax in zip(regimes, axes[0]):
        a = analyse(run, source)
        low, high = arena_limits(run)
        image = ax.imshow(occupancy(run, a["child"], ~a["stuck"]), origin="lower", cmap="hot",
                          extent=[low[0], high[0], low[1], high[1]])
        ax.set_title(f"{name}\n$\\epsilon_{{Av}}$={run.avoidant:g}, $\\epsilon_{{Am}}$={run.ambivalent:g}", fontsize=10)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save(fig, "paper_style", "fig12_heatmaps")


def paper_fig9(regimes, source):
    """Trajectories, distance with vocalisation, and needs over time."""
    fig, axes = plt.subplots(len(regimes), 3, figsize=(15, 3.4 * len(regimes)),
                             gridspec_kw={"width_ratios": [1, 2, 2]}, squeeze=False)
    for (name, run), (ax_traj, ax_dist, ax_need) in zip(regimes, axes):
        a = analyse(run, source)
        ax_traj.plot(a["child"][:, 0], a["child"][:, 1], color="tab:blue", linewidth=0.8, label="child")
        ax_traj.plot(a["parent"][:, 0], a["parent"][:, 1], color="k", linewidth=0.8, label="caregiver")
        draw_arena(ax_traj, run, a["parent"], improved=False)
        ax_traj.set_title(f"({run.avoidant:g}, {run.ambivalent:g}) {name}", fontsize=10)
        ax_traj.legend(fontsize=7, loc="upper right")

        normalised = np.minimum(1.0, a["distance"] / run.arena_size)
        ax_dist.fill_between(run.t, 0, run.vocal_parent * 0.3, step="mid", color="0.6", label="caregiver vocalisation")
        ax_dist.fill_between(run.t, 0, run.vocal_child * 0.2, step="mid", color="tab:blue", label="child vocalisation")
        ax_dist.plot(run.t, normalised, color="tab:orange", label="distance")
        ax_dist.set_ylim(0, 1)
        ax_dist.set_xlabel("time (s)")
        ax_dist.set_ylabel("normalised distance")
        ax_dist.legend(fontsize=7, loc="upper right")

        ax_need.plot(run.t, run.child_need, color="tab:blue", label="child")
        ax_need.plot(run.t, run.parent_need, color="k", label="caregiver")
        ax_need.set_ylim(-1.6, 1.6)
        ax_need.set_xlabel("time (s)")
        ax_need.set_ylabel("need" + (" (model diverged, off scale)" if run.diverged else ""))
        ax_need.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "paper_style", "fig9_trajectory_distance_needs")


def paper_fig3(runs):
    """Need correlation and oscillation frequency as each parameter increases."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    for ax, parameter, symbol in ((axes[0], "ambivalent", "Am"), (axes[1], "avoidant", "Av")):
        sweep = sweep_runs(runs, parameter)
        ax.plot([getattr(r, parameter) for r in sweep], [metrics.need_correlation(r) for r in sweep], "o-", color="k")
        ax.axhline(0, color="0.7", linewidth=0.8)
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel(f"$\\epsilon_{{{symbol}}}$")
        ax.set_ylabel("correlation of needs $r(x_c, x_p)$")
        ax.set_title(f"{parameter.capitalize()} regime")
    for parameter, colour in (("ambivalent", "tab:blue"), ("avoidant", "tab:orange")):
        sweep = sweep_runs(runs, parameter)
        axes[2].plot([getattr(r, parameter) for r in sweep],
                     [metrics.oscillation_frequency(r.child_need, r.duration) for r in sweep],
                     "o-", color=colour, label=parameter)
    axes[2].set_xlabel("parameter value")
    axes[2].set_ylabel("frequency from zero crossings (Hz)")
    axes[2].set_title("Oscillation frequency")
    axes[2].legend()
    fig.tight_layout()
    save(fig, "paper_style", "fig3_correlation_frequency")


# ----------------------------------------------------------------------------- improved figures

def improved_regimes(regimes, source):
    """Per regime: path to scale, distance in metres with behaviour shading, vocalisation raster."""
    n = len(regimes)
    fig = plt.figure(figsize=(12, 4.2 * n), facecolor=SURFACE)
    grid = fig.add_gridspec(2 * n, 2, width_ratios=[1, 2.4], height_ratios=[5, 1] * n, hspace=0.9, wspace=0.16)
    max_distance = max(analyse(run, source)["distance"].max() for _, run in regimes)

    for i, (name, run) in enumerate(regimes):
        a = analyse(run, source)
        ax_traj = fig.add_subplot(grid[2 * i:2 * i + 2, 0])
        ax_dist = fig.add_subplot(grid[2 * i, 1])
        ax_vocal = fig.add_subplot(grid[2 * i + 1, 1], sharex=ax_dist)
        for ax in (ax_traj, ax_dist, ax_vocal):
            ax.set_facecolor(SURFACE)

        # Path, with stuck stretches de-emphasised
        path = np.where(a["stuck"][:, None], np.nan, a["child"])
        ax_traj.plot(a["child"][:, 0], a["child"][:, 1], color=FAINT, linewidth=0.8)
        ax_traj.plot(path[:, 0], path[:, 1], color=CHILD, linewidth=1.0)
        ax_traj.plot(*a["child"][0], "o", color=CHILD, markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax_traj.annotate("start", a["child"][0], xytext=(6, 6), textcoords="offset points", fontsize=8, color=MUTED)
        draw_arena(ax_traj, run, a["parent"])
        clean_axes(ax_traj, grid_axis=None)
        ax_traj.set_xlabel("x (m)", fontsize=8, color=MUTED)
        ax_traj.set_ylabel("y (m)", fontsize=8, color=MUTED)
        ax_traj.set_title(name, fontsize=11, color=INK, loc="left", fontweight="bold", pad=18)
        ax_traj.text(0, 1.02, f"avoidance {run.avoidant:g}, ambivalence {run.ambivalent:g}"
                     + (" · model diverged" if run.diverged else ""),
                     transform=ax_traj.transAxes, fontsize=8, color=MUTED, va="bottom", ha="left")

        # Distance, shaded while the child's controller is in approach
        for start, end in spans(run.child_state == 0, run.t):
            ax_dist.axvspan(start, end, color=CHILD, alpha=0.10, linewidth=0)
        for start, end in spans(a["stuck"], run.t):
            ax_dist.axvspan(start, end, facecolor="none", edgecolor=FAINT, hatch="////", linewidth=0)
        ax_dist.plot(run.t, a["distance"], color=INK, linewidth=1.4)
        ax_dist.set_ylim(0, max_distance * 1.05)
        ax_dist.set_ylabel("distance to caregiver (m)", fontsize=8, color=MUTED)
        clean_axes(ax_dist)
        plt.setp(ax_dist.get_xticklabels(), visible=False)
        share_near = 100 * a["near"][~a["stuck"]].mean()
        ax_dist.set_title(f"{share_near:.0f}% of the time inside the dotted square  ·  "
                          f"{metrics.vocal_bouts(run.vocal_child)} child care requests  ·  "
                          f"{100 * a['stuck'].mean():.0f}% stuck (hatched, excluded)",
                          fontsize=8, color=MUTED, loc="left")

        # Vocalisation raster, one row per robot
        for row, (vocal, colour) in enumerate(((run.vocal_child, CHILD), (run.vocal_parent, PARENT))):
            for start, end in spans(vocal > 0, run.t):
                ax_vocal.add_patch(Rectangle((start, row + 0.15), max(end - start, 0.1), 0.7, color=colour, linewidth=0))
        ax_vocal.set_ylim(2, 0)
        ax_vocal.set_yticks([0.5, 1.5])
        ax_vocal.set_yticklabels(["child calls", "caregiver calls"])
        ax_vocal.set_xlim(0, run.t[-1])
        ax_vocal.set_xlabel("time (s)", fontsize=8, color=MUTED)
        clean_axes(ax_vocal, grid_axis=None)

    height = fig.get_figheight()
    fig.subplots_adjust(top=1 - 1.25 / height, bottom=0.6 / height)
    fig.text(0.125, 1 - 0.3 / height, "Child path and distance to the caregiver in the attachment regimes",
             fontsize=13, color=INK, fontweight="bold", va="top")
    fig.text(0.125, 1 - 0.58 / height, "Blue line: child path. Orange square: caregiver box. Dotted square: close to "
             "the caregiver. Blue wash: child in approach mode.", fontsize=9, color=MUTED, va="top")
    save(fig, "improved", "regimes_path_distance_vocalisation")


def improved_heatmaps(regimes, source):
    """Occupancy on one shared colour scale, with the caregiver box and proximity square."""
    maps = []
    for name, run in regimes:
        a = analyse(run, source)
        maps.append((name, run, a, 100 * occupancy(run, a["child"], ~a["stuck"])))
    vmax = np.percentile(np.concatenate([m[3].ravel() for m in maps]), 99.5)

    fig, axes = plt.subplots(1, len(maps), figsize=(4.3 * len(maps) + 0.8, 4.6), facecolor=SURFACE, squeeze=False)
    for (name, run, a, image), ax in zip(maps, axes[0]):
        low, high = arena_limits(run)
        mesh = ax.imshow(image, origin="lower", cmap=OCCUPANCY, vmin=0, vmax=vmax,
                         extent=[low[0], high[0], low[1], high[1]])
        draw_arena(ax, run, a["parent"])
        clean_axes(ax, grid_axis=None)
        ax.set_title(name, fontsize=11, color=INK, loc="left", fontweight="bold", pad=18)
        ax.text(0, 1.02, f"avoidance {run.avoidant:g}, ambivalence {run.ambivalent:g}", transform=ax.transAxes,
                fontsize=8, color=MUTED, va="bottom")
        ax.set_xlabel("x (m)", fontsize=8, color=MUTED)
    axes[0][0].set_ylabel("y (m)", fontsize=8, color=MUTED)
    bar = fig.colorbar(mesh, ax=list(axes[0]), fraction=0.025, pad=0.02, extend="max")
    bar.set_label("share of run time per cell (%)", fontsize=8, color=MUTED)
    bar.ax.tick_params(labelsize=8, colors=MUTED)
    bar.outline.set_visible(False)
    fig.suptitle("Where the child spends its time", fontsize=13, color=INK, fontweight="bold", x=0.07, ha="left", y=1.04)
    save(fig, "improved", "heatmaps_shared_scale")


def improved_proximity(regimes, source, tests):
    """Share of time close to the caregiver, with block-bootstrap intervals."""
    fig, ax = plt.subplots(figsize=(7.5, 1.1 + 0.75 * len(regimes)), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    names = []
    for row, (name, run) in enumerate(regimes):
        a = analyse(run, source)
        near = a["near"][~a["stuck"]].astype(float)
        share = 100 * near.mean()
        low, high = 100 * metrics.block_bootstrap_ci(near, block=BLOCK_SECONDS * 10)
        ax.barh(row, share, height=0.42, color=CHILD)
        ax.plot([low, high], [row, row], color=INK, linewidth=1.4)
        ax.text(max(high, share) + 1.5, row, f"{share:.0f}%", va="center", fontsize=9, color=INK)
        names.append(f"{name}\n(avoidance {run.avoidant:g}, ambivalence {run.ambivalent:g})"
                     + ("\nmodel diverged" if run.diverged else ""))
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of time inside the square around the caregiver (%)", fontsize=8, color=MUTED)
    clean_axes(ax, grid_axis="x")
    ax.set_title("Time spent close to the caregiver", fontsize=12, color=INK, fontweight="bold", loc="left", pad=24)
    if tests is not None:
        chi2, dof, n, p, _ = tests
        p_text = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
        ax.text(0, 1.04, f"Black lines: 95% interval from {BLOCK_SECONDS} s blocks. "
                f"Chi-squared = {chi2:.1f}, df = {dof}, N = {n}, {p_text}.",
                transform=ax.transAxes, fontsize=8, color=MUTED)
    save(fig, "improved", "time_near_caregiver")


def improved_sweep_distance(runs, source):
    """Small multiples: distance over time for every value of each parameter."""
    for parameter in ("avoidant", "ambivalent"):
        sweep = sweep_runs(runs, parameter)
        if not sweep:
            continue
        columns = min(4, len(sweep))
        rows = int(np.ceil(len(sweep) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.2 * rows), facecolor=SURFACE,
                                 sharex=True, sharey=True, squeeze=False)
        for ax in axes.ravel()[len(sweep):]:
            ax.set_visible(False)
        for run, ax in zip(sweep, axes.ravel()):
            a = analyse(run, source)
            ax.set_facecolor(SURFACE)
            for start, end in spans(run.child_state == 0, run.t):
                ax.axvspan(start, end, color=CHILD, alpha=0.10, linewidth=0)
            ax.plot(run.t, a["distance"], color=INK, linewidth=1.1)
            ax.set_title(f"{NOUN[parameter]} {getattr(run, parameter):g}" + (" · model diverged" if run.diverged else ""),
                         fontsize=9, color=INK, loc="left")
            clean_axes(ax)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)", fontsize=8, color=MUTED)
        for ax in axes[:, 0]:
            ax.set_ylabel("distance (m)", fontsize=8, color=MUTED)
        fig.suptitle(f"Distance to the caregiver as {NOUN[parameter]} increases (blue wash: child in approach mode)",
                     fontsize=12, color=INK, fontweight="bold", x=0.06, ha="left", y=1.0)
        fig.tight_layout()
        save(fig, "improved", f"sweep_distance_{parameter}")


def improved_sweep_metrics(runs, source):
    """Synchrony, rhythm and proximity against each parameter; legacy runs shown for reference."""
    panels = [
        ("correlation of needs r(x_c, x_p)", lambda r, a: metrics.need_correlation(r), False),
        ("oscillation frequency (Hz)", lambda r, a: metrics.oscillation_frequency(r.child_need, r.duration), False),
        ("child calling (% of time)", lambda r, a: 100 * r.vocal_child.mean(), False),
        ("time near caregiver (%)", lambda r, a: 100 * a["near"][~a["stuck"]].mean(), True),
        ("mean distance (m)", lambda r, a: a["distance"][~a["stuck"]].mean(), True),
    ]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels), 6), facecolor=SURFACE, sharex=True)
    for row, parameter in enumerate(("avoidant", "ambivalent")):
        other = "ambivalent" if parameter == "avoidant" else "avoidant"
        sweep = sweep_runs(runs, parameter)
        legacy = [r for r in runs if r.legacy and getattr(r, other) == 0]
        for (title, measure, needs_position), ax in zip(panels, axes[row]):
            ax.set_facecolor(SURFACE)
            if not needs_position and legacy:
                ax.plot([getattr(r, parameter) for r in legacy], [measure(r, None) for r in legacy], "o",
                        markerfacecolor="none", markeredgecolor=FAINT, markersize=5, label="earlier runs (no positions)")
            analysed = [analyse(r, source) for r in sweep]
            ax.plot([getattr(r, parameter) for r in sweep], [measure(r, a) for r, a in zip(sweep, analysed)],
                    "o-", color=CHILD, linewidth=2, markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2,
                    label="runs with recorded paths")
            if "correlation" in title:
                ax.axhline(0, color=FAINT, linewidth=0.8)
                ax.set_ylim(-1.05, 1.05)
            clean_axes(ax)
            if row == 0:
                ax.set_title(title, fontsize=9, color=INK, loc="left")
            else:
                ax.set_xlabel(f"{NOUN[parameter]}", fontsize=8, color=MUTED)
        axes[row][0].set_ylabel(f"{NOUN[parameter]} sweep", fontsize=9, color=INK)
    axes[0][0].legend(fontsize=7, frameon=False, loc="lower left")
    for ax in axes[0]:
        ax.set_xlabel("avoidance", fontsize=8, color=MUTED)
        ax.tick_params(labelbottom=True)
    fig.suptitle("How the dyad changes as each attachment parameter increases (other parameter at 0)",
                 fontsize=12, color=INK, fontweight="bold", x=0.05, ha="left")
    fig.tight_layout()
    save(fig, "improved", "sweep_metrics")


def improved_odometry_error(runs, regimes):
    """How far the wheel odometry drifts from the true position."""
    tracked = [r for r in runs if r.has_positions]
    fig, (ax_path, ax_err) = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE,
                                          gridspec_kw={"width_ratios": [1, 1.6]})
    name, run = regimes[0]
    ax_path.set_facecolor(SURFACE)
    ax_path.plot(run.child_gt[:, 0], run.child_gt[:, 1], color=CHILD, linewidth=1.2, label="true path (Gazebo)")
    ax_path.plot(run.child_odom[:, 0], run.child_odom[:, 1], color=PARENT, linewidth=1.2, label="wheel odometry")
    draw_arena(ax_path, run, run.parent_gt)
    ax_path.autoscale()
    ax_path.set_aspect("equal")
    clean_axes(ax_path, grid_axis=None)
    ax_path.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    ax_path.set_title(f"Child path, {name.lower()} run", fontsize=10, color=INK, loc="left")
    ax_path.set_xlabel("x (m)", fontsize=8, color=MUTED)
    ax_path.set_ylabel("y (m)", fontsize=8, color=MUTED)

    ax_err.set_facecolor(SURFACE)
    grid_t = np.arange(0, min(r.duration for r in tracked), 1.0)
    errors = []
    for r in tracked:
        error = np.hypot(*(r.child_odom - r.child_gt).T)
        ax_err.plot(r.t, error, color=FAINT, linewidth=0.7)
        errors.append(np.interp(grid_t, r.t, error))
    ax_err.plot(grid_t, np.median(errors, axis=0), color=INK, linewidth=2)
    ax_err.axhline(tracked[0].arena_size, color=MUTED, linewidth=0.8, linestyle=":")
    ax_err.text(0, tracked[0].arena_size, " arena width", fontsize=8, color=MUTED, va="bottom")
    ax_err.set_xlabel("time (s)", fontsize=8, color=MUTED)
    ax_err.set_ylabel("child odometry error (m)", fontsize=8, color=MUTED)
    ax_err.set_title(f"Odometry error, {len(tracked)} run(s) (black: median)", fontsize=10, color=INK, loc="left")
    clean_axes(ax_err)
    fig.suptitle("Wheel odometry drifts away from the true position", fontsize=12, color=INK,
                 fontweight="bold", x=0.06, ha="left")
    fig.tight_layout()
    save(fig, "improved", "odometry_error")


# ----------------------------------------------------------------------------- statistics

def write_results(runs, regimes, source):
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)

    # Per-run summary of every experiment
    summary = os.path.join(out_dir, "run_summary.csv")
    with open(summary, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "avoidant", "ambivalent", "duration_s", "labels_corrected", "model_diverged",
                         "need_correlation",
                         "frequency_hz", "relative_phase_deg", "child_calling_pct", "child_call_bouts",
                         "near_caregiver_pct", "mean_distance_m", "stuck_pct", "odometry_error_end_m"])
        for r in runs:
            row = [r.index, r.avoidant, r.ambivalent, round(r.duration, 1), r.legacy, r.diverged,
                   round(metrics.need_correlation(r), 3),
                   round(metrics.oscillation_frequency(r.child_need, r.duration), 4),
                   round(metrics.relative_phase(r), 1), round(100 * r.vocal_child.mean(), 1),
                   metrics.vocal_bouts(r.vocal_child)]
            if r.has_positions:
                a = analyse(r, source)
                keep = ~a["stuck"]
                row += [round(100 * a["near"][keep].mean(), 1), round(a["distance"][keep].mean(), 3),
                        round(100 * a["stuck"].mean(), 1),
                        round(float(np.hypot(*(r.child_odom[-1] - r.child_gt[-1]))), 3)]
            else:
                row += ["", "", "", ""]
            writer.writerow(row)
    fix_owner(summary)
    print("  wrote results/run_summary.csv")

    # Time near the caregiver across the paper's regimes
    tests = None
    if len(regimes) >= 2:
        counts, lines = {}, []
        for name, run in regimes:
            a = analyse(run, source)
            keep = ~a["stuck"]
            counts[name] = (int(a["near"][keep].sum()), int(keep.sum()))
        tests = metrics.proximity_tests(counts)
        chi2, dof, n, p, pairwise = tests
        lines += ["# Time spent close to the caregiver", "",
                  f"Position source: {'Gazebo ground truth' if source == 'gt' else 'wheel odometry'}. "
                  "Close means inside the square around the caregiver whose side is half the arena width. "
                  "Stuck periods are excluded. Every 10 Hz sample counts as one observation, as in the paper, "
                  "so the p-values overstate the evidence from a single run per regime.", "",
                  "| Regime | Run | Samples | Close to caregiver |", "|---|---|---|---|"]
        for name, run in regimes:
            near, total = counts[name]
            lines.append(f"| {name} (avoidance {run.avoidant:g}, ambivalence {run.ambivalent:g}"
                         f"{'; oscillator model diverged' if run.diverged else ''}) "
                         f"| experiment_data_{run.index} | {total} | {100 * near / total:.1f}% |")
        lines += ["", f"Chi-squared = {chi2:.2f}, df = {dof}, N = {n}, p = {p:.3g}", "",
                  "| Pair | z | p (Bonferroni) | Differ at 0.05 |", "|---|---|---|---|"]
        for a_name, b_name, z, p_pair in pairwise:
            lines.append(f"| {a_name} vs {b_name} | {z:.2f} | {p_pair:.3g} | {'yes' if p_pair < 0.05 else 'no'} |")
        stats_file = os.path.join(out_dir, "proximity_stats.md")
        with open(stats_file, "w") as f:
            f.write("\n".join(lines) + "\n")
        fix_owner(stats_file)
        print("  wrote results/proximity_stats.md")
    fix_owner(out_dir)
    return tests


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="data directory (default: the package's data folder)")
    parser.add_argument("--source", choices=("gt", "odom"), default="gt",
                        help="robot positions from Gazebo ground truth (default) or wheel odometry")
    args = parser.parse_args()

    data_dir = args.data or default_data_dir()
    runs = load_all(data_dir)
    tracked = [r for r in runs if r.has_positions]
    print(f"Loaded {len(runs)} runs from {data_dir} ({len(tracked)} with recorded paths)")

    regimes = regime_runs(runs)
    missing = [name for name, _, _ in REGIMES if name not in [n for n, _ in regimes]]
    if missing:
        print(f"No run with recorded paths for: {', '.join(missing)}")

    tests = write_results(runs, regimes, args.source)
    if not tracked:
        print("No runs with recorded paths: record new experiments to produce the figures.")
        return

    if regimes:
        paper_fig11(regimes, args.source)
        paper_fig12(regimes, args.source)
        paper_fig9(regimes, args.source)
        improved_regimes(regimes, args.source)
        improved_heatmaps(regimes, args.source)
        improved_proximity(regimes, args.source, tests)
        improved_odometry_error(runs, regimes)
    paper_fig3(runs)
    improved_sweep_distance(runs, args.source)
    improved_sweep_metrics(runs, args.source)


if __name__ == "__main__":
    main()
