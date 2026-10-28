#!/usr/bin/env python3
"""
Numbers and tables quoted in the TCDS paper. Writes results/numbers.json and the LaTeX tables in
analysis/tcds/tables/. Run after canonical.py, ssp_sim.py, check_runs.py, gazebo.py, robustness.py,
estimator_eval.py, inverse.py and calibration.py.
"""
import os
import sys
import csv
import json

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
sys.path.insert(0, ANALYSIS)

from load import load_all          # noqa: E402
import gazebo as gz                # noqa: E402
from figures import gazebo_table, select, by_value, mean_ci, MODES, ANCHOR_DESIGNS, READOUT_VARIANTS, readout_runs   # noqa: E402

RES = os.path.join(HERE, "results")
TABLES = os.path.join(HERE, "tables")


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def med(a):
    a = np.asarray(a, float)
    return r3(np.nanmedian(a)) if np.isfinite(a).any() else None


def replay_summary():
    with open(os.path.join(RES, "run_checks.csv")) as f:
        rows = list(csv.DictReader(f))
    out = {"n_runs": len(rows), "n_pass": sum(r["ok"] == "True" for r in rows), "by_controller": {}}
    for c in sorted({r["controller"] for r in rows}):
        sel = [r for r in rows if r["controller"] == c]
        out["by_controller"][c] = dict(n=len(sel), passed=sum(r["ok"] == "True" for r in sel),
                                       diverged=sum(r["diverged"] == "True" for r in sel))
    rtf = [float(r["rtf"]) for r in rows if r["rtf"]]
    out["rtf_median"], out["rtf_min"] = r3(np.median(rtf)), r3(min(rtf))
    return out


def offline_strips():
    out = {}
    for tag in ("strips180", "strips"):
        d = np.load(os.path.join(RES, f"robust_{tag}.npz"))
        v = d["values"]
        for axis in ("avoidant", "ambivalent"):
            for e in (0.1, 0.2, 0.3, 0.4, 0.6, 0.8):
                i = int(np.argmin(np.abs(v - e)))
                cell = {}
                for kind in ("coupled", "decoupled"):
                    cell[kind] = dict(r=med(d[f"{axis}/{kind}/r"][i]),
                                      r_iqr=[r3(np.nanpercentile(d[f"{axis}/{kind}/r"][i], q)) if np.isfinite(d[f"{axis}/{kind}/r"][i]).any() else None for q in (25, 75)],
                                      escaped=r3(d[f"{axis}/{kind}/escaped"][i].mean()))
                cell["decoupled"].update(belief=med(d[f"{axis}/decoupled/belief_c"][i]),
                                         track=med(d[f"{axis}/decoupled/track_r"][i]),
                                         track_rmse=med(d[f"{axis}/decoupled/track_rmse"][i]))
                out[f"{tag}/{axis}/{e}"] = cell
    return out


def stability():
    s = np.load(os.path.join(RES, "robust_stability.npz"))
    return {"sizes": s["sizes"].tolist(), "eavs": s["eavs"].tolist(),
            "decoupled_break_s": np.round(s["decoupled/t_break_s"], 0).tolist(),
            "decoupled_r_end": np.round(s["decoupled/r_end"], 2).tolist(),
            "coupled_r_end": np.round(s["coupled/r_end"], 2).tolist()}


def grid():
    g = np.load(os.path.join(RES, "robust_grid.npz"))
    out = {}
    for kind in ("coupled", "decoupled"):
        esc = g[f"{kind}/escaped"].astype(bool)
        quench = ((g[f"{kind}/freq_c"] == 0) | (g[f"{kind}/freq_p"] == 0)) & ~esc
        out[kind] = dict(escaped=r3(esc.mean()), quenched=r3(quench.mean()), oscillating=r3((~esc & ~quench).mean()),
                         r_osc_abs_median=med(np.abs(np.where(~esc & ~quench, g[f"{kind}/r"], np.nan))))
    return out


def nets_ablation():
    d = np.load(os.path.join(RES, "robust_nets.npz"))
    v = d["values"]
    out = {}
    for axis in ("avoidant", "ambivalent"):
        for e in (0.1, 0.2, 0.3):
            i = int(np.argmin(np.abs(v - e)))
            out[f"{axis}/{e}"] = dict(r=med(d[f"{axis}/decoupled/r"][i]), belief=med(d[f"{axis}/decoupled/belief_c"][i]),
                                      escaped=r3(d[f"{axis}/decoupled/escaped"][i].mean()))
    return out


def notebook_offline():
    d = np.load(os.path.join(RES, "robust_notebook.npz"))
    v = d["values"]
    out = {}
    for axis in ("avoidant", "ambivalent"):
        sel = v > 0
        for mode in ("analytic", "learned"):
            out[f"{axis}/{mode}"] = dict(r=[med(x) for x in d[f"{axis}/{mode}/r"]],
                                         est_r_child=med(d[f"{axis}/{mode}/est_r_c"][sel]),
                                         est_r_carer=med(d[f"{axis}/{mode}/est_r_p"][sel]))
    out["values"] = v.tolist()
    return out


def estimator_offline():
    e = np.load(os.path.join(RES, "estimator_eval.npz"))
    vals = e["open/values"]
    rng = (vals > 0.05) & (vals < 0.7)
    out = {}
    for kind, abl in (("physical", "distance"), ("emotional", "vocal")):
        for agent in ("child", "parent"):
            for tag in ("full", abl, "own_state"):
                out[f"{kind}/{agent}/{tag}"] = dict(r2=med(e[f"open/{kind}/{tag}/{agent}/r2"][rng]),
                                                    rmse=med(e[f"open/{kind}/{tag}/{agent}/rmse"][vals >= 0.1]))
    return out


def group_stats(rows, controller, start, axis, eps, key, distance=""):
    runs = [r for r in select(rows, controller, start, axis, distance) if r[axis] == eps]
    vals = [r[key] for r in runs if r["regime"] != "escaped" or key in ("near", "child_calling", "carer_calling", "reciprocity")]
    m, ci = mean_ci(vals)
    v = np.asarray(vals, float)
    return dict(n=len(runs), mean=r3(m), ci=r3(ci), sd=r3(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else None,
                escaped=r3(np.mean([r["regime"] == "escaped" for r in runs])) if runs else None)


def gazebo_summary(rows):
    out = {}
    for axis in ("avoidant", "ambivalent"):
        for eps in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            for (controller, distance) in MODES:
                for start in ("random", "identical"):
                    cell = {key: group_stats(rows, controller, start, axis, eps, key, distance)
                            for key in ("r", "belief_c", "track_r", "est_r_c", "est_r_p", "near", "child_calling",
                                        "carer_calling", "reciprocity")}
                    if cell["near"]["n"]:
                        out[f"{controller}{'-' + distance if distance else ''}/{start}/{axis}/{eps}"] = cell
    return out


def long_runs():
    out = {}
    for r in load_all():
        if r.batch != "E4":
            continue
        w = gz.windowed(r, window=120.0, step=15.0)
        rec = np.array([x["reciprocity"] for x in w], float)
        rr = np.array([x["r"] for x in w], float)
        out[f"{r.controller}/av{r.avoidant}/am{r.ambivalent}/seed{r.seed}"] = dict(
            index=r.index, reciprocity_mean=r3(np.nanmean(rec)), reciprocity_min=r3(np.nanmin(rec)),
            reciprocity_max=r3(np.nanmax(rec)), r_mean=r3(np.nanmean(rr)), r_min=r3(np.nanmin(rr)), r_max=r3(np.nanmax(rr)))
    return out


SHORT = {("decoupled", ""): "blind / self", ("observed", "gt"): "perceiving / self",
         ("observed", "odom"): "perceiving (odometry) / self", ("baseline", ""): "blind / reference",
         ("baseline-observed", "gt"): "perceiving / reference", ("notebook", "gt"): "no partner model"}


def estimation_table(rows):
    """Mean over the runs of each design on each strip (eps > 0, random starts, not escaped)."""
    lines = [r"\begin{tabular}{@{}lcccccccc@{}}", r"\toprule",
             r" & \multicolumn{2}{c}{Escaped} & \multicolumn{2}{c}{Child estimate $r$} & "
             r"\multicolumn{2}{c}{Caregiver estimate $r$} & \multicolumn{2}{c}{Partner tracking $r$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(l){8-9}",
             r"Design & Av & Am & Av & Am & Av & Am & Av & Am \\", r"\midrule"]
    out = {}
    fmt = lambda x: "---" if x is None or not np.isfinite(x) else f"${x:+.2f}$"
    for (controller, distance) in MODES:
        if controller == "coupled":
            continue
        cells = {}
        for axis in ("avoidant", "ambivalent"):
            runs = [r for r in select(rows, controller, "random", axis, distance) if r[axis] > 0]
            if not runs:
                continue
            ok = [r for r in runs if r["regime"] != "escaped"]
            cell = dict(n=len(runs), escaped=r3(np.mean([r["regime"] == "escaped" for r in runs])),
                        est_c=r3(np.nanmean([r["est_r_c"] for r in ok])) if ok else None,
                        est_p=r3(np.nanmean([r["est_r_p"] for r in ok])) if ok else None,
                        track=r3(np.nanmean([r["track_r"] for r in ok])) if ok and controller != "notebook" else None)
            out[f"{controller}{'-' + distance if distance else ''}/{axis}"] = cell
            cells[axis] = cell
        if len(cells) < 2:
            continue
        av, am = cells["avoidant"], cells["ambivalent"]
        lines.append(f"{SHORT[(controller, distance)]} & {100 * av['escaped']:.0f}\\% & {100 * am['escaped']:.0f}\\% & "
                     f"{fmt(av['est_c'])} & {fmt(am['est_c'])} & {fmt(av['est_p'])} & {fmt(am['est_p'])} & "
                     f"{fmt(av['track'])} & {fmt(am['track'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "estimation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def anchoring(rows):
    """Pooled over 0.1 <= eps_Am <= 0.4 (random starts, runs that did not escape), by design (Fig. anchoring)."""
    out = {}
    for controller, distance in READOUT_VARIANTS:
        runs = readout_runs(rows, controller, distance)
        cell = {"n": len(runs)}
        for key in ("r", "belief_c", "track_r", "track_rmse_x", "detune_model", "detune_own", "reciprocity", "est_r_c"):
            v = np.array([r.get(key, np.nan) for r in runs], float)
            if key.startswith("detune"):
                v = v * 1000            # mHz
            m, ci = mean_ci(v)
            cell[key] = dict(mean=r3(m), ci=r3(ci))
        out[f"{controller}{'-' + distance if distance else ''}"] = cell
    # avoidance at 0.1, the only avoidance value at which every private design stays bounded
    for controller, distance in READOUT_VARIANTS:
        runs = [r for r in select(rows, controller, "random", "avoidant", distance)
                if r["avoidant"] == 0.1 and r["regime"] != "escaped"]
        out[f"{controller}{'-' + distance if distance else ''}/avoidant0.1"] = {
            key: r3(np.nanmean([r.get(key, np.nan) for r in runs])) if runs else None
            for key in ("r", "belief_c", "track_r", "track_rmse_x", "est_r_c", "reciprocity")}
    return out


NAMES = {"coupled": "shared needs", "decoupled": "private models"}


def conditions_table(rows):
    """Behaviour at three conditions for the coupled and decoupled controllers (random starts)."""
    lines = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule",
             r"$(\varepsilon_{Av},\varepsilon_{Am})$ & Design & $n$ & $r(x_c,x_p)$ & Near (\%) & Child calls (\%) & Carer calls (\%) & Reciprocity \\",
             r"\midrule"]
    for av, am in ((0.2, 0.0), (0.0, 0.2), (0.0, 0.7)):
        axis = "avoidant" if av > 0 else "ambivalent"
        eps = av or am
        for controller in ("coupled", "decoupled"):
            s = {k: group_stats(rows, controller, "random", axis, eps, k)
                 for k in ("r", "near", "child_calling", "carer_calling", "reciprocity")}
            f = lambda c, scale=1, d=2: "---" if c["mean"] is None else f"${scale * c['mean']:.{d}f} \\pm {scale * (c['sd'] or 0):.{d}f}$"
            lines.append(f"({av:g}, {am:g}) & {NAMES[controller]} & {s['near']['n']} & {f(s['r'])} & {f(s['near'], 100, 0)} & "
                         f"{f(s['child_calling'], 100, 0)} & {f(s['carer_calling'], 100, 0)} & {f(s['reciprocity'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "gazebo_conditions.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def inverse_summary():
    inv = np.load(os.path.join(RES, "inverse.npz"))
    out = {"n_train": int(len(inv["Ytr"])), "n_test": int(len(inv["Yte"])),
           "n_test_random": int(inv["gazebo_random_start"].sum()),
           "n_test_bounded": int(inv["gazebo_bounded"].sum()),
           "test_counts": inv["test_counts"].tolist(), "test_counts_bounded": inv["test_counts_bounded"].tolist(),
           "majority": r3(inv["majority"]), "majority_bounded": r3(inv["majority_bounded"])}
    for name in ("child", "dyad"):
        out[name] = dict(sim_cv=r3(inv[f"{name}/cv_acc"]), transfer=r3(inv[f"{name}/test_acc"]),
                         transfer_identical=r3(inv[f"{name}/test_acc_identical"]),
                         transfer_random=r3(inv[f"{name}/test_acc_random"]), gazebo_cv=r3(inv[f"{name}/gazebo_cv_acc"]),
                         transfer_bounded=r3(inv[f"{name}/test_acc_bounded"]),
                         mae=[r3(x) for x in inv[f"{name}/test_mae"]],
                         mae_bounded=[r3(x) for x in inv[f"{name}/test_mae_bounded"]],
                         confusion=inv[f"{name}/confusion"].tolist(),
                         importance=dict(zip([str(inv["feature_names"][c]) for c in inv[f"{name}/cols"]],
                                             [r3(x) for x in inv[f"{name}/importance"]])))
    cal_file = os.path.join(RES, "calibration.json")
    if os.path.exists(cal_file):
        with open(cal_file) as f:
            out["calibration"] = json.load(f)
    n, nb = out["n_test"], out["n_test_bounded"]
    pct = lambda x: f"{100 * x:.0f}\\%"
    rows = [f"{'Child only' if k == 'child' else 'Both agents'} & {pct(out[k]['sim_cv'])} & {pct(out[k]['transfer'])} & "
            f"{pct(out[k]['transfer_bounded'])} & {pct(out[k]['gazebo_cv'])} & "
            f"{out[k]['mae'][0]:.2f} & {out[k]['mae'][1]:.2f} \\\\" for k in ("child", "dyad")]
    rows.append(f"Majority class & --- & {pct(out['majority'])} & {pct(out['majority_bounded'])} & {pct(out['majority'])} & --- & --- \\\\")
    with open(os.path.join(TABLES, "inverse.tex"), "w") as f:
        f.write("\n".join([
            r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
            r" & Stand-in & \multicolumn{2}{c}{Transfer} & Gazebo & \multicolumn{2}{c}{MAE} \\",
            r"Observer & CV & all & bounded & CV & $\varepsilon_{Av}$ & $\varepsilon_{Am}$ \\",
            rf"$n$ & {out['n_train']} & {n} & {nb} & {n} & \multicolumn{{2}}{{c}}{{{n}}} \\",
            r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"]) + "\n")
    return out


def odometry():
    errs = [float(np.hypot(*(r.child_odom[min(len(r.t), 1800) - 1] - r.child_gt[min(len(r.t), 1800) - 1])))
            for r in load_all() if r.batch == "E1r"]
    return dict(n=len(errs), median_end_error_m=r3(np.median(errs)))


def dynamics(form="canonical"):
    """The coupled-oscillator dynamics of Section III (canonical model) and S2 (distance form)."""
    path = os.path.join(RES, "canonical.npz" if form == "canonical" else "canonical_distance.npz")
    if not os.path.exists(path):
        return None
    c = np.load(path)
    out = {}
    v = c["strips/values"]
    for axis in ("avoidant", "ambivalent"):
        rr = c[f"strips/{axis}/random/r"]
        reg = c[f"strips/{axis}/random/regime"]
        for val in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
            j = int(np.argmin(np.abs(v - val)))
            ok = np.isfinite(rr[j])
            out[f"{axis}/{val}"] = dict(r_median=r3(np.median(rr[j][ok])) if ok.any() else None,
                                        steady=r3(np.mean(reg[j] == 1)), escaped=r3(np.mean(reg[j] == 2)),
                                        anti_r=r3(c[f"strips/{axis}/anti/r"][j]),
                                        identical_r=r3(c[f"strips/{axis}/identical/r"][j]))
        # first value at which at least half of the random starts reach a steady state or escape
        frac = (reg != 0).mean(1)
        out[f"{axis}/oscillation_ends"] = r3(v[np.flatnonzero(frac >= 0.5)[0]]) if (frac >= 0.5).any() else None
        ff = c[f"strips/{axis}/random/freq"]
        med = np.nanmedian(ff, 1)
        out[f"{axis}/freq_at_0"] = r3(med[0]); 
        last = np.flatnonzero(np.isfinite(med))[-1]
        out[f"{axis}/freq_last_oscillating"] = [r3(v[last]), r3(med[last])]
        # irregular band: >= 3 distinct section values among >= 8 crossings
        pv = c["poincare/values"]; counts = c[f"poincare/{axis}/counts"]; pts = c[f"poincare/{axis}/points"]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        irregular = np.array([counts[k] >= 8 and len(np.unique(np.round(pts[starts[k]:starts[k] + counts[k]], 3))) >= 3
                              for k in range(len(pv))])
        out[f"{axis}/irregular"] = [r3(x) for x in pv[irregular]][:: max(1, irregular.sum() // 12)] if irregular.any() else []
        out[f"{axis}/irregular_range"] = [r3(pv[irregular].min()), r3(pv[irregular].max())] if irregular.any() else None
        steady = (counts == 0) & ~c[f"poincare/{axis}/escaped"]
        out[f"{axis}/steady_from_anti_start"] = r3(pv[np.flatnonzero(steady)[0]]) if steady.any() else None
        out[f"{axis}/escape_from_anti_start"] = (r3(pv[np.flatnonzero(c[f"poincare/{axis}/escaped"])[0]])
                                                 if c[f"poincare/{axis}/escaped"].any() else None)
    reg = c["plane/regime"]
    out["plane"] = dict(oscillating=r3(np.mean(reg == 0)), steady=r3(np.mean(reg == 1)), escaped=r3(np.mean(reg == 2)),
                        nonperiodic=r3(np.mean((reg == 0) & (c["plane/n_section_values"] >= 3))))
    if "fixed/values" in c:
        fv = c["fixed/values"]
        for axis in ("ambivalent", "avoidant"):
            st = c[f"fixed/{axis}/max_real"]
            out[f"{axis}/fixed_point_stable_from"] = r3(fv[np.flatnonzero(st < 0)[0]]) if (st < 0).any() else None
            ok = np.isfinite(c[f"fixed/{axis}/x"])
            out[f"{axis}/fixed_point_exists_from"] = r3(fv[ok][0]) if ok.any() else None
    return out


def sim_ssp():
    path = os.path.join(RES, "sim_ssp.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    names = [str(n) for n in d["names"]]
    return dict(near={n: r3(x) for n, x in zip(names, d["near"])}, r={n: r3(x) for n, x in zip(names, d["r"])},
                chi2=r3(float(d["chi2"])), dof=int(d["dof"]), N=int(d["counts"].sum()),
                pairs=[dict(a=names[int(i)], b=names[int(j)], z=r3(z), p=float(f"{pb:.3g}")) for i, j, z, pb in d["pairs"]],
                child_calls={n: r3(x) for n, x in zip(names, d["voc_c"].mean(0))},
                carer_calls={n: r3(x) for n, x in zip(names, d["voc_p"].mean(0))})


if __name__ == "__main__":
    np.seterr(all="ignore")
    os.makedirs(TABLES, exist_ok=True)
    rows = gazebo_table()
    numbers = dict(replay=replay_summary(), offline=offline_strips(), stability=stability(), grid=grid(),
                   nets=nets_ablation(), notebook_offline=notebook_offline(), estimator_offline=estimator_offline(),
                   gazebo=gazebo_summary(rows), long_runs=long_runs(), estimation=estimation_table(rows),
                   anchoring=anchoring(rows), dynamics=dynamics("canonical"), dynamics_distance=dynamics("distance"),
                   sim_ssp=sim_ssp(),
                   inverse=inverse_summary(), odometry=odometry(),
                   stuck_median=r3(np.median([r["stuck"] if isinstance(r["stuck"], float) else float(r["stuck"]) for r in rows])),
                   n_gazebo_runs=len(rows))
    conditions_table(rows)
    with open(os.path.join(RES, "numbers.json"), "w") as f:
        json.dump(numbers, f, indent=1)
    print(json.dumps({k: numbers[k] for k in ("anchoring", "replay", "grid", "estimation", "inverse", "odometry", "stuck_median",
                                              "n_gazebo_runs", "nets", "stability")}, indent=1)[:9000])
