#!/usr/bin/env python3
"""
The canonical coupled-oscillator model of attachment (the npj formulation) and the distance form
used by the robot controllers, analysed side by side.

Canonical model, as implemented in github.com/aung9htet/attachment_model_inference
(src/child.py, src/parent.py) and analysed in the npj draft (Sections 3-4):
    x_i'  = x_i - 2 x_i^3 - y_i
    y_c'  = eps (b x_c - Am Phi_p - Av Phi_e + delta k0 Av)
    y_p'  = eps (b x_p + Am Phi_p - Av Phi_e + delta k0 Av)
    Phi_p = (y_c + y_p)/2 + y_c            (physical distance)
    Phi_e = k0 ((y_c - y_p)/2 + y_c)       (emotional distance)
with b = 0.5, k0 = 0.8, delta = 0.1 and eps = 0.1. Its inhomogeneous steady state under
ambivalence, x^2 = (Am - b)/(2 Am), and its avoidant fixed point, the root of
x^3 + (1 - q)/(2q) x + delta/2 = 0 with q = k0 Av / b, are those of the npj analysis.

Distance form: the robot controllers' equations (model.vector_field), with the analytic distances
of model.analytic_distances (the shared-needs dyad).

    python3 analysis/tcds/canonical.py [strips plane poincare fixed]

writes results/canonical.npz (canonical model) and results/canonical_distance.npz (distance form):
    strips    correlation, frequency, relative phase, phase locking and regime along both strips,
              from the anti-synchronised start of npj Fig. 4, the identical start and random starts
    plane     relative phase and regime over the avoidance-ambivalence plane (npj Fig. 4, left)
    poincare  bifurcation diagram: x_p at upward crossings of y_c = 0 (npj Fig. 4, right)
    fixed     the steady states of the canonical model and their stability
"""
import os
import sys
import time

import numpy as np
from scipy.signal import hilbert

import model

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

CANON = dict(b=0.5, k0=0.8, delta=0.1, eps=0.1)
ANTI = (1.0, 0.0, -1.0, 0.0)          # anti-synchronised start (npj Fig. 4)
IDENTICAL = (-1.0, 1.0, -1.0, 1.0)     # the identical start used throughout the article
BLOWUP = 10.0


# ---------------------------------------------------------------- vector fields

def canonical_field(s, av, am, p=CANON):
    xc, yc, xp, yp = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    phi_p = (yc + yp) / 2.0 + yc
    phi_e = p["k0"] * ((yc - yp) / 2.0 + yc)
    dyc = p["eps"] * (p["b"] * xc - am * phi_p - av * phi_e + p["delta"] * p["k0"] * av)
    dyp = p["eps"] * (p["b"] * xp + am * phi_p - av * phi_e + p["delta"] * p["k0"] * av)
    return np.stack([xc - 2 * xc ** 3 - yc, dyc, xp - 2 * xp ** 3 - yp, dyp], axis=-1)


def distance_field(s, av, am):
    dp, de = model.analytic_distances(s)
    return model.vector_field(s, av, am, dp, de)


FORMS = {"canonical": dict(field=canonical_field, dt=0.02, T=3000.0, burn=1500.0),
         "distance": dict(field=distance_field, dt=0.01, T=600.0, burn=300.0)}


# ---------------------------------------------------------------- integration

def integrate(form, av, am, init, T=None, burn=None, every=5, section=False):
    """RK4 for N runs at once. Returns the trajectory after burn-in [K, N, 4] (sampled every `every`
    steps, NaN once a run escapes), the escape mask, and, with section=True, the values of x_p at the
    upward crossings of y_c = 0 after burn-in (one list per run)."""
    spec = FORMS[form]
    f, dt = spec["field"], spec["dt"]
    T = spec["T"] if T is None else T
    burn = spec["burn"] if burn is None else burn
    av = np.atleast_1d(np.asarray(av, float))
    am = np.atleast_1d(np.asarray(am, float))
    av, am = np.broadcast_arrays(av, am)
    s = np.array(np.broadcast_to(np.asarray(init, float), av.shape + (4,)))
    n, nb = int(round(T / dt)), int(round(burn / dt))
    escaped = np.zeros(av.shape, bool)
    out = []
    pts = [[] for _ in range(av.size)] if section else None
    prev = s[..., 1].copy()
    with np.errstate(all="ignore"):
        for i in range(n):
            k1 = f(s, av, am)
            k2 = f(s + dt / 2 * k1, av, am)
            k3 = f(s + dt / 2 * k2, av, am)
            k4 = f(s + dt * k3, av, am)
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bad = ~np.isfinite(s).all(-1) | (np.abs(s) > BLOWUP).any(-1)
            if bad.any():
                escaped |= bad
                s[bad] = 0.0
            if i >= nb:
                if i % every == 0:
                    frame = s.copy()
                    frame[escaped] = np.nan
                    out.append(frame)
                if section:
                    up = (prev < 0) & (s[..., 1] >= 0) & ~escaped
                    for j in np.flatnonzero(up.ravel()):
                        pts[j].append(float(s.reshape(-1, 4)[j, 2]))
            prev = s[..., 1].copy()
    return np.array(out), escaped, pts


# ---------------------------------------------------------------- measures

def measures(traj, escaped, dt_sample):
    """Per-run correlation, frequency (cycles per unit model time), mean relative phase (degrees,
    0 = in phase, 180 = anti-phase), phase locking (0..1) and regime:
    0 oscillating, 1 steady state, 2 escaped."""
    K, N = traj.shape[0], traj.shape[1]
    r = np.full(N, np.nan); freq = np.full(N, np.nan); phase = np.full(N, np.nan); lock = np.full(N, np.nan)
    regime = np.zeros(N, int)
    for j in range(N):
        if escaped[j]:
            regime[j] = 2
            continue
        xc, xp = traj[:, j, 0], traj[:, j, 2]
        if xc.std() < 1e-3 or xp.std() < 1e-3:
            regime[j] = 1
            continue
        r[j] = np.corrcoef(xc, xp)[0, 1]
        freq[j] = (np.diff(np.sign(xc)) != 0).sum() / (2 * K * dt_sample)
        dphi = np.angle(hilbert(xc - xc.mean())) - np.angle(hilbert(xp - xp.mean()))
        z = np.exp(1j * dphi)[K // 10: -K // 10 or None]       # trim the Hilbert edge effects
        phase[j] = np.degrees(np.abs(np.angle(z.mean())))
        lock[j] = np.abs(z.mean())
    return dict(r=r, freq=freq, phase=phase, lock=lock, regime=regime)


def sample_dt(form, every=5):
    return FORMS[form]["dt"] * every


# ---------------------------------------------------------------- analyses

def strips(form, step=0.01, n_random=20, seed=1):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    starts = np.random.default_rng(seed).uniform(-1, 1, (n_random, 4))
    out = {"values": values, "random_starts": starts}
    for axis in ("avoidant", "ambivalent"):
        av = values if axis == "avoidant" else np.zeros_like(values)
        am = values if axis == "ambivalent" else np.zeros_like(values)
        for name, init in (("anti", ANTI), ("identical", IDENTICAL)):
            t0 = time.time()
            traj, esc, _ = integrate(form, av, am, init)
            for k, v in measures(traj, esc, sample_dt(form)).items():
                out[f"{axis}/{name}/{k}"] = v
            print(f"  {form} {axis} {name}: {time.time() - t0:.0f}s", flush=True)
        # random starts: all values x all starts in one batch
        t0 = time.time()
        AV = np.repeat(av, n_random); AM = np.repeat(am, n_random)
        init = np.tile(starts, (len(values), 1))
        traj, esc, _ = integrate_many(form, AV, AM, init)
        m = measures(traj, esc, sample_dt(form))
        for k, v in m.items():
            out[f"{axis}/random/{k}"] = v.reshape(len(values), n_random)
        print(f"  {form} {axis} random: {time.time() - t0:.0f}s", flush=True)
    return out


def integrate_many(form, av, am, init, **kw):
    """As integrate, with one starting state per run."""
    spec = FORMS[form]
    f, dt = spec["field"], spec["dt"]
    T, burn = kw.get("T", spec["T"]), kw.get("burn", spec["burn"])
    every = kw.get("every", 5)
    s = np.array(init, float)
    av = np.asarray(av, float); am = np.asarray(am, float)
    n, nb = int(round(T / dt)), int(round(burn / dt))
    escaped = np.zeros(len(av), bool)
    out = []
    with np.errstate(all="ignore"):
        for i in range(n):
            k1 = f(s, av, am)
            k2 = f(s + dt / 2 * k1, av, am)
            k3 = f(s + dt / 2 * k2, av, am)
            k4 = f(s + dt * k3, av, am)
            s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            bad = ~np.isfinite(s).all(-1) | (np.abs(s) > BLOWUP).any(-1)
            if bad.any():
                escaped |= bad
                s[bad] = 0.0
            if i >= nb and i % every == 0:
                frame = s.copy()
                frame[escaped] = np.nan
                out.append(frame)
    return np.array(out), escaped, None


def plane(form, step=0.025):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 4)
    AV, AM = np.meshgrid(values, values, indexing="ij")
    t0 = time.time()
    traj, esc, pts = integrate(form, AV.ravel(), AM.ravel(), ANTI, section=True)
    m = measures(traj, esc, sample_dt(form))
    n_unique = np.array([len(np.unique(np.round(p, 3))) if len(p) >= 4 else 0 for p in pts])
    print(f"  {form} plane: {time.time() - t0:.0f}s", flush=True)
    out = {"values": values, "n_section_values": n_unique.reshape(AV.shape)}
    for k, v in m.items():
        out[k] = v.reshape(AV.shape)
    return out


def poincare(form, step=0.0025, T=None, burn=None):
    values = np.round(np.arange(0.0, 1.0 + 1e-9, step), 5)
    out = {"values": values}
    spec = FORMS[form]
    T = 2 * spec["T"] if T is None else T
    burn = spec["T"] if burn is None else burn
    for axis in ("avoidant", "ambivalent"):
        av = values if axis == "avoidant" else np.zeros_like(values)
        am = values if axis == "ambivalent" else np.zeros_like(values)
        t0 = time.time()
        traj, esc, pts = integrate(form, av, am, ANTI, T=T, burn=burn, every=50, section=True)
        lengths = np.array([len(p) for p in pts])
        flat = np.concatenate([np.asarray(p, float) for p in pts]) if lengths.sum() else np.zeros(0)
        out[f"{axis}/points"] = flat
        out[f"{axis}/counts"] = lengths
        out[f"{axis}/escaped"] = esc
        last = traj[-1] if len(traj) else np.full((len(values), 4), np.nan)
        out[f"{axis}/final"] = last
        print(f"  {form} poincare {axis}: {time.time() - t0:.0f}s", flush=True)
    return out


def jacobian(f, s, av, am, h=1e-6):
    J = np.zeros((4, 4))
    for k in range(4):
        e = np.zeros(4); e[k] = h
        J[:, k] = (f(s + e, av, am) - f(s - e, av, am)) / (2 * h)
    return J


def fixed_points(step=0.0025):
    """Steady states of the canonical model along the two strips and their stability
    (largest real part of the Jacobian's eigenvalues)."""
    p = CANON
    values = np.round(np.arange(step, 1.0 + 1e-9, step), 5)
    out = {"values": values}
    # ambivalence: inhomogeneous steady state (x, y, -x, -y), x^2 = (Am - b)/(2 Am), y = b x / Am
    xs, stab = np.full(len(values), np.nan), np.full(len(values), np.nan)
    for j, am in enumerate(values):
        if am <= p["b"]:
            continue
        x = np.sqrt((am - p["b"]) / (2 * am)); y = p["b"] * x / am
        s = np.array([x, y, -x, -y])
        assert np.abs(canonical_field(s, 0.0, am)).max() < 1e-9
        xs[j] = x
        stab[j] = np.linalg.eigvals(jacobian(canonical_field, s, 0.0, am)).real.max()
    out["ambivalent/x"], out["ambivalent/max_real"] = xs, stab
    # avoidance: shared fixed point (x, y, x, y), root of x^3 + (1-q)/(2q) x + delta/2 = 0
    xs, stab = np.full(len(values), np.nan), np.full(len(values), np.nan)
    for j, av in enumerate(values):
        q = p["k0"] * av / p["b"]
        roots = np.roots([1.0, 0.0, (1 - q) / (2 * q), p["delta"] / 2])
        real = roots[np.abs(roots.imag) < 1e-9].real
        if len(real) != 1:
            continue
        x = real[0]; y = x - 2 * x ** 3
        s = np.array([x, y, x, y])
        assert np.abs(canonical_field(s, av, 0.0)).max() < 1e-9
        xs[j] = x
        stab[j] = np.linalg.eigvals(jacobian(canonical_field, s, av, 0.0)).real.max()
    out["avoidant/x"], out["avoidant/max_real"] = xs, stab
    return out


def summary(c):
    """Numbers quoted in the article, from the saved results."""
    s = {}
    fp = c
    v = fp["fixed/values"]
    st_am = fp["fixed/ambivalent/max_real"]; st_av = fp["fixed/avoidant/max_real"]
    s["ihss_exists_from"] = float(CANON["b"])
    s["ihss_stable_from"] = float(v[np.flatnonzero(st_am < 0)[0]]) if (st_am < 0).any() else None
    s["avoidant_fp_stable_from"] = float(v[np.flatnonzero(st_av < 0)[0]]) if (st_av < 0).any() else None
    return s


if __name__ == "__main__":
    parts = sys.argv[1:] or ["fixed", "strips", "plane", "poincare"]
    for form in ("canonical", "distance"):
        path = os.path.join(RES, "canonical.npz" if form == "canonical" else "canonical_distance.npz")
        data = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
        for part in parts:
            if part == "fixed" and form != "canonical":
                continue
            res = {"strips": strips, "plane": plane, "poincare": poincare}.get(part)
            res = fixed_points() if part == "fixed" else res(form)
            data.update({f"{part}/{k}": v for k, v in res.items()})
        np.savez(path, **data)
        print(f"wrote {path}", flush=True)
