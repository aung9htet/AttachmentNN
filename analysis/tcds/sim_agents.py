#!/usr/bin/env python3
"""
The agent-based strange-situation simulator of attachment_with_nn.ipynb, vectorised over runs.

This is where the coupling networks were trained: two point agents in a [-1, 1]^2 arena,
the caregiver held near the centre, the child exploring the corners or approaching. Each
agent couples to the other either through the analytic coupling functions, which read the
partner's accumulated need directly, or through the learned networks, which only see
signals available to the agent itself: the inter-agent distance, the vocal exchange
(own calls, partner calls, evidence trace), its own need state and the coupling parameter.

Defaults follow the notebook: dt = 0.01, b = 0.5, k0 = 0.8, delta = 0.1, step 0.03,
kp = ke = 0.1, kappa = 0.2, child starts at (1, 1) need state, caregiver at (-1, -0.3).
"""
import numpy as np

from nnweights import load_networks

NB = dict(dt=0.01, b=0.5, k0=0.8, delta=0.1, epsilon=1.0, dx=0.03, kp=0.1, ke=0.1, kappa=0.2)


def coupling_targets(yc, yp, p=NB):
    """Analytic coupling functions (the networks' training targets)."""
    phys = (yc + yp) / 2.0 + yc
    emo = p["k0"] * ((yc - yp) / 2.0 + yc) + p["delta"] * p["k0"]
    return phys, emo


def filter_kernel(size=20, tau=10.0):
    k = np.exp(-np.arange(size) / tau)
    return k / k.sum()


def simulate(ambivalence, avoidance, T, seed=0, learned=(), nets=None, p=NB, record=True, initial=None):
    """Run N dyads (one per parameter pair) for T steps.

    learned: subset of {'physical', 'emotional'}; those couplings use the networks.
    initial: starting needs (x_c, y_c, x_p, y_p), one for all runs or one row per run; default
    the notebook's child (1, 1) and caregiver (-1, -0.3).
    Returns a dict of arrays [T, N] (needs, positions, calls, evidence, targets, network outputs).
    """
    rng = np.random.default_rng(seed)
    amb = np.atleast_1d(np.asarray(ambivalence, float))
    av = np.atleast_1d(np.asarray(avoidance, float))
    amb, av = np.broadcast_arrays(amb, av)
    n = len(amb)
    nets = nets or (load_networks() if learned else None)

    if initial is None:
        initial = (1.0, 1.0, -1.0, -0.3)
    init = np.broadcast_to(np.asarray(initial, float), (n, 4))
    xc, yc, xp, yp = (init[:, k].copy() for k in range(4))
    parent_pos = np.zeros((n, 2))
    child_pos = rng.random((n, 2)) * 2 - 1
    ngoal = np.ones((n, 2))
    call_c = np.zeros(n); call_p = np.zeros(n)
    ev_c = np.zeros(n); ev_p = np.zeros(n)
    voc_c_prev = np.zeros(n); voc_p_prev = np.zeros(n)

    keys = ("xc", "yc", "xp", "yp", "dist", "voc_c", "voc_p", "ev_c", "ev_p", "target_phys", "target_emo",
            "net_phys_c", "net_phys_p", "net_emo_c", "net_emo_p")
    rec = {k: np.empty((T, n)) for k in keys} if record else None
    rec_pos = {"child": np.empty((T, n, 2)), "parent": np.empty((T, n, 2))} if record else None

    def dist_now():
        return np.sqrt(np.linalg.norm(parent_pos - child_pos, axis=1))   # as Simulator.physical_distance

    for t in range(T):
        d = dist_now()
        # ---- child step (Child.step) ----
        p_call = np.maximum(0.0, 1.0 - call_c)
        voc_c = ((xc > 0) & (rng.random(n) < p_call)) | ((voc_p_prev > 0) & (xc > 0))
        voc_c = voc_c.astype(float)
        ev_c = (1 - p["ke"]) * ev_c + p["kappa"] * voc_p_prev * call_c
        call_c = (1 - p["kp"]) * call_c + p["kappa"] * voc_c
        phys_a, emo_a = coupling_targets(yc, yp, p)
        if "physical" in learned:
            pc = nets[("physical", "child")](np.stack([d, xc, yc, amb], 1))
        else:
            pc = phys_a
        if "emotional" in learned:
            ec = nets[("emotional", "child")](np.stack([ev_c, voc_p_prev, voc_c, xc, yc, av], 1))
        else:
            ec = emo_a
        xc = xc + p["dt"] * (xc - 2 * xc ** 3 - yc)
        yc = yc + p["epsilon"] * p["dt"] * (p["b"] * xc - amb * pc - av * ec)

        # ---- parent step (Parent.step), sees the child's updated state ----
        p_call = np.maximum(0.0, 1.0 - call_p)
        voc_p = ((xp > 0) & (rng.random(n) < p_call)) | ((voc_c_prev > 0) & (xp > 0))
        voc_p = voc_p.astype(float)
        ev_p = (1 - p["ke"]) * ev_p + p["kappa"] * voc_c_prev * call_p
        call_p = (1 - p["kp"]) * call_p + p["kappa"] * voc_p
        phys_a, emo_a = coupling_targets(yc, yp, p)
        if "physical" in learned:
            pp = nets[("physical", "parent")](np.stack([d, xp, yp, amb], 1))
        else:
            pp = phys_a
        if "emotional" in learned:
            ep = nets[("emotional", "parent")](np.stack([ev_p, voc_c_prev, voc_p, xp, yp, av], 1))
        else:
            ep = emo_a
        xp = xp + p["dt"] * (xp - 2 * xp ** 3 - yp)
        yp = yp + p["epsilon"] * p["dt"] * (p["b"] * xp + amb * pp - av * ep)
        tgt_phys, tgt_emo = coupling_targets(yc, yp, p)

        if record:
            rec["dist"][t] = d
            rec["xc"][t], rec["yc"][t], rec["xp"][t], rec["yp"][t] = xc, yc, xp, yp
            rec["voc_c"][t], rec["voc_p"][t], rec["ev_c"][t], rec["ev_p"][t] = voc_c, voc_p, ev_c, ev_p
            rec["target_phys"][t], rec["target_emo"][t] = tgt_phys, tgt_emo
            rec["net_phys_c"][t], rec["net_phys_p"][t] = pc, pp
            rec["net_emo_c"][t], rec["net_emo_p"][t] = ec, ep
        voc_c_prev, voc_p_prev = voc_c, voc_p

        # ---- movement (Simulator.step) ----
        act_c = xc > 0                     # 1 approach
        vp = (0.0 - parent_pos) + (rng.random((n, 2)) * 2 - 1) * 0.1
        vc_app = -(child_pos - parent_pos) + (rng.random((n, 2)) * 2 - 1) * 0.5
        new_goal = np.sign(rng.random((n, 2)) - 0.5) + (rng.random((n, 2)) * 2 - 1) * 0.5
        vc_exp = (ngoal - child_pos) + (rng.random((n, 2)) * 2 - 1) * 0.5
        ngoal = np.where(act_c[:, None], new_goal, ngoal)
        vc = np.where(act_c[:, None], vc_app, vc_exp)
        vp /= np.linalg.norm(vp, axis=1, keepdims=True)
        vc /= np.linalg.norm(vc, axis=1, keepdims=True)
        parent_pos = parent_pos + p["dx"] * vp
        bounce = np.abs(child_pos + p["dx"] * vc).sum(axis=1) < 0.1
        vc = np.where(bounce[:, None], -vc, vc)
        child_pos = child_pos + p["dx"] * vc
        parent_pos = np.clip(parent_pos, -1, 1)
        child_pos = np.clip(child_pos, -1, 1)
        if record:
            rec_pos["child"][t], rec_pos["parent"][t] = child_pos, parent_pos

    if record:
        rec["child_pos"], rec["parent_pos"] = rec_pos["child"], rec_pos["parent"]
        rec["amb"], rec["av"] = amb, av
    return rec


def filtered_distance(child_pos, parent_pos):
    """Training input of the physical networks: Euclidean distance through the notebook's filter.

    filter_dist applies F.conv1d with the exponential kernel and left padding. conv1d is a
    cross-correlation, so kernel[0] (the largest weight) multiplies the oldest sample of the
    20-step window: out[t] = sum_j kernel[j] * d[t - 19 + j], with zeros before the start.
    """
    dist = np.linalg.norm(child_pos - parent_pos, axis=-1)          # [T, N]
    k = filter_kernel()
    taps = len(k)
    out = np.zeros_like(dist)
    for lag in range(taps):                     # sample d[t - lag] gets weight k[taps - 1 - lag]
        out[lag:] += k[taps - 1 - lag] * dist[:len(dist) - lag]
    return out
