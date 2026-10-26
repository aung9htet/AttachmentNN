#!/usr/bin/env python3
"""
Inverse problem: infer the attachment parameters from observed behaviour.

Training data: runs of the kinematic stand-in (sim_kinematic) driven by the decoupled
controller, both parameter axes 0-0.9 in steps of 0.05, 30 seeds (10 from the identical start,
20 from random starting needs), 180 s each.
Test data: every Gazebo run of the decoupled controller on the two strips (identical and random
starts), first 180 s. A second evaluation cross-validates within the Gazebo runs (folds by seed).
Two observers: one that sees only the child (path, distance, calls), one that also hears
the caregiver. Writes results/inverse.npz.
"""
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from load import load_all
import sim_kinematic as sk
import model

HERE = os.path.dirname(os.path.abspath(__file__))
SECONDS = 180
SEEDS = 30
CHILD_ONLY = [0, 1, 2, 3, 7]          # near, mean distance, distance s.d., child calling, cycles/min
DYAD = [0, 1, 2, 3, 5, 6, 7]          # + caregiver calling, caregiver call bouts/min
CLASSES = ["secure", "avoidant", "ambivalent"]


def label(eAv, eAm):
    if max(eAv, eAm) <= 0.1 + 1e-9:
        return 0
    return 1 if eAv > eAm else 2


def training_set():
    values = np.round(np.arange(0.0, 0.9001, 0.05), 2)
    conds = [(v, 0.0) for v in values] + [(0.0, v) for v in values if v > 0]
    eAv = np.array([c[0] for c in conds]); eAm = np.array([c[1] for c in conds])
    # 10 seeds from the identical start, 20 from random starts (the same starts in every condition)
    starts = np.tile(np.array(model.INITIAL, float), (SEEDS, 1))
    starts[10:] = np.random.default_rng(7).uniform(-1.0, 1.0, (SEEDS - 10, 4))
    needs = sk.needs_10hz(np.repeat(eAv, SEEDS), np.repeat(eAm, SEEDS), SECONDS,
                          initial=np.tile(starts, (len(conds), 1)))
    X, Y, groups = [], [], []
    for i, (av, am) in enumerate(conds):
        for s in range(SEEDS):
            b = sk.run_behaviour(needs[:, i * SEEDS + s], seed=1000 + s)
            X.append(sk.features(np.stack([b["x"], b["y"]], 1), np.tile(sk.CAREGIVER, (len(b["x"]), 1)),
                                 b["call_c"], b["call_p"]))
            Y.append((av, am, label(av, am)))
            groups.append(s)
    return np.array(X), np.array(Y), np.array(groups)


def gazebo_set():
    """Decoupled-controller Gazebo runs on the two strips."""
    X, Y, idx, seeds, random_start = [], [], [], [], []
    for r in load_all():
        if not r.has_positions or r.controller != "decoupled" or min(r.avoidant, r.ambivalent) > 0:
            continue
        n = min(int(SECONDS * 10), len(r.t))
        X.append(sk.features(r.child_gt[:n], r.parent_gt[:n], r.vocal_child[:n], r.vocal_parent[:n]))
        Y.append((r.avoidant, r.ambivalent, label(r.avoidant, r.ambivalent)))
        idx.append(r.index)
        seeds.append(r.seed)
        random_start.append(r.random_start)
    return np.array(X), np.array(Y), np.array(idx), np.array(seeds), np.array(random_start)


def fit_eval(Xtr, Ytr, groups, Xte, Yte, cols):
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=3, class_weight="balanced", random_state=0)
    reg = RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0)
    # cross-validation inside the simulator, folds by seed
    cv_pred = np.zeros(len(Ytr), int)
    for tr, te in GroupKFold(n_splits=5).split(Xtr, groups=groups):
        clf.fit(Xtr[tr][:, cols], Ytr[tr, 2].astype(int))
        cv_pred[te] = clf.predict(Xtr[te][:, cols])
    clf.fit(Xtr[:, cols], Ytr[:, 2].astype(int))
    reg.fit(Xtr[:, cols], Ytr[:, :2])
    return {
        "cv_acc": float(np.mean(cv_pred == Ytr[:, 2])),
        "test_pred": clf.predict(Xte[:, cols]),
        "test_proba": clf.predict_proba(Xte[:, cols]),
        "test_eps": reg.predict(Xte[:, cols]),
        "importance": clf.feature_importances_,
    }


def confusion(true, pred, k=3):
    m = np.zeros((k, k), int)
    for t, p in zip(true, pred):
        m[int(t), int(p)] += 1
    return m


def gazebo_cv(Xte, Yte, seeds, cols):
    """Accuracy of a classifier trained and tested within the Gazebo runs, folds by run seed."""
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced", random_state=0)
    pred = np.zeros(len(Yte), int)
    folds = GroupKFold(n_splits=min(5, len(np.unique(seeds))))
    for tr, te in folds.split(Xte, groups=seeds):
        clf.fit(Xte[tr][:, cols], Yte[tr, 2].astype(int))
        pred[te] = clf.predict(Xte[te][:, cols])
    return pred


if __name__ == "__main__":
    np.seterr(all="ignore")
    Xtr, Ytr, groups = training_set()
    Xte, Yte, idx, seeds, rand = gazebo_set()
    out = {"Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte, "gazebo_index": idx, "gazebo_seed": seeds,
           "gazebo_random_start": rand, "feature_names": np.array(sk.FEATURE_NAMES), "classes": np.array(CLASSES)}
    true = Yte[:, 2].astype(int)
    for name, cols in (("child", CHILD_ONLY), ("dyad", DYAD)):
        res = fit_eval(Xtr, Ytr, groups, Xte, Yte, cols)
        acc = float(np.mean(res["test_pred"] == true))
        acc_by_start = {k: float(np.mean(res["test_pred"][rand == v] == true[rand == v]))
                        for k, v in (("identical", False), ("random", True)) if (rand == v).any()}
        mae = np.abs(res["test_eps"] - Yte[:, :2]).mean(0)
        cv_pred = gazebo_cv(Xte, Yte, seeds, cols)
        cv_acc = float(np.mean(cv_pred == true))
        print(f"{name}: sim CV {res['cv_acc']:.3f} | Gazebo transfer {acc:.3f} ({int(round(acc * len(true)))}/{len(true)}; "
              f"{', '.join(f'{k} {v:.2f}' for k, v in acc_by_start.items())}) | Gazebo CV {cv_acc:.3f} "
              f"| MAE eps_Av {mae[0]:.3f} eps_Am {mae[1]:.3f}")
        print(confusion(true, res["test_pred"]))
        for k, v in res.items():
            out[f"{name}/{k}"] = v
        out[f"{name}/cols"] = np.array(cols)
        out[f"{name}/test_acc"] = acc
        out[f"{name}/test_acc_identical"] = acc_by_start.get("identical", np.nan)
        out[f"{name}/test_acc_random"] = acc_by_start.get("random", np.nan)
        out[f"{name}/test_mae"] = mae
        out[f"{name}/confusion"] = confusion(true, res["test_pred"])
        out[f"{name}/gazebo_cv_pred"] = cv_pred
        out[f"{name}/gazebo_cv_acc"] = cv_acc
        out[f"{name}/gazebo_cv_confusion"] = confusion(true, cv_pred)
    np.savez(os.path.join(HERE, "results", "inverse.npz"), **out)
