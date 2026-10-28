#!/usr/bin/env python3
"""
Write the run lists for the TCDS validation experiments (stdlib only).

    python3 experiments/make_runs.py        # writes tcds_runs.csv, tcds_runs_e6.csv and tcds_runs_e7.csv

Random starting needs are drawn from the run seed alone (utils.initial_state), so seed s
gives the same start in every condition and controller: comparisons between controllers
are paired. Seeds 101-103 are used for runs from the identical start (-1, 1, -1, 1), where
the seed only drives the robots' random exploration.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FIELDS = ["id", "batch", "controller", "distance", "avoidant", "ambivalent", "duration", "seed", "initial", "networks"]
VALUES = [round(0.1 * k, 1) for k in range(1, 10)]
STRIPS = [(0.0, 0.0)] + [(v, 0.0) for v in VALUES] + [(0.0, v) for v in VALUES]
OFF_AXIS = [(0.1, 0.1), (0.1, 0.2), (0.2, 0.1), (0.2, 0.2), (0.3, 0.1), (0.1, 0.3)]
RANDOM_SEEDS = [1, 2, 3, 4, 5]
IDENTICAL_SEEDS = [101, 102, 103]


def run(batch, controller, av, am, seed, duration=180, distance="", random_start=True, tag="", networks=""):
    initial = "random" if random_start else ""
    rid = (f"{batch}_{controller}{'-' + distance if distance else ''}_av{av:.1f}_am{am:.1f}"
           f"_{'r' if random_start else 'i'}{seed}{tag}{'_' + networks if networks else ''}")
    return dict(id=rid, batch=batch, controller=controller, distance=distance, avoidant=av,
                ambivalent=am, duration=duration, seed=seed, initial=initial, networks=networks)


def main_runs():
    runs = []
    # E4 first: the 900 s runs, so that they do not finish last
    for av, am in ((0.2, 0.0), (0.0, 0.2)):
        for controller in ("decoupled", "coupled"):
            runs += [run("E4", controller, av, am, s, duration=900) for s in RANDOM_SEEDS[:2]]
    runs += [run("E1s", "decoupled", av, am, s, random_start=False) for av, am in STRIPS for s in IDENTICAL_SEEDS]
    runs += [run("E1r", "decoupled", av, am, s) for av, am in STRIPS for s in RANDOM_SEEDS]
    runs += [run("E2", "coupled", av, am, s) for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    runs += [run("E2", "coupled", av, am, IDENTICAL_SEEDS[0], random_start=False) for av, am in STRIPS]
    runs += [run("E3", c, av, am, s) for av, am in OFF_AXIS for c in ("decoupled", "coupled") for s in RANDOM_SEEDS[:3]]
    runs += [run("E5a", "observed", av, am, s, distance="gt") for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    runs += [run("E5b", "observed", av, am, s, distance="odom") for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    runs += [run("E5c", "notebook", av, am, s, distance="gt") for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    return runs


def baseline_runs():
    """E6: the partner model follows the standard model (baseline); own half learned."""
    runs = [run("E6a", "baseline", av, am, s) for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    runs += [run("E6b", "baseline-observed", av, am, s, distance="gt") for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    return runs


def own_network_runs():
    """E7: the blind controllers with each robot's own networks, so that the blind and perceiving
    variants differ only in what the estimates are fed (network-role ablation)."""
    runs = [run("E7a", "decoupled", av, am, s, networks="own") for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    runs += [run("E7b", "baseline", av, am, s, networks="own") for av, am in STRIPS for s in RANDOM_SEEDS[:3]]
    return runs


def write(name, runs):
    ids = [r["id"] for r in runs]
    assert len(ids) == len(set(ids)), "run ids must be unique"
    with open(os.path.join(HERE, name), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(runs)
    by_batch = {}
    for r in runs:
        by_batch[r["batch"]] = by_batch.get(r["batch"], 0) + 1
    hours = sum(float(r["duration"]) + 30 for r in runs) / 3600
    print(f"{name}: {len(runs)} runs {by_batch}, {hours:.1f} h of simulation one at a time")


if __name__ == "__main__":
    write("tcds_runs.csv", main_runs())
    write("tcds_runs_e6.csv", baseline_runs())
    write("tcds_runs_e7.csv", own_network_runs())
