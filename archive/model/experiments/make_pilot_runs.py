#!/usr/bin/env python3
"""
Write the run lists of the 60 s pilot runs that preceded the TCDS batches (stdlib only).
Archived: the pilots checked the controllers and the batch runner and are not used in the paper.

    python3 archive/model/experiments/make_pilot_runs.py   # writes pilot_runs.csv and pilot_runs_e6.csv here

The E7 pilot list (pilot_runs_e7.csv) was written by hand. The recordings are in pilot_data/ (not in git):
    python3 experiments/run_batch.py archive/model/experiments/pilot_runs.csv --workers 4 \
        --out archive/model/experiments/pilot_data
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "experiments"))
import make_runs                    # noqa: E402
from make_runs import run           # noqa: E402


def pilot_runs():
    d = 60
    return [
        run("P", "decoupled", 0.0, 0.2, 7, d, random_start=False, tag="_rep1"),
        run("P", "decoupled", 0.0, 0.2, 7, d, random_start=False, tag="_rep2"),
        run("P", "decoupled", 0.2, 0.0, 1, d),
        run("P", "decoupled", 0.0, 0.2, 1, d),
        run("P", "coupled", 0.2, 0.0, 1, d),
        run("P", "coupled", 0.0, 0.2, 1, d),
        run("P", "observed", 0.2, 0.0, 1, d, distance="gt"),
        run("P", "observed", 0.0, 0.2, 1, d, distance="gt"),
        run("P", "observed", 0.0, 0.2, 1, d, distance="odom"),
        run("P", "notebook", 0.2, 0.0, 1, d, distance="gt"),
        run("P", "notebook", 0.0, 0.2, 1, d, distance="gt"),
        run("P", "coupled", 0.0, 0.0, 101, d, random_start=False),
    ]


if __name__ == "__main__":
    make_runs.HERE = HERE           # write the lists next to this script
    make_runs.write("pilot_runs.csv", pilot_runs())
    make_runs.write("pilot_runs_e6.csv", [run("P6", "baseline", 0.0, 0.2, 1, 60),
                                          run("P6", "baseline-observed", 0.2, 0.0, 1, 60, distance="gt")])
