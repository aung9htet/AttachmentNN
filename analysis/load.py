#!/usr/bin/env python3
"""
Load recorded attachment experiments (data/experiment_data_{index}).
"""
import os
import re
import glob
import json
from dataclasses import dataclass, field

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Inside the container the package is mounted in the catkin workspace
_CANDIDATE_DATA_DIRS = [
    os.path.join(REPO_ROOT, "ros", "Attachment-Model", "data"),
    "/root/mdk/catkin_ws/src/Attachment-Model/data",
]

# Arena geometry used when a run has no meta.json (worlds/attachment.world)
ARENA_CENTRE = (0.325, 0.325)
ARENA_SIZE = 2.61
BOX_SIZE = 0.3
SAMPLE_PERIOD = 0.1  # agents record at 10 Hz


def default_data_dir():
    for path in _CANDIDATE_DATA_DIRS:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError("Could not find the data directory; pass --data")


@dataclass
class Run:
    index: int
    path: str
    avoidant: float            # strength of the avoidant (emotional-distance) coupling actually applied
    ambivalent: float          # strength of the ambivalent (physical-distance) coupling actually applied
    legacy: bool               # recorded before odometry logging, with the two parameters swapped
    t: np.ndarray              # time (s), starts at 0
    child_need: np.ndarray     # x_c
    child_acc: np.ndarray      # y_c
    parent_need: np.ndarray    # x_p
    parent_acc: np.ndarray     # y_p
    child_state: np.ndarray    # 0 approach, 1 explore
    parent_state: np.ndarray
    vocal_child: np.ndarray
    vocal_parent: np.ndarray
    child_gt: np.ndarray = None     # [T, 2] ground-truth path (m), None for legacy runs
    parent_gt: np.ndarray = None
    child_odom: np.ndarray = None   # [T, 2] odometry path in the world frame (m)
    parent_odom: np.ndarray = None
    controller: str = "decoupled"   # decoupled, coupled, observed, notebook; "legacy" for runs 0-33
    distance_source: str = None     # gt or odom (observed and notebook controllers)
    networks: str = "default"       # "own": decoupled/baseline robots load their own role's networks
    seed: int = -1
    initial: tuple = (-1.0, 1.0, -1.0, 1.0)   # starting needs (x_c, y_c, x_p, y_p)
    batch: str = ""
    logs: dict = None               # controller logs {name: {key: array}}, from experiment_data_54 on
    arena_centre: tuple = ARENA_CENTRE
    arena_size: float = ARENA_SIZE
    box_size: float = BOX_SIZE
    meta: dict = field(default_factory=dict)

    @property
    def random_start(self):
        return tuple(np.round(self.initial, 6)) != (-1.0, 1.0, -1.0, 1.0)

    @property
    def has_positions(self):
        return self.child_gt is not None

    @property
    def diverged(self):
        """True if the oscillator model blew up during the run (needs are bounded near +/-1)."""
        needs = np.concatenate((self.child_need, self.parent_need, self.child_acc, self.parent_acc))
        return bool(not np.isfinite(needs).all() or np.abs(needs).max() > 10)

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0]) if len(self.t) > 1 else 0.0

    @property
    def label(self):
        return f"avoidant {self.avoidant:g}, ambivalent {self.ambivalent:g}"

    def positions(self, source="gt"):
        """Return (child_xy, parent_xy) from ground truth ('gt') or odometry ('odom')."""
        if source == "odom":
            return self.child_odom, self.parent_odom
        return self.child_gt, self.parent_gt

    def distance(self, source="gt"):
        """Child-caregiver distance (m)."""
        child, parent = self.positions(source)
        return np.hypot(*(child - parent).T)


def _merge(child_data, parent_data):
    """Merge child and parent dictionaries (same rule as read_recorded_data.load_robot_data)."""
    merged = {}
    for key in set(child_data) | set(parent_data):
        child_val = child_data.get(key, [])
        parent_val = parent_data.get(key, [])
        if len(child_val) and not len(parent_val):
            merged[key] = child_val
        else:
            merged[key] = parent_val
    return merged


def load_run(path):
    """Load one experiment folder. Returns None if it is incomplete."""
    child_file = os.path.join(path, "child_data.npy")
    parent_file = os.path.join(path, "parent_data.npy")
    if not (os.path.exists(child_file) and os.path.exists(parent_file)):
        return None
    child = np.load(child_file, allow_pickle=True).item()
    parent = np.load(parent_file, allow_pickle=True).item()
    data = _merge(child, parent)

    meta = {}
    meta_file = os.path.join(path, "meta.json")
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            meta = json.load(f)
    legacy = meta.get("param_mapping") != "fixed"

    # The two robots save independently, so their series can differ by a sample
    n = min(len(data["Xc"]), len(data["Xp"]))
    if n < 2:
        return None
    arr = lambda key, source=data: np.asarray(source[key][:n], dtype=float)

    Xc, Xp = arr("Xc"), arr("Xp")
    recorded_avoidance = float(np.median(data["avoidance"]))
    recorded_ambivalence = float(np.median(data["ambivalence"]))
    if legacy:
        # Legacy controller: epsilonAv <- ambivalent, epsilonAm <- avoidant
        avoidant, ambivalent = recorded_ambivalence, recorded_avoidance
    else:
        avoidant, ambivalent = recorded_avoidance, recorded_ambivalence

    run = Run(
        index=int(re.search(r"(\d+)$", path.rstrip("/")).group(1)),
        path=path, avoidant=round(avoidant, 3), ambivalent=round(ambivalent, 3), legacy=legacy,
        t=np.arange(n) * SAMPLE_PERIOD,
        child_need=Xc[:, 0], child_acc=Xc[:, 1], parent_need=Xp[:, 0], parent_acc=Xp[:, 1],
        child_state=arr("child_states"), parent_state=arr("parent_states"),
        vocal_child=arr("vocal_child"), vocal_parent=arr("vocal_parent"),
        meta=meta,
        controller=meta.get("controller", "legacy" if legacy else "decoupled"),
        distance_source=meta.get("distance"),
        networks=meta.get("networks") or "default",
        seed=int(meta.get("seed", -1)),
        initial=tuple(meta.get("initial", (-1.0, 1.0, -1.0, 1.0))),
        batch=meta.get("batch", ""),
    )
    logs = {}
    for name in ("child", "parent", "central"):
        log_file = os.path.join(path, f"controller_{name}.npy")
        if os.path.exists(log_file):
            log = np.load(log_file, allow_pickle=True).item()
            # saved while the control thread may still append: cut every series to a common length
            n_log = min(len(v) for k, v in log.items() if k != "meta")
            logs[name] = {k: (v if k == "meta" else v[:n_log]) for k, v in log.items()}
    run.logs = logs or None

    if "child_gt" in child and len(child["child_gt"]):
        # Positions of both robots come from the child's file so they share its clock
        t = arr("time", child)
        run.t = t - t[0]
        run.child_gt, run.parent_gt = arr("child_gt", child), arr("parent_gt", child)
        run.child_odom, run.parent_odom = arr("child_odom", child)[:, :2], arr("parent_odom", child)[:, :2]
        run.arena_centre = tuple(meta.get("arena_centre", ARENA_CENTRE))
        run.arena_size = meta.get("arena_size", ARENA_SIZE)
        run.box_size = meta.get("box_size", BOX_SIZE)
        # Drop the first samples, taken before Gazebo published any model state
        valid = ~np.isnan(run.child_gt).any(axis=1) & ~np.isnan(run.parent_gt).any(axis=1)
        if not valid.all():
            for name in ("t", "child_need", "child_acc", "parent_need", "parent_acc", "child_state",
                         "parent_state", "vocal_child", "vocal_parent", "child_gt", "parent_gt",
                         "child_odom", "parent_odom"):
                setattr(run, name, getattr(run, name)[valid])
    return run


def load_all(data_dir=None):
    """Load every complete experiment, ordered by index."""
    data_dir = data_dir or default_data_dir()
    runs = [load_run(p) for p in glob.glob(os.path.join(data_dir, "experiment_data_*"))]
    return sorted((r for r in runs if r is not None), key=lambda r: r.index)


def find_run(runs, avoidant, ambivalent, tolerance=0.01):
    """Longest run with positions recorded at the given parameters, or None."""
    matches = [r for r in runs if r.has_positions
               and abs(r.avoidant - avoidant) < tolerance and abs(r.ambivalent - ambivalent) < tolerance]
    return max(matches, key=lambda r: r.duration) if matches else None


if __name__ == "__main__":
    for run in load_all():
        kind = "legacy (labels corrected)" if run.legacy else "odometry"
        print(f"experiment_data_{run.index:<3d} {run.duration:6.1f}s  {run.label:34s} {kind}")
