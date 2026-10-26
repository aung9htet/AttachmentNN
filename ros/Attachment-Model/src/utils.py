#!/usr/bin/env python3
"""
MiRo utility constants and configurations.
"""
import numpy as np

# Audio constants
BLOCK_SAMPLES = 500
BLOCK_RATE = 40
MICS = 4
TONE_FREQUENCY = {'min': 50, 'max': 2000}
TONE_VOLUME = {'min': 0, 'max': 255}

# Visual salience map dimensions
PRI = {'width': 178, 'height': 100}
PRIW = {'width': 256, 'height': 1}

# Color definitions (BGR tuples and ARGB words)
BGR_TUPLE = {
    'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0),
    'yellow': (0, 255, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255),
    'white': (255, 255, 255), 'black': (0, 0, 0),
}

ARGB_WORD = {
    'red': 0xFFFF0000, 'green': 0xFF00FF00, 'blue': 0xFF0000FF,
    'yellow': 0xFFFFFF00, 'cyan': 0xFF00FFFF, 'magenta': 0xFFFF00FF,
    'white': 0xFFFFFFFF, 'black': 0x00000000, 'off': 0x00000000,
}

# Camera calibration parameters
MTX = np.array([
    [1.04358065e+03, 0, 3.29969935e+02],
    [0, 1.03845278e+03, 1.68243114e+02],
    [0, 0, 1]
])
DIST = np.array([[-3.63299415e+00, 1.52661324e+01, -7.23780207e-03, 
                   -7.48630198e-04, -3.20700124e+01]])
FOCAL_LENGTH = 330

# Arena geometry in the Gazebo world frame (worlds/attachment.world)
ARENA_CENTRE = (0.325, 0.325)   # caregiver box centre (m)
ARENA_SIZE = 2.61               # wall-to-wall side length L (m)
BOX_SIZE = 0.3                  # caregiver box side length (m)
# Spawn poses (x, y, yaw in rad). MiRo odometry starts at zero in the robot's own
# start frame, so these place the odometry path in the world frame.
SPAWN_POSE = {'child': (1.0, 0.325, 3.14), 'parent': (0.375, 0.325, 0.0)}


def odom_to_world(robot_name, x, y, yaw_deg):
    """Convert a robot's raw odometry pose to the world frame: [x, y, yaw in degrees]."""
    x0, y0, yaw0 = SPAWN_POSE[robot_name]
    wx = x0 + np.cos(yaw0) * x - np.sin(yaw0) * y
    wy = y0 + np.sin(yaw0) * x + np.cos(yaw0) * y
    return [round(float(wx), 4), round(float(wy), 4), round(float((yaw_deg + np.degrees(yaw0)) % 360), 4)]


DEFAULT_INITIAL = (-1.0, 1.0, -1.0, 1.0)   # (x_c, y_c, x_p, y_p) used by all earlier runs


def initial_state(spec="", seed=-1):
    """Starting needs (x_c, y_c, x_p, y_p) for a run.

    spec: "" for the default (-1, 1, -1, 1), "random" for U(-1, 1) drawn from `seed`
    (seed must be >= 0 so that every node computes the same state), or four numbers.
    """
    spec = str(spec).strip()
    if not spec:
        return DEFAULT_INITIAL
    if spec == "random":
        if int(seed) < 0:
            raise ValueError("initial:=random needs seed >= 0")
        return tuple(float(v) for v in np.random.default_rng(int(seed)).uniform(-1.0, 1.0, 4))
    values = tuple(float(v) for v in spec.replace(",", " ").split())
    if len(values) != 4:
        raise ValueError(f"initial needs four numbers, got {spec!r}")
    return values


def data_root():
    """Return the package data directory, creating it if needed."""
    import os
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(root, exist_ok=True)
    return root


def latest_experiment_dir():
    """Return the latest existing experiment_data_{index} folder (created if none)."""
    import os
    root = data_root()
    index = 0
    last_existing = None
    while True:
        candidate = os.path.join(root, f"experiment_data_{index}")
        if os.path.exists(candidate):
            last_existing = candidate
            index += 1
        else:
            break
    if last_existing is None:
        last_existing = os.path.join(root, "experiment_data_0")
        os.makedirs(last_existing, exist_ok=True)
        fix_owner(last_existing)
    return last_existing


def fix_owner(path):
    """Give a file written by the (root) container the owner of the data folder.

    The package is bind-mounted from the host, so this keeps recordings
    editable from the host without sudo.
    """
    import os
    try:
        st = os.stat(data_root())
        os.chown(path, st.st_uid, st.st_gid)
    except OSError:
        pass
