#!/usr/bin/env python3
"""
Run a batch of Gazebo attachment experiments in parallel Docker containers (host side).

Each worker is its own `docker run --rm` of the MiRo image, without --net=host, so every
simulation has its own ROS master and Gazebo ports. It mounts a private copy of
ros/Attachment-Model (with an empty data/ folder), the X display (the robots' cameras need
rendering) and the GPU. After each run the worker checks the run folder and moves it to the
output directory as experiment_data_{next index}. A manifest records every attempt, so an
interrupted batch can be resumed by running the same command again.

    python3 experiments/run_batch.py experiments/tcds_runs.csv --workers 5
    python3 experiments/run_batch.py experiments/pilot_runs.csv --workers 4 --out experiments/pilot_data

The run list is a CSV with columns: id, batch, controller, distance, avoidant, ambivalent,
duration, seed, initial (see make_runs.py). Only the standard library is used.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE = os.path.join(REPO, "ros", "Attachment-Model")
DEFAULT_OUT = os.path.join(PACKAGE, "data")
IMAGE = "aung9htet/ubuntu-20.04:miro"
CONTAINER_PKG = "/root/mdk/catkin_ws/src/Attachment-Model"
REQUIRED = ("child_data.npy", "parent_data.npy", "meta.json", "camera_recording.mp4")

print_lock = threading.Lock()
index_lock = threading.Lock()


def say(message):
    with print_lock:
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def next_index(out_dir):
    index = 0
    while os.path.exists(os.path.join(out_dir, f"experiment_data_{index}")):
        index += 1
    return index


def read_runs(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def read_manifest(path):
    done = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["status"] == "ok":
                    done[row["id"]] = row
    return done


def append_manifest(path, row):
    new = not os.path.exists(path)
    with print_lock, open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


def remove_tree(path):
    """Remove a folder, using a container for files the (root) containers left behind."""
    try:
        shutil.rmtree(path)
    except PermissionError:
        parent, name = os.path.split(path)
        subprocess.call(["docker", "run", "--rm", "-v", f"{parent}:/w", IMAGE, "rm", "-rf", f"/w/{name}"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def prepare_worker(work_dir, worker):
    """Private copy of the ROS package with an empty data folder."""
    pkg = os.path.join(work_dir, f"w{worker}", "Attachment-Model")
    if os.path.exists(pkg):
        remove_tree(pkg)
    shutil.copytree(PACKAGE, pkg, ignore=shutil.ignore_patterns("data", "__pycache__", "*.pyc"))
    os.makedirs(os.path.join(pkg, "data"))
    return pkg


def launch_command(run):
    args = [f"avoidant:={float(run['avoidant'])}", f"ambivalent:={float(run['ambivalent'])}",
            f"duration:={float(run['duration'])}", "gui:=false",
            f"controller:={run['controller']}", f"distance:={run['distance'] or 'gt'}",
            f"seed:={int(run['seed'])}", f"batch:={run['batch']}"]
    if run["initial"]:
        args.append(f"initial:={run['initial']}")
    timeout = int(float(run["duration"])) + 240
    return (f"cd {CONTAINER_PKG} && timeout {timeout} roslaunch attachment_model relationship.launch "
            + " ".join(args))


def check_run(folder, run):
    """Return (ok, message) for a finished run folder."""
    missing = [name for name in REQUIRED if not os.path.exists(os.path.join(folder, name))]
    expected_logs = {"coupled": ["controller_central.npy"]}.get(run["controller"],
                                                                 ["controller_child.npy", "controller_parent.npy"])
    missing += [name for name in expected_logs if not os.path.exists(os.path.join(folder, name))]
    if missing:
        return False, "missing " + ", ".join(missing)
    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)
    recorded = float(meta.get("recorded_duration_s") or 0)
    if recorded < 0.97 * float(run["duration"]):
        return False, f"recorded {recorded:.1f}s of {run['duration']}s"
    if meta.get("controller") != run["controller"] or int(meta.get("seed", -2)) != int(run["seed"]):
        return False, "meta does not match the requested run"
    wall = meta.get("wall_duration_s")
    rtf = recorded / wall if wall else float("nan")
    return True, f"{recorded:.0f}s recorded, real-time factor {rtf:.2f}"


def worker_loop(worker, queue, args, manifest, display):
    pkg = prepare_worker(args.work, worker)
    data = os.path.join(pkg, "data")
    log_dir = os.path.join(args.work, "logs")
    os.makedirs(log_dir, exist_ok=True)
    name = f"attnn_w{worker}"
    while True:
        with index_lock:
            if not queue:
                return
            run = queue.pop(0)
        for leftover in os.listdir(data):
            remove_tree(os.path.join(data, leftover))
        cmd = ["docker", "run", "--rm", "--name", name,
               "-e", f"DISPLAY={display}", "-e", "QT_X11_NO_MITSHM=1",
               "-e", "NVIDIA_DRIVER_CAPABILITIES=all", "-e", "NVIDIA_VISIBLE_DEVICES=all",
               "-e", "PYTHONDONTWRITEBYTECODE=1",
               "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw", "-v", f"{pkg}:{CONTAINER_PKG}",
               "--gpus", "all", IMAGE, "bash", "-ic", launch_command(run)]
        say(f"w{worker} start {run['id']}")
        started = time.time()
        with open(os.path.join(log_dir, f"{run['id']}.log"), "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        folders = sorted(d for d in os.listdir(data) if d.startswith("experiment_data_"))
        if len(folders) == 1:
            ok, message = check_run(os.path.join(data, folders[0]), run)
        else:
            ok, message = False, f"{len(folders)} run folders (exit code {code})"
        index = ""
        if ok:
            with index_lock:
                index = next_index(args.out)
                shutil.move(os.path.join(data, folders[0]), os.path.join(args.out, f"experiment_data_{index}"))
        say(f"w{worker} {'ok  ' if ok else 'FAIL'} {run['id']}: {message}"
            + (f" -> experiment_data_{index}" if ok else ""))
        append_manifest(manifest, dict(id=run["id"], status="ok" if ok else "failed", index=index,
                                       message=message, wall_s=round(time.time() - started, 1),
                                       finished=f"{datetime.now():%Y-%m-%d %H:%M:%S}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", help="CSV run list")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default=DEFAULT_OUT, help="where finished run folders go")
    parser.add_argument("--work", default=os.path.join(REPO, "experiments", "work"),
                        help="worker copies of the package and run logs")
    parser.add_argument("--only", default="", help="comma-separated batch labels to run")
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)
    args.work = os.path.abspath(args.work)
    os.makedirs(args.out, exist_ok=True)
    display = os.environ.get("DISPLAY")
    if not display:
        sys.exit("DISPLAY is not set: the robots' cameras need an X display for Gazebo rendering")

    manifest = os.path.join(args.work, "manifest_" + os.path.splitext(os.path.basename(args.runs))[0] + ".csv")
    os.makedirs(args.work, exist_ok=True)
    done = read_manifest(manifest)
    runs = [r for r in read_runs(args.runs) if r["id"] not in done]
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if r["batch"] in wanted]
    total = sum(float(r["duration"]) + 30 for r in runs) / 60
    say(f"{len(runs)} runs to do ({len(done)} already done), ~{total / max(1, args.workers):.0f} min "
        f"with {args.workers} workers; output {args.out}")
    threads = [threading.Thread(target=worker_loop, args=(w, runs, args, manifest, display))
               for w in range(args.workers)]
    for t in threads:
        t.start()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        say("interrupted: stopping containers")
        for w in range(args.workers):
            subprocess.call(["docker", "kill", f"attnn_w{w}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise
    say("batch finished")


if __name__ == "__main__":
    main()
