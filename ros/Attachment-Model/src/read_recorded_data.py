#!/usr/bin/env python3

import os
import sys
import glob
import numpy as np


def load_robot_data(path):
    """Load child and parent .npy files from the experiment directory.

    Returns (merged_data_dict, [filepaths]).
    """
    if os.path.isdir(path):
        child_file = os.path.join(path, "child_data.npy")
        parent_file = os.path.join(path, "parent_data.npy")
        
        if not os.path.exists(child_file) or not os.path.exists(parent_file):
            raise FileNotFoundError(f"Missing child_data.npy or parent_data.npy in: {path}")
        
        # Load both files
        child_data = np.load(child_file, allow_pickle=True).item()
        parent_data = np.load(parent_file, allow_pickle=True).item()
        
        # Merge the data intelligently - combine lists for matching keys
        merged_data = {}
        all_keys = set(child_data.keys()) | set(parent_data.keys())
        
        for key in all_keys:
            child_val = child_data.get(key, [])
            parent_val = parent_data.get(key, [])
            
            # If both have data, prefer the non-empty one
            # Otherwise combine them intelligently
            if child_val and not parent_val:
                merged_data[key] = child_val
            elif parent_val and not child_val:
                merged_data[key] = parent_val
            elif child_val and parent_val:
                # Both have data - keep both (this handles shared keys like ambivalence)
                merged_data[key] = parent_val  # Use parent since they should be the same for shared state
            else:
                merged_data[key] = []
        
        return merged_data, [child_file, parent_file]
    else:
        # Single file path provided
        filepath = path
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File does not exist: {filepath}")
        
        data = np.load(filepath, allow_pickle=True).item()
        return data, [filepath]


def print_robot_summary(name, data):
    """Print a simple text summary for the loaded robot data dict."""
    print(f"\n===== {name.upper()} DATA SUMMARY =====")

    def length(key):
        return len(data.get(key, []))

    # Estimate timesteps from the longest relevant series
    series_lengths = [
        length('child_states'),
        length('parent_states'),
        length('ambivalence'),
        length('avoidance'),
    ]
    total_timesteps = max(series_lengths) if series_lengths else 0
    print(f"Total timesteps: {total_timesteps}")

    for key in [
        'Xc', 'Xp', 'ambivalence', 'avoidance',
        'physical_coupling_targets', 'emotional_coupling_targets',
        'child_states', 'parent_states',
        'vocal_child', 'vocal_parent',
        'child_evidence_trace', 'parent_evidence_trace',
    ]:
        print(f"- {key}: {length(key)} entries")

    # Show small samples if available
    if data.get('Xc'):
        print("\nExample child state (Xc[0]):", data['Xc'][0])
    if data.get('Xp'):
        print("Example parent state (Xp[0]):", data['Xp'][0])
    if data.get('ambivalence'):
        print("First 5 ambivalence:", data['ambivalence'][:5])
    if data.get('avoidance'):
        print("First 5 avoidance:", data['avoidance'][:5])


def is_experiment_complete(data):
    """Return True if both child and parent data are present; else False."""

    has_child = bool(data.get('Xc')) or bool(data.get('child_states'))
    has_parent = bool(data.get('Xp')) or bool(data.get('parent_states'))
    return has_child and has_parent


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    # Interactive prompt for experiment index
    try:
        user_input = input("Enter experiment index (default 0): ").strip()
    except EOFError:
        user_input = ""

    if user_input == "":
        idx = 0
    else:
        try:
            idx = int(user_input)
        except ValueError:
            print(f"Invalid index '{user_input}'. Please enter an integer.")
            return 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_data_dir = os.path.join(base_dir, "..", "data")
    exp_dir = os.path.join(root_data_dir, f"experiment_data_{idx}")

    if not os.path.isdir(exp_dir):
        print(f"Experiment folder does not exist: {exp_dir}")
        return 1

    try:
        data, filepaths = load_robot_data(exp_dir)
    except Exception as e:
        print(f"Error loading data from {exp_dir}: {e}")
        return 1

    name = f"Experiment {idx}"
    print(f"\nLoaded files: {', '.join(filepaths)}")
    print_robot_summary(name, data)

    if not is_experiment_complete(data):
        print("\nWARNING: Incomplete data detected (missing child or parent).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
