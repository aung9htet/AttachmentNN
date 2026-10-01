#!/usr/bin/env python3
"""
Experiment timer - ends the run after a fixed simulation time and saves run metadata.
"""
import os
import json
from datetime import datetime

import rospy
from utils import ARENA_CENTRE, ARENA_SIZE, BOX_SIZE, latest_experiment_dir, fix_owner


class ExperimentTimer:
    """Stop the launch after `~duration` simulation seconds (0 = run until Ctrl+C)."""

    def __init__(self):
        rospy.init_node('experiment_timer')
        self.duration = float(rospy.get_param("~duration", 0.0))
        self.avoidant = float(rospy.get_param("/attachment/avoidant", 0.0))
        self.ambivalent = float(rospy.get_param("/attachment/ambivalent", 0.0))
        self.start_time = None
        rospy.on_shutdown(self._save_meta)

    def _save_meta(self):
        """Save run parameters next to the recorded data."""
        elapsed = 0.0 if self.start_time is None else rospy.get_time() - self.start_time
        meta = {
            'avoidant': self.avoidant,
            'ambivalent': self.ambivalent,
            'requested_duration_s': self.duration,
            'recorded_duration_s': round(elapsed, 2),
            'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # 'fixed': ambivalent drives the physical-distance coupling and avoidant the
            # emotional-distance coupling. Runs without meta.json were recorded with the two swapped.
            'param_mapping': 'fixed',
            'arena_centre': list(ARENA_CENTRE),
            'arena_size': ARENA_SIZE,
            'box_size': BOX_SIZE,
        }
        filename = os.path.join(latest_experiment_dir(), "meta.json")
        with open(filename, "w") as f:
            json.dump(meta, f, indent=2)
        fix_owner(filename)
        rospy.loginfo(f"Saved run metadata to {filename}")

    def run(self):
        rate = rospy.Rate(10)
        # Wait for the simulation clock
        while not rospy.is_shutdown() and rospy.get_time() == 0:
            rate.sleep()
        self.start_time = rospy.get_time()

        while not rospy.is_shutdown():
            if self.duration > 0 and rospy.get_time() - self.start_time >= self.duration:
                rospy.loginfo(f"Experiment duration of {self.duration}s reached, shutting down")
                break
            rate.sleep()


if __name__ == '__main__':
    try:
        ExperimentTimer().run()
    except rospy.ROSInterruptException:
        pass
