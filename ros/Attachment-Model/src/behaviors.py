#!/usr/bin/env python3
"""
    Behavior nodes for MiRo: exploration and tag approach using dual cameras.
"""
import os
import rospy
import numpy as np
from geometry_msgs.msg import TwistStamped
from subscribers import RangeSubscriber, PerceptionSubscriber

class Movement:
    """
        Basic movement control for MiRo.
    """
    def __init__(self, robot_name):
        self.wheel_pub = rospy.Publisher(f"/{robot_name}/control/cmd_vel", TwistStamped, queue_size=0)
        self.cmd = TwistStamped()

    def move(self, linear=0.0, angular=0.0):
        self.cmd.twist.linear.x = linear
        self.cmd.twist.angular.z = angular
        self.wheel_pub.publish(self.cmd)

    def stop(self):
        self.move(0.0, 0.0)


class Explorer:
    """
        Random exploration with obstacle avoidance.

        This is adapted from the original RobotExplore behavior, but designed
        to be called iteratively from an external control loop.
    """

    def __init__(self, robot_name, speed=0.2, min_wall_distance=0.105):
        self.movement = Movement(robot_name)
        self.range = RangeSubscriber(robot_name)

        # Motion parameters
        self.speed = speed
        self.min_wall_distance = min_wall_distance
        self.turn_angle = 0.0

        # State for exploration/turning
        self.start_time = rospy.get_rostime()
        self.is_exploring = True
        # Initial turn direction: +/- 60 degrees (2/3 of 90deg)
        self.angle_turn = np.random.choice([(np.pi / 2) * 2 / 3,
                                            -(np.pi / 2) * 2 / 3])

    def _time_elapsed(self):
        """Return time elapsed since start in seconds (int)."""
        now = rospy.get_rostime()
        return now.secs - self.start_time.secs

    def _should_turn(self):
        """Decide whether the robot should be turning to avoid obstacles.

        - If already turning (is_exploring is False), keep turning for a
          while (based on time modulo 5s).
        - If exploring and wall is closer than min_wall_distance, start turn.
        """
        time_elapsed = self._time_elapsed()

        # Currently in a turn phase
        if not self.is_exploring:
            if (time_elapsed % 5) == 0:
                # After a multiple of 5 seconds, resume exploration
                self.is_exploring = True
            return True

        # Not currently turning: check obstacle distance
        if self.range.range < self.min_wall_distance:
            # Start a new turn phase
            self.is_exploring = False
            return True

        return False

    def _explore(self):
        """Perform a single exploration step.

        Call this iteratively from your main loop. It decides whether the
        robot should go straight with small random angular variation, or
        rotate in place to avoid obstacles.
        """
        distance = self.range.range
        rospy.logdebug(f"Explorer: range = {distance:.3f} m")

        time_elapsed = self._time_elapsed()

        if not self._should_turn():
            # Straight exploration with occasional random heading changes
            if (time_elapsed % 2) == 0:
                max_angle_turn = np.pi / 2  # up to +/- 90 degrees
                rand_turn = np.random.uniform(-max_angle_turn, max_angle_turn)
                self.turn_angle = rand_turn
            self.movement.move(linear=self.speed, angular=self.turn_angle)
        else:
            # Rotate in place to search for a clearer direction
            if (time_elapsed % 5) == 0:
                # Occasionally re-sample turn direction
                self.angle_turn = np.random.choice([(np.pi / 2) * 2 / 3,
                                                    -(np.pi / 2) * 2 / 3])
            self.movement.move(linear=0.0, angular=self.angle_turn * 2)


class Approacher:
    """
        Track and approach AprilTags using dual cameras.
    """
    def __init__(self, robot_name, target_tag_id=None, camera_index=0):
        self.movement = Movement(robot_name)
        self.perception = PerceptionSubscriber(robot_name)
        self.range = RangeSubscriber(robot_name)
        self.found_tag = False

    def _approach(self):
        print(f"Left Detection: {self.perception.left_detection}, Right Detection: {self.perception.right_detection}")
        print(f"Range: {self.range.range}")
        if self.perception.left_detection and not self.perception.right_detection:
            self.found_tag = True
            self.movement.move(linear=0.1, angular=0.2)
            return True
        elif not self.perception.left_detection and self.perception.right_detection:
            self.found_tag = True
            self.movement.move(linear=0.1, angular=-0.2)
            return True
        elif self.perception.left_detection and self.perception.right_detection:
            self.found_tag = True
            if self.range.range < 0.2:
                self.movement.stop()
            else:
                self.movement.move(linear=0.1, angular=0.0)
            return True
        else:
            self.found_tag = False
            self.movement.move(linear=0.0, angular=0.3)
            return False