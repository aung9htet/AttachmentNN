#!/usr/bin/env python3

import os
import rospy
import numpy as np
from datetime import datetime
from attachment_model.msg import RobotPub
from subscribers import OdomSubscriber, ActionSubscriber, EmotionSubscriber, AudioSubscriber, GroundTruthSubscriber
from utils import ARENA_SIZE, odom_to_world, latest_experiment_dir, fix_owner
from behaviors import Explorer, Approacher
from actuators import TailController, IlluminationController, AudioController, NeckController

class AttachmentAgent:
    """
        General agent for attachment modelling
    """
    
    def __init__(self, robot_name, model_separation=True):
        self.robot_name = robot_name
        rospy.init_node(f'{self.robot_name}_node')
        # The coupled controller publishes both robots' actions on /central/controller
        self.model_separation = rospy.get_param("/attachment/controller", "decoupled") != "coupled"
        model_separation = self.model_separation
        # Seed the random exploration (child: seed, caregiver: seed + 1) when a run seed is given
        seed = int(rospy.get_param("/attachment/seed", -1))
        if seed >= 0:
            np.random.seed(seed + (0 if robot_name == "child" else 1))
        
        # Publishers
        self.state = rospy.Publisher(f"/{self.robot_name}/state", RobotPub, queue_size=10)
        self.robot_pub = RobotPub()
        
        # State subscribers
        self.emotion = EmotionSubscriber()
        self.odom = OdomSubscriber(self.robot_name)
        # Odometry and ground truth of both robots, for path and distance recording
        self.child_odom = self.odom if self.robot_name == "child" else OdomSubscriber("child")
        self.parent_odom = self.odom if self.robot_name == "parent" else OdomSubscriber("parent")
        self.child_gt = GroundTruthSubscriber("child")
        self.parent_gt = GroundTruthSubscriber("parent")
        if model_separation:
            self.action_sub = ActionSubscriber(self.robot_name)
        else:
            self.action_sub = ActionSubscriber("central")
        # Listen to the partner's calls (the child listened to its own channel until audio_fix)
        self.audio_detector = AudioSubscriber("parent" if self.robot_name == "child" else "child")
        
        # Behavior
        self.explorer = Explorer(self.robot_name)
        self.approacher = Approacher(self.robot_name, target_tag_id=0)

        # State Controllers
        self.tail = TailController(self.robot_name)
        self.lights = IlluminationController(self.robot_name)
        self.audio = AudioController(self.robot_name)
        self.neck = NeckController(self.robot_name)
        
        # State variables
        self.evidence_trace = 0.0
        self.call_trace = 0.0
        self.kp = 0.1  # Call trace decay
        self.ke = 0.1  # Evidence trace decay
        self.kappa = 0.5  # Evidence scaling

        # Robot state variables
        self.action = getattr(self.action_sub, self.robot_name, 0)
        self.produce_sound = False
        self.sound_detected = False
        self.state_msg = RobotPub()
        self.state_msg.pos_x = getattr(self.odom, 'posx', 0.0)
        self.state_msg.pos_y = getattr(self.odom, 'posy', 0.0)
        self.start_time = rospy.get_rostime()
        self.robot_found = False

        # Dictionary to store robot data
        self.robot_data = {
            'Xc': [],
            'Xp': [],
            'ambivalence': [],
            'avoidance': [],
            'physical_coupling_targets': [],
            'emotional_coupling_targets': [],
            'child_states': [],
            'parent_states': [],
            'vocal_child': [],
            'vocal_parent': [],
            'child_evidence_trace': [],
            'parent_evidence_trace': [],
            'time': [],                 # simulation time (s)
            'child_odom': [],           # [x, y, yaw] odometry path in the world frame
            'parent_odom': [],          # [x, y, yaw] odometry path in the world frame
            'child_odom_raw': [],       # [x, y, yaw] as published on /child/sensors/odom
            'parent_odom_raw': [],      # [x, y, yaw] as published on /parent/sensors/odom
            'child_gt': [],             # [x, y] Gazebo ground truth
            'parent_gt': [],            # [x, y] Gazebo ground truth
            'odom_distance': [],        # child-parent distance from odometry (m)
            'odom_distance_norm': [],   # min(1, distance / arena size)
            'gt_distance': [],          # child-parent distance from ground truth (m)
            'gt_distance_norm': []      # min(1, distance / arena size)
        }
        
        rospy.on_shutdown(self._save_data)

    def _detect_sound(self, detection = False):
        """Simple sound detection and evidence trace based on boolean message."""
        if detection is True:
            self.sound_detected = getattr(self.audio_detector, 'audio', False)
        else:
            self.sound_detected = False
        self.call_trace = (1 - self.kp) * self.call_trace + self.kappa * self.produce_sound
        self.evidence_trace = (1 - self.ke) * self.evidence_trace + self.kappa * self.sound_detected * self.call_trace

    def _update_color(self):
        """Update LED color based on emotional distance."""
        emotional_dist = getattr(self.action_sub, 'emotional_distance', 0.0) or 0.0
        # Clamp to [0,1]
        emotional_dist = max(0.0, min(1.0, float(emotional_dist)))
        red = int(max(0, min(255, emotional_dist * 255)))
        green = int(max(0, min(255, (1.0 - emotional_dist) * 255)))
        self.lights.set_color(red=red, green=green, blue=0)

    def _wag_tail(self):
        """Sinusoidal tail wagging pattern."""
        time_elapsed = rospy.Time.now().to_sec() - self.start_time.to_sec()
        wag_value = np.sin(time_elapsed * 10.0)
        self.tail.wag(wag_value)

    def _update_data(self):
        if self.robot_name == "child":
            self.robot_data['Xc'].append([self.action_sub.child_need, self.action_sub.child_accumulated_need])
            self.robot_data['child_states'].append(self.action)
            self.robot_data['vocal_child'].append(1 if self.state_msg.robot_sound else 0)
            self.robot_data['child_evidence_trace'].append(self.evidence_trace)
        elif self.robot_name == "parent":
            self.robot_data['Xp'].append([self.action_sub.parent_need, self.action_sub.parent_accumulated_need])
            self.robot_data['parent_states'].append(self.action)
            self.robot_data['vocal_parent'].append(1 if self.state_msg.robot_sound else 0)
            self.robot_data['parent_evidence_trace'].append(self.evidence_trace)
        
        self.robot_data['ambivalence'].append(self.emotion.ambivalent)
        self.robot_data['avoidance'].append(self.emotion.avoidant)
        self.robot_data['physical_coupling_targets'].append(self.action_sub.physical_distance)
        self.robot_data['emotional_coupling_targets'].append(self.action_sub.emotional_distance)

        # Robot paths and inter-robot distance
        child_raw = [self.child_odom.posx, self.child_odom.posy, self.child_odom.yaw]
        parent_raw = [self.parent_odom.posx, self.parent_odom.posy, self.parent_odom.yaw]
        child_pose = odom_to_world("child", *child_raw)
        parent_pose = odom_to_world("parent", *parent_raw)
        distance = float(np.hypot(child_pose[0] - parent_pose[0], child_pose[1] - parent_pose[1]))
        self.robot_data['time'].append(rospy.get_time())
        self.robot_data['child_odom'].append(child_pose)
        self.robot_data['parent_odom'].append(parent_pose)
        self.robot_data['child_odom_raw'].append(child_raw)
        self.robot_data['parent_odom_raw'].append(parent_raw)
        self.robot_data['child_gt'].append([self.child_gt.posx, self.child_gt.posy])
        self.robot_data['parent_gt'].append([self.parent_gt.posx, self.parent_gt.posy])
        self.robot_data['odom_distance'].append(distance)
        self.robot_data['odom_distance_norm'].append(min(1.0, distance / ARENA_SIZE))
        gt_distance = float(np.hypot(self.child_gt.posx - self.parent_gt.posx,
                                     self.child_gt.posy - self.parent_gt.posy))
        self.robot_data['gt_distance'].append(gt_distance)
        self.robot_data['gt_distance_norm'].append(min(1.0, gt_distance / ARENA_SIZE))

    def _save_data(self):
        """Save robot data on shutdown."""
        # Save into the latest experiment_data_{index} folder (created by the camera recorder)
        exp_dir = latest_experiment_dir()

        filename = os.path.join(exp_dir, f"{self.robot_name}_data.npy")

        # Save as numpy dictionary
        np.save(filename, self.robot_data, allow_pickle=True)
        fix_owner(filename)
        rospy.loginfo(f"Saved {self.robot_name} data to {filename}")
