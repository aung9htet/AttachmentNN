#!/usr/bin/env python3
"""
ROS subscribers for robot state and control data.
"""
import os
import rospy
from std_msgs.msg import Bool
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Range
from gazebo_msgs.msg import ModelStates
from tf.transformations import euler_from_quaternion
from math import degrees
from attachment_model.msg import Action, RobotPub, Emotion, Perception

class OdomSubscriber:
    """
        Subscribe to odometry data and track robot position/orientation.
    """
    def __init__(self, robot_name):
        self.posx = 0.0
        self.posy = 0.0
        self.yaw = 0.0
        self.subscriber = rospy.Subscriber(f"/{robot_name}/sensors/odom", Odometry, self._callback)

    def _callback(self, msg):
        orientation = msg.pose.pose.orientation
        position = msg.pose.pose.position
        (_, _, yaw) = euler_from_quaternion(
            [orientation.x, orientation.y, orientation.z, orientation.w], 'sxyz'
        )
        self.yaw = round(degrees(yaw) + (360 if yaw < 0 else 0), 4)
        self.posx = round(position.x, 4)
        self.posy = round(position.y, 4)


class GroundTruthSubscriber:
    """
        Subscribe to Gazebo model states and track a model's true world position.
    """
    def __init__(self, robot_name):
        self.robot_name = robot_name
        self.posx = float('nan')
        self.posy = float('nan')
        self.subscriber = rospy.Subscriber("/gazebo/model_states", ModelStates, self._callback, queue_size=1)

    def _callback(self, msg):
        if self.robot_name in msg.name:
            position = msg.pose[msg.name.index(self.robot_name)].position
            self.posx = round(position.x, 4)
            self.posy = round(position.y, 4)


class RangeSubscriber:
    """
        Subscribe to sonar range data.
    """
    def __init__(self, robot_name):
        self.range = 0.0
        self.subscriber = rospy.Subscriber(f"/{robot_name}/sensors/sonar", Range, self._callback)

    def _callback(self, msg):
        self.range = msg.range


class ActionSubscriber:
    """
        Subscribe to central controller action commands.
    """
    def __init__(self, robot_name):
        self.parent = 0
        self.child = 0
        self.emotional_distance = 0
        self.physical_distance = 0
        self.child_need = 0
        self.parent_need = 0
        self.child_accumulated_need = 0
        self.parent_accumulated_need = 0
        self.subscriber = rospy.Subscriber(f"{robot_name}/controller", Action, self._callback)

    def _callback(self, msg):
        self.parent = msg.parent
        self.child = msg.child
        self.emotional_distance = msg.emotional_distance
        self.physical_distance = msg.physical_distance
        self.child_need = msg.child_need
        self.parent_need = msg.parent_need
        self.child_accumulated_need = msg.child_accumulated_need
        self.parent_accumulated_need = msg.parent_accumulated_need


class RobotStateSubscriber:
    """
        Subscribe to robot state (child or parent).
    """
    def __init__(self, robot_name):
        self.pos_x = 0.0
        self.pos_y = 0.0
        self.robot_sound = False
        self.subscriber = rospy.Subscriber(f"/{robot_name}/state", RobotPub, self._callback)

    def _callback(self, msg):
        self.pos_x = msg.pos_x
        self.pos_y = msg.pos_y
        self.robot_sound = msg.robot_sound

class EmotionSubscriber:
    """
        Subscribe to emotion parameters.
    """
    def __init__(self):
        # Set per run with: roslaunch attachment_model relationship.launch avoidant:=.. ambivalent:=..
        self.avoidant = float(rospy.get_param("/attachment/avoidant", 0.0))
        self.ambivalent = float(rospy.get_param("/attachment/ambivalent", 0.0))
        self.subscriber = rospy.Subscriber("/emotion", Emotion, self._callback)

    def _callback(self, msg):
        self.avoidant = msg.avoidant
        self.ambivalent = msg.ambivalent

class AudioSubscriber:
    """
        Detect audio and process for simulation
    """
    
    def __init__(self, robot_name):
        self.subscriber = rospy.Subscriber(f"/{robot_name}/audio_simulation", Bool, self._callback)
        self.audio = False

    def _callback(self, msg):
        data = msg.data
        self.audio = data

class PerceptionSubscriber:
    """
        Subscribe to perception data.
    """
    def __init__(self, robot_name):
        self.subscriber = rospy.Subscriber(
            f"/{robot_name}/tag_detection", 
            Perception, 
            self._callback,
            queue_size=1,
            tcp_nodelay=True
        )
        self.left_detection = False
        self.right_detection = False

    def _callback(self, msg):
        self.left_detection = msg.left_detection
        self.right_detection = msg.right_detection
