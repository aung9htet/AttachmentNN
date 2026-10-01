#!/usr/bin/env python3
"""
MiRo actuator control for cosmetic actions, sounds, and visual displays.
"""
import os
import rospy
from std_msgs.msg import UInt32MultiArray, UInt16MultiArray, Float32MultiArray, Int16MultiArray, Bool
from sensor_msgs.msg import JointState

class TailController:
    """
        Control tail wagging.
    """
    def __init__(self, robot_name):
        self.publisher = rospy.Publisher(f"/{robot_name}/control/cosmetic_joints", Float32MultiArray, queue_size=0)
        self.msg = Float32MultiArray()
        self.msg.data = [0.0, 0.5, 0.0, 0.0, 0.0, 0.0]

    def wag(self, value):
        """Set tail wag position (-1 to 1)."""
        self.msg.data[1] = value
        self.publisher.publish(self.msg)

class AudioController:
    """Creates a simulated audio output"""
    
    def __init__(self, robot_name):
        self.publisher = rospy.Publisher(f"/{robot_name}/audio_simulation", Bool, queue_size=0)

    def play_audio(self, audio):
        """Play a tone with specified frequency, volume, and duration."""
        msg = Bool(data=audio)
        self.publisher.publish(msg)

class NeckController:
    """
        Control neck movements.
    """
    def __init__(self, robot_name):
        self.neck_publisher = rospy.Publisher(f"/{robot_name}/control/kinematic_joints", JointState, queue_size=0)
        self.msg = JointState()
        self.msg.position = [0.0, 0.0, 0.0, 0.5]

    def reset(self):
        self.neck_publisher.publish(self.msg)

class IlluminationController:
    """
        Control LED illumination.
    """
    def __init__(self, robot_name):
        self.publisher = rospy.Publisher(f"/{robot_name}/control/illum", UInt32MultiArray, queue_size=0)
        self.msg = UInt32MultiArray()
        self.msg.data = [0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 
                         0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF]

    def set_color(self, red=0, green=0, blue=255):
        """Set all LEDs to RGB color (0-255 per channel)."""
        r, g, b = int(red), int(green), int(blue)
        color = 0xFF000000 | (r << 16) | (g << 8) | b
        self.msg.data = [color] * 6
        self.publisher.publish(self.msg)
