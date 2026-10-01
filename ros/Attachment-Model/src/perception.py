#!/usr/bin/env python3
"""
Perception module for AprilTag detection across dual cameras.
Standalone node that detects tags and reports [Bool, Bool] for left/right eyes.
"""
import sys
sys.path.insert(0, '/workspace/.venv/lib/python3.8/site-packages')
import os
import cv2
import rospy
import threading
from dt_apriltags import Detector
from sensor_msgs.msg import CompressedImage
from attachment_model.msg import Perception
from cv_bridge import CvBridge, CvBridgeError


class AprilTagPerception:
    """AprilTag detector supporting all tag families."""
    
    def __init__(self):
        try:
            # Using the stable tag36h11 family to avoid native-library segfaults
            self.detector = Detector(
                families='tag36h11',
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                nthreads=1,
                debug=0,
            )
            self.lock = threading.Lock()
            rospy.loginfo("AprilTag detector initialized for tag36h11")
        except Exception as e:
            rospy.logerr(f"Failed to initialize AprilTag detector: {e}")
            self.detector = None

    def detect_tags(self, image):
        """Detect AprilTags in image and return list of detections."""
        if self.detector is None:
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        with self.lock:
            detections = self.detector.detect(gray)
        
        tags = []
        for detection in detections:
            tag = Tag(
                tag_id=detection.tag_id,
                center=detection.center,
                corners=detection.corners
            )
            tags.append(tag)
        return tags


class Tag:
    """Simple tag dataclass."""
    
    def __init__(self, tag_id, center, corners):
        self.tag_id = tag_id
        self.center = center
        self.corners = corners


class TagDetectionNode:
    """
    Standalone ROS node for dual-camera AprilTag detection.
    Detects tags in left/right cameras and reports presence as [Bool, Bool].
    """
    
    def __init__(self):
        rospy.init_node('tag_detection_node')
        print("Tag Detection Node Starting")
        
        rospy.sleep(2.0)
        
        # Initialize perception
        self.image_convert = CvBridge()
        self.atp = AprilTagPerception()
        
        # Tag storage for parent and child (index 0=left, 1=right)
        self.parent_tags = [None, None]
        self.child_tags = [None, None]
        
        # Subscribe to parent cameras
        self.sub_parent_caml = rospy.Subscriber(
            "/parent/sensors/caml/compressed",
            CompressedImage,
            lambda msg: self.callback_cam(msg, "parent", 0),
            queue_size=1,
            tcp_nodelay=True
        )

        self.sub_parent_camr = rospy.Subscriber(
            "/parent/sensors/camr/compressed",
            CompressedImage,
            lambda msg: self.callback_cam(msg, "parent", 1),
            queue_size=1,
            tcp_nodelay=True
        )
        
        # Subscribe to child cameras
        self.sub_child_caml = rospy.Subscriber(
            "/child/sensors/caml/compressed",
            CompressedImage,
            lambda msg: self.callback_cam(msg, "child", 0),
            queue_size=1,
            tcp_nodelay=True
        )
        
        self.sub_child_camr = rospy.Subscriber(
            "/child/sensors/camr/compressed",
            CompressedImage,
            lambda msg: self.callback_cam(msg, "child", 1),
            queue_size=1,
            tcp_nodelay=True
        )
        
        # Publishers for tag detection status
        self.parent_tag_pub = rospy.Publisher('/parent/tag_detection', Perception, queue_size=10)
        self.child_tag_pub = rospy.Publisher('/child/tag_detection', Perception, queue_size=10)
        
        rospy.loginfo("Tag detection node initialized for parent and child")
    
    def callback_cam(self, ros_image, agent, index):
        """Process camera image and detect tags for specific agent and camera."""
        try:
            image = self.image_convert.compressed_imgmsg_to_cv2(ros_image, "bgr8")
            tags = self.atp.detect_tags(image)
            
            if agent == "parent":
                self.parent_tags[index] = tags
            elif agent == "child":
                self.child_tags[index] = tags
        except CvBridgeError as e:
            rospy.logwarn(f"CvBridge error on {agent} camera {index}: {e}")
        except Exception as e:
            rospy.logwarn(f"Error processing {agent} camera {index}: {e}")
    
    def run(self):
        """Main loop - check for tags and report detection status."""
        rospy.loginfo("Starting main perception loop...")
        rate = rospy.Rate(100)
        
        while not rospy.is_shutdown():
            try:
                # Check parent tag detection status [left, right]
                parent_left_detected = bool(self.parent_tags[0]) if self.parent_tags[0] is not None else False
                parent_right_detected = bool(self.parent_tags[1]) if self.parent_tags[1] is not None else False
                
                # Check child tag detection status [left, right]
                child_left_detected = bool(self.child_tags[0]) if self.child_tags[0] is not None else False
                child_right_detected = bool(self.child_tags[1]) if self.child_tags[1] is not None else False
                
                # Publish parent [Bool, Bool] message here
                parent_msg = Perception()
                parent_msg.left_detection = parent_left_detected
                parent_msg.right_detection = parent_right_detected
                self.parent_tag_pub.publish(parent_msg)
                
                # Publish child [Bool, Bool] message here
                child_msg = Perception()
                child_msg.left_detection = child_left_detected
                child_msg.right_detection = child_right_detected
                self.child_tag_pub.publish(child_msg)
                
                # Debug output
                if parent_left_detected or parent_right_detected:
                    rospy.loginfo_throttle(1.0, f"Parent Tags - Left: {parent_left_detected}, Right: {parent_right_detected}")
                if child_left_detected or child_right_detected:
                    rospy.loginfo_throttle(1.0, f"Child Tags - Left: {child_left_detected}, Right: {child_right_detected}")
                
                rate.sleep()
            except Exception as e:
                rospy.logerr(f"Error in perception loop: {e}")
                rate.sleep()


if __name__ == "__main__":
    try:
        node = TagDetectionNode()
        node.run()
    except rospy.ROSInterruptException:
        pass
