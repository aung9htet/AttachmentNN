#!/usr/bin/env python3

"""Minimal camera streaming node using OpenCV, with recording to MP4."""

import os
from datetime import datetime

import rospy
import cv2
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from utils import fix_owner

class CameraRecorder(object):
    """Subscribe to a compressed camera topic and display it with OpenCV."""

    def __init__(self):
        self.label = rospy.get_param("~label", "camera")
        self.camera_topic = rospy.get_param(
            "~camera_topic", "/camera/image_raw/compressed"
        )
        self.fps = float(rospy.get_param("~fps", 30.0))
        self.show = bool(rospy.get_param("~show", True))

        # Video writer initialised lazily on first frame
        self._writer = None
        self._video_path = None

        if cv2 is None:
            rospy.logerr(
                "OpenCV (cv2) is not available. Install it to enable camera streaming."
            )

        self._bridge = CvBridge()

        self._sub = rospy.Subscriber(
            self.camera_topic, CompressedImage, self._image_cb, queue_size=1
        )

        rospy.loginfo(
            f"CameraRecorder listening on '{self.camera_topic}' with label '{self.label}'"
        )

    def _image_cb(self, msg):
        if cv2 is None:
            return

        try:
            frame = self._bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn(f"CvBridgeError converting compressed image: {exc}")
            return

        # Initialise writer once we know frame size
        if self._writer is None:
            self._init_writer(frame)

        if self._writer is not None:
            self._writer.write(frame)

        if not self.show:
            return

        cv2.imshow(f"{self.label} camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            rospy.signal_shutdown("User requested shutdown from OpenCV window")

    def _init_writer(self, frame):
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_data_dir = os.path.join(base_dir, "..", "data")
        os.makedirs(root_data_dir, exist_ok=True)

        # Find next experiment_data_{index} folder
        index = 0
        while True:
            exp_dir = os.path.join(root_data_dir, f"experiment_data_{index}")
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
                fix_owner(exp_dir)
                break
            index += 1

        timestamp = datetime.now().strftime("%m-%d-%Y_%H-%M-%S")
        self._video_path = os.path.join(
            exp_dir, "camera_recording.mp4"
        )

        self._writer = cv2.VideoWriter(
            self._video_path, fourcc, self.fps, (width, height)
        )

        if not self._writer.isOpened():
            rospy.logerr(f"Failed to open VideoWriter for {self._video_path}")
            self._writer = None
        else:
            rospy.loginfo(f"Recording camera stream to {self._video_path}")

    def shutdown(self):
        if self._writer is not None:
            self._writer.release()
            fix_owner(self._video_path)
            rospy.loginfo(f"Closed camera recording file {self._video_path}")
        if cv2 is not None:
            cv2.destroyAllWindows()


def main():
    rospy.init_node("camera_recording")
    recorder = CameraRecorder()
    rospy.on_shutdown(recorder.shutdown)
    rospy.loginfo("camera_recording node started; waiting for images...")
    rospy.spin()


if __name__ == "__main__":
    main()
