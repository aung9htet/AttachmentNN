#!/usr/bin/env python3
"""
Child robot node - reactive attachment behavior.
"""
import rospy
from agents import AttachmentAgent

class ChildRobot(AttachmentAgent):
    """
    Child robot with proximity-seeking attachment behavior.
    """

    def __init__(self, model_separation):
        AttachmentAgent.__init__(self, robot_name="child", model_separation=model_separation)

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            self.neck.reset()  # Keep neck neutral
            self.action = getattr(self.action_sub, self.robot_name, 0)
            
            # Update state message
            self.state_msg.pos_x = getattr(self.odom, 'posx', 0.0)
            self.state_msg.pos_y = getattr(self.odom, 'posy', 0.0)

            # Update LED color
            self._update_color()
            
            # proceed on actions, 0 is to approach the miros, 1 is to explore
            if self.action == 0:  # Approach mode
                print("Child Robot Approaching")

                # Detect sound
                self._detect_sound(True)
                
                # Search for the other agent
                self.robot_found = self.approacher._approach()

                if self.robot_found == False:
                    self.audio.play_audio(False)
                    self.state_msg.robot_sound = False
                else:
                    self.audio.play_audio(True)
                    self.state_msg.robot_sound = True

                if self.sound_detected is True:
                    self._wag_tail()
            else:  # Explore mode
                print("Child Robot Exploring")
                self.explorer._explore()
                self.state_msg.robot_sound = False
                self.audio.play_audio(False)
                self.robot_found = False
            
            # Publish state and record position
            self.state.publish(self.state_msg)
            self._update_data()

            rate.sleep()

if __name__ == '__main__':
    try:
        robot = ChildRobot(model_separation=True)
        robot.run()
    except rospy.ROSInterruptException:
        pass
