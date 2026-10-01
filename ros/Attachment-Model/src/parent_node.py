#!/usr/bin/env python3
"""
Parent robot node - caregiving attachment behavior.
"""
import rospy
from agents import AttachmentAgent

class ParentRobot(AttachmentAgent):
    """
        Parent robot with caregiving attachment behavior.
    """
    
    def __init__(self, model_separation):
        AttachmentAgent.__init__(self, robot_name="parent", model_separation=model_separation)

    def run(self):
        """Main control loop."""
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            action = getattr(self.action_sub, 'parent', 0)
            self.action = action  # recorded as parent_states
            
            # Update state message
            self.state_msg.pos_x = getattr(self.odom, 'posx', 0.0)
            self.state_msg.pos_y = getattr(self.odom, 'posy', 0.0)
            
            # Update LED color
            self._update_color()
            
            if action == 0:  # Approach mode
                print("Parent Robot Approaching")
                self.audio.play_audio(True)
                self.state_msg.robot_sound = True
                self.robot_found = True
                self._wag_tail()
            else:  # Explore mode
                print("Parent Robot Exploring")
                self.audio.play_audio(False)
                self.state_msg.robot_sound = False
                self.robot_found = False
            
            # Publish state and record position
            self.state.publish(self.state_msg)
            self._update_data()

            rate.sleep()

if __name__ == '__main__':
    try:
        robot = ParentRobot(model_separation=True)
        robot.run()
    except rospy.ROSInterruptException:
        pass
